from __future__ import annotations

import json
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import joblib
import numpy as np
import pandas as pd
import requests
import tldextract
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.service_account import Credentials
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT_DIR / "models")))
MODEL_PATH = MODEL_DIR / "pipeline.joblib"

DEFAULT_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
DEFAULT_SOURCE = os.getenv("DEFAULT_SOURCE", "github-pages")
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]

EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=False)
_NEUTRAL_SUBS = {"www", "www2", "m", "mobile", "wap", "docs", "api", "cdn", "static", "dev"}

_BUNDLE: dict[str, Any] | None = None


class CheckRequest(BaseModel):
    url: str
    source: str | None = None


def _load_bundle() -> dict[str, Any]:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict):
            _BUNDLE = artifact
        else:
            _BUNDLE = {"pipe": artifact}
        _BUNDLE["_path"] = str(MODEL_PATH)
        return _BUNDLE

    raise FileNotFoundError(f"No model artifact found at {MODEL_PATH}. Train the model first.")


def _normalize_url(url: str) -> str:
    value = str(url).strip()
    if not value.startswith(("http://", "https://", "ftp://")):
        value = "http://" + value
    try:
        ext = EXTRACT(value)
        if ext.subdomain.lower() in _NEUTRAL_SUBS and ext.domain and ext.suffix:
            value = value.replace(f"{ext.subdomain}.{ext.domain}.{ext.suffix}", f"{ext.domain}.{ext.suffix}", 1)
    except Exception:
        pass
    return value


def _extract_features(url: str, top_domains: set[str] | None = None) -> dict[str, Any]:
    url = _normalize_url(url)
    feat: dict[str, Any] = {
        "url_length": len(url),
        "path_length": 0,
        "query_length": 0,
        "hostname_length": 0,
        "has_port": 0,
        "n_subdomains": 0,
    }

    try:
        parsed = urlparse(url)
        feat["path_length"] = len(parsed.path)
        feat["query_length"] = len(parsed.query)
        feat["hostname_length"] = len(parsed.hostname or "")
        feat["has_port"] = int(bool(parsed.port))
        hostname = parsed.hostname or ""
        feat["n_subdomains"] = max(len(hostname.split(".")) - 2, 0)
    except Exception:
        pass

    feat["count_dots"] = url.count(".")
    feat["count_hyphens"] = url.count("-")
    feat["count_underscores"] = url.count("_")
    feat["count_slash"] = url.count("/")
    feat["count_at"] = url.count("@")
    feat["count_question"] = url.count("?")
    feat["count_equals"] = url.count("=")
    feat["count_ampersand"] = url.count("&")
    feat["count_percent"] = url.count("%")
    feat["count_digits"] = sum(c.isdigit() for c in url)
    feat["count_params"] = len(re.findall(r"[?&][^=&]+=", url))

    n = max(len(url), 1)
    feat["digit_ratio"] = feat["count_digits"] / n
    feat["special_ratio"] = (feat["count_dots"] + feat["count_hyphens"] + feat["count_at"] + feat["count_percent"]) / n

    feat["has_ip"] = int(bool(re.search(r"(?:^|//)\d{1,3}(\.\d{1,3}){3}", url)))
    feat["has_at"] = int("@" in url)
    feat["has_double_slash"] = int("//" in url[8:])
    feat["has_hex"] = int("%" in url)
    feat["has_shortener"] = int(bool(re.search(r"bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co|is\.gd|buff\.ly|adf\.ly|rebrand\.ly|cutt\.ly|v\.gd", url, re.I)))
    feat["https_in_path"] = int("https" in url[8:].lower())
    feat["has_login_kw"] = int(bool(re.search(r"login|signin|sign-in|verify|secure|update|account|banking|paypal|ebay|amazon|apple|microsoft|facebook|webscr|iniciar-sesion|contrasena|clave|acceso|validar|confirmar|netflix|spotify|mercadolibre|bancolombia|davivienda|nequi|instagram|whatsapp|telegram|coinbase|binance|wallet|crypto", url, re.I)))
    feat["suspicious_tld"] = int(bool(re.search(r"\.(tk|ml|ga|cf|gq|xyz|top|club|work|date|download|racing|science|bid|win|stream|loan|review|trade|click|buzz|cyou|cfd|sbs|hair|skin|beauty|mom|bond|lol|monster|observer|quest|rest|zip|mov)(?:/|$)", url, re.I)))
    feat["has_punycode"] = int("xn--" in url.lower())
    feat["max_token_len"] = len(max(re.split(r"[.\-_/?=&]", url) or [""], key=len))
    feat["longest_digit_seq"] = len(max(re.findall(r"\d+", url) or [""], key=len))

    try:
        ext = EXTRACT(url)
        feat["domain_len"] = len(ext.domain)
        feat["tld_len"] = len(ext.suffix)
        feat["subdomain_len"] = len(ext.subdomain)
        feat["n_dots_sub"] = ext.subdomain.count(".") if ext.subdomain else 0
        feat["domain_digit"] = int(any(c.isdigit() for c in ext.domain))
        feat["domain_hyphen"] = int("-" in ext.domain)

        domain_name = ext.domain.lower()
        vowels = set("aeiou")
        if domain_name:
            consonant_count = sum(1 for c in domain_name if c.isalpha() and c not in vowels)
            feat["consonant_ratio"] = consonant_count / max(len(domain_name), 1)
            max_consec = 0
            cur = 0
            for c in domain_name:
                if c.isalpha() and c not in vowels:
                    cur += 1
                    max_consec = max(max_consec, cur)
                else:
                    cur = 0
            feat["max_consonant_seq"] = max_consec
        else:
            feat["consonant_ratio"] = 0.0
            feat["max_consonant_seq"] = 0
    except Exception:
        feat["domain_len"] = 0
        feat["tld_len"] = 0
        feat["subdomain_len"] = 0
        feat["n_dots_sub"] = 0
        feat["domain_digit"] = 0
        feat["domain_hyphen"] = 0
        feat["consonant_ratio"] = 0.0
        feat["max_consonant_seq"] = 0

    try:
        host = url.split("//")[-1].split("/")[0]
        if host:
            probs = [v / len(host) for v in Counter(host).values()]
            feat["hostname_entropy"] = float(-sum(p * np.log2(p) for p in probs if p > 0))
        else:
            feat["hostname_entropy"] = 0.0
    except Exception:
        feat["hostname_entropy"] = 0.0

    try:
        ext2 = EXTRACT(url)
        root = f"{ext2.domain}.{ext2.suffix}".lower()
        feat["domain_in_top10k"] = int(bool(top_domains and root in top_domains))
    except Exception:
        feat["domain_in_top10k"] = 0

    return feat


def _build_flags(features: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if features["has_ip"]:
        flags.append("IP in URL")
    if features["has_at"]:
        flags.append("@ symbol")
    if features["suspicious_tld"]:
        flags.append("suspicious TLD")
    if features["has_login_kw"]:
        flags.append("login/account keywords")
    if features["has_shortener"]:
        flags.append("URL shortener")
    if features["count_dots"] > 5:
        flags.append(f"{features['count_dots']} dots")
    if features["has_double_slash"]:
        flags.append("double-slash in path")
    if features["hostname_entropy"] > 4.0:
        flags.append(f"high entropy ({features['hostname_entropy']:.2f})")
    if features["has_punycode"]:
        flags.append("punycode domain")
    if features["domain_digit"]:
        flags.append("digit in domain name")
    return flags


def _append_sheet_row(url: str, verdict: str, probability: float, flags: list[str], request_id: str, source: str) -> tuple[bool, str | None]:
    spreadsheet_id = os.getenv("GSHEETS_SPREADSHEET_ID", "").strip()
    if not spreadsheet_id:
        return False, "Sheets logging disabled"

    try:
        creds_raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
        if not creds_raw:
            return False, "GOOGLE_SERVICE_ACCOUNT_JSON not configured"
        creds = Credentials.from_service_account_info(
            json.loads(creds_raw), scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        if not creds.valid or creds.expired:
            creds.refresh(GoogleAuthRequest())

        sheet_name = os.getenv("GSHEETS_WORKSHEET_NAME", "detections")
        timeout_s = float(os.getenv("GSHEETS_TIMEOUT_SECONDS", "4"))
        range_name = quote(f"{sheet_name}!A:G", safe="!:")
        endpoint = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}:append"
        row = [
            datetime.now(timezone.utc).isoformat(),
            url,
            verdict,
            round(float(probability), 6),
            json.dumps(flags),
            request_id,
            source,
        ]
        response = requests.post(
            endpoint,
            params={"valueInputOption": "USER_ENTERED", "insertDataOption": "INSERT_ROWS"},
            json={"values": [row]},
            headers={"Authorization": f"Bearer {creds.token}"},
            timeout=timeout_s,
        )
        response.raise_for_status()
        return True, None
    except Exception as exc:
        return False, str(exc)


def _explain(url: str) -> dict[str, Any]:
    bundle = _load_bundle()
    pipe = bundle["pipe"]
    features_order = bundle.get("features")
    top_domains = bundle.get("tranco_top10k", set())
    threshold = float(bundle.get("threshold", DEFAULT_THRESHOLD))

    feats = _extract_features(url, top_domains)

    try:
        ext = EXTRACT(url)
        root = f"{ext.domain}.{ext.suffix}".lower()
        if top_domains and root in top_domains:
            if feats.get("has_shortener", 0) == 0 and feats.get("has_login_kw", 0) == 0:
                return {
                    "url": url,
                    "probability": 0.01,
                    "phishing": False,
                    "verdict": "LEGITIMA",
                    "flags": [],
                    "note": "Dominio en Top-10k - override aplicado",
                }
    except Exception:
        pass
    frame = pd.DataFrame([feats])
    if features_order:
        frame = frame.reindex(columns=features_order, fill_value=0)

    probability = float(pipe.predict_proba(frame)[:, 1][0])
    phishing = bool(probability >= threshold)
    return {
        "url": url,
        "probability": round(probability, 4),
        "phishing": phishing,
        "verdict": "PHISHING" if phishing else "LEGITIMA",
        "flags": _build_flags(feats),
    }


app = FastAPI(title="Phishing Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    bundle = _load_bundle()
    return {
        "ok": True,
        "model_path": bundle.get("_path"),
    }


@app.get("/api/check")
def check_get(url: str = Query(...), source: str = Query(DEFAULT_SOURCE)) -> dict[str, Any]:
    return _check(url=url, source=source)


@app.post("/api/check")
def check_post(payload: CheckRequest) -> dict[str, Any]:
    source = payload.source or DEFAULT_SOURCE
    return _check(url=payload.url, source=source)


def _check(url: str, source: str) -> dict[str, Any]:
    request_id = uuid.uuid4().hex
    result = _explain(url)
    logged, log_error = _append_sheet_row(
        url=url,
        verdict=result["verdict"],
        probability=result["probability"],
        flags=result.get("flags", []),
        request_id=request_id,
        source=source,
    )

    result["request_id"] = request_id
    result["source"] = source
    result["logging"] = {
        "status": "ok" if logged else "warning",
        "message": None if logged else log_error,
    }
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
