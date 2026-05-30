# Vercel environment variables (minimal)

Backend file: `api/index.py` (FastAPI app).

## Required for model loading

- `MODEL_PATH` (optional): explicit full path to a `.joblib` artifact.
- `MODEL_FILENAME` (optional): default `pipeline_final.joblib`.
- `MODEL_FALLBACK_FILENAME` (optional): default `pipeline_fixed.joblib`.
- `MODEL_DIR` (optional): default `models`.
- `DECISION_THRESHOLD` (optional): fallback threshold if artifact does not include one, default `0.5`.

## CORS and request metadata

- `ALLOWED_ORIGINS`: comma-separated origins. Default `*`.
- `DEFAULT_SOURCE`: default request source label, default `github-pages`.

## Google Sheets logging (best-effort)

- `GSHEETS_SPREADSHEET_ID`
- `GSHEETS_WORKSHEET_NAME` (optional, default `detections`)
- `GOOGLE_SERVICE_ACCOUNT_JSON`: full service-account JSON as one env var
- `GSHEETS_TIMEOUT_SECONDS` (optional, default `4`)

## Behavior

- The API always classifies first.
- Sheets append is best-effort; if logging fails, prediction still returns.
