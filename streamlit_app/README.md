# Streamlit Demo

Live app: **https://digital-image-forgery-detection-hpbpmrn8y9elvxtaaab2dt.streamlit.app/**

`app.py` is a self-contained Streamlit script (built by a colleague, adapted
for this repo) — it does not import from `src/`, it has its own copies of
`generate_ela_image`, image preprocessing, and model loading. If
`src/model_architecture.py` or the ELA parameters in `src/config.py` ever
change, this file needs to be updated separately to match.

## How it works

1. On first run, `load_my_model()` downloads `best_dual_model_robust_head.keras`
   (the Methodology 1 / `main` model) from Google Drive via `gdown`, using the
   file ID hardcoded in `GDRIVE_FILE_ID` near the top of `app.py`, and caches
   it at `models/best_dual_model_robust_head.keras`.
2. The uploaded image is converted to an ELA map (same quality=90, scale=15
   as `src/config.py`) and, together with the original RGB image, passed to
   the model.
3. A prediction is "forged" if the model's score is ≥ the value chosen with
   the sidebar's decision-threshold slider (default 0.45).

## Redeploying elsewhere / after retraining

- **New best checkpoint**: re-upload it to the same Google Drive file (same
  share link), keeping the sharing setting as **"Anyone with the link"** —
  the next redeploy/reboot will pick it up automatically. If you upload it as
  a *new* Drive file instead, update `GDRIVE_FILE_ID` in `app.py` to the new
  file's ID.
- **Deploying to a different Streamlit account**: share.streamlit.io → New app
  → repository `Mariam6600/Digital-image-forgery-detection`, branch `main`,
  file path `streamlit_app/app.py`.
- **Running locally**:
  ```bash
  cd Digital-image-forgery-detection   # main branch, repo root
  pip install -r streamlit_app/requirements.txt
  streamlit run streamlit_app/app.py
  ```

## Notes

- `models/` is git-ignored on purpose (the `.keras` file is too large for a
  normal git repo); download-on-first-run is the standard way to serve large
  model files from a small Streamlit Cloud deployment.
- `streamlit_app/requirements.txt` was updated after the first deployment
  (newer `tensorflow`/`streamlit`/etc. versions) because the pinned original
  versions had no wheels for the Python version Streamlit Cloud provisioned.
  If a future redeploy fails with a similar dependency error, check
  Streamlit Cloud's current default Python version against the pins here.
