# Streamlit Demo — Deployment Guide

`app.py` in this folder is the interface built by Mariam's colleague. It is a
**self-contained script** — it does NOT import from `src/`, it has its own
copies of `generate_ela_image`, `preprocess_image_rgb/ela`, and model loading.
That's fine (nothing to wire up), but it does mean: if `src/model_architecture.py`
or the ELA parameters ever change, this file must be updated separately.

It loads `best_dual_model_robust_head.keras` (the Methodology 1 / `main`
branch model — the best-performing one) and expects it at
`models/best_dual_model_robust_head.keras`, downloading it from Google Drive
on first run via `gdown` if it isn't already there.

## 1. Get YOUR OWN Google Drive file ID

The file already in your Drive (`best_dual_model_robust_head.keras`) needs to
be shared and its ID copied:

1. Open Google Drive, right-click the file → **Share**.
2. Under "General access", change to **Anyone with the link**.
3. Copy the link. It looks like:
   ```
   https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz123456/view?usp=sharing
                                     └──────────── this is the file ID ────────────┘
   ```
4. Copy just the ID part (between `/d/` and `/view`).

## 2. Put your file ID into app.py

Open `app.py` and find this line near the top (search for `GDRIVE_FILE_ID`):

```python
GDRIVE_FILE_ID = 'PUT_YOUR_OWN_GOOGLE_DRIVE_FILE_ID_HERE'
```

Replace the placeholder with the ID you copied in step 1, e.g.:

```python
GDRIVE_FILE_ID = '1AbCdEfGhIjKlMnOpQrStUvWxYz123456'
```

This is the **only required code change**. Everything else in `app.py` — the
UI, the ELA generation, the RTL styling, the sidebar — already works and
already credits Mariam Abd Alaal.

## 3. Two small things worth deciding (not required, just FYI)

- **Decision threshold**: this app uses `threshold = 0.45` (image flagged as
  "forged" if the model's score is ≥ 0.45), while every notebook's own
  evaluation uses the standard `0.5`. This isn't derived from anything in the
  original notebooks — it looks like a manual choice. Keep it or change the
  `threshold = 0.45` line to `0.5` to match the reported metrics exactly.
- **Sidebar slider**: there's a "Decision threshold" slider in the sidebar
  (bottom of `main()`) that is currently **not wired to the actual decision
  logic** — moving it has no effect. It's harmless, just cosmetic. Ask if you
  want it made functional.

## 4. Push to your `main` branch

```bash
git checkout main
# copy this whole streamlit_app/ folder into the repo root, alongside src/, notebooks/, etc.
git add streamlit_app/
git commit -m "Add Streamlit demo app"
git push
```

Final layout on `main`:

```
main/  (repo root)
├── src/
├── notebooks/
├── streamlit_app/
│   ├── app.py
│   ├── requirements.txt
│   ├── packages.txt
│   ├── .python-version
│   └── .streamlit/config.toml
├── README.md
└── ...
```

## 5. Deploy under YOUR OWN Streamlit account

This is the actual fix for the "someone else's account shows up" problem —
it happens because the app is currently deployed under your **friend's**
Streamlit Community Cloud account, not because of anything in the code.

1. Go to **share.streamlit.io** and sign in with **your own** GitHub account
   (`Mariam6600`) — not your friend's.
2. Click **New app**.
3. Repository: `Mariam6600/Digital-image-forgery-detection`, branch: `main`,
   main file path: `streamlit_app/app.py`.
4. Click **Deploy**.

Because the whole chain (GitHub repo → Streamlit account) is now yours, there
is no more "friend's account" anywhere in it.

## 6. Test locally first (optional but recommended)

```bash
cd Digital-image-forgery-detection   # main branch, repo root
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

## Notes

- Make sure the Drive file's sharing is **"Anyone with the link"** —
  "Restricted" will fail to download on Streamlit Cloud (no logged-in Google
  account there).
- `models/` is git-ignored on purpose (the `.keras` file is too large for a
  normal git repo); the download-on-first-run pattern is the standard way to
  serve large model files from a small Streamlit Cloud deployment.
- If you retrain and get a new best checkpoint, re-upload it to the same
  Drive file (same share link/ID) — next redeploy will pick it up. If you
  upload as a *new* Drive file, update `GDRIVE_FILE_ID` in `app.py` again.
