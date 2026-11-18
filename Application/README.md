# EEG model evaluation App

The React app provides a local single-recording inference screen with two
selectable models:

- Stress: upload one SAM40 `.mat` file for the ShallowConvNet.
- Intent: upload one BCI IV 2a `.gdf` file for the EEGNet motor-imagery model.

The Intent result shows four-class scores and individual trial predictions. The
displayed model scores are not validation accuracy. See
[`../backend/README.md`](../backend/README.md) for supported model files,
recording details, preprocessing, and API routes.

## Run locally

1. Start the API from `backend`:

   ```powershell
   python -m pip install -r requirements.txt
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

2. Start the UI from `frontend`:

   ```powershell
   npm install
   npm run dev
   ```

The frontend connects to `http://127.0.0.1:8000` by default. Copy `.env.example`
to `.env.local` to use a different API URL.

## Source layout

- `src/pages/` contains the overview and evaluation pages.
- `src/components/` contains the upload and task-specific results UI.
- `src/services/evaluationApi.js` owns communication with the FastAPI endpoints.
- `src/evaluation.css` styles the evaluation screen; `src/styles.css` styles the overview.
- `public/assets/evaluate/` contains the evaluation screen's local assets.
