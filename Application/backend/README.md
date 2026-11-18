# Local EEG inference backend

The backend runs one local model per uploaded recording. Keep exactly one `.pth`
checkpoint in the folder for its task:

- Stress: `backend/models/stress/` — SAM40 ShallowConvNet.
- Intent: `backend/models/intent/` — BCI IV 2a EEGNet checkpoint.

The app does not upload recordings to a remote service. Intent GDF files are
temporarily written to a private system temporary directory for MNE to read and
are removed when processing finishes.

## Start

From this `backend` directory:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the API interface.

## Models and recording formats

### Stress

- Checkpoint folder: `models/stress/` (one `.pth` file).
- Upload: one SAM40 `.mat` file.
- Expected array: 32 channels by samples, in a `Clean_data`, `Data`, or `data`
  variable.
- The model scores 640-sample windows. Its output scores are predictions, not
  measured accuracy or clinical confidence.

### Intent

- Checkpoint folder: `models/intent/` (one `.pth` file).
- Upload: one BCI Competition IV 2a `.gdf` file sampled at 250 Hz.
- Expected signals: 22 EEG channels plus optional `EOG-left`, `EOG-central`,
  and `EOG-right` channels.
- The GDF must contain class cues 769-772 or unlabelled evaluation cues 783.
  Complete four-second epochs are classified as Left hand, Right hand, Feet, or
  Tongue.
- Preprocessing follows the checkpoint notebook: drop the three EOG channels,
  apply a 4-40 Hz IIR bandpass and 50 Hz notch filter, epoch from 0 to 4 seconds
  at each cue, then z-score the uploaded recording's trials together.

The intent loader validates that the checkpoint is a tensor-only state
dictionary matching the EEGNet architecture and input dimensions found in the
training notebook. It will reject a different or incompatible `.pth` file.

## Endpoints

- `GET /api/health`
- `GET /api/profiles`
- `POST /api/inference/stress` — multipart field `recording`, one `.mat`.
- `POST /api/inference/intent` — multipart field `recording`, one `.gdf`.

Older ZIP-based dataset evaluation endpoints remain available for the previous
SAM40 and WESAD evaluation workflow. The current UI uses the single-recording
inference endpoints above.
