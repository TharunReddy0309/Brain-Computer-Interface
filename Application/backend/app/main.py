"""FastAPI application for laptop-only SAM40 EEG evaluation."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4
import zipfile

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from .dataset import read_sam40_mat, read_sam40_zip
from .evaluation import evaluate, evaluate_wesad
from .intent import load_intent_model, predict_intent
from .model import load_model, load_sklearn_model, load_wesad_model
from .wesad import identify_wesad_subjects, read_wesad_zip
from .evaluation import predict_stress


MAX_MODEL_BYTES = 500_000_000
MAX_DATASET_BYTES = 1_000_000_000
MAX_RECORDING_BYTES = 100_000_000
MODELS_DIRECTORY = Path(__file__).resolve().parent.parent / "models"
STRESS_MODEL_DIRECTORY = MODELS_DIRECTORY / "stress"
INTENT_MODEL_DIRECTORY = MODELS_DIRECTORY / "intent"
MODEL_DIRECTORY = STRESS_MODEL_DIRECTORY
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eeg-evaluation")
jobs: dict[str, dict[str, Any]] = {}
queues: dict[str, asyncio.Queue] = {}

app = FastAPI(title="Local EEG Model Evaluator", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "execution": "local"}


@app.get("/api/profiles")
async def profiles() -> dict[str, list[dict[str, Any]]]:
    return {"profiles": [
        {
            "id": "stress",
            "name": "Stress detection",
            "framework": "PyTorch",
            "model_directory": "backend/models/stress/",
            "model_format": ".pth",
            "dataset_upload": "One 32-channel SAM40 .mat recording",
            "input_shape": "32 x 640 EEG windows",
            "classes": {"0": "Relax", "1": "Stress / Arithmetic"},
            "available": True,
        },
        {
            "id": "intent",
            "name": "Intent detection",
            "framework": "PyTorch",
            "model_directory": "backend/models/intent/",
            "model_format": ".pth",
            "dataset_upload": "One .gdf recording",
            "input_shape": "22 EEG channels x 1001 samples per trial",
            "classes": {"0": "Left hand", "1": "Right hand", "2": "Feet", "3": "Tongue"},
            "available": len(list(INTENT_MODEL_DIRECTORY.glob("*.pth"))) == 1,
            "message": "Requires the BCI IV 2a EEGNet checkpoint and matching GDF preprocessing.",
        },
    ]}


def _stress_model_path() -> Path:
    candidates = sorted(STRESS_MODEL_DIRECTORY.glob("*.pth"))
    if not candidates:
        raise HTTPException(
            503,
            "Stress model is missing. Place the standard SAM40 .pth model in backend/models/stress/.",
        )
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates)
        raise HTTPException(
            503,
            f"Expected one stress model in backend/models/stress/, found: {names}.",
        )
    model_path = candidates[0]
    if model_path.stat().st_size == 0:
        raise HTTPException(503, f"The stress model {model_path.name} is empty.")
    if model_path.stat().st_size > MAX_MODEL_BYTES:
        raise HTTPException(413, "The stress model exceeds the 500 MB limit.")
    try:
        load_model(model_path, torch.device("cpu"))
    except (ValueError, OSError) as exc:
        raise HTTPException(422, f"Invalid stress model {model_path.name}: {exc}") from exc
    return model_path


@app.post("/api/inference/stress")
async def stress_inference(recording: UploadFile = File(...)) -> dict[str, Any]:
    if not recording.filename or not recording.filename.lower().endswith(".mat"):
        raise HTTPException(400, "Stress detection requires one .mat EEG recording.")
    contents = await recording.read(MAX_RECORDING_BYTES + 1)
    if len(contents) > MAX_RECORDING_BYTES:
        raise HTTPException(413, "The .mat recording exceeds the 100 MB limit.")
    if not contents:
        raise HTTPException(400, "The .mat recording is empty.")
    model_path = await run_in_threadpool(_stress_model_path)
    try:
        signal = await run_in_threadpool(read_sam40_mat, contents, recording.filename)
        return await run_in_threadpool(predict_stress, model_path, signal, recording.filename)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/inference/intent")
async def intent_inference(recording: UploadFile = File(...)) -> dict[str, Any]:
    if not recording.filename or not recording.filename.lower().endswith(".gdf"):
        raise HTTPException(400, "Intent detection requires one .gdf EEG recording.")
    contents = await recording.read(MAX_RECORDING_BYTES + 1)
    if len(contents) > MAX_RECORDING_BYTES:
        raise HTTPException(413, "The .gdf recording exceeds the 100 MB limit.")
    if not contents:
        raise HTTPException(400, "The .gdf recording is empty.")

    candidates = sorted(INTENT_MODEL_DIRECTORY.glob("*.pth"))
    if not candidates:
        raise HTTPException(
            503,
            "Intent model is missing. Place the BCI IV 2a EEGNet .pth model in backend/models/intent/.",
        )
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates)
        raise HTTPException(
            503,
            f"Expected one intent model in backend/models/intent/, found: {names}.",
        )
    model_path = candidates[0]
    if model_path.stat().st_size == 0:
        raise HTTPException(503, f"The intent model {model_path.name} is empty.")
    if model_path.stat().st_size > MAX_MODEL_BYTES:
        raise HTTPException(413, "The intent model exceeds the 500 MB limit.")
    try:
        await run_in_threadpool(load_intent_model, model_path, torch.device("cpu"))
        return await run_in_threadpool(
            predict_intent, model_path, contents, recording.filename
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _configured_model(dataset_kind: str) -> Path:
    candidates = sorted(
        path for path in MODEL_DIRECTORY.iterdir()
        if path.is_file() and path.suffix.lower() in {".pth", ".pkl"}
    ) if MODEL_DIRECTORY.is_dir() else []
    if not candidates:
        raise HTTPException(
            503,
            "No model found in backend/model/.",
        )
    compatible: list[Path] = []
    rejected: list[str] = []
    for model_path in candidates:
        expected_suffix = ".pkl" if dataset_kind == "wesad" else ".pth"
        if model_path.suffix.lower() != expected_suffix:
            continue
        if model_path.stat().st_size == 0:
            rejected.append(f"{model_path.name}: file is empty")
            continue
        if model_path.stat().st_size > MAX_MODEL_BYTES:
            rejected.append(f"{model_path.name}: exceeds the 500 MB limit")
            continue
        try:
            if dataset_kind == "wesad":
                load_wesad_model(model_path)
            elif model_path.suffix.lower() == ".pkl":
                load_sklearn_model(model_path)
            else:
                load_model(model_path, torch.device("cpu"))
            compatible.append(model_path)
        except (ValueError, OSError) as exc:
            rejected.append(f"{model_path.name}: {exc}")

    if not compatible:
        details = " ".join(rejected)
        raise HTTPException(
            422,
            f"No model compatible with the {dataset_kind} dataset found in backend/model/. "
            f"{'Add a WESAD feature model as .pkl.' if dataset_kind == 'wesad' else 'Add a SAM40 EEG model as .pth.'} {details}",
        )
    if len(compatible) > 1:
        names = ", ".join(path.name for path in compatible)
        raise HTTPException(
            503,
            f"Multiple compatible models found in backend/model/: {names}. Keep one model for this dataset.",
        )
    return compatible[0]


def _dataset_kind(contents: bytes) -> str:
    if identify_wesad_subjects(contents):
        try:
            with zipfile.ZipFile(BytesIO(contents)) as archive:
                has_mat = any(
                    not item.is_dir() and item.filename.lower().endswith(".mat")
                    for item in archive.infolist()
                )
        except zipfile.BadZipFile as exc:
            raise ValueError("Dataset upload must be a valid ZIP archive.") from exc
        if has_mat:
            raise ValueError("Upload either a WESAD subject ZIP or a SAM40 EEG ZIP, not both.")
        return "wesad"
    try:
        with zipfile.ZipFile(BytesIO(contents)) as archive:
            has_mat = any(
                not item.is_dir() and item.filename.lower().endswith(".mat")
                for item in archive.infolist()
            )
    except zipfile.BadZipFile as exc:
        raise ValueError("Dataset upload must be a valid ZIP archive.") from exc
    if has_mat:
        return "sam40"
    raise ValueError(
        "ZIP contents are not recognized. For SAM40, include labelled .mat recordings. "
        "For WESAD, include raw subject files such as S2/S2.pkl."
    )


async def _run_job(job_id: str, dataset_zip: bytes, dataset_kind: str, model_path: Path) -> None:
    queue = queues[job_id]
    jobs[job_id]["status"] = "running"
    loop = asyncio.get_running_loop()

    def report(stage: str, percent: int, message: str) -> None:
        update = {"job_id": job_id, "status": "running", "stage": stage, "percent": percent, "message": message}
        loop.call_soon_threadsafe(queue.put_nowait, update)

    try:
        if dataset_kind == "wesad":
            report("validation", 5, "Loading WESAD wrist signals and extracting model features")
            records = await run_in_threadpool(read_wesad_zip, dataset_zip)
            result = await loop.run_in_executor(executor, evaluate_wesad, model_path, records, report)
        else:
            report("validation", 5, "Validating SAM40 dataset ZIP and recording labels")
            records = await run_in_threadpool(read_sam40_zip, dataset_zip)
            result = await loop.run_in_executor(executor, evaluate, model_path, records, report)
        jobs[job_id].update(status="complete", result=result)
        report("complete", 100, "Evaluation complete")
    except Exception as exc:
        jobs[job_id].update(status="failed", error=str(exc))
        report("failed", 100, str(exc))
    finally:
        # Input bytes live only in this task; no upload files are persisted.
        dataset_zip = b""


@app.post("/api/evaluations", status_code=202)
async def create_evaluation(
    background_tasks: BackgroundTasks,
    dataset: UploadFile = File(...),
) -> dict[str, str]:
    if not dataset.filename or not dataset.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Dataset upload must be a .zip file.")
    data = await dataset.read(MAX_DATASET_BYTES + 1)
    if len(data) > MAX_DATASET_BYTES:
        raise HTTPException(413, "Dataset exceeds the 1 GB upload limit.")
    if not data:
        raise HTTPException(400, "The dataset upload must contain data.")
    try:
        dataset_kind = await run_in_threadpool(_dataset_kind, data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    model_path = await run_in_threadpool(_configured_model, dataset_kind)
    job_id = str(uuid4())
    jobs[job_id] = {"status": "queued"}
    queues[job_id] = asyncio.Queue()
    background_tasks.add_task(_run_job, job_id, data, dataset_kind, model_path)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/evaluations/{job_id}")
async def evaluation_status(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Evaluation job not found.")
    return {"job_id": job_id, "status": job["status"], **({"error": job["error"]} if "error" in job else {})}


@app.get("/api/evaluations/{job_id}/result")
async def evaluation_result(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Evaluation job not found.")
    if job["status"] == "failed":
        raise HTTPException(422, job["error"])
    if job["status"] != "complete":
        raise HTTPException(202, "Evaluation is still running.")
    return job["result"]


@app.get("/api/evaluations/{job_id}/download/metrics")
async def download_metrics(job_id: str) -> Response:
    job = jobs.get(job_id)
    if job is None or job.get("status") != "complete":
        raise HTTPException(404, "Completed evaluation not found.")
    result = job["result"]
    report = {key: value for key, value in result.items() if key != "predictions"}
    return Response(json.dumps(report, indent=2), media_type="application/json",
                    headers={"Content-Disposition": "attachment; filename=metrics.json"})


@app.get("/api/evaluations/{job_id}/download/predictions")
async def download_predictions(job_id: str) -> Response:
    job = jobs.get(job_id)
    if job is None or job.get("status") != "complete":
        raise HTTPException(404, "Completed evaluation not found.")
    output = StringIO()
    rows = job["result"]["predictions"]
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return Response(output.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=recording_predictions.csv"})


@app.delete("/api/evaluations/{job_id}", status_code=204)
async def delete_evaluation(job_id: str) -> Response:
    if job_id not in jobs:
        raise HTTPException(404, "Evaluation job not found.")
    jobs.pop(job_id, None)
    queues.pop(job_id, None)
    return Response(status_code=204)


@app.websocket("/api/evaluations/{job_id}/progress")
async def evaluation_progress(websocket: WebSocket, job_id: str) -> None:
    if job_id not in jobs:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    queue = queues[job_id]
    try:
        await websocket.send_json({"job_id": job_id, "status": jobs[job_id]["status"], "percent": 0})
        while True:
            event = await queue.get()
            await websocket.send_json(event)
            if event["status"] in {"complete", "failed"}:
                break
    except WebSocketDisconnect:
        return
