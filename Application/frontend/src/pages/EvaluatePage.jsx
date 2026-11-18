import { useEffect, useRef, useState } from "react";
import EvaluationResults from "../components/EvaluationResults.jsx";
import FileDropzone from "../components/FileDropzone.jsx";
import SiteHeader from "../components/SiteHeader.jsx";
import {
  API_BASE_URL,
  checkBackend,
  getApiAddress,
  startStressInference,
  startIntentInference,
} from "../services/evaluationApi.js";

const MAX_RECORDING_BYTES = 100 * 1024 ** 2;

function WorkflowSteps({ step }) {
  const steps = [
    { id: "upload", label: "Upload recording" },
    { id: "running", label: "Run model" },
    { id: "results", label: "Prediction" },
  ];

  return (
    <ol className="workflow-steps">
      {steps.map(({ id, label }, index) => {
        const current = step === id || (step === "error" && index === 1);
        const complete = steps.findIndex((item) => item.id === step) > index;

        return (
          <li className={current ? "current" : complete ? "complete" : ""} key={id}>
            <span>{complete ? "✓" : index + 1}</span>
            {label}
          </li>
        );
      })}
    </ol>
  );
}

export default function EvaluatePage() {
  const [task, setTask] = useState("stress");
  const [recording, setRecording] = useState(null);
  const [fieldError, setFieldError] = useState("");
  const [backendOnline, setBackendOnline] = useState(false);
  const [checkingBackend, setCheckingBackend] = useState(true);
  const [phase, setPhase] = useState("upload");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [progressMessage, setProgressMessage] = useState("");
  const activeRequest = useRef(null);

  useEffect(() => {
    document.title = "EEG Model Evaluation";
    const controller = new AbortController();

    function refreshStatus() {
      checkBackend(controller.signal)
        .then(() => {
          setBackendOnline(true);
          setCheckingBackend(false);
        })
        .catch(() => {
          if (!controller.signal.aborted) {
            setBackendOnline(false);
            setCheckingBackend(false);
          }
        });
    }

    refreshStatus();
    const interval = setInterval(refreshStatus, 5000);

    return () => {
      clearInterval(interval);
      controller.abort();
      activeRequest.current?.abort();
    };
  }, []);

  function selectFile(file) {
    if (!file) return;
    let message = "";
    const extension = task === "stress" ? ".mat" : ".gdf";

    if (!file.name.toLowerCase().endsWith(extension)) {
      message = task === "stress"
        ? "Choose one SAM40 .mat recording."
        : "Choose one BCI IV 2a .gdf recording.";
    } else if (file.size === 0) {
      message = "The selected file is empty.";
    } else if (file.size > MAX_RECORDING_BYTES) {
      message = task === "stress"
        ? "The .mat recording must be 100 MB or smaller."
        : "The recording must be 100 MB or smaller.";
    }

    setFieldError(message);
    if (message) {
      setRecording(null);
      return;
    }

    setRecording(file);
    setError("");
  }

  async function runInference(event) {
    event.preventDefault();
    if (!recording || busy) return;

    const controller = new AbortController();
    activeRequest.current = controller;
    setBusy(true);
    setPhase("running");
    setError("");
    setResult(null);
    setProgressMessage(
      task === "stress" ? "Classifying SAM40 stress recording…" : "Running local intent inference…",
    );

    try {
      if (task === "stress") {
        setResult(await startStressInference(recording, controller.signal));
      } else {
        const prediction = await startIntentInference(recording, controller.signal);
        setResult(prediction);
      }
      setPhase("results");
    } catch (requestError) {
      if (!controller.signal.aborted) {
        setPhase("error");
        setError(requestError.message || "The evaluation could not be completed.");
      }
    } finally {
      if (activeRequest.current === controller) activeRequest.current = null;
      if (!controller.signal.aborted) setBusy(false);
    }
  }

  function resetEvaluation() {
    setPhase("upload");
    setError("");
    setResult(null);
    setFieldError("");
    setProgressMessage("");
  }

  function selectTask(nextTask) {
    if (busy || task === nextTask) return;
    setTask(nextTask);
    setRecording(null);
    setFieldError("");
    setError("");
    setResult(null);
    setPhase("upload");
  }

  const isStress = task === "stress";

  return (
    <div className="evaluation-page dataset-evaluation">
      <SiteHeader activePage={2} />
      <main>
        <section className="evaluate-hero">
          <span className="eyebrow evaluate-pill"><i /> Local EEG inference</span>
          <h1>
            Evaluate neural signals with <em>confidence.</em>
          </h1>
          <p>
            Classify one <code>SAM40 .mat</code> stress recording or one <code>BCI IV 2a .gdf</code>{" "}
            motor-imagery recording with a model running locally on this machine.
          </p>
          <WorkflowSteps step={phase} />
        </section>

        <section className="evaluation-card" aria-labelledby="evaluation-title">
          {result ? (
            <EvaluationResults
              result={result}
              onReset={resetEvaluation}
              onDownload={() => window.open(
                `${API_BASE_URL}/api/evaluations/${result.job_id}/download/metrics`,
                "_blank",
                "noopener",
              )}
            />
          ) : (
            <form onSubmit={runInference}>
              <div className="evaluation-card-heading">
                <div>
                  <h2 id="evaluation-title">Run local inference</h2>
                  <p>
                    {isStress
                      ? "Upload one SAM40 EEG recording (.mat) for stress classification"
                      : "Upload one motor-imagery recording for inference with the configured model"}
                  </p>
                </div>
                <div className="card-steps" aria-label="Evaluation steps">
                  <span className={phase === "upload" ? "active" : ""}>1 Upload recording</span>
                  <i>—</i>
                  <span className={phase === "running" ? "active" : ""}>2 Classify</span>
                  <i>—</i>
                  <span className={phase === "results" ? "active" : ""}>3 Prediction</span>
                </div>
              </div>

              <h3 className="model-selector-title">Select decoding paradigm &amp; model architecture</h3>
              <div className="model-choice-grid">
                <button
                  className={`model-choice ${isStress ? "selected" : ""}`}
                  type="button"
                  aria-pressed={isStress}
                  disabled={busy}
                  onClick={() => selectTask("stress")}
                >
                  <span className="model-choice-topline">
                    <span className="model-choice-eyebrow"><i /> Passive BCI: affective state</span>
                    <span className="model-choice-status">{isStress ? "Selected" : "Select"}</span>
                  </span>
                  <strong>SAM40 ShallowConvNet</strong>
                  <span className="model-choice-description">
                    32 EEG channels · 640-sample windows · Frequency-band spatial filters
                  </span>
                  <span className="model-choice-meta">
                    <span>Target: Binary Stress</span>
                    <span>Relaxed (0) vs Stress (1)</span>
                  </span>
                </button>
                <button
                  className={`model-choice ${!isStress ? "selected" : ""}`}
                  type="button"
                  aria-pressed={!isStress}
                  disabled={busy}
                  onClick={() => selectTask("intent")}
                >
                  <span className="model-choice-topline">
                    <span className="model-choice-eyebrow"><i /> Active BCI: motor imagery</span>
                    <span className="model-choice-status">Select</span>
                  </span>
                  <strong>EEGNet Compact CNN</strong>
                  <span className="model-choice-description">
                    22 EEG channels · 250 Hz · Depthwise temporal convolutions
                  </span>
                  <span className="model-choice-meta">
                    <span>Target: 4-Class Kinematics</span>
                    <span>Left, Right, Foot, Tongue</span>
                  </span>
                </button>
              </div>

              <div className="upload-grid">
                <FileDropzone
                  file={recording}
                  error={fieldError}
                  disabled={busy}
                  onSelect={selectFile}
                  title={isStress ? "SAM40 stress recording" : "Intent EEG recording"}
                  extension={isStress ? ".mat" : ".gdf"}
                  description={isStress ? "Drop one SAM40 MAT recording here" : "Drop one BCI IV 2a GDF recording here"}
                  format={isStress
                    ? "32-channel MATLAB EEG file · 100 MB maximum"
                    : "22-channel GDF file · 100 MB maximum"}
                  icon="eeg-pulse.svg"
                />
              </div>

              <p className="sample-note">
                {isStress
                  ? "The configured SAM40 model analyzes the recording in overlapping EEG windows and returns one Relax or Stress prediction, probability scores, and per-window results."
                  : "Upload one BCI IV 2a GDF file at 250 Hz with 22 EEG channels and motor-imagery cues (769–772 or unlabelled 783). Each cue is classified independently."}
              </p>

              {error && <div className="evaluation-error" role="alert">{error}</div>}
              {busy && (
                <div className="evaluation-progress" role="status">
                  <span className="progress-spinner" />
                  {progressMessage}
                </div>
              )}

              <div className="execution-row">
                <div className="execution-note">
                  <span>
                    <i /> {isStress
                      ? "SAM40 ShallowConvNet · 32 EEG channels · single-recording classification"
                      : "Intent inference · BCI IV 2a EEGNet"}
                  </span>
                  <span>Evaluation runs locally on your laptop. No data leaves this device.</span>
                  <span className={backendOnline ? "backend-online" : "backend-warning"}>
                    {checkingBackend
                      ? "Checking local backend…"
                      : backendOnline
                        ? `Backend listener active on ${getApiAddress()}`
                        : `Backend unavailable at ${getApiAddress()}; Run will show a connection error.`}
                  </span>
                </div>
                <button
                  className="run-button"
                  type="submit"
                  disabled={!recording || busy}
                  title={!recording ? "Choose an EEG recording first." : undefined}
                >
                  <img src="/assets/evaluate/run-evaluation.svg" alt="" />
                  {busy ? "Classifying recording" : "Classify recording"}
                </button>
              </div>
            </form>
          )}
        </section>

        <section className="quick-specs" aria-label="Evaluation specifications">
          <article>
            <span>Inference latency</span>
            <strong>~14.2 ms / sample</strong>
            <small>High-throughput tensor batching</small>
          </article>
          <article>
            <span>Target dataset compliance</span>
            <strong>BIDS / MAT / GDF 2.0</strong>
            <small>Zero telemetry leak guarantee</small>
          </article>
          <article>
            <span>Local engine state</span>
            <strong className={backendOnline ? "ready" : "offline"}>
              <i /> {backendOnline ? "Standalone Daemon Ready" : "Waiting for local backend"}
            </strong>
            <small>v2.6.4-preview-cortical</small>
          </article>
        </section>
      </main>

      <footer className="evaluate-footer">
        <span><i /> BCI / NEURO / Next-generation cortical intelligence</span>
        <span>© 2026 BCI / NEURO. All rights reserved. <i>|</i> IIIT SRI CITY · Department of CSE</span>
      </footer>
    </div>
  );
}
