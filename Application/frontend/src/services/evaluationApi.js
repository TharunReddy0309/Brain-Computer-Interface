export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  const body = await response.json();

  if (!response.ok) {
    throw new Error(body.detail || `The evaluation API returned ${response.status}.`);
  }

  return body;
}

export function getApiAddress() {
  return new URL(API_BASE_URL).host;
}

export function checkBackend(signal) {
  return request("/api/health", { signal });
}

export async function startEvaluation(dataset, signal) {
  const form = new FormData();
  form.append("dataset", dataset);
  return request("/api/evaluations", { method: "POST", body: form, signal });
}

export async function startStressInference(recording, signal) {
  const form = new FormData();
  form.append("recording", recording);
  return request("/api/inference/stress", { method: "POST", body: form, signal });
}

export async function startIntentInference(recording, signal) {
  const form = new FormData();
  form.append("recording", recording);
  return request("/api/inference/intent", { method: "POST", body: form, signal });
}

function wait(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, milliseconds);

    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(signal.reason);
      },
      { once: true },
    );
  });
}

export async function waitForEvaluation(jobId, onStatus, signal) {
  while (true) {
    const job = await request(`/api/evaluations/${jobId}`, { signal });
    onStatus(job);

    if (job.status === "failed") {
      throw new Error(job.error || "The evaluation failed.");
    }

    if (job.status === "complete") {
      return request(`/api/evaluations/${jobId}/result`, { signal });
    }

    await wait(1200, signal);
  }
}
