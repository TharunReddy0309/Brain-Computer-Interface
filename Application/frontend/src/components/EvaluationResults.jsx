function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function EvaluationResults({ result, onReset, onDownload }) {
  if (result.task === "evaluation") {
    return <DatasetResults result={result} onReset={onReset} onDownload={onDownload} />;
  }

  if (result.task === "intent") {
    return <IntentResults result={result} onReset={onReset} />;
  }

  return (
    <section className="results" id="results" aria-labelledby="results-title">
      <div className="results-heading">
        <div>
          <p className="eyebrow">Inference complete</p>
          <h2 id="results-title">Stress prediction</h2>
        </div>
        <button className="secondary-button" type="button" onClick={onReset}>
          Analyze another recording
        </button>
      </div>

      <div className="inference-summary">
        <span>Predicted state</span>
        <strong>{result.predicted_class}</strong>
        <span>Model score: {percent(result.confidence)}</span>
      </div>
      <p className="sample-note">{result.interpretation_note}</p>

      <div className="metric-grid">
        <article className="metric-card">
          <span>Relax score</span>
          <strong>{percent(result.probability_relax)}</strong>
        </article>
        <article className="metric-card">
          <span>Stress score</span>
          <strong>{percent(result.probability_stress)}</strong>
        </article>
        <article className="metric-card">
          <span>Windows analyzed</span>
          <strong>{result.windows}</strong>
        </article>
      </div>

      <div className="result-meta">
        <span>Recording: {result.filename}</span>
        <span>Model: {result.profile}</span>
        <span>{result.device} · {result.duration_seconds}s</span>
      </div>

      <h3>Window predictions</h3>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Window</th>
              <th scope="col">Start sample</th>
              <th scope="col">Prediction</th>
              <th scope="col">Relax score</th>
              <th scope="col">Stress score</th>
            </tr>
          </thead>
          <tbody>
            {result.window_predictions.map((window) => (
              <tr key={window.window}>
                <td>{window.window}</td>
                <td>{window.start_sample}</td>
                <td>{window.predicted_class}</td>
                <td>{percent(window.probability_relax)}</td>
                <td>{percent(window.probability_stress)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function DatasetResults({ result, onReset, onDownload }) {
  const { metrics } = result;
  const confusion = metrics.confusion_matrix;

  return (
    <section className="results dataset-results" id="results" aria-labelledby="results-title">
      <div className="results-heading">
        <div>
          <p className="eyebrow">Metrics // Evaluation complete</p>
          <h2 id="results-title">Dataset evaluation</h2>
        </div>
        <div className="result-actions">
          <button className="secondary-button" type="button" onClick={onDownload}>
            Download metrics
          </button>
          <button className="secondary-button" type="button" onClick={onReset}>
            Analyze another dataset
          </button>
        </div>
      </div>

      <div className="metric-grid">
        <article className="metric-card">
          <span>Recording accuracy</span>
          <strong>{percent(metrics.accuracy)}</strong>
        </article>
        <article className="metric-card">
          <span>Balanced accuracy</span>
          <strong>{percent(metrics.balanced_accuracy)}</strong>
        </article>
        <article className="metric-card">
          <span>Macro F1</span>
          <strong>{percent(metrics.macro_f1)}</strong>
        </article>
        <article className="metric-card">
          <span>Recordings analyzed</span>
          <strong>{result.recordings}</strong>
        </article>
      </div>

      <div className="result-meta">
        <span>Model: {result.profile}</span>
        <span>Evaluation unit: {result.evaluation_unit}</span>
        <span>{result.device} · {result.duration_seconds}s</span>
      </div>

      <h3>Confusion matrix</h3>
      <div className="table-scroll">
        <table>
          <caption>Rows are true labels; columns are predicted labels.</caption>
          <thead>
            <tr>
              <th scope="col">True / Predicted</th>
              {confusion.labels.map((label) => <th scope="col" key={label}>{label}</th>)}
            </tr>
          </thead>
          <tbody>
            {confusion.values.map((row, index) => (
              <tr key={confusion.labels[index]}>
                <th scope="row">{confusion.labels[index]}</th>
                {row.map((count, column) => <td key={confusion.labels[column]}>{count}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>Recording predictions</h3>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Recording</th>
              <th scope="col">Actual</th>
              <th scope="col">Predicted</th>
              <th scope="col">Relax score</th>
              <th scope="col">Stress score</th>
              <th scope="col">Windows</th>
            </tr>
          </thead>
          <tbody>
            {result.predictions.map((prediction) => (
              <tr key={prediction.filename}>
                <td>{prediction.filename}</td>
                <td>{prediction.true_class}</td>
                <td>{prediction.predicted_class}</td>
                <td>{percent(prediction.probability_relax)}</td>
                <td>{percent(prediction.probability_stress)}</td>
                <td>{prediction.windows_used}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function IntentResults({ result, onReset }) {
  return (
    <section className="results" id="results" aria-labelledby="results-title">
      <div className="results-heading">
        <div>
          <p className="eyebrow">Inference complete</p>
          <h2 id="results-title">Intent prediction</h2>
        </div>
        <button className="secondary-button" type="button" onClick={onReset}>
          Analyze another recording
        </button>
      </div>

      <div className="inference-summary">
        <span>Most likely intent</span>
        <strong>{result.predicted_class}</strong>
        <span>Mean class score: {percent(result.confidence)}</span>
      </div>
      <p className="sample-note">{result.interpretation_note}</p>

      <div className="metric-grid">
        {Object.entries(result.class_probabilities).map(([label, probability]) => (
          <article className="metric-card" key={label}>
            <span>{label} score</span>
            <strong>{percent(probability)}</strong>
          </article>
        ))}
        <article className="metric-card">
          <span>Trials analyzed</span>
          <strong>{result.trials}</strong>
        </article>
      </div>

      <div className="result-meta">
        <span>Recording: {result.filename}</span>
        <span>Model: {result.profile}</span>
        <span>{result.device} · {result.duration_seconds}s</span>
      </div>

      <h3>Trial predictions</h3>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Trial</th>
              <th scope="col">Start sample</th>
              <th scope="col">Predicted intent</th>
              <th scope="col">Model score</th>
            </tr>
          </thead>
          <tbody>
            {result.trial_predictions.map((trial) => (
              <tr key={trial.trial}>
                <td>{trial.trial}</td>
                <td>{trial.start_sample}</td>
                <td>{trial.predicted_class}</td>
                <td>{percent(trial.confidence)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
