import SiteHeader from "../components/SiteHeader.jsx";

const literature = [
  {
    title: "Standardized 3-Stage BCI Taxonomy",
    tag: "Foundational",
    description:
      "Established our structural framework: Acquisition, Feature Extraction, and Decision Logic. Codified signal handling baselines for non-invasive electrophysiology.",
    note: "Informed: 3-stage pipeline blueprint",
  },
  {
    title: "Classical Feature-Based EEG Classification",
    tag: "Benchmark",
    description:
      "Statistical and spectral features provide a useful classical baseline for EEG classification. Reported scores should only be compared when dataset splits and evaluation protocols match.",
    note: "Informed: baseline comparison",
  },
  {
    title: "SAM40: 32-Channel Affective Corpus",
    tag: "Dataset",
    description:
      "The primary training corpus: 40 subjects under multimodal cognitive friction (Color-Word Stroop and timed mental arithmetic) captured at 128 Hz.",
    note: "Informed: primary data ground truth",
  },
  {
    title: "Deep Learning with ConvNets in EEG",
    tag: "Architecture",
    description:
      "Introduced ShallowConvNet: temporal convolution followed by spatial convolution that mirrors the neurophysiologically sound Filter Bank Common Spatial Pattern.",
    note: "Informed: shallow convolution design",
  },
  {
    title: "EEGNet: Compact CNN for BCI",
    tag: "Architecture",
    description:
      "Demonstrated depthwise and separable convolutions as an efficient alternative to large models, with compact parameter counts for EEG classification.",
    note: "Informed: efficient intent decoder",
  },
  {
    title: "BCI Competition IV-2a Dataset",
    tag: "Dataset",
    description:
      "Nine subjects perform four motor-imagery tasks across 22 EEG channels, providing a standardized benchmark for motor-intent decoding.",
    note: "Informed: active BCI evaluation",
  },
];

const samFacts = [
  ["Cohort", "40 Subjects"],
  ["Model rate", "320 Hz"],
  ["Channels", "32 EEG"],
  ["Input", "Single .MAT"],
];

const intentFacts = [
  ["Cohort", "9 Subjects"],
  ["Sampling", "250 Hz"],
  ["Channels", "22 Ag/AgCl"],
  ["Trials", "288 / Subj"],
];

const preprocessing = [
  {
    step: "01",
    title: "Raw Ingestion",
    description:
      "Ingestion of multi-format bio-data (.mat for SAM40 and .gdf for BCI Competition IV-2a) into unified MNE raw data structures.",
    footer: "Preserves bio-timestamps",
  },
  {
    step: "02",
    title: "Channel Validation",
    description:
      "The stress pipeline validates 32 EEG channels. For motor imagery, recognized EOG channels are removed before the 22 EEG channels are selected.",
    footer: "Task-specific input checks",
  },
  {
    step: "03",
    title: "Frequency Filtering",
    description:
      "The multiband stress profile derives 0.5–45 Hz, theta, alpha, and beta views. Motor imagery uses a 4–40 Hz IIR bandpass and a 50 Hz notch.",
    footer: "Profile-specific filters",
  },
  {
    step: "04",
    title: "Epoch Segmentation",
    description:
      "Stress inference uses 640-sample windows at a 320-sample step. Motor imagery is segmented into cue-locked epochs from 0 to 4 seconds.",
    footer: "50% stress-window overlap",
  },
  {
    step: "05",
    title: "Normalization & Aggregation",
    description:
      "Stress EEG is standardized per channel and recording. Window probabilities are averaged to produce one stress prediction per recording.",
    footer: "Recording-level output",
  },
];

const architectures = [
  {
    id: "system-one",
    eyebrow: "System 1 // Passive BCI",
    title: "ShallowConvNet: Stress Detection Engine",
    accent: "cyan",
    badges: ["32-channel SAM40", "Binary classifier", "CPU / GPU ready"],
    stages: [
      ["01", "EEG input", "32 × 640", "32 sensor leads"],
      ["02", "Temporal conv", "40 filters", "Band features"],
      ["03", "Spatial conv", "Depthwise", "Channel maps"],
      ["04", "Square & pool", "Mean pool", "Log variance"],
      ["05", "Dense layer", "2 classes", "Relax / stress"],
      ["06", "Softmax", "Probability", "State output"],
    ],
    mechanisms: [
      ["Temporal filters", "Learn frequency-selective features directly from raw EEG."],
      ["Spatial projection", "Combines channel-specific activations across the sensor montage."],
      ["Log-variance pooling", "Captures band power while keeping the network compact."],
      ["Recording-level output", "Aggregates overlapping windows into one prediction per recording."],
    ],
  },
  {
    id: "system-two",
    eyebrow: "System 2 // Active BCI",
    title: "EEGNet: Motor Imagery Intent",
    accent: "amber",
    badges: ["22-channel BCI IV-2a", "Four-class intent", "Low-latency inference"],
    stages: [
      ["01", "EEG input", "22 × 1000", "Four-second trial"],
      ["02", "Temporal conv", "16 filters", "Learned bands"],
      ["03", "Depthwise spatial", "D = 2", "Across electrodes"],
      ["04", "Feature pooling", "ELU / pool", "Temporal summary"],
      ["05", "Separable conv", "32 filters", "Compact features"],
      ["06", "Classification", "4 classes", "Motor intent"],
    ],
    mechanisms: [
      ["Depthwise spatial filters", "Learn motor-rhythm combinations independently across EEG leads."],
      ["Temporal integration", "Pool learned feature maps to capture sustained event-related desynchronization."],
      ["Separable convolution", "Decouples spatial and temporal processing while keeping parameter count low."],
      ["Intent mapping", "Classifies left hand, right hand, feet, or tongue imagery per trial."],
    ],
  },
];

const evaluationOutputs = [
  ["Predicted state", "Relax / stress"],
  ["Model score", "Per recording"],
  ["Class probabilities", "Relax + stress"],
  ["Window outputs", "Per segment"],
];

function SectionHeading({ phase, label, title, description }) {
  return (
    <header className="research-section-heading">
      <p className="eyebrow">{phase} · {label}</p>
      <h2>{title}</h2>
      {description && <p className="research-description">{description}</p>}
    </header>
  );
}

function MetricTiles({ facts }) {
  return (
    <div className="research-facts">
      {facts.map(([label, value]) => (
        <div className="research-fact" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

function DatasetCard({ title, badge, color, facts, bullets, footer, footerValue }) {
  return (
    <article className={`research-dataset ${color}`}>
      <div className="research-dataset-heading">
        <h3><i />{title}</h3>
        <span className="research-tag">{badge}</span>
      </div>
      <MetricTiles facts={facts} />
      <ul className="research-bullets">
        {bullets.map(([label, text]) => (
          <li key={label}><strong>{label}:</strong> {text}</li>
        ))}
      </ul>
      <div className="research-domain">
        <span>{footer}</span>
        <strong>{footerValue}</strong>
      </div>
    </article>
  );
}

function ArchitectureCard({ architecture }) {
  return (
    <article className={`architecture-card ${architecture.accent}`} id={architecture.id}>
      <div className="architecture-heading">
        <div>
          <p className="eyebrow">{architecture.eyebrow}</p>
          <h3>{architecture.title}</h3>
        </div>
        <div className="architecture-badges">
          {architecture.badges.map((badge) => <span key={badge}>{badge}</span>)}
        </div>
      </div>
      <div className="architecture-visual">
        <div className="architecture-visual-meta">
          <strong><i /> Layer-by-layer computation</strong>
          <span>EEG signal → learned features → class output</span>
        </div>
        <div className="architecture-stages">
          {architecture.stages.map(([number, title, shape, note], index) => (
            <div className="architecture-stage-wrap" key={number}>
              <article className="architecture-stage">
                <span className="architecture-stage-index">{number}</span>
                <div className={`architecture-block block-${index + 1}`}>
                  <div className="block-lines"><i /><i /><i /></div>
                  <strong>{shape}</strong>
                </div>
                <h4>{title}</h4>
                <p>{note}</p>
              </article>
              {index < architecture.stages.length - 1 && <span className="stage-arrow">›</span>}
            </div>
          ))}
        </div>
        <div className="architecture-legend">
          <span>Input tensor</span>
          <span>Convolution</span>
          <span>Feature map</span>
          <span>Classification head</span>
        </div>
      </div>
      <div className="architecture-mechanics">
        {architecture.mechanisms.map(([title, text]) => (
          <article key={title}>
            <span>Design rationale</span>
            <h4>{title}</h4>
            <p>{text}</p>
          </article>
        ))}
      </div>
    </article>
  );
}

export default function ResearchPage() {
  return (
    <div className="research-page">
      <SiteHeader activePage={1} />
      <main>
        <section className="research-hero">
          <div className="research-breadcrumb">
            <span>Research // Methodology</span><i>·</i><span>Dual-paradigm BCI</span>
          </div>
          <div className="research-hero-copy">
            <h1>Our Research <em>Journey.</em></h1>
            <p>
              From a conceptual framework to a functioning AI system — our six-step, data-driven,
              end-to-end methodology.
            </p>
          </div>
          <article className="research-inquiry">
            <p className="eyebrow">Core motivating inquiry</p>
            <blockquote>“Can a machine know you are stressed before you even realise it yourself?”</blockquote>
            <p>
              Cortical electroencephalography captures micro-volt potential fluctuations with
              sub-millisecond fidelity. By synthesizing passive cognitive state tracking with active
              motor intent decoding, our investigation constructs a dual-paradigm neural coprocessor.
            </p>
          </article>
          <div className="research-roster">
            <article>
              <span>Faculty mentorship</span>
              <strong>Dr. Annushree Bablani</strong>
              <small>Assistant Professor, IIIT Sri City</small>
            </article>
            <article>
              <span>Student investigators</span>
              <strong>B. Tharun Reddy &amp; K. Raghuveer</strong>
              <small>Undergraduate Researchers (B.Tech ’28)</small>
            </article>
            <article>
              <span>Dual paradigm scope</span>
              <strong>Passive &amp; Active BCI</strong>
              <small>Affective State Estimation · Real-Time Motor Kinematics</small>
            </article>
          </div>
          <div className="research-system-summary">
            <article className="research-summary-card cyan">
              <div className="research-summary-top">
                <span>System 01 · Passive BCI</span><span>SAM40</span>
              </div>
              <h3>Passive BCI: Acute Stress Detection</h3>
              <p>
                Decodes autonomic sympathetic response versus baseline relaxation under induced
                cognitive friction. Focuses on frontal sub-band power spectrum dynamics and
                frequency-band feature convergence.
              </p>
              <div><span>Core classifier</span><strong>ShallowConvNet</strong></div>
            </article>
            <article className="research-summary-card amber">
              <div className="research-summary-top">
                <span>System 02 · Active BCI</span><span>BCI IV-2a</span>
              </div>
              <h3>Active BCI: Motor Imagery Intent</h3>
              <p>
                Classifies imagined movement into four discrete intents using compact temporal and
                depthwise spatial filters across the sensorimotor cortex.
              </p>
              <div><span>Core classifier</span><strong>EEGNet</strong></div>
            </article>
          </div>
        </section>

        <section className="research-section">
          <SectionHeading
            phase="Phase 1"
            label="Literature Foundation"
            title="Foundational Insights from Literature"
            description="A rigorous review of EEG-based brain-computer interfaces shaped our taxonomy, model selection, data strategy, and evaluation targets."
          />
          <div className="literature-grid">
            {literature.map((paper, index) => (
              <article className="literature-card" key={paper.title}>
                <div className="literature-meta">
                  <span>Reference 0{index + 1}</span><span>{paper.tag}</span>
                </div>
                <h3>{paper.title}</h3>
                <p>{paper.description}</p>
                <footer>{paper.note}</footer>
              </article>
            ))}
          </div>
        </section>

        <section className="research-section dataset-section">
          <SectionHeading
            phase="Phase 2"
            label="Data Understanding"
            title="Benchmark Cohorts &amp; Sensor Topologies"
            description="Deep characterization of the two foundational datasets powering the passive affective pipeline and active kinematic mobility system."
          />
          <div className="research-datasets">
            <DatasetCard
              title="SAM40 Affective Dataset"
              badge="Passive stress engine"
              color="cyan"
              facts={samFacts}
              bullets={[
                ["Stimulus protocol", "Four distinct task modalities including Stroop color-word conflict, mental arithmetic countdowns, mirror-image recognition, and baseline relaxation."],
                ["Primary focus", "Prefrontal and frontal cortex electrodes capturing autonomic valence, stress-induced beta elevation, and theta suppression."],
                ["Input representation", "The supported stress-model profile uses 640-sample windows with a 320-sample step."],
              ]}
              footer="Domain: Cognitive stress vs rest"
              footerValue="Binary classification"
            />
            <DatasetCard
              title="BCI Competition IV-2a"
              badge="Active kinematics"
              color="amber"
              facts={intentFacts}
              bullets={[
                ["Stimulus protocol", "Cue-guided four-class motor imagery: Left Hand, Right Hand, Both Feet, and Tongue movements."],
                ["Epoch window", "Four-second cue-locked execution interval with monopolar EOG channels used for artifact rejection."],
                ["Electrode grid", "Standard 10-20 layout clustered across sensorimotor cortex to monitor contralateral ERD/ERS."],
              ]}
              footer="Domain: 4-way intent direction"
              footerValue="Motor kinematics"
            />
          </div>
        </section>

        <section className="research-section preprocessing-section">
          <SectionHeading
            phase="Phase 3"
            label="Signal Processing"
            title="End-to-End Preprocessing Flow"
            description="A deterministic conditioning pipeline transforms raw non-stationary voltage micro-signals into regularized tensor batches."
          />
          <div className="preprocessing-grid">
            {preprocessing.map((stage) => (
              <article className="preprocessing-card" key={stage.step}>
                <span className="step-label">Step {stage.step}</span>
                <h3>{stage.title}</h3>
                <p>{stage.description}</p>
                <footer>{stage.footer}</footer>
              </article>
            ))}
          </div>
        </section>

        <section className="research-section architecture-section">
          <SectionHeading
            phase="Phase 4"
            label="Model Building & Architecture"
            title="Deep Cortical Neural Architectures"
            description="Two compact convolutional systems transform electrophysiological signals into affective state estimates and actionable motor intent."
          />
          {architectures.map((architecture) => (
            <ArchitectureCard architecture={architecture} key={architecture.id} />
          ))}
        </section>

        <section className="research-section evaluation-section">
          <SectionHeading
            phase="Phase 5"
            label="Inference & Evaluation"
            title="Single-Recording Stress Classification"
            description="The local workflow accepts one SAM40 .mat recording and classifies its overlapping EEG windows with the configured stress model."
          />
          <article className="research-panel benchmark-panel">
            <div className="research-panel-heading">
              <div><span className="research-tag">Inference flow</span><span>Recording-level prediction</span></div>
              <small>One SAM40 .mat recording</small>
            </div>
            <p>
              The recording is segmented into overlapping EEG windows. Window probabilities are
              aggregated into one Relax or Stress prediction, while individual window outputs remain
              available for inspection. Model scores are not measured accuracy or clinical confidence.
            </p>
            <div className="research-table-wrap">
              <table>
                <thead><tr><th>Output</th><th>Meaning</th><th>Granularity</th><th>Input</th></tr></thead>
                <tbody>
                  <tr><td>Predicted state</td><td>Relax or Stress</td><td>Recording</td><td>One .mat file</td></tr>
                  <tr><td>Class probabilities</td><td>Aggregated model output</td><td>Recording</td><td>EEG windows</td></tr>
                  <tr><td>Model score</td><td>Score for predicted state</td><td>Recording</td><td>EEG windows</td></tr>
                  <tr><td>Window predictions</td><td>Per-window class and scores</td><td>Window</td><td>EEG windows</td></tr>
                </tbody>
              </table>
            </div>
          </article>
          <div className="fold-panel">
            <div className="research-panel-heading">
              <div><span className="research-tag">Per recording</span><span>Model outputs</span></div>
              <small>Generated after single-file upload</small>
            </div>
            <div className="fold-grid evaluation-outputs">
              {evaluationOutputs.map(([name, description]) => (
                <article key={name}><span>{name}</span><strong>{description}</strong></article>
              ))}
            </div>
          </div>
          <article className="refinements-panel">
            <div className="research-panel-heading">
              <div><span className="research-tag amber-tag">Interpretation</span><span>What the metrics do and do not mean</span></div>
              <small>Read results in context</small>
            </div>
            <div className="refinements-grid">
              <article><span>01 / Input</span><h3>One recording</h3><p>Each run processes one 32-channel SAM40 .mat EEG file.</p></article>
              <article><span>02 / Aggregation</span><h3>Window to file</h3><p>Overlapping window probabilities are aggregated into one recording prediction.</p></article>
              <article><span>03 / Validation</span><h3>No CV claim</h3><p>The app does not report a precomputed five-fold or held-out-subject score.</p></article>
              <article><span>04 / Use</span><h3>Research only</h3><p>Predictions are not a clinical diagnosis or a measure of clinical confidence.</p></article>
            </div>
          </article>
        </section>

        <section className="research-section validation-section">
          <SectionHeading
            phase="Phase 6"
            label="Results & Generalization"
            title="From Local Prediction to Validation"
            description="The application predicts stress state for one uploaded recording. Broader claims about held-out subjects, external datasets, or cross-paradigm transfer require separate experiments."
          />
          <article className="fold-panel final-results">
            <div className="research-panel-heading">
              <div><span className="research-tag">SAM40</span><span>Stress classification</span></div>
              <small>Prediction generated from one .mat recording</small>
            </div>
            <div className="fold-grid evaluation-outputs">
              {evaluationOutputs.map(([name, description]) => (
                <article key={name}><span>{name}</span><strong>{description}</strong></article>
              ))}
            </div>
          </article>
          <div className="generalization-grid">
            <article className="generalization-card cyan">
              <div className="research-panel-heading">
                <span className="research-tag">Transfer study</span><small>Stress representation</small>
              </div>
              <h3>External dataset transfer</h3>
              <p>
                Transfer to another dataset or sensor montage is a separate validation study.
                This application does not currently run or report a SAM40-to-STEW transfer experiment.
              </p>
              <div className="generalization-stat"><span>Transfer result</span><strong>Not measured</strong></div>
            </article>
            <article className="generalization-card amber">
              <div className="research-panel-heading">
                <span className="research-tag amber-tag">Cross-paradigm</span><small>Intent representation</small>
              </div>
              <h3>Motor imagery four-class performance</h3>
              <p>
                The intent workflow accepts a BCI Competition IV-2a recording and reports its
                predicted motor-imagery class. Subject-wise benchmark accuracy is not bundled.
              </p>
              <div className="generalization-stat"><span>Benchmark result</span><strong>Not reported</strong></div>
            </article>
          </div>
          <div className="research-conclusion">
            <p className="eyebrow">Research outcome</p>
            <h3>From cortical signals to interpretable, task-specific neural predictions.</h3>
            <a href="/evaluate">Explore the local classification workflow <span>→</span></a>
          </div>
        </section>
      </main>
      <footer className="research-footer">
        <span><strong>BCI / NEURO</strong>　/　 Next-generation cortical intelligence</span>
        <span>© 2026 BCI / Neuro. All rights reserved.</span>
      </footer>
    </div>
  );
}
