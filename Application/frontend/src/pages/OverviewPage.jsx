import SiteHeader from "../components/SiteHeader.jsx";

const cards = [
  {
    number: "01",
    mode: "Passive paradigm",
    title: "Affective Decoding",
    description:
      "Autonomous cognitive state discernment. Quantifying workload inflection points, autonomic valence, and attention synchronization through subtle cortical desynchrony.",
    label: "Frontal theta // alpha coherence",
    status: "Phase synced",
    color: "cyan",
    chart: ["affective-wave.svg", "affective-wave-secondary.svg"],
    footer: ["Spatio-temporal convolution", "Continuous latent stream"],
  },
  {
    number: "02",
    mode: "Active paradigm",
    title: "Motor Synthesis",
    description:
      "Sensorimotor rhythm classification transforming voluntary kinematic imagery into deterministic spatial trajectories for robotic steering and external intent articulation.",
    label: "Sensorimotor μ / β ERD / ERS",
    status: "Trajectory resolved",
    color: "amber",
    chart: ["motor-wave.svg"],
    footer: ["Depthwise separable spatial", "Zero-lag execution"],
  },
];

function Card({ card }) {
  return (
    <article className="card">
      <div className="card-meta">
        <span>Pillar {card.number}</span>
        <span className={card.color}>{card.mode}</span>
      </div>
      <h3>{card.title}</h3>
      <p className="description">{card.description}</p>
      <div className="chart">
        <div className="chart-meta">
          <span>{card.label}</span>
          <span className={card.color}>{card.status}</span>
        </div>
        <div className="chart-lines">
          {card.chart.map((image) => (
            <img key={image} src={`/assets/${image}`} alt="" />
          ))}
        </div>
      </div>
      <div className="card-footer">
        <span>{card.footer[0]}</span>
        <span>{card.footer[1]}</span>
      </div>
    </article>
  );
}

export default function App() {
  return (
    <div id="overview">
      <SiteHeader activePage={0} />

      <main>
        <section className="hero">
          <img className="hero-art" src="/assets/hero-network.svg" alt="" />
          <div className="hero-copy">
            <span className="eyebrow pill">Non-invasive cortical architecture</span>
            <h1>
              Neural <em>Intelligence</em>.
            </h1>
            <p>
              Decoding the human mind through next-generation cortical architectures.
              Synthesizing cognitive dynamics and autonomous intent with pristine fidelity.
            </p>
            <div className="actions">
              <a className="primary-button" href="#research">
                Explore architecture <img src="/assets/arrow-down.svg" alt="" />
              </a>
              <a className="text-button" href="/research">
                Spectral research
              </a>
              <a className="text-button" href="/evaluate">
                Upload &amp; analysis
              </a>
            </div>
          </div>
        </section>

        <section className="section" id="pillars">
          <p className="eyebrow">Core pillars</p>
          <h2>
            Bimodal <em>Cognitive Paradigms</em>.
          </h2>
          <div className="cards">
            {cards.map((card) => (
              <Card card={card} key={card.number} />
            ))}
          </div>
        </section>

        <section className="section research" id="research">
          <div className="research-heading">
            <div>
              <p className="eyebrow">Cortical intelligence // deep neural engines</p>
              <h2>
                Neural Predictive <em>Synthesis</em>.
              </h2>
            </div>
            <p className="description">
              Datasets trained on deep neural networks delivering high-fidelity cognitive prediction
              and real-time intent decoding across continuous cortical streams.
            </p>
          </div>
          <div className="manifold" id="intelligence">
            <div className="manifold-meta">
              <span><i /> Deep latent embedding engine</span>
              <span className="cyan">Manifold topology: Riemannian</span>
              <span className="amber">Inference latency: &lt; 0.8 ms</span>
            </div>
            <div className="manifold-art">
              <img src="/assets/manifold-grid-horizontal.svg" alt="" />
              <img src="/assets/manifold-grid-vertical.svg" alt="" />
              <img src="/assets/manifold-line-one.svg" alt="" />
              <img src="/assets/manifold-line-two.svg" alt="" />
              <img src="/assets/manifold-line-three.svg" alt="" />
              <img src="/assets/manifold-line-four.svg" alt="" />
            </div>
            <div className="manifold-footer">
              <span>D-512 latent space // continuous tensor flow</span>
              <span className="cyan">Zero-phase alignment: 99.84% convergence</span>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <span><strong>BCI / NEURO</strong>　/　 Next-generation cortical intelligence</span>
        <span>© 2026 BCI / Neuro. All rights reserved.</span>
      </footer>
    </div>
  );
}
