import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getMetrics, staticUrl } from "../api";
import AnalysisPlayer from "../components/AnalysisPlayer";
import ShapBars from "../components/ShapBars";
import { featureLabel, genreColor, pretty } from "../genres";
import { useApp } from "../state";
import type { Metrics } from "../types";

export default function Explain() {
  const { current } = useApp();
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [cls, setCls] = useState<string | null>(null);
  const [slice, setSlice] = useState<number | null>(null);
  useEffect(() => { getMetrics().then(setMetrics); }, []);

  const a = current?.analysis;
  const shap = a?.shap;
  const rfPred = a?.models.rf?.pred;
  const target = cls && shap?.by_class[cls] ? cls : rfPred ?? (shap ? Object.keys(shap.by_class)[0] : null);
  useEffect(() => setCls(null), [a?.id, a?.title]);
  const block = target && shap ? shap.by_class[target] : null;
  const groups = block ? Object.entries(block.groups) : [];
  const gmax = Math.max(...groups.map(([, v]) => Math.abs(v)), 1e-9);

  return (
    <div className="page">
      <header className="stack" style={{ gap: 8 }}>
        <div className="eyebrow">Why this genre?</div>
        <h1>What the features say, and when the network listened</h1>
        <p className="lede">Two kinds of explanation for the same clip. TreeSHAP tells us <b>which sound properties</b> pushed the
          Random Forest to its answer. CRNN attention tells us <b>which moments</b> the deep model relied on.</p>
      </header>

      {!a && <div className="panel">No clip loaded yet. <Link to="/lab">Open the Genre Lab</Link> and pick a song first.</div>}

      {a && (
        <section className="grid-2" style={{ alignItems: "start" }}>
          <div className="panel stack">
            <div className="panel-head" style={{ marginBottom: 0 }}>
              <div>
                <div className="eyebrow">Which features · Random Forest + TreeSHAP</div>
                <h3 style={{ marginTop: 4 }}>{a.title ?? a.id}</h3>
              </div>
              {rfPred && <span className="chip"><span className="dot" style={{ background: genreColor(rfPred) }} />RF says {pretty(rfPred)}</span>}
            </div>
            {shap ? (
              <>
                <label className="small muted row" htmlFor="shap-class">
                  Explain the score for
                  <select id="shap-class" value={target ?? ""} onChange={(e) => setCls(e.target.value)}
                          style={{ padding: "4px 8px", borderRadius: 6, border: "1px solid var(--line)", background: "var(--surface)", color: "var(--ink)" }}>
                    {Object.keys(shap.by_class).map((g) => <option key={g} value={g}>{pretty(g)}{g === rfPred ? " (RF prediction)" : ""}{g === a.true_label ? " (true label)" : ""}</option>)}
                  </select>
                </label>
                {block && <ShapBars features={block.top} />}
                {block && (
                  <div className="stack" style={{ gap: 6 }}>
                    <h3>By feature family</h3>
                    {groups.map(([g, v]) => (
                      <div key={g} className="shap-row">
                        <span className="name">{g}</span>
                        <div className="shap-axis">
                          <div className={`shap-bar ${v >= 0 ? "pos" : "neg"}`}
                               style={v >= 0 ? { left: "50%", width: `${(Math.abs(v) / gmax) * 50}%` } : { right: "50%", width: `${(Math.abs(v) / gmax) * 50}%` }} />
                        </div>
                        <span className="v">{v >= 0 ? "+" : ""}{v.toFixed(3)}</span>
                      </div>
                    ))}
                    <p className="small muted">Base rate for {pretty(target!)}: {(block.base * 100).toFixed(1)}%. The bars add up to how far this clip moves away from it.</p>
                  </div>
                )}
              </>
            ) : <p className="muted">This clip has no SHAP values (the Random Forest was not loaded).</p>}
          </div>

          <div className="panel stack">
            <div>
              <div className="eyebrow">Which moments · Attention-CRNN</div>
              <h3 style={{ marginTop: 4 }}>Attention over the spectrogram</h3>
            </div>
            <AnalysisPlayer current={current!} selectedSlice={slice} onSelectSlice={setSlice} compact />
            <p className="small muted">Low frequencies sit at the bottom. Bright = loud. The green curve rises where the CRNN looked hardest within
              each 3 s slice. SHAP explains the whole clip at once; attention explains it moment by moment. Where they agree (for example a
              high spectral-contrast feature and attention on a drum hit), we can trust the explanation more.</p>
          </div>
        </section>
      )}

      {metrics?.shap_global && (
        <section className="panel">
          <div className="panel-head">
            <h3>Most important features overall</h3>
            <p>Mean |SHAP| across the 200 test songs and all 10 genres</p>
          </div>
          <div className="bars">
            {metrics.shap_global.overall.slice(0, 12).map((f) => {
              const m = metrics.shap_global!.overall[0].value;
              return (
                <div key={f.name} className="bar" style={{ gridTemplateColumns: "minmax(140px, 200px) 1fr 60px" }}>
                  <span title={f.group}>{featureLabel(f.name)}</span>
                  <div className="bar-track"><div className="bar-fill" style={{ width: `${(f.value / m) * 100}%`, background: "var(--plum)" }} /></div>
                  <span className="v">{f.value.toFixed(4)}</span>
                </div>
              );
            })}
          </div>
        </section>
      )}

      <section className="panel stack">
        <div className="panel-head" style={{ marginBottom: 0 }}>
          <h3>Case study from the report</h3>
          <p>Top: both models right. Bottom: RF wrong, CRNN right.</p>
        </div>
        <img className="figure" src={staticUrl("reports/shap_vs_attention_comparison.png")} alt="Local TreeSHAP bars next to CRNN attention over the spectrogram for two test songs" loading="lazy" />
      </section>
    </div>
  );
}
