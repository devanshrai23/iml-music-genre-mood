import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getMetrics, staticUrl } from "../api";
import ConfusionMatrix from "../components/ConfusionMatrix";
import { BENCHMARKS, PER_GENRE } from "../data";
import { MODEL_LABELS, genreColor, pct, pretty } from "../genres";
import { useApp } from "../state";
import type { Metrics, ModelKey } from "../types";

const STATIC_CM: Record<ModelKey, string> = {
  rf: "confusion_RandomForest.png", svm: "confusion_SVM.png", cnn: "confusion_CNN.png", crnn: "confusion_CRNN.png",
};

export default function Results() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [model, setModel] = useState<ModelKey>("crnn");
  const [cell, setCell] = useState<[number, number] | null>(null);
  const { samples, loadSample } = useApp();
  const nav = useNavigate();
  useEffect(() => { getMetrics().then(setMetrics); }, []);

  const m = metrics?.models[model];
  const genres = metrics?.genres ?? [];
  const tracks = useMemo(() => {
    if (!cell || !metrics?.test_tracks) return [];
    return metrics.test_tracks.filter((t) => t.label === genres[cell[0]] && t.preds[model] === genres[cell[1]]);
  }, [cell, metrics, model, genres]);
  const sampleIds = new Set(samples.map((s) => s.id));

  return (
    <div className="page">
      <header className="stack" style={{ gap: 8 }}>
        <div className="eyebrow">Results</div>
        <h1>From 74.5% to 87% on GTZAN</h1>
        <p className="lede">Handcrafted features with classic models, then spectrogram networks with attention. Each step is evaluated
          without leakage: the deep models split by song before slicing.</p>
      </header>

      <section className="panel">
        <div className="panel-head"><h3>Benchmarks</h3><p>As reported in the project README</p></div>
        <div className="tbl-wrap">
          <table>
            <thead><tr><th>Task</th><th>Model</th><th>Validation</th><th className="num">Accuracy</th><th className="num">Macro P</th><th className="num">Macro R</th><th className="num">Macro F1</th></tr></thead>
            <tbody>
              {BENCHMARKS.map((b) => (
                <tr key={b.model} className={b.best ? "best" : ""}>
                  <td>{b.task}</td><td>{b.model}</td><td className="muted">{b.split}</td>
                  <td className="num">{pct(b.acc, 1)}</td><td className="num">{b.prec.toFixed(2)}</td>
                  <td className="num">{b.rec.toFixed(2)}</td><td className="num">{b.f1.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head"><h3>F1 per genre</h3><p>Darker cell = higher F1</p></div>
        <div className="tbl-wrap">
          <table>
            <thead><tr><th>Genre</th><th className="num">RF</th><th className="num">SVM</th><th className="num">CNN</th><th className="num">CRNN</th><th>What changed</th></tr></thead>
            <tbody>
              {PER_GENRE.map(([g, ...rest]) => {
                const vals = rest.slice(0, 4) as number[];
                const best = Math.max(...vals);
                return (
                  <tr key={g}>
                    <td><span className="chip"><span className="dot" style={{ background: genreColor(g) }} />{pretty(g)}</span></td>
                    {vals.map((v, i) => (
                      <td key={i} className="num" style={{ background: `color-mix(in srgb, var(--plum) ${Math.max(0, (v - 0.45) * 90)}%, transparent)`, fontWeight: v === best ? 700 : 400, color: v > 0.85 ? "var(--surface)" : undefined }}>{v.toFixed(2)}</td>
                    ))}
                    <td className="muted small" style={{ whiteSpace: "normal", minWidth: 220 }}>{rest[4] as string}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel stack">
        <div className="panel-head" style={{ marginBottom: 0 }}>
          <div>
            <h3>Confusion matrix</h3>
            <p className="small muted">{m ? "Held-out 200-song test split, re-evaluated by the website's own code. Click a cell to list its songs." : "Figures from the report."}</p>
          </div>
          <div className="tabs" role="group" aria-label="Model">
            {(["rf", "svm", "cnn", "crnn"] as ModelKey[]).map((k) => (
              <button key={k} className={model === k ? "on" : ""} onClick={() => { setModel(k); setCell(null); }}>{MODEL_LABELS[k].short}</button>
            ))}
          </div>
        </div>
        {m ? (
          <div className="split split-b">
            <ConfusionMatrix labels={genres} matrix={m.confusion} selected={cell} onSelect={setCell} />
            <div className="stack">
              <div className="row" style={{ gap: 24 }}>
                <div className="kpi"><b>{pct(m.accuracy, 1)}</b><span>test accuracy</span></div>
                <div className="kpi"><b>{m.macro_f1.toFixed(3)}</b><span>macro F1</span></div>
              </div>
              {cell ? (
                <div className="stack" style={{ gap: 6 }}>
                  <h3>True {pretty(genres[cell[0]])}, predicted {pretty(genres[cell[1]])}: {tracks.length} songs</h3>
                  <div className="row" style={{ gap: 6 }}>
                    {tracks.map((t) => {
                      const id = t.file.replace(/\.wav$/, "");
                      return sampleIds.has(id)
                        ? <button key={id} className="chip" style={{ cursor: "pointer", borderColor: "var(--accent)" }}
                                  onClick={() => { loadSample(id); nav("/lab"); }}>▶ {id}</button>
                        : <span key={id} className="chip mono">{id}</span>;
                    })}
                  </div>
                  <p className="small muted">Songs with ▶ are in the demo gallery and open in the Genre Lab.</p>
                </div>
              ) : <p className="small muted">The diagonal (plum) is correct predictions. Amber cells are mistakes; rock and country are the usual trouble spots.</p>}
            </div>
          </div>
        ) : (
          <img className="figure" style={{ maxWidth: 620 }} src={staticUrl(`reports/${STATIC_CM[model]}`)} alt={`Confusion matrix for ${MODEL_LABELS[model].name}`} />
        )}
        {m && (model === "rf" || model === "svm") && (
          <p className="small muted">The website's RF and SVM are trained on the same 80% song split as the CRNN, so this number is a single
            hold-out score and can differ slightly from the 5-fold CV figure in the table above.</p>
        )}
      </section>
    </div>
  );
}
