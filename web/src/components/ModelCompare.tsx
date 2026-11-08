import { MODEL_LABELS, genreColor, pct, pretty, topK } from "../genres";
import type { Analysis, ModelKey } from "../types";

const ORDER: ModelKey[] = ["crnn", "cnn", "svm", "rf"];

export default function ModelCompare({ analysis }: { analysis: Analysis }) {
  const truth = analysis.true_label;
  return (
    <div className="grid-4">
      {ORDER.filter((k) => analysis.models[k]).map((k) => {
        const m = analysis.models[k]!;
        const top = topK(m.probs, analysis.genres, 3);
        const ok = truth ? m.pred === truth : null;
        return (
          <div key={k} className="panel stack" style={{ padding: 14, gap: 8 }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h3>{MODEL_LABELS[k].name}</h3>
              {ok !== null && <span className={`chip ${ok ? "ok" : "no"}`}>{ok ? "correct" : "wrong"}</span>}
            </div>
            <div className="small muted">{MODEL_LABELS[k].note}</div>
            <div className="row" style={{ gap: 8 }}>
              <span className="chip"><span className="dot" style={{ background: genreColor(m.pred) }} />{pretty(m.pred)}</span>
              <span className="mono small">{pct(top[0].p)}</span>
            </div>
            <div className="small muted">
              {top.slice(1).map((t) => `${pretty(t.g)} ${pct(t.p)}`).join(" · ")}
            </div>
          </div>
        );
      })}
    </div>
  );
}
