import { featureLabel } from "../genres";
import type { ShapFeature } from "../types";

interface Props { features: ShapFeature[]; label?: (n: string) => string }

/** Diverging bars: amber pushes towards the class, plum pushes away. */
export default function ShapBars({ features, label = featureLabel }: Props) {
  const max = Math.max(...features.map((f) => Math.abs(f.shap)), 1e-9);
  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="shap">
        {features.map((f) => {
          const w = (Math.abs(f.shap) / max) * 50;
          return (
            <div className="shap-row" key={f.name} title={`${f.name} = ${f.value}`}>
              <span className="name">{label(f.name)}</span>
              <div className="shap-axis">
                <div className={`shap-bar ${f.shap >= 0 ? "pos" : "neg"}`}
                     style={f.shap >= 0 ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }} />
              </div>
              <span className="v">{f.shap >= 0 ? "+" : ""}{f.shap.toFixed(3)}</span>
            </div>
          );
        })}
      </div>
      <div className="legend">
        <span><i className="pos" /> pushes towards this class</span>
        <span><i className="neg" /> pushes away</span>
      </div>
    </div>
  );
}
