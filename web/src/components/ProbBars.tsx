import { genreColor, pct, pretty } from "../genres";

interface Props { probs: number[]; labels: string[]; k?: number; colorFor?: (l: string) => string }

export default function ProbBars({ probs, labels, k, colorFor = genreColor }: Props) {
  const rows = probs.map((p, i) => ({ l: labels[i], p })).sort((a, b) => b.p - a.p).slice(0, k ?? probs.length);
  return (
    <div className="bars">
      {rows.map((r, i) => (
        <div key={r.l} className={`bar ${i === 0 ? "top" : ""}`}>
          <span>{pretty(r.l)}</span>
          <div className="bar-track"><div className="bar-fill" style={{ width: pct(r.p, 1), background: colorFor(r.l) }} /></div>
          <span className="v">{pct(r.p, r.p < 0.1 ? 1 : 0)}</span>
        </div>
      ))}
    </div>
  );
}
