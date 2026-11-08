import { pretty } from "../genres";

interface Props {
  labels: string[];
  matrix: number[][];
  selected?: [number, number] | null;
  onSelect?: (cell: [number, number]) => void;
}

/** Rows = true class, columns = predicted class. Cells shaded by row share. */
export default function ConfusionMatrix({ labels, matrix, selected, onSelect }: Props) {
  const n = labels.length;
  return (
    <div className="tbl-wrap">
      <div className="cm" style={{ gridTemplateColumns: `76px repeat(${n}, minmax(26px, 1fr))`, minWidth: 360 }}>
        <div />
        {labels.map((l) => <div key={l} className="lab collab">{pretty(l)}</div>)}
        {matrix.map((row, i) => {
          const total = row.reduce((a, b) => a + b, 0) || 1;
          return [
            <div key={`l${i}`} className="lab">{pretty(labels[i])}</div>,
            ...row.map((v, j) => {
              const share = v / total;
              const diag = i === j;
              const bg = v === 0 ? "var(--sunken)" : diag
                ? `color-mix(in srgb, var(--plum) ${20 + share * 80}%, var(--surface))`
                : `color-mix(in srgb, var(--accent) ${15 + share * 85}%, var(--surface))`;
              const on = selected && selected[0] === i && selected[1] === j;
              const dark = share > 0.45;
              return (
                <button key={`${i}-${j}`} className={`cell ${on ? "on" : ""}`}
                        style={{ background: bg, color: dark ? "var(--surface)" : "var(--ink)" }}
                        onClick={() => onSelect?.([i, j])}
                        aria-label={`True ${labels[i]}, predicted ${labels[j]}: ${v}`}>
                  {v || ""}
                </button>
              );
            }),
          ];
        })}
      </div>
      <div className="small muted" style={{ marginTop: 6 }}>Rows: true genre · Columns: predicted genre</div>
    </div>
  );
}
