import type { Analysis } from "../types";

interface Props {
  analysis: Analysis;
  time: number;
  onSeek: (t: number) => void;
  showAttention: "crnn" | "cnn" | "none";
}

const FRAMES_SECONDS = 1280 * 512 / 22050; // the 1280 frames the models see (~29.7 s)

/** Mel-spectrogram with the model's temporal attention drawn on top and a playhead. */
export default function SpectrogramView({ analysis, time, onSeek, showAttention }: Props) {
  const span = FRAMES_SECONDS;
  let curve: number[] = [];
  if (showAttention === "crnn" && analysis.models.crnn?.attention) {
    curve = (analysis.models.crnn.attention as number[][]).flat();
  } else if (showAttention === "cnn" && analysis.models.cnn?.attention) {
    curve = analysis.models.cnn.attention as number[];
  }
  const max = Math.max(...curve, 1e-9);
  const n = curve.length;
  const pts = curve.map((v, i) => `${((i + 0.5) / n) * 1000},${100 - (v / max) * 58}`);
  const path = n ? `M0,100 L${pts.join(" L")} L1000,100 Z` : "";
  const line = n ? `M${pts.join(" L")}` : "";
  const sliceCount = 10;

  const click = (e: React.MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    onSeek(((e.clientX - r.left) / r.width) * span);
  };

  return (
    <div className="spec" onClick={click} role="slider" aria-label="Seek in clip"
         aria-valuemin={0} aria-valuemax={Math.round(span)} aria-valuenow={Math.round(time)} tabIndex={0}
         onKeyDown={(e) => { if (e.key === "ArrowRight") onSeek(time + 3); if (e.key === "ArrowLeft") onSeek(time - 3); }}>
      <img src={analysis.spectrogram} alt="Log-mel spectrogram of the analysed 30 second clip" />
      <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true">
        {Array.from({ length: sliceCount - 1 }, (_, i) => (
          <line key={i} x1={((i + 1) / sliceCount) * 1000} x2={((i + 1) / sliceCount) * 1000} y1={0} y2={100}
                stroke="rgba(255,255,255,0.28)" strokeWidth={1} strokeDasharray="2 3" vectorEffect="non-scaling-stroke" />
        ))}
        {n > 0 && <path d={path} fill="rgba(120,255,214,0.14)" />}
        {n > 0 && <path d={line} fill="none" stroke="#78ffd6" strokeWidth={2} vectorEffect="non-scaling-stroke" />}
      </svg>
      <div className="playhead" style={{ left: `${Math.min(100, (time / span) * 100)}%` }} />
    </div>
  );
}
