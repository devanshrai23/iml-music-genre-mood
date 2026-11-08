import { useEffect, useState } from "react";
import { getMetrics, getMoodPresets, predictMood, staticUrl } from "../api";
import ProbBars from "../components/ProbBars";
import ShapBars from "../components/ShapBars";
import { MOOD_COLORS, pct, pretty } from "../genres";
import { useApp } from "../state";
import type { Metrics, MoodPreset, MoodResult } from "../types";

const FEATURES: { key: string; label: string; min: number; max: number; step: number }[] = [
  { key: "danceability", label: "Danceability", min: 0, max: 1, step: 0.01 },
  { key: "loudness", label: "Loudness (dB)", min: -30, max: 0, step: 0.5 },
  { key: "tempo", label: "Tempo (BPM)", min: 60, max: 200, step: 1 },
  { key: "acousticness", label: "Acousticness", min: 0, max: 1, step: 0.01 },
  { key: "instrumentalness", label: "Instrumentalness", min: 0, max: 1, step: 0.01 },
  { key: "speechiness", label: "Speechiness", min: 0, max: 0.6, step: 0.01 },
  { key: "liveness", label: "Liveness", min: 0, max: 1, step: 0.01 },
  { key: "mode", label: "Mode (0 minor, 1 major)", min: 0, max: 1, step: 1 },
  { key: "key", label: "Key (0 = C … 11 = B)", min: 0, max: 11, step: 1 },
];
const DEFAULTS: Record<string, number> = { danceability: 0.58, loudness: -7, tempo: 120, acousticness: 0.17, instrumentalness: 0, speechiness: 0.05, liveness: 0.13, mode: 1, key: 5 };
const moodLabel = (m: string) => pretty(m);
const moodColor = (m: string) => MOOD_COLORS[m] ?? "#888";

function Circumplex({ pred, point }: { pred?: string; point?: { valence: number; energy: number } }) {
  // x = valence (sad/tense on the left), y = energy (high at the top)
  const q = [
    { m: "tense", x: 0, y: 0, label: "Tense", sub: "low valence · high energy" },
    { m: "happy", x: 100, y: 0, label: "Happy", sub: "high valence · high energy" },
    { m: "sad", x: 0, y: 100, label: "Sad", sub: "low valence · low energy" },
    { m: "calm", x: 100, y: 100, label: "Calm", sub: "high valence · low energy" },
  ];
  return (
    <svg className="circ" viewBox="-14 -6 228 226" role="img" aria-label={`Russell circumplex, predicted ${pred ?? "none"}`}>
      {q.map((c) => (
        <g key={c.m}>
          <rect x={c.x} y={c.y} width={100} height={100} rx={8}
                fill={pred === c.m ? moodColor(c.m) : "var(--sunken)"} opacity={pred === c.m ? 0.9 : 1} />
          <text x={c.x + 50} y={c.y + 48} textAnchor="middle" fontSize="15" fontWeight={700}
                fill={pred === c.m ? "#fff" : "var(--ink)"} style={{ fontFamily: "var(--display)" }}>{c.label}</text>
          <text x={c.x + 50} y={c.y + 64} textAnchor="middle" fontSize="7" fill={pred === c.m ? "#fff" : "var(--muted)"}>{c.sub}</text>
        </g>
      ))}
      {point && <circle cx={point.valence * 200} cy={(1 - point.energy) * 200} r={5} fill="var(--ink)" stroke="var(--surface)" strokeWidth={2} />}
      <text x={100} y={214} textAnchor="middle" fontSize="8" fill="var(--muted)">valence →</text>
      <text x={-6} y={100} textAnchor="middle" fontSize="8" fill="var(--muted)" transform="rotate(-90 -6 100)">energy →</text>
    </svg>
  );
}

export default function Mood() {
  const { health } = useApp();
  const live = !!health?.mood;
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [presets, setPresets] = useState<MoodPreset[]>([]);
  const [chosen, setChosen] = useState<number | null>(null);
  const [vals, setVals] = useState<Record<string, number>>(DEFAULTS);
  const [result, setResult] = useState<MoodResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getMetrics().then(setMetrics);
    getMoodPresets().then((p) => { setPresets(p); if (p[0]) { setChosen(0); setVals(p[0].features); setResult(p[0].result ?? null); } });
  }, []);

  useEffect(() => {
    if (!live) return;
    const t = setTimeout(() => {
      predictMood(vals).then((r) => { setResult(r); setErr(null); }).catch((e) => setErr(String(e.message ?? e)));
    }, 180);
    return () => clearTimeout(t);
  }, [vals, live]);

  const preset = chosen !== null ? presets[chosen] : null;
  const mood = metrics?.mood;

  return (
    <div className="page">
      <header className="stack" style={{ gap: 8 }}>
        <div className="eyebrow">Mood</div>
        <h1>Four moods from Spotify audio descriptors</h1>
        <p className="lede">Each of 89,741 Spotify tracks gets a mood from Russell's circumplex: valence ≥ 0.5 and energy ≥ 0.5 split the
          plane into Happy, Tense, Sad and Calm. Valence and energy are then <b>removed</b> from the inputs, so the model cannot just read the answer back.</p>
      </header>

      <section className="grid-3">
        <div className="panel"><div className="kpi"><b>{mood ? pct(mood.calibrated.accuracy, 1) : "70.1%"}</b><span>accuracy (calibrated RF)</span></div></div>
        <div className="panel"><div className="kpi"><b>{mood ? mood.calibrated.macro_f1.toFixed(3) : "0.667"}</b><span>macro F1</span></div></div>
        <div className="panel"><div className="kpi"><b>0.36 → 0.48</b><span>calm recall after the 1.45× prior boost</span></div></div>
      </section>

      <section className="grid-3" style={{ gridTemplateColumns: "minmax(0, 0.9fr) minmax(0, 1.2fr) minmax(0, 1fr)", alignItems: "start" }}>
        <div className="panel stack">
          <h3>Example tracks</h3>
          {presets.length ? (
            <div className="presets">
              {presets.map((p, i) => (
                <button key={i} className={`preset ${chosen === i ? "on" : ""}`}
                        onClick={() => { setChosen(i); setVals(p.features); if (!live) setResult(p.result ?? null); }}>
                  <span style={{ minWidth: 0 }}>
                    <span className="t" style={{ display: "block" }}>{p.title}</span>
                    <span className="a" style={{ display: "block" }}>{p.artist}</span>
                  </span>
                  <span className="chip"><span className="dot" style={{ background: moodColor(p.mood) }} />{moodLabel(p.mood)}</span>
                </button>
              ))}
            </div>
          ) : <p className="small muted">Run <code>scripts/build_demo_samples.py</code> to add example tracks.</p>}
        </div>

        <div className="panel stack">
          <div className="panel-head" style={{ marginBottom: 0 }}>
            <h3>What-if sliders</h3>
            {!live && <span className="chip">needs the backend</span>}
          </div>
          {FEATURES.map((f) => (
            <div className="slider" key={f.key}>
              <label htmlFor={`m-${f.key}`}>{f.label}</label>
              <input id={`m-${f.key}`} type="range" min={f.min} max={f.max} step={f.step} disabled={!live}
                     value={vals[f.key] ?? DEFAULTS[f.key]}
                     onChange={(e) => { setChosen(null); setVals({ ...vals, [f.key]: Number(e.target.value) }); }} />
              <span className="v">{(vals[f.key] ?? 0).toFixed(f.step >= 1 ? 0 : 2)}</span>
            </div>
          ))}
          {err && <p className="error small">{err}</p>}
          {!live && <p className="small muted">The sliders call the mood model live. Without the server you can still click the example tracks.</p>}
        </div>

        <div className="panel stack">
          <h3>Prediction</h3>
          <Circumplex pred={result?.pred} point={preset ? { valence: preset.valence, energy: preset.energy } : undefined} />
          {preset && <p className="small muted">Dot = the track's real valence/energy (true mood: {moodLabel(preset.mood)}).</p>}
          {result && (
            <>
              <ProbBars probs={result.probs_calibrated} labels={result.classes} colorFor={moodColor} />
              {result.pred !== result.pred_uncalibrated && (
                <p className="small muted">Without calibration this would be {moodLabel(result.pred_uncalibrated)}. The calm boost changed the answer.</p>
              )}
            </>
          )}
        </div>
      </section>

      {result && (
        <section className="grid-2">
          <div className="panel">
            <div className="panel-head"><h3>Why {moodLabel(result.pred)}?</h3><p>TreeSHAP for this input</p></div>
            <ShapBars features={result.shap.top} label={(n) => n.replace(/_/g, " ")} />
          </div>
          <div className="panel stack">
            <h3>Confusion matrix (report)</h3>
            <img className="figure" src={staticUrl("reports/confusion_Mood_Final.png")} alt="Mood confusion matrix for the calibrated RF" loading="lazy" />
          </div>
        </section>
      )}
    </div>
  );
}
