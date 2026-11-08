import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { analyzeFile } from "../api";
import AnalysisPlayer from "../components/AnalysisPlayer";
import ModelCompare from "../components/ModelCompare";
import ProbBars from "../components/ProbBars";
import ShowcaseBanner from "../components/Showcase";
import { genreColor, pct, pretty } from "../genres";
import { useApp } from "../state";

export default function Lab() {
  const { health, samples, current, setCurrent, loadSample } = useApp();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const [offset, setOffset] = useState("");
  const [slice, setSlice] = useState<number | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const live = !!health && health.genre_models.length > 0;

  const run = async (file: File) => {
    setErr(null); setBusy(true); setSlice(null);
    try {
      const off = offset.trim() === "" ? undefined : Number(offset);
      const a = await analyzeFile(file, off);
      setCurrent({ analysis: a, audioSrc: URL.createObjectURL(file), audioOffset: a.clip.offset });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally { setBusy(false); }
  };

  const a = current?.analysis;
  const crnn = a?.models.crnn;
  const shown = crnn && slice !== null && crnn.slices ? crnn.slices[slice].probs : (crnn ?? a?.models.cnn ?? a?.models.svm)?.probs;

  return (
    <div className="page">
      <header className="stack" style={{ gap: 8 }}>
        <div className="eyebrow">Genre Lab</div>
        <h1>Play a clip. Watch the CRNN listen.</h1>
        <p className="lede">The Attention-CRNN cuts 30 seconds of audio into ten 3-second slices, predicts each one, and averages them.
          The green curve shows where inside each slice its attention went.</p>
      </header>
      <ShowcaseBanner feature="uploading your own audio" />

      <section className="grid-2" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1.6fr)" }}>
        <div className={`drop ${over ? "over" : ""}`}
             onDragOver={(e) => { e.preventDefault(); if (live) setOver(true); }}
             onDragLeave={() => setOver(false)}
             onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f && live) run(f); }}>
          <h3>Analyse your own song</h3>
          <p className="small muted">WAV, MP3, FLAC, OGG or M4A, up to 30 MB. Long songs are cut to a 30 s window from the middle.</p>
          <input ref={input} id="upload" type="file" accept="audio/*" hidden
                 onChange={(e) => { const f = e.target.files?.[0]; if (f) run(f); e.target.value = ""; }} />
          <div className="row" style={{ justifyContent: "center" }}>
            <button className="btn primary" disabled={!live || busy} onClick={() => input.current?.click()}>
              {busy ? <><span className="spinner" /> Analysing…</> : "Choose audio file"}
            </button>
          </div>
          <label className="small muted row" htmlFor="offset" style={{ justifyContent: "center" }}>
            Start at
            <input id="offset" type="number" min={0} step={1} placeholder="auto" value={offset}
                   onChange={(e) => setOffset(e.target.value)}
                   style={{ width: 70, padding: "4px 6px", borderRadius: 6, border: "1px solid var(--line)", background: "var(--surface)", color: "var(--ink)" }} />
            seconds
          </label>
          {err && <p className="error small">{err}</p>}
        </div>

        <div className="panel">
          <div className="panel-head">
            <h3>Or pick a test-set song</h3>
            <p>{samples.length ? "None of these were seen during training." : "No pre-computed samples yet."}</p>
          </div>
          <div className="gallery">
            {samples.map((s) => (
              <button key={s.id} className={`sample ${a?.id === s.id ? "on" : ""}`}
                      onClick={() => { setSlice(null); loadSample(s.id); }}>
                <span className="t" style={{ borderLeft: `3px solid ${genreColor(s.label)}`, paddingLeft: 6 }}>{pretty(s.label)}</span>
                <span className="s mono">{s.id}</span>
                <span className="s">CRNN: {s.crnn ? pretty(s.crnn) : "–"} · RF: {s.rf ? pretty(s.rf) : "–"}</span>
              </button>
            ))}
          </div>
        </div>
      </section>

      {!a && <div className="panel muted">Pick a song or upload a clip to see the analysis.</div>}

      {a && (
        <>
          <section className="panel stack" style={{ gap: 16 }}>
            <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
              <div className="stack" style={{ gap: 4 }}>
                <div className="eyebrow">{a.id === "upload" ? "Your upload" : "GTZAN test song"}</div>
                <h2 style={{ wordBreak: "break-word" }}>{a.title ?? a.id}</h2>
                {a.true_label && <span className="small muted">Labelled genre: <b>{pretty(a.true_label)}</b></span>}
              </div>
              {crnn && (
                <div className="kpi" style={{ textAlign: "right" }}>
                  <span>CRNN verdict</span>
                  <b style={{ color: genreColor(crnn.pred) }}>{pretty(crnn.pred)}</b>
                  <span>{pct(Math.max(...crnn.probs))} after soft voting{a.elapsed_ms ? ` · ${(a.elapsed_ms / 1000).toFixed(1)} s` : ""}</span>
                </div>
              )}
            </div>
            <AnalysisPlayer current={current!} selectedSlice={slice} onSelectSlice={setSlice} />
          </section>

          <section className="grid-2">
            <div className="panel">
              <div className="panel-head">
                <h3>{slice === null ? "Song-level probabilities (CRNN)" : `Slice ${slice + 1} on its own`}</h3>
                <p>{slice === null ? "Average of the 10 slice predictions" : "Click the slice again to go back"}</p>
              </div>
              {shown && <ProbBars probs={shown} labels={a.genres} />}
            </div>
            <div className="panel stack">
              <h3>How to read this</h3>
              <ul className="small" style={{ margin: 0, paddingLeft: 18, display: "flex", flexDirection: "column", gap: 6 }}>
                <li>The coloured strip under each slice is that slice's own guess. Disagreeing slices are normal; voting smooths them out.</li>
                <li>The green curve is attention <i>inside</i> each slice: which of its 16 time steps (≈0.19 s each) fed the decision most.</li>
                <li>Peaks usually sit on drum hits, vocal entries or chord changes.</li>
                <li>The CNN overlay shows the older model, which attends over the whole 30 s at once in 40 steps.</li>
              </ul>
              <Link className="btn" to="/explain" style={{ alignSelf: "flex-start" }}>Why this genre? See SHAP →</Link>
            </div>
          </section>

          <section className="stack">
            <h2>Four models, same clip</h2>
            <ModelCompare analysis={a} />
          </section>
        </>
      )}
    </div>
  );
}
