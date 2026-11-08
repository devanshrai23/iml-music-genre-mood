import { Link } from "react-router-dom";
import SpectrogramView from "../components/SpectrogramView";
import { genreColor, pct, pretty } from "../genres";
import { useApp } from "../state";

export default function Home() {
  const { current } = useApp();
  const a = current?.analysis;
  const crnn = a?.models.crnn;

  return (
    <div className="page" style={{ gap: 44 }}>
      <section className="hero">
        <div className="stack" style={{ gap: 18 }}>
          <div className="eyebrow">IML group project · Explainable music classification</div>
          <h1>Which genre is this song, and why does the model think so?</h1>
          <p className="lede">We classify music into 10 GTZAN genres and 4 Spotify moods, then explain every prediction two ways: TreeSHAP on
            89 handcrafted audio features, and learned temporal attention on mel-spectrograms.</p>
          <div className="row">
            <Link className="btn primary" to="/lab">Try the Genre Lab</Link>
            <Link className="btn" to="/results">See the results</Link>
          </div>
        </div>
        <div className="hero-viz">
          {a ? (
            <>
              <SpectrogramView analysis={a} time={0} onSeek={() => undefined} showAttention="crnn" />
              <div className="row" style={{ justifyContent: "space-between" }}>
                <span className="cap">{a.title ?? a.id}{a.true_label ? ` · labelled ${a.true_label}` : ""}</span>
                {crnn && <span className="chip" style={{ background: "#1b1726", color: "#fff", borderColor: genreColor(crnn.pred) }}>
                  <span className="dot" style={{ background: genreColor(crnn.pred) }} />CRNN: {pretty(crnn.pred)} {pct(Math.max(...crnn.probs))}</span>}
              </div>
              <span className="cap">Mel-spectrogram (low notes at the bottom) with the CRNN's attention in green.</span>
            </>
          ) : <span className="cap">Loading an example…</span>}
        </div>
      </section>

      <section className="grid-4">
        <div className="panel kpi"><b>87.0%</b><span>genre accuracy, Attention-CRNN</span></div>
        <div className="panel kpi"><b>+12.5 pts</b><span>over the Random Forest baseline</span></div>
        <div className="panel kpi"><b>0.667</b><span>mood macro F1, without valence or energy</span></div>
        <div className="panel kpi"><b>89</b><span>handcrafted features explained with SHAP</span></div>
      </section>

      <section className="stack" style={{ gap: 18 }}>
        <h2>How it works</h2>
        <div className="steps">
          <div className="step"><h3>Listen</h3><p>Each 30 s clip is resampled to 22,050 Hz and turned into a 128-band log-mel spectrogram, plus 89 summary features (MFCCs, chroma, spectral shape, tempo).</p></div>
          <div className="step"><h3>Slice</h3><p>Songs are split 80/20 <i>before</i> slicing, so no song leaks into both sets. Each spectrogram becomes ten 3 s slices of 128 × 128.</p></div>
          <div className="step"><h3>Classify</h3><p>A 4-block CNN feeds a bidirectional GRU. Attention weights the 16 time steps of each slice, and the 10 slice votes are averaged.</p></div>
          <div className="step"><h3>Explain</h3><p>TreeSHAP shows which features moved the Random Forest. Attention shows which moments mattered to the CRNN. We compare both on the same songs.</p></div>
        </div>
      </section>

      <section className="grid-3">
        <Link to="/lab" className="panel stack" style={{ textDecoration: "none", color: "inherit" }}>
          <div className="eyebrow">Genre Lab</div><h3>Upload a song or pick a test clip</h3>
          <p className="small muted">Play it with the attention curve, see every slice's vote, compare four models.</p>
        </Link>
        <Link to="/explain" className="panel stack" style={{ textDecoration: "none", color: "inherit" }}>
          <div className="eyebrow">Why this genre?</div><h3>SHAP next to attention</h3>
          <p className="small muted">Which sound properties pushed the decision, and when the network was listening.</p>
        </Link>
        <Link to="/mood" className="panel stack" style={{ textDecoration: "none", color: "inherit" }}>
          <div className="eyebrow">Mood</div><h3>Happy, tense, sad or calm</h3>
          <p className="small muted">A calibrated Random Forest on Spotify descriptors, with what-if sliders.</p>
        </Link>
      </section>
    </div>
  );
}
