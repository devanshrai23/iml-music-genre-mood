import { useState } from "react";
import { genreColor, pct, pretty } from "../genres";
import type { Current } from "../state";
import SpectrogramView from "./SpectrogramView";
import { useAudio } from "./useAudio";

interface Props { current: Current; selectedSlice: number | null; onSelectSlice: (i: number | null) => void; compact?: boolean }

const fmt = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

export default function AnalysisPlayer({ current, selectedSlice, onSelectSlice, compact }: Props) {
  const { analysis, audioSrc, audioOffset } = current;
  const dur = analysis.clip.duration;
  const audio = useAudio(audioSrc, audioOffset, dur);
  const [attn, setAttn] = useState<"crnn" | "cnn" | "none">(analysis.models.crnn ? "crnn" : "cnn");
  const crnn = analysis.models.crnn;
  const sliceSec = analysis.slice_seconds || 2.97;
  const playingSlice = audio.playing ? Math.min(9, Math.floor(audio.time / sliceSec)) : -1;

  return (
    <div className="stack">
      <div className="player">
        <button className="play" onClick={audio.toggle} disabled={!audio.hasAudio}
                aria-label={audio.playing ? "Pause" : "Play"} title={audio.hasAudio ? "" : "No audio for this clip"}>
          {audio.playing
            ? <svg width="16" height="16" viewBox="0 0 16 16"><rect x="3" y="2" width="4" height="12" rx="1" fill="currentColor" /><rect x="9" y="2" width="4" height="12" rx="1" fill="currentColor" /></svg>
            : <svg width="16" height="16" viewBox="0 0 16 16"><path d="M4 2.5v11l9-5.5z" fill="currentColor" /></svg>}
        </button>
        <span className="time">{fmt(audio.time)} / {fmt(dur)}</span>
        {!audio.hasAudio && <span className="small muted">Audio isn't included in this online version. Click the spectrogram or slices to explore.</span>}
        {analysis.clip.offset > 0 && <span className="small muted">window starts at {fmt(analysis.clip.offset)} of {fmt(analysis.clip.total_duration ?? 0)}</span>}
        {!compact && (
          <div className="tabs" style={{ marginLeft: "auto" }} role="group" aria-label="Attention overlay">
            {analysis.models.crnn && <button className={attn === "crnn" ? "on" : ""} onClick={() => setAttn("crnn")}>CRNN attention</button>}
            {analysis.models.cnn && <button className={attn === "cnn" ? "on" : ""} onClick={() => setAttn("cnn")}>CNN attention</button>}
            <button className={attn === "none" ? "on" : ""} onClick={() => setAttn("none")}>Off</button>
          </div>
        )}
      </div>
      <SpectrogramView analysis={analysis} time={audio.time} onSeek={audio.seek} showAttention={attn} />
      {crnn?.slices && (
        <div className="slices" role="list" aria-label="CRNN prediction for each 3 second slice">
          {crnn.slices.map((s, i) => {
            const conf = Math.max(...s.probs);
            return (
              <button key={i} role="listitem"
                      className={`slice ${selectedSlice === i ? "on" : ""} ${playingSlice === i ? "playing" : ""}`}
                      style={{ borderTop: `3px solid ${genreColor(s.pred)}` }}
                      onClick={() => { onSelectSlice(selectedSlice === i ? null : i); audio.seek(i * sliceSec); }}>
                <span className="muted">{Math.round(i * sliceSec)}–{Math.round((i + 1) * sliceSec)}s</span>
                <b>{pretty(s.pred)}</b>
                <span className="muted">{pct(conf)}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
