import { useEffect, useRef, useState } from "react";

/** Plays one analysis window of an audio file. Times are relative to the window start. */
export function useAudio(src: string | null, offset: number, duration: number) {
  const ref = useRef<HTMLAudioElement | null>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setTime(0); setPlaying(false); setReady(false);
    if (!src) { ref.current = null; return; }
    const a = new Audio(src);
    a.preload = "auto";
    ref.current = a;
    let raf = 0;
    const tick = () => {
      const t = a.currentTime - offset;
      if (t >= duration) { a.pause(); a.currentTime = offset; setTime(0); setPlaying(false); return; }
      setTime(Math.max(0, t));
      raf = requestAnimationFrame(tick);
    };
    const onPlay = () => { setPlaying(true); raf = requestAnimationFrame(tick); };
    const onPause = () => { setPlaying(false); cancelAnimationFrame(raf); };
    const onReady = () => { setReady(true); if (a.currentTime < offset) a.currentTime = offset; };
    a.addEventListener("play", onPlay);
    a.addEventListener("pause", onPause);
    a.addEventListener("ended", onPause);
    a.addEventListener("loadedmetadata", onReady);
    return () => { cancelAnimationFrame(raf); a.pause(); a.src = ""; };
  }, [src, offset, duration]);

  const seek = (t: number) => {
    const a = ref.current;
    const clamped = Math.min(Math.max(0, t), duration - 0.05);
    setTime(clamped);
    if (a) a.currentTime = offset + clamped;
  };
  const toggle = () => {
    const a = ref.current;
    if (!a) return;
    if (a.paused) { if (a.currentTime < offset || a.currentTime > offset + duration) a.currentTime = offset; a.play().catch(() => undefined); }
    else a.pause();
  };
  return { time, playing, ready: ready && !!src, seek, toggle, hasAudio: !!src };
}
