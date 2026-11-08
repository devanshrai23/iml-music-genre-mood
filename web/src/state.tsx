import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { getHealth, getSample, getSamples, staticUrl, type Health } from "./api";
import type { Analysis, SampleEntry } from "./types";

export interface Current { analysis: Analysis; audioSrc: string | null; audioOffset: number }

interface Ctx {
  health: Health | null;
  checked: boolean;
  samples: SampleEntry[];
  current: Current | null;
  setCurrent: (c: Current | null) => void;
  loadSample: (id: string) => Promise<void>;
}
const AppCtx = createContext<Ctx | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<Health | null>(null);
  const [checked, setChecked] = useState(false);
  const [samples, setSamples] = useState<SampleEntry[]>([]);
  const [current, setCurrentState] = useState<Current | null>(null);

  const setCurrent = (c: Current | null) => {
    setCurrentState((prev) => {
      if (prev?.audioSrc?.startsWith("blob:") && prev.audioSrc !== c?.audioSrc) URL.revokeObjectURL(prev.audioSrc);
      return c;
    });
  };

  const loadSample = async (id: string) => {
    const a = await getSample(id);
    setCurrent({ analysis: a, audioSrc: a.audio ? staticUrl(a.audio) : null, audioOffset: 0 });
  };

  useEffect(() => {
    getHealth().then((h) => { setHealth(h); setChecked(true); });
    getSamples().then((s) => {
      setSamples(s);
      // Open on a realistic example: prefer a "rescue" case (RF wrong, CRNN right).
      const first = s.find((x) => x.tag === "rescue") ?? s[0];
      if (first) loadSample(first.id).catch(() => undefined);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <AppCtx.Provider value={{ health, checked, samples, current, setCurrent, loadSample }}>
      {children}
    </AppCtx.Provider>
  );
}

export function useApp() {
  const c = useContext(AppCtx);
  if (!c) throw new Error("useApp outside AppProvider");
  return c;
}
