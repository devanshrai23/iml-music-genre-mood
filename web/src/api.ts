import type { Analysis, Metrics, MoodPreset, MoodResult, SampleEntry } from "./types";

const API = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const body = await r.json(); if (body?.detail) msg = String(body.detail); } catch { /* not json */ }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

export interface Health { status: string; genre_models: string[]; mood: boolean; errors: Record<string, string> }

export async function getHealth(): Promise<Health | null> {
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 2500);
    const h = await json<Health>(`${API}/api/health`, { signal: ctl.signal });
    clearTimeout(t);
    return h?.status === "ok" ? h : null;
  } catch { return null; }
}

export const analyzeFile = (file: File, offset?: number) => {
  const fd = new FormData();
  fd.append("file", file);
  if (offset !== undefined && !Number.isNaN(offset)) fd.append("offset", String(offset));
  return json<Analysis>(`${API}/api/analyze`, { method: "POST", body: fd });
};

export const predictMood = (features: Record<string, number>) =>
  json<MoodResult>(`${API}/api/mood/predict`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ features }),
  });

// Static files shipped with the site (work on Vercel without a backend).
const base = import.meta.env.BASE_URL;
export const staticUrl = (p: string) => `${base}${p.replace(/^\//, "")}`;
export const getSamples = () => json<SampleEntry[]>(staticUrl("samples/index.json")).catch(() => [] as SampleEntry[]);
export const getSample = (id: string) => json<Analysis>(staticUrl(`samples/${id}.json`));
export const getMetrics = () => json<Metrics>(staticUrl("data/metrics.json")).catch(() => null);
export const getMoodPresets = () => json<MoodPreset[]>(staticUrl("samples/mood.json")).catch(() => [] as MoodPreset[]);
