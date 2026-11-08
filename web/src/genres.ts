// One fixed colour per genre, used everywhere a genre appears.
export const GENRE_COLORS: Record<string, string> = {
  blues: "#3b6fd8", classical: "#b08a2e", country: "#c2662d", disco: "#d6409f", hiphop: "#7b4fd6",
  jazz: "#178f86", metal: "#5d6470", pop: "#e0507a", reggae: "#3f9a3a", rock: "#c23b30",
};
export const MOOD_COLORS: Record<string, string> = {
  happy: "#e2a12b", tense: "#d0493a", sad: "#4a72c9", calm: "#3d9b7c",
};
export const genreColor = (g: string) => GENRE_COLORS[g] ?? "#888";
export const pretty = (g: string) => (g === "hiphop" ? "Hip-hop" : g.charAt(0).toUpperCase() + g.slice(1));
export const pct = (v: number, d = 0) => `${(v * 100).toFixed(d)}%`;
export const MODEL_LABELS = {
  crnn: { name: "Attention-CRNN", short: "CRNN", note: "CNN + BiGRU + attention on 10 × 3 s slices" },
  cnn: { name: "Attention-CNN", short: "CNN", note: "CNN + attention on the full 30 s" },
  svm: { name: "RBF-SVM", short: "SVM", note: "89 handcrafted features" },
  rf: { name: "Random Forest", short: "RF", note: "89 handcrafted features, explained with SHAP" },
} as const;

export function topK(probs: number[], genres: string[], k = 3) {
  return probs.map((p, i) => ({ g: genres[i], p })).sort((a, b) => b.p - a.p).slice(0, k);
}

// Readable names for the 89 librosa features.
export function featureLabel(name: string) {
  const m = name.match(/^(mfcc|chroma|spec_contrast)_(\d+)_(mean|std)$/);
  const stat = (s: string) => (s === "mean" ? "avg" : "var.");
  const pitch = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
  if (m) {
    const [, kind, idx, s] = m;
    if (kind === "mfcc") return `MFCC ${idx} ${stat(s)}`;
    if (kind === "chroma") return `Chroma ${pitch[+idx - 1] ?? idx} ${stat(s)}`;
    return `Contrast band ${idx} ${stat(s)}`;
  }
  const simple: Record<string, string> = {
    spec_centroid: "Spectral centroid", spec_bandwidth: "Bandwidth", spec_rolloff: "Roll-off",
    zcr: "Zero-crossing rate", rms: "RMS energy",
  };
  const s = name.match(/^(.*)_(mean|std)$/);
  if (s && simple[s[1]]) return `${simple[s[1]]} ${stat(s[2])}`;
  if (name === "tempo") return "Tempo (BPM)";
  return name.replace(/_/g, " ");
}
