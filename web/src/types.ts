export type ModelKey = "crnn" | "cnn" | "rf" | "svm";

export interface SliceResult { probs: number[]; pred: string }
export interface ModelResult {
  probs: number[];
  pred: string;
  slices?: SliceResult[];
  attention?: number[][] | number[];
}
export interface ShapFeature { name: string; value: number; shap: number; group?: string }
export interface ShapClass { base: number; top: ShapFeature[]; groups: Record<string, number> }

export interface Analysis {
  id: string;
  title: string | null;
  true_label?: string | null;
  audio?: string | null;
  clip: { duration: number; offset: number; total_duration?: number };
  genres: string[];
  waveform: number[];
  spectrogram: string;
  slice_seconds: number;
  models: Partial<Record<ModelKey, ModelResult>>;
  shap?: { model: string; by_class: Record<string, ShapClass> };
  elapsed_ms?: number;
}

export interface SampleEntry {
  id: string; title: string; label: string; tag: "hit" | "rescue" | "miss";
  crnn?: string; rf?: string; audio?: string | null;
}

export interface MetricsBlock {
  accuracy: number; macro_f1: number;
  per_class: Record<string, { precision: number; recall: number; f1: number }>;
  confusion: number[][];
}

export interface MoodResult {
  classes: string[]; probs_raw: number[]; probs_calibrated: number[];
  pred: string; pred_uncalibrated: string;
  shap: { class: string; base: number; top: ShapFeature[] };
}
export interface MoodPreset {
  title: string; artist: string; genre: string; mood: string;
  valence: number; energy: number; features: Record<string, number>; result?: MoodResult;
}

export interface Metrics {
  genres: string[];
  models: Partial<Record<ModelKey, MetricsBlock>>;
  shap_global?: { overall: { name: string; value: number; group: string }[];
                  by_class: Record<string, { name: string; value: number }[]> };
  test_tracks?: { file: string; label: string; preds: Partial<Record<ModelKey, string>> }[];
  mood?: {
    classes: string[]; calibration: Record<string, number>; rows: number;
    raw: MetricsBlock; calibrated: MetricsBlock;
    features: Record<string, { min: number; max: number; median: number }>;
    shap_global: Record<string, { name: string; value: number }[]>;
  };
}
