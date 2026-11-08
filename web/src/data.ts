// Numbers from the project README (final results).
export const BENCHMARKS = [
  { task: "Genre", model: "Random Forest (300 trees)", split: "5-fold stratified CV", acc: 0.745, prec: 0.74, rec: 0.74, f1: 0.740 },
  { task: "Genre", model: "RBF-SVM (C = 10)", split: "5-fold stratified CV", acc: 0.776, prec: 0.78, rec: 0.78, f1: 0.776 },
  { task: "Genre", model: "Attention-CNN (full 30 s)", split: "80/20 song split", acc: 0.830, prec: 0.83, rec: 0.83, f1: 0.826 },
  { task: "Genre", model: "Attention-CRNN (10 × 3 s, soft vote)", split: "80/20 song split", acc: 0.870, prec: 0.87, rec: 0.87, f1: 0.869, best: true },
  { task: "Mood", model: "Random Forest baseline (9 features)", split: "5-fold stratified CV", acc: 0.705, prec: 0.69, rec: 0.65, f1: 0.660 },
  { task: "Mood", model: "Calibrated RF + interactions", split: "5-fold stratified CV", acc: 0.701, prec: 0.67, rec: 0.67, f1: 0.667, best: true },
  { task: "Mood", model: "HistGradBoost (class-balanced)", split: "5-fold stratified CV", acc: 0.660, prec: 0.63, rec: 0.66, f1: 0.636 },
  { task: "Mood", model: "Deep MLP (128 → 64)", split: "5-fold stratified CV", acc: 0.665, prec: 0.63, rec: 0.60, f1: 0.608 },
];

// genre: [RF, SVM, CNN, CRNN] F1, plus the takeaway from the README
export const PER_GENRE: [string, number, number, number, number, string][] = [
  ["classical", 0.90, 0.89, 0.98, 0.98, "Low zero-crossing rate and high spectral contrast"],
  ["metal", 0.86, 0.87, 0.86, 0.97, "+0.11 F1 on the CRNN, with 1.00 precision"],
  ["jazz", 0.85, 0.86, 0.88, 0.95, "BiGRU picks up phrasing and brass/woodwind transitions"],
  ["blues", 0.81, 0.82, 0.82, 0.92, "+0.10 F1 from 3 s slice soft voting"],
  ["country", 0.71, 0.78, 0.74, 0.88, "Better at telling acoustic country from rock"],
  ["hiphop", 0.74, 0.72, 0.88, 0.86, "0.95 recall on rhythmic vocal cadence"],
  ["disco", 0.61, 0.67, 0.82, 0.81, "Spectrograms keep the 4/4 groove that summary stats lose"],
  ["reggae", 0.65, 0.69, 0.86, 0.81, "Off-beat syncopation is visible over time"],
  ["pop", 0.79, 0.80, 0.85, 0.76, "High-frequency vocal production"],
  ["rock", 0.50, 0.66, 0.57, 0.74, "+0.17 F1 on the hardest umbrella genre"],
];
