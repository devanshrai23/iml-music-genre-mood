// Copies the figures in ../reports into public/reports so the site can show them.
import { cpSync, existsSync, mkdirSync, readdirSync } from "node:fs";
const src = new URL("../../reports/", import.meta.url);
const dst = new URL("../public/reports/", import.meta.url);
if (existsSync(src)) {
  mkdirSync(dst, { recursive: true });
  for (const f of readdirSync(src)) if (f.endsWith(".png")) cpSync(new URL(f, src), new URL(f, dst));
}
