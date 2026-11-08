import { useApp } from "../state";

/** Explains what works when the site runs without the Python backend (e.g. on Vercel). */
export default function ShowcaseBanner({ feature }: { feature: string }) {
  const { health, checked } = useApp();
  if (!checked || health) return null;
  return (
    <div className="banner">
      <b>Showcase mode.</b> The analysis server is not reachable, so {feature} is turned off. The pre-computed
      songs below still show real model output. To go live, run <code>python -m api</code> on the laptop with the models.
    </div>
  );
}
