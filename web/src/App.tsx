import { HashRouter, NavLink, Route, Routes } from "react-router-dom";
import Explain from "./pages/Explain";
import Home from "./pages/Home";
import Lab from "./pages/Lab";
import Mood from "./pages/Mood";
import Results from "./pages/Results";
import { AppProvider, useApp } from "./state";

const REPO = "https://github.com/devanshrai23/iml-music-genre-mood";

function Shell() {
  const { health, checked } = useApp();
  return (
    <>
      <header className="topbar">
        <div className="topbar-in">
          <NavLink to="/" className="brand">
            <span className="brand-mark" aria-hidden="true">
              {[8, 15, 11, 19, 13].map((h, i) => <i key={i} style={{ height: h }} />)}
            </span>
            Hearing Genre
          </NavLink>
          <nav className="nav">
            <NavLink to="/" end>Overview</NavLink>
            <NavLink to="/lab">Genre Lab</NavLink>
            <NavLink to="/explain">Why this genre?</NavLink>
            <NavLink to="/results">Results</NavLink>
            <NavLink to="/mood">Mood</NavLink>
          </nav>
          <span className={`status ${health ? "on" : ""}`} title={health ? `Models: ${health.genre_models.join(", ")}` : "Pre-computed examples only"}>
            <i />{!checked ? "checking…" : health ? "live models" : "showcase"}
          </span>
        </div>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/lab" element={<Lab />} />
          <Route path="/explain" element={<Explain />} />
          <Route path="/results" element={<Results />} />
          <Route path="/mood" element={<Mood />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </main>
      <footer className="footer">
        <div className="page">
          <span>Explainable Music Genre &amp; Mood Classification · IML group project</span>
          <a href={REPO} target="_blank" rel="noreferrer">Source on GitHub</a>
        </div>
      </footer>
    </>
  );
}

export default function App() {
  return (
    <HashRouter>
      <AppProvider><Shell /></AppProvider>
    </HashRouter>
  );
}
