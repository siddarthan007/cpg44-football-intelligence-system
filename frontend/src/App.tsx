import React, { useEffect, useState } from 'react';
import { flushSync } from 'react-dom';
import { Navigate, NavLink, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { LivePage } from './pages/LivePage';
import { WearablePage } from './pages/WearablePage';
import { TrainingHubPage } from './pages/TrainingHubPage';
import { MatchUploadPage } from './pages/MatchUploadPage';
import { CameraNetworkPage } from './pages/CameraNetworkPage';
import { PlayerMappingPage } from './pages/PlayerMappingPage';
import { HardwareFlasherPage } from './pages/HardwareFlasherPage';
import { PlayerTaggingPage } from './pages/PlayerTaggingPage';
import { ReviewPage } from './pages/ReviewPage';
import { SessionPage } from './pages/SessionPage';
import { CameraCheckPage } from './pages/CameraCheckPage';
import { EvidencePage } from './pages/EvidencePage';
import { PitchSetupPage } from './pages/PitchSetupPage';
import { AdaptPage } from './pages/AdaptPage';
import { Glyph, type GlyphName } from './components/Glyph';
import { listAnalyses } from './lib/analysis';

type Item = { to: string; label: string; g: GlyphName; badge?: number };

export const App: React.FC = () => {
  const loc = useLocation();
  const navigate = useNavigate();
  const [open, setOpen] = useState(0);
  // Same-document view transition on rail clicks; plain navigation where the
  // API is missing, for modified clicks and when motion is reduced.
  const go = (e: React.MouseEvent<HTMLAnchorElement>, to: string) => {
    const d = document as Document & { startViewTransition?: (cb: () => void) => unknown };
    if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || !d.startViewTransition
      || matchMedia('(prefers-reduced-motion: reduce)').matches || to === loc.pathname) return;
    e.preventDefault();
    d.startViewTransition(() => flushSync(() => navigate(to)));
  };
  // After a navigation, move focus to the new page's heading so keyboard and
  // screen-reader users land on the new content, not on a vanished link.
  const first = React.useRef(true);
  useEffect(() => {
    if (first.current) { first.current = false; return; }
    const id = requestAnimationFrame(() => {
      const h = document.querySelector<HTMLElement>('#main h1');
      if (h) { h.tabIndex = -1; h.focus({ preventScroll: true }); }
    });
    return () => cancelAnimationFrame(id);
  }, [loc.pathname]);
  useEffect(() => {
    listAnalyses().then(rows => setOpen(rows.reduce((n, r) => n + (r.checks_pending ?? 0), 0))).catch(() => setOpen(0));
  }, [loc.pathname]);

  const groups: Item[][] = [
    [{ to: '/', label: 'Home', g: 'home' }],
    // in the order a coach works: set up, play, the vest, then the lab
    [
      { to: '/workshop/cameras', label: 'Connect camera', g: 'sources' },
      { to: '/prepare/camera', label: 'Camera position', g: 'camera' },
      { to: '/prepare/pitch', label: 'Pitch setup', g: 'pitch' },
      { to: '/prepare/kits', label: 'Kits', g: 'kits' },
      { to: '/squad', label: 'Squad and vests', g: 'squad' },
    ],
    [
      { to: '/live', label: 'Live', g: 'live' },
      { to: '/review', label: 'Matches', g: 'review', badge: open },
      { to: '/ingest', label: 'Add footage', g: 'upload' },
    ],
    [
      { to: '/vest', label: 'Vest readings', g: 'vest' },
      { to: '/vest/setup', label: 'Flash a vest', g: 'chip' },
    ],
    [
      { to: '/workshop/adapt', label: 'Adapt to ground', g: 'snap' },
      { to: '/workshop/training', label: 'Models', g: 'model' },
      { to: '/evidence', label: 'Evidence', g: 'evidence' },
    ],
  ];

  return <div className="shell">
    <a className="skip" href="#main">Skip to content</a>
    <aside className="rail" aria-label="Workspace">
      <NavLink to="/" className="brand"><span className="brand-mark" aria-hidden="true">44</span><b>CPG44</b></NavLink>
      <nav>
        {groups.map((items, i) => <div className="rail-group" key={i}>
          {items.map(item => <NavLink key={item.to} to={item.to} end={item.to === '/' || item.to === '/vest'} onClick={e => go(e, item.to)}
            className={({ isActive }) => `rail-link ${isActive ? 'on' : ''}`}>
            <Glyph g={item.g} /><span>{item.label}</span>
            {item.badge ? <span className="badge" aria-label={`${item.badge} to check`}>{item.badge}</span> : null}
          </NavLink>)}
        </div>)}
      </nav>
    </aside>
    <main id="main" className="desk">
      <Routes>
        <Route path="/" element={<SessionPage />} />
        <Route path="/prepare/camera" element={<CameraCheckPage />} />
        <Route path="/prepare/pitch" element={<PitchSetupPage />} />
        <Route path="/prepare/kits" element={<PlayerTaggingPage />} />
        <Route path="/squad" element={<PlayerMappingPage />} />
        <Route path="/live" element={<LivePage />} />
        <Route path="/review" element={<ReviewPage />} />
        <Route path="/review/:id" element={<ReviewPage />} />
        <Route path="/ingest" element={<MatchUploadPage />} />
        <Route path="/evidence" element={<EvidencePage />} />
        <Route path="/vest" element={<WearablePage />} />
        <Route path="/vest/setup" element={<HardwareFlasherPage />} />
        <Route path="/workshop/adapt" element={<AdaptPage />} />
        <Route path="/workshop/training" element={<TrainingHubPage />} />
        <Route path="/workshop/cameras" element={<CameraNetworkPage />} />
        <Route path="/workshop/runtime" element={<Navigate to="/workshop/training" replace />} />
        <Route path="/prepare/calibrate" element={<Navigate to="/prepare/pitch" replace />} />
        <Route path="/recordings" element={<Navigate to="/live" replace />} />
        <Route path="/biometrics" element={<Navigate to="/vest" replace />} />
        <Route path="/analytics" element={<Navigate to="/review" replace />} />
        <Route path="/roster" element={<Navigate to="/squad" replace />} />
        <Route path="/tagging" element={<Navigate to="/prepare/kits" replace />} />
        <Route path="/flasher" element={<Navigate to="/vest/setup" replace />} />
        <Route path="/training" element={<Navigate to="/workshop/training" replace />} />
        <Route path="/cameras" element={<Navigate to="/workshop/cameras" replace />} />
        <Route path="/settings" element={<Navigate to="/workshop/training" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </main>
  </div>;
};

export default App;
