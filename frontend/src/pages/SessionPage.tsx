import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Glyph, type GlyphName } from '../components/Glyph';
import { api, apiUrl } from '../lib/api';
import type { AnalysisSummary } from '../lib/analysis';
import { clock, listAnalyses } from '../lib/analysis';

type State = 'done' | 'warn' | 'todo';
interface Step { to: string; g: GlyphName; title: string; status: string; state: State }

export const SessionPage = () => {
  const [calibs, setCalibs] = useState<Array<{ id: string }>>();
  const [vest, setVest] = useState<Record<string, unknown>>();
  const [teams, setTeams] = useState<Record<string, unknown>>();
  const [recent, setRecent] = useState<AnalysisSummary[]>();
  const [backend, setBackend] = useState<boolean>();

  useEffect(() => {
    api.health().then(() => setBackend(true)).catch(() => setBackend(false));
    fetch(apiUrl('/api/v1/calibrations')).then(r => r.json()).then(setCalibs).catch(() => setCalibs([]));
    api.wearableStatus().then(setVest).catch(() => setVest({ connected: false }));
    api.teamProfiles().then(setTeams).catch(() => setTeams({}));
    listAnalyses().then(setRecent).catch(() => setRecent([]));
  }, []);

  const kits = Boolean((teams as { calibrated?: boolean } | undefined)?.calibrated);
  const steps: Step[] = [
    { to: '/prepare/camera', g: 'camera', title: 'Camera', status: 'check the view', state: 'todo' },
    { to: '/prepare/pitch', g: 'pitch', title: 'Pitch', status: calibs?.length ? `${calibs.length} saved` : 'not calibrated', state: calibs?.length ? 'done' : 'todo' },
    { to: '/prepare/kits', g: 'kits', title: 'Kits', status: kits ? 'colours set' : 'automatic', state: kits ? 'done' : 'warn' },
    { to: '/squad', g: 'vest', title: 'Vests', status: vest?.connected ? (vest.clock_synced ? 'streaming' : 'syncing clock') : 'none streaming', state: vest?.connected ? (vest.clock_synced ? 'done' : 'warn') : 'todo' },
    { to: recent?.length ? `/review/${recent[0].id}` : '/ingest', g: 'review', title: 'Matches', status: recent?.length ? `${recent.length} analysed` : 'add footage', state: recent?.length ? 'done' : 'todo' },
  ];

  return <div className="page home">
    <div className="top"><h1>Matchday</h1>
      <div style={{ display: 'flex', gap: 8 }}><Link className="btn ghost" to="/live"><Glyph g="live" size={16} />Live</Link><Link className="btn" to="/ingest"><Glyph g="upload" size={16} />Add footage</Link></div>
    </div>
    {backend === false && <p className="notice bad">The analysis service is not running. Start it with scripts/demo.sh</p>}

    <nav className="flow" aria-label="Session steps">
      {steps.map(s => <Link key={s.title} to={s.to} className={s.state}>
        <span className="ico"><Glyph g={s.g} size={20} /></span>
        <strong>{s.title}</strong><small>{s.status}</small>
        {s.state === 'done' && <span className="tick state-ok"><Glyph g="check" size={16} /></span>}
      </Link>)}
    </nav>

    <section aria-label="Recent matches">
      <div className="panel-head"><h2>Recent matches</h2></div>
      {!recent?.length ? <div className="empty panel"><Glyph g="review" size={34} />No matches yet<Link className="btn ghost small" to="/ingest">Add footage</Link></div>
        : <div className="matches">{recent.map(r => <Link key={r.id} to={`/review/${r.id}`} className="match-card">
          {r.has_video ? <img src={apiUrl(`/api/v1/analyses/${r.id}/thumb`)} alt="" loading="lazy" /> : <img alt="" />}
          <div><strong>{r.name}</strong><span>{clock(r.duration_s)}{r.checks_pending ? <b className="state-warn" title="to check"> {r.checks_pending}</b> : null}</span></div>
        </Link>)}</div>}
    </section>
  </div>;
};
