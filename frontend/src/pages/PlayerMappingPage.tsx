import { useEffect, useMemo, useState } from 'react';
import { Glyph } from '../components/Glyph';
import { apiUrl } from '../lib/api';

type Row = { name: string; number: number | null; position: string | null; vest: number | null };
type Squad = { team_name: string; players: Row[] };

const POSITIONS = ['GK', 'DF', 'MF', 'FW'];
const blank = (): Row => ({ name: '', number: null, position: null, vest: null });

export const PlayerMappingPage = () => {
  const [squad, setSquad] = useState<Squad>({ team_name: '', players: [] });
  const [saved, setSaved] = useState<string>('');
  const [status, setStatus] = useState<{ ok: boolean; text: string }>();

  useEffect(() => {
    fetch(apiUrl('/api/v1/squad')).then(r => r.json()).then((s: Squad) => {
      const next = s.players.length ? s : { ...s, players: [blank()] };
      setSquad(next); setSaved(JSON.stringify(s));
    }).catch(() => setStatus({ ok: false, text: 'The API is not running' }));
  }, []);

  const rows = squad.players;
  const dupes = useMemo(() => {
    const seen = new Map<string, number>();
    rows.forEach(r => { if (r.number != null) seen.set(`n${r.number}`, (seen.get(`n${r.number}`) ?? 0) + 1); if (r.vest != null) seen.set(`v${r.vest}`, (seen.get(`v${r.vest}`) ?? 0) + 1); });
    return seen;
  }, [rows]);
  const set = (i: number, patch: Partial<Row>) => setSquad({ ...squad, players: rows.map((r, k) => k === i ? { ...r, ...patch } : r) });
  const num = (v: string) => v === '' ? null : Math.max(0, Math.min(9999, Math.round(Number(v))));
  const cleaned = { ...squad, players: rows.filter(r => r.name.trim()) };
  const dirty = JSON.stringify(cleaned) !== saved;
  const vests = rows.filter(r => r.vest != null).length;

  const save = async () => {
    const r = await fetch(apiUrl('/api/v1/squad'), { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cleaned) });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) { setStatus({ ok: false, text: (body as { detail?: string }).detail ?? `HTTP ${r.status}` }); return; }
    setSaved(JSON.stringify(body)); setStatus({ ok: true, text: 'Saved' });
  };

  return <div className="page">
    <header className="page-header"><h1>Squad and vests</h1>
      <p>Shirt numbers let the camera name players; a vest number links each vest to its wearer.</p></header>

    <section className="panel">
      <div className="panel-head">
        <input className="squad-team" aria-label="Team name" placeholder="Team name" value={squad.team_name}
          onChange={e => setSquad({ ...squad, team_name: e.target.value })} />
        <span className="num muted">{cleaned.players.length} players, {vests} with a vest</span>
      </div>
      <div className="squad-grid" role="table" aria-label="Squad">
        <div className="squad-row head" role="row"><span>No.</span><span>Name</span><span>Position</span><span>Vest</span><span /></div>
        {rows.map((r, i) => <div key={i} className="squad-row" role="row">
          <input aria-label="Shirt number" inputMode="numeric" className={`num ${r.number != null && (dupes.get(`n${r.number}`) ?? 0) > 1 ? 'bad' : ''}`}
            value={r.number ?? ''} onChange={e => set(i, { number: num(e.target.value) })} placeholder="7" />
          <input aria-label="Name" value={r.name} onChange={e => set(i, { name: e.target.value })} placeholder="Player name" />
          <div className="seg small" role="group" aria-label="Position">{POSITIONS.map(p =>
            <button key={p} aria-pressed={r.position === p} onClick={() => set(i, { position: r.position === p ? null : p })}>{p}</button>)}</div>
          <input aria-label="Vest number" inputMode="numeric" className={`num vest ${r.vest != null && (dupes.get(`v${r.vest}`) ?? 0) > 1 ? 'bad' : ''}`}
            value={r.vest ?? ''} onChange={e => set(i, { vest: num(e.target.value) })} placeholder="none" />
          <button className="icon-btn" aria-label={`Remove ${r.name || 'row'}`} onClick={() => setSquad({ ...squad, players: rows.filter((_, k) => k !== i) })}><Glyph g="x" size={14} /></button>
        </div>)}
      </div>
      <div className="squad-actions">
        <button className="btn ghost small" onClick={() => setSquad({ ...squad, players: [...rows, blank()] })}>Add player</button>
        {status && <span className={status.ok ? 'state-ok' : 'state-bad'} role="status">{status.text}</span>}
        <button className="btn small" disabled={!dirty || [...dupes.values()].some(v => v > 1)} onClick={() => void save()} style={{ marginLeft: 'auto' }}>
          <Glyph g="save" size={14} />Save squad</button>
      </div>
    </section>

    <section className="panel squad-how">
      <div><Glyph g="eye" size={20} /><p><b>Camera</b> reads shirt numbers over the whole match and names the player once enough clear views agree.</p></div>
      <div><Glyph g="vest" size={20} /><p><b>Vest</b> is matched to the player whose running rises and falls with it, and its clock is aligned from the same match.</p></div>
      <div><Glyph g="check" size={20} /><p><b>Both</b> must agree. If the vest's player shows a different shirt number, the match goes to To check.</p></div>
    </section>
  </div>;
};
