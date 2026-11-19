import { useEffect, useRef, useState } from 'react';
import { Glyph } from '../components/Glyph';
import { api } from '../lib/api';

type Team = { name: string; short_name: string; color: string };

const rgbToHex = (rgb: unknown) => Array.isArray(rgb) && rgb.length === 3
  ? `#${rgb.map((v) => Math.round(Number(v)).toString(16).padStart(2, '0')).join('')}` : '#706e68';
const hexToRgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

/** Median colour of a 9 x 9 patch: robust to a stripe, a number or a shadow. */
function patchColour(ctx: CanvasRenderingContext2D, x: number, y: number) {
  const r = 4, d = ctx.getImageData(Math.max(0, x - r), Math.max(0, y - r), 2 * r + 1, 2 * r + 1).data;
  const ch = [0, 1, 2].map(c => { const v: number[] = []; for (let i = c; i < d.length; i += 4) v.push(d[i]); v.sort((a, b) => a - b); return v[v.length >> 1]; });
  return rgbToHex(ch);
}

export const PlayerTaggingPage = () => {
  const [teams, setTeams] = useState<[Team, Team]>([{ name: 'Home', short_name: 'H', color: '#2251a3' }, { name: 'Away', short_name: 'A', color: '#d1492e' }]);
  const [saved, setSaved] = useState('');
  const [active, setActive] = useState<0 | 1>(0);
  const [frame, setFrame] = useState<string>();
  const [status, setStatus] = useState<{ ok: boolean; text: string }>();
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    api.teamProfiles().then(p => {
      const one = p.team_1 as Record<string, unknown> | undefined, two = p.team_2 as Record<string, unknown> | undefined;
      const next: [Team, Team] = [
        one ? { name: String(one.name ?? 'Home'), short_name: String(one.short_name ?? 'H'), color: rgbToHex(one.primary_color_rgb) } : teams[0],
        two ? { name: String(two.name ?? 'Away'), short_name: String(two.short_name ?? 'A'), color: rgbToHex(two.primary_color_rgb) } : teams[1]];
      setTeams(next); setSaved(JSON.stringify(next));
    }).catch(() => setStatus({ ok: false, text: 'The API is not running' }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!frame || !canvas.current) return;
    const img = new Image();
    img.onload = () => {
      const c = canvas.current!; c.width = img.naturalWidth; c.height = img.naturalHeight;
      c.getContext('2d', { willReadFrequently: true })!.drawImage(img, 0, 0);
    };
    img.src = frame;
  }, [frame]);

  const set = (k: 0 | 1, patch: Partial<Team>) => setTeams(t => (k === 0 ? [{ ...t[0], ...patch }, t[1]] : [t[0], { ...t[1], ...patch }]));
  const pick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const c = canvas.current!, r = c.getBoundingClientRect();
    const x = Math.round((e.clientX - r.left) * c.width / r.width), y = Math.round((e.clientY - r.top) * c.height / r.height);
    set(active, { color: patchColour(c.getContext('2d', { willReadFrequently: true })!, x, y) });
    setActive(active === 0 ? 1 : 0);
  };
  const save = async () => {
    try {
      await api.saveTeamProfiles({
        team_1: { name: teams[0].name, short_name: teams[0].short_name, primary_color_rgb: hexToRgb(teams[0].color) },
        team_2: { name: teams[1].name, short_name: teams[1].short_name, primary_color_rgb: hexToRgb(teams[1].color) },
      });
      setSaved(JSON.stringify(teams)); setStatus({ ok: true, text: 'Saved. Tracking uses these colours from the next match.' });
    } catch (err) { setStatus({ ok: false, text: err instanceof Error ? err.message : 'Not saved' }); }
  };

  return <div className="page">
    <header className="page-header"><h1>Kits</h1><p>Each side's shirt colour, picked from your own footage.</p></header>

    <div className="kits">
      {teams.map((t, k) => <section key={k} className={`panel kit ${active === k ? 'picking' : ''}`} aria-label={`Side ${k + 1}`}>
        <button type="button" className="kit-swatch" style={{ background: t.color }} onClick={() => setActive(k as 0 | 1)}
          aria-pressed={active === k} aria-label={`Pick the ${t.name} colour from the frame`}>
          <Glyph g="kits" size={44} /></button>
        <div className="kit-fields">
          <label htmlFor={`kit-name-${k}`}>Team</label>
          <input id={`kit-name-${k}`} value={t.name} onChange={e => set(k as 0 | 1, { name: e.target.value })} />
          <label htmlFor={`kit-col-${k}`}>Colour</label>
          <div className="kit-colour"><input id={`kit-col-${k}`} type="color" value={t.color} onChange={e => set(k as 0 | 1, { color: e.target.value })} />
            <span className="num">{t.color}</span></div>
        </div>
      </section>)}
    </div>

    <section className="panel" aria-label="Pick from a frame">
      <div className="panel-head"><h2>Pick from a frame</h2>
        <label className="btn ghost small"><input type="file" accept="image/*" hidden onChange={e => { const f = e.target.files?.[0]; if (f) setFrame(URL.createObjectURL(f)); }} />Choose a frame</label></div>
      {frame ? <>
        <p className="muted pick-hint">Click a shirt of <b style={{ color: teams[active].color }}>{teams[active].name}</b>. Pick a plain part of the shirt in daylight, not the number.</p>
        <canvas ref={canvas} className="kit-frame" onClick={pick} />
      </> : <div className="empty"><Glyph g="camera" size={28} />A screenshot from the match footage works best: the same light and camera as the analysis.</div>}
    </section>

    <div className="kit-actions">
      {status && <span className={status.ok ? 'state-ok' : 'state-bad'} role="status">{status.text}</span>}
      <button className="btn" disabled={JSON.stringify(teams) === saved} onClick={() => void save()}><Glyph g="save" size={15} />Save kits</button>
    </div>
  </div>;
};
