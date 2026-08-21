import { useEffect, useState } from 'react';
import { EmptyState, ErrorBanner, PageHeader, StatusBadge } from '../components/common';
import { api } from '../lib/api';

type TeamForm = { name: string; short_name: string; color: string };
type EventRow = { id: string; timestamp_s: number; time_str: string; type: string; team?: number | null; player_jersey?: number | null; description: string; source: string };

const rgbToHex = (rgb: unknown) => Array.isArray(rgb) && rgb.length === 3
  ? `#${rgb.map((value) => Number(value).toString(16).padStart(2, '0')).join('')}`
  : '#64748b';
const hexToRgb = (hex: string) => [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16));

export const PlayerTaggingPage = () => {
  const [team1, setTeam1] = useState<TeamForm>({ name: 'Team 1', short_name: 'T1', color: '#2563eb' });
  const [team2, setTeam2] = useState<TeamForm>({ name: 'Team 2', short_name: 'T2', color: '#b91c1c' });
  const [calibrated, setCalibrated] = useState(false);
  const [events, setEvents] = useState<EventRow[]>([]);
  const [eventType, setEventType] = useState('pass');
  const [eventTeam, setEventTeam] = useState(1);
  const [jersey, setJersey] = useState('');
  const [seconds, setSeconds] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const [profiles, eventRows] = await Promise.all([api.teamProfiles(), api.events()]);
      const one = profiles.team_1 as Record<string, unknown> | undefined;
      const two = profiles.team_2 as Record<string, unknown> | undefined;
      if (one) setTeam1({ name: String(one.name ?? 'Team 1'), short_name: String(one.short_name ?? 'T1'), color: rgbToHex(one.primary_color_rgb) });
      if (two) setTeam2({ name: String(two.name ?? 'Team 2'), short_name: String(two.short_name ?? 'T2'), color: rgbToHex(two.primary_color_rgb) });
      setCalibrated(profiles.calibrated === true);
      setEvents(eventRows as unknown as EventRow[]);
      setError(null);
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Tagging API unavailable'); }
  };
  useEffect(() => { void refresh(); }, []);

  const saveTeams = async () => {
    try {
      await api.saveTeamProfiles({
        team_1: { name: team1.name, short_name: team1.short_name, primary_color_rgb: hexToRgb(team1.color) },
        team_2: { name: team2.name, short_name: team2.short_name, primary_color_rgb: hexToRgb(team2.color) },
      });
      await refresh();
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Profiles were not saved'); }
  };

  const addEvent = async () => {
    try {
      await api.addEvent({
        type: eventType,
        team: eventTeam,
        player_jersey: jersey || null,
        timestamp_s: Number(seconds),
        description: description || eventType,
      });
      setDescription(''); setJersey('');
      await refresh();
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Event was not saved'); }
  };

  const teamEditor = (team: TeamForm, setTeam: (value: TeamForm) => void) => (
    <div className="team-editor">
      <input className="color-input" type="color" value={team.color} onChange={(event) => setTeam({ ...team, color: event.target.value })} aria-label={`${team.name} primary colour`} />
      <label className="field"><span>Team name</span><input value={team.name} onChange={(event) => setTeam({ ...team, name: event.target.value })} /></label>
      <label className="field short"><span>Code</span><input value={team.short_name} maxLength={8} onChange={(event) => setTeam({ ...team, short_name: event.target.value })} /></label>
    </div>
  );

  return (
    <div className="page">
      <PageHeader eyebrow="HUMAN REVIEW" title="Team and event tagging" description="Calibrate kit colours from the actual match, then add time-coded observations with explicit manual provenance." actions={<StatusBadge status={calibrated ? 'ready' : 'unavailable'} />} />
      <ErrorBanner message={error} />
      <div className="two-column">
        <section className="card-clean">
          <div className="section-heading"><div><span className="eyebrow">KIT CALIBRATION</span><h2>Team colour references</h2></div></div>
          <p className="body-copy">Pick representative shirt colours from a well-lit frame. The backend converts RGB to CIELAB, reports colour distance, and abstains when the two profiles are ambiguous.</p>
          <div className="team-editor-list">{teamEditor(team1, setTeam1)}{teamEditor(team2, setTeam2)}</div>
          <button className="btn-solid" onClick={saveTeams}>Save calibrated profiles</button>
        </section>
        <section className="card-clean">
          <div className="section-heading"><div><span className="eyebrow">MATCH TIMELINE</span><h2>Add reviewed event</h2></div></div>
          <div className="form-grid">
            <label className="field"><span>Event</span><select value={eventType} onChange={(event) => setEventType(event.target.value)}><option value="pass">Pass</option><option value="shot">Shot</option><option value="tackle">Tackle</option><option value="interception">Interception</option><option value="goal">Goal</option><option value="sprint">Sprint</option><option value="note">Note</option></select></label>
            <label className="field"><span>Team</span><select value={eventTeam} onChange={(event) => setEventTeam(Number(event.target.value))}><option value={1}>{team1.name}</option><option value={2}>{team2.name}</option></select></label>
            <label className="field"><span>Time in seconds</span><input type="number" min="0" value={seconds} onChange={(event) => setSeconds(event.target.value)} placeholder="e.g. 372.4" /></label>
            <label className="field"><span>Jersey (optional)</span><input type="number" min="0" value={jersey} onChange={(event) => setJersey(event.target.value)} /></label>
            <label className="field wide"><span>Description</span><input value={description} onChange={(event) => setDescription(event.target.value)} /></label>
          </div>
          <button className="btn-solid" disabled={seconds === '' || Number(seconds) < 0} onClick={addEvent}>Save manual event</button>
        </section>
      </div>
      <section className="card-clean">
        <div className="section-heading"><div><span className="eyebrow">REVIEW LOG</span><h2>Verified timeline</h2></div><span className="muted">{events.length} events</span></div>
        {!events.length ? <EmptyState title="No verified events. Seeded demonstration events are intentionally excluded." /> : (
          <div className="event-list">{events.map((event) => <article key={event.id}><time>{event.time_str}</time><div><strong>{event.description}</strong><span>{event.type} | team {event.team ?? '-'} | jersey {event.player_jersey ?? '-'} | {event.source}</span></div></article>)}</div>
        )}
      </section>
    </div>
  );
};
