import { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { PassingPitch } from '../components/PassingPitch';
import { EmptyState, ErrorBanner, PageHeader, ProvenanceBanner, StatusBadge, Value } from '../components/common';
import { IconDownload, IconPause, IconPlay, IconStepBack5, IconStepForward5 } from '../components/Icons';
import { apiUrl } from '../lib/api';
import { useLive } from '../lib/useLive';

type PitchNode = { id: number; name: string; x: number; y: number; size: number; passes: number };
type PitchLink = { source: number; target: number; weight: number };

export const LivePage = () => {
  const { data, connected, error } = useLive('live');
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [playing, setPlaying] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [view, setView] = useState<'map' | 'network'>('map');
  const [selected, setSelected] = useState<number>();
  const [frameNonce, setFrameNonce] = useState(0);
  const [frameReady, setFrameReady] = useState(false);

  useEffect(() => {
    if (!data?.provenance.live) return;
    const timer = window.setInterval(() => setFrameNonce(Date.now()), 160);
    return () => window.clearInterval(timer);
  }, [data?.provenance.live]);

  const nodes = useMemo<PitchNode[]>(() => (data?.players ?? [])
    .filter((player) => typeof player.x === 'number' && typeof player.y === 'number')
    .map((player) => ({
      id: player.track_id,
      name: player.jersey ? `#${player.jersey}` : player.global_player_id,
      x: player.x as number,
      y: player.y as number,
      size: 24,
      passes: 0,
    })), [data]);

  const links = useMemo<PitchLink[]>(() => {
    const network = data?.passing_network as { team_1?: { links?: PitchLink[] } } | undefined;
    return network?.team_1?.links ?? [];
  }, [data]);

  const possession1 = data?.possession_pct?.['1'] ?? 0;
  const possession2 = data?.possession_pct?.['2'] ?? 0;
  const processedFootage = data?.source_kind === 'recorded_file' || data?.provenance.input_kind === 'recorded_file';
  const formatClock = (seconds: number) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;

  const exportSnapshot = () => {
    if (!data) return;
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const href = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = href;
    anchor.download = `cpg44-${data.match_id}-${Math.floor(data.timestamp)}.json`;
    anchor.click();
    URL.revokeObjectURL(href);
  };

  return (
    <div className="page">
      <PageHeader
        eyebrow="MATCH WORKSPACE"
        title={data?.match?.name ?? 'Campus football session'}
        description="Measured vision and wearable evidence, with source age and calibration state kept visible."
        actions={(
          <>
            <Link className="btn-outline" to="/tagging">Open tagging</Link>
            <button className="btn-solid with-icon" onClick={exportSnapshot} disabled={!data}>
              <IconDownload size={16} /> Export snapshot
            </button>
          </>
        )}
      />
      <ErrorBanner message={error} />
      <ProvenanceBanner snapshot={data} />

      <div className="metric-strip">
        <div><span>API</span><strong>{connected ? 'Connected' : 'Offline'}</strong></div>
        <div><span>Source</span><strong>{processedFootage ? 'Processed footage' : data?.provenance.live ? 'Live camera' : data?.mode ?? '-'}</strong></div>
        <div><span>Players in evidence</span><strong>{data?.players.length ?? 0}</strong></div>
        <div><span>Possession</span><strong>{possession1.toFixed(0)}-{possession2.toFixed(0)}%</strong></div>
      </div>

      {(data?.data_quality?.warnings?.length ?? 0) > 0 && (
        <div className="quality-callout">
          <strong>Data quality review</strong>
          <ul>{data?.data_quality.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
        </div>
      )}

      <div className="workspace-grid">
        <section className="card-clean pitch-card">
          <div className="section-heading">
            <div>
              <span className="eyebrow">TACTICAL POSITION</span>
              <h2>Pitch view</h2>
            </div>
            <div className="segmented">
              <button className={view === 'map' ? 'active' : ''} onClick={() => setView('map')}>Map</button>
              <button className={view === 'network' ? 'active' : ''} onClick={() => setView('network')} disabled={!links.length}>Network</button>
            </div>
          </div>
          <div className="pitch-container">
            {nodes.length ? (
              <PassingPitch nodes={nodes} links={links} mode={view} selectedPlayerId={selected} onSelectPlayer={setSelected} />
            ) : (
              <EmptyState title="No metric player coordinates are available. Calibrate the camera and publish a live frame." />
            )}
          </div>
        </section>

        <section className="video-player-card">
          <div className="section-heading video-heading">
            <div>
              <span className="eyebrow">VIDEO EVIDENCE</span>
              <h2>{processedFootage ? 'Footage with live markings' : data?.provenance.live ? 'Live annotated field view' : 'Recorded match clip'}</h2>
            </div>
            <StatusBadge status={processedFootage ? 'processing' : data?.provenance.live ? 'live' : 'recorded'} />
          </div>
          <div className="video-frame-box">
            {data?.provenance.live ? (
              <>
                {!frameReady && <div className="frame-wait">Waiting for an annotated frame…</div>}
                <img className="live-annotated-frame" src={`${apiUrl('/api/v1/live/frame')}?t=${frameNonce}`} alt="Live annotated match frame" onLoad={() => setFrameReady(true)} onError={() => setFrameReady(false)} />
              </>
            ) : data?.provenance.kind === 'recorded_analysis' ? (
              <video
                ref={videoRef}
                src={apiUrl(`/api/v1/video/stream/${encodeURIComponent(data.match_id)}`)}
                autoPlay muted loop playsInline
                onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
              />
            ) : (
              <EmptyState title="No live or recorded video evidence is selected." />
            )}
            {!data?.provenance.live && data?.provenance.kind === 'recorded_analysis' && <div className="floating-video-controls">
              <button className="ctrl-btn" aria-label="Back five seconds" onClick={() => { if (videoRef.current) videoRef.current.currentTime -= 5; }}><IconStepBack5 size={18} /></button>
              <button className="ctrl-btn play" aria-label={playing ? 'Pause' : 'Play'} onClick={() => {
                const video = videoRef.current;
                if (!video) return;
                if (video.paused) {
                  void video.play();
                  setPlaying(true);
                } else {
                  video.pause();
                  setPlaying(false);
                }
              }}>{playing ? <IconPause size={18} /> : <IconPlay size={18} />}</button>
              <button className="ctrl-btn" aria-label="Forward five seconds" onClick={() => { if (videoRef.current) videoRef.current.currentTime += 5; }}><IconStepForward5 size={18} /></button>
            </div>}
          </div>
          {!data?.provenance.live && data?.provenance.kind === 'recorded_analysis' && <div className="timeline-scrubber-card">
            <div className="timeline-info-row"><span>Recorded source</span><span>{formatClock(currentTime)} / {formatClock(duration)}</span></div>
            <input className="video-scrubber" type="range" min="0" max={duration || 1} value={currentTime} onChange={(event) => {
              const next = Number(event.target.value);
              if (videoRef.current) videoRef.current.currentTime = next;
              setCurrentTime(next);
            }} />
          </div>}
        </section>
      </div>

      <section className="card-clean">
        <div className="section-heading">
          <div><span className="eyebrow">TRACK AUDIT</span><h2>Current player evidence</h2></div>
          <span className="muted">Track IDs are not jersey identities until reviewed.</span>
        </div>
        {!data?.players.length ? <EmptyState title="No player observations are available." /> : (
          <div className="table-scroll">
            <table>
              <thead><tr><th>Identity</th><th>Team</th><th>Vest</th><th>Distance (m)</th><th>Top speed (m/s)</th><th>Review score</th></tr></thead>
              <tbody>{data.players.slice(0, 30).map((player) => (
                <tr key={player.global_player_id}>
                  <td><strong>{player.jersey ? `#${player.jersey}` : player.global_player_id}</strong></td>
                  <td>{player.team || 'unassigned'}</td>
                  <td>{player.wearable ? 'linked' : '-'}</td>
                  <td><Value value={player.distance_m} /></td>
                  <td><Value value={player.top_speed_mps} digits={2} /></td>
                  <td>{player.load_indicator?.score == null ? '-' : `${Math.round(player.load_indicator.score * 100)} / 100 ${player.load_indicator.severity ?? ''}`}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
};
