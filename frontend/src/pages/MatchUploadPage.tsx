import { useEffect, useState } from 'react';
import { EmptyState, ErrorBanner, PageHeader, StatusBadge } from '../components/common';
import { IconClips } from '../components/Icons';
import { api } from '../lib/api';

type Job = { match_id: string; status: string; progress_pct?: number | null; current_frame?: number; total_frames?: number; fps?: number | null; error_message?: string | null; output_video?: string | null; output_stats?: string | null; log_path?: string | null };

export const MatchUploadPage = () => {
  const [file, setFile] = useState<File | null>(null);
  const [calibration, setCalibration] = useState<File | null>(null);
  const [name, setName] = useState('Campus match');
  const [job, setJob] = useState<Job | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!job || ['completed', 'failed'].includes(job.status)) return;
    let cancelled = false;
    const timer = window.setInterval(() => api.progress(job.match_id).then((body) => {
      if (!cancelled) setJob(body as unknown as Job);
    }).catch((reason: unknown) => { if (!cancelled) setError(reason instanceof Error ? reason.message : 'Progress endpoint unavailable'); }), 1500);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [job?.match_id, job?.status]);

  const upload = async () => {
    if (!file) return;
    setUploading(true); setError(null); setJob(null);
    const form = new FormData();
    form.append('file', file);
    form.append('match_name', name);
    if (calibration) form.append('calibration', calibration);
    try {
      const response = await api.uploadMatch(form);
      setJob({ match_id: response.match_id, status: response.status, progress_pct: 0 });
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Upload failed'); }
    finally { setUploading(false); }
  };

  return (
    <div className="page">
      <PageHeader eyebrow="VIDEO PIPELINE" title="Ingest match footage" description="Upload a real clip and follow the detector, ByteTrack, Re-ID and analytics subprocess through to its artifacts." actions={job && <StatusBadge status={job.status} />} />
      <ErrorBanner message={error ?? job?.error_message} />
      <div className="two-column">
        <section className="card-clean">
          <div className="section-heading"><div><span className="eyebrow">SOURCE</span><h2>Video file</h2></div></div>
          <label className="field"><span>Session name</span><input value={name} onChange={(event) => setName(event.target.value)} /></label>
          <label className="drop-zone">
            <input type="file" accept="video/mp4,video/quicktime,video/x-matroska,video/webm" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
            <IconClips size={30} />
            <strong>{file?.name ?? 'Choose match footage'}</strong>
            <span>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : 'MP4, MOV, MKV or WebM'}</span>
          </label>
          <button className="btn-solid" disabled={!file || uploading || Boolean(job && !['completed', 'failed'].includes(job.status))} onClick={upload}>{uploading ? 'Uploading…' : 'Upload and start analysis'}</button>
        </section>
        <section className="card-clean">
          <div className="section-heading"><div><span className="eyebrow">CALIBRATION</span><h2>Metric geometry</h2></div></div>
          <div className="quality-callout neutral"><strong>Camera-specific evidence</strong><p>Create and review a landmark file, then attach it to this job:</p><code>python -m soccer_analytics.calibrate --video footage.mp4 --out camera.yaml</code></div>
          <label className="field"><span>Calibration YAML (optional)</span><input type="file" accept=".yaml,.yml,application/yaml" onChange={(event) => setCalibration(event.target.files?.[0] ?? null)} /></label>
          <p className="form-help">Selected: {calibration?.name ?? 'none. Analysis will stay in pixel mode'}. A calibration must match this camera view.</p>
        </section>
      </div>
      <section className="card-clean">
        <div className="section-heading"><div><span className="eyebrow">PROCESS</span><h2>Analysis job</h2></div>{job && <StatusBadge status={job.status} />}</div>
        {!job ? <EmptyState title="No upload job in this browser session." /> : (
          <div className="job-card">
            <div className="progress"><span style={{ width: `${job.progress_pct ?? 0}%` }} /></div>
            <div className="compact-metrics"><span>Progress <strong>{job.progress_pct == null ? 'starting' : `${job.progress_pct.toFixed(1)}%`}</strong></span><span>Frames <strong>{job.current_frame ?? 0} / {job.total_frames || '-'}</strong></span><span>Throughput <strong>{job.fps ? `${job.fps.toFixed(2)} fps` : '-'}</strong></span></div>
            {job.output_stats && <p className="artifact-path"><strong>Stats:</strong> {job.output_stats}</p>}
            {job.output_video && <p className="artifact-path"><strong>Video:</strong> {job.output_video}</p>}
            {job.log_path && <p className="artifact-path"><strong>Log:</strong> {job.log_path}</p>}
          </div>
        )}
      </section>
    </div>
  );
};
