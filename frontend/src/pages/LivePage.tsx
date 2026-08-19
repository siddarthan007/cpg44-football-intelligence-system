import React, { useEffect, useRef, useState } from "react";
import { PassingPitch } from "../components/PassingPitch";
import {
  IconDownload,
  IconBookmark,
  IconAnnotate,
  IconJersey,
  IconPlay,
  IconPause,
  IconStepBack5,
  IconStepForward5,
  IconInfo,
} from "../components/Icons";

export const LivePage: React.FC = () => {
  const [tacticalTab, setTacticalTab] = useState<"map" | "network" | "heatmap">("network");
  const [period, setPeriod] = useState<"1st" | "2nd">("2nd");
  const [isPlaying, setIsPlaying] = useState(true);
  const [currentTime, setCurrentTime] = useState(2492); // 41:32 in seconds
  const [selectedPlayer, setSelectedPlayer] = useState<number>(27);
  const [matchData, setMatchData] = useState<any>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    // Fetch live state from backend
    const poll = setInterval(() => {
      fetch(`http://${window.location.hostname}:8000/api/v1/matches/live/analytics`)
        .then((r) => r.json())
        .then((d) => setMatchData(d))
        .catch(() => {});
    }, 500);
    return () => clearInterval(poll);
  }, []);

  const passingNodes = matchData?.passing_network?.team_1?.nodes || [
    { id: 27, name: "R. Edwards", x: 52.5, y: 34.0, size: 36, passes: 48 },
    { id: 20, name: "K. Koffie", x: 42.0, y: 24.0, size: 28, passes: 32 },
    { id: 85, name: "N. Hackshaw", x: 32.0, y: 36.0, size: 26, passes: 29 },
    { id: 5, name: "J. Cochran", x: 62.0, y: 22.0, size: 27, passes: 31 },
    { id: 17, name: "M. Arteaga", x: 68.0, y: 42.0, size: 30, passes: 38 },
    { id: 29, name: "S. Guenzatti", x: 54.0, y: 52.0, size: 24, passes: 22 },
  ];

  const passingLinks = matchData?.passing_network?.team_1?.links || [
    { source: 27, target: 20, weight: 14 },
    { source: 27, target: 85, weight: 18 },
    { source: 27, target: 5, weight: 12 },
    { source: 27, target: 17, weight: 22 },
    { source: 27, target: 29, weight: 11 },
  ];

  const timelineTags = matchData?.timeline_tags || [
    { id: "t1", time: 240, label: "Pass" },
    { id: "t2", time: 720, label: "Shot" },
    { id: "t3", time: 1850, label: "Sprint" },
    { id: "t4", time: 2492, label: "Current Play" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Top Action Toolbar (matching reference image) */}
      <div className="top-action-bar">
        <div className="tool-buttons">
          <button className="icon-btn" title="Export Match JSON">
            <IconDownload size={16} />
          </button>
          <button className="icon-btn" title="Bookmark Tag">
            <IconBookmark size={16} />
          </button>
          <button className="icon-btn" title="Annotate Draw">
            <IconAnnotate size={16} />
          </button>
          <button className="icon-btn" title="Jersey Mapping">
            <IconJersey size={16} />
          </button>
        </div>

        <div className="match-header-info">
          <span>ATL vs IND - 8/8</span>
          <IconInfo size={18} color="#64748b" />
        </div>
      </div>

      {/* Main Tactical + Video Grid */}
      <div className="match-view-grid">
        {/* Left Side Tactical Panel */}
        <div className="tactical-side-panel">
          <div className="panel-header-tabs">
            <div className="dropdown-title">
              Pass data <span style={{ fontSize: "0.75rem" }}>▼</span>
            </div>
            <div className="tab-pill-group">
              <button
                className={`tab-pill ${tacticalTab === "map" ? "active" : ""}`}
                onClick={() => setTacticalTab("map")}
              >
                Map
              </button>
              <button
                className={`tab-pill ${tacticalTab === "network" ? "active" : ""}`}
                onClick={() => setTacticalTab("network")}
              >
                Network
              </button>
              <button
                className={`tab-pill ${tacticalTab === "heatmap" ? "active" : ""}`}
                onClick={() => setTacticalTab("heatmap")}
              >
                Heatmap
              </button>
            </div>
          </div>

          <div className="team-period-controls">
            <select className="select-input">
              <option>Indy Eleven</option>
              <option>Louisville City</option>
            </select>
            <button
              className={`btn-period ${period === "1st" ? "active" : ""}`}
              onClick={() => setPeriod("1st")}
            >
              1st
            </button>
            <button
              className={`btn-period ${period === "2nd" ? "active" : ""}`}
              onClick={() => setPeriod("2nd")}
            >
              2nd
            </button>
          </div>

          {/* Vertical Passing Pitch */}
          <div className="pitch-container">
            <PassingPitch
              nodes={passingNodes}
              links={passingLinks}
              mode={tacticalTab}
              selectedPlayerId={selectedPlayer}
              onSelectPlayer={(id) => setSelectedPlayer(id)}
            />
          </div>
        </div>

        {/* Center / Right Video Player & Scrubber */}
        <div className="video-player-card">
          <div className="video-frame-box">
            {/* HTML5 Video Streaming Endpoint */}
            <video
              ref={videoRef}
              src={`http://${window.location.hostname}:8000/api/v1/video/stream/live`}
              style={{ width: "100%", height: "100%", objectFit: "cover" }}
              autoPlay
              muted
              loop
              playsInline
            />

            {/* Overlaid Floating Playback Controls (Matching Reference Image) */}
            <div className="floating-video-controls">
              <button
                className="ctrl-btn"
                onClick={() => {
                  if (videoRef.current) videoRef.current.currentTime -= 5;
                }}
              >
                <IconStepBack5 size={18} />
              </button>
              <button
                className="ctrl-btn play"
                onClick={() => {
                  if (videoRef.current) {
                    if (isPlaying) videoRef.current.pause();
                    else videoRef.current.play();
                    setIsPlaying(!isPlaying);
                  }
                }}
              >
                {isPlaying ? <IconPause size={18} /> : <IconPlay size={18} />}
              </button>
              <button
                className="ctrl-btn"
                onClick={() => {
                  if (videoRef.current) videoRef.current.currentTime += 5;
                }}
              >
                <IconStepForward5 size={18} />
              </button>
            </div>

            {/* Floating Zoom Controls */}
            <div className="zoom-controls">
              <button className="zoom-btn">-</button>
              <button className="zoom-btn">+</button>
            </div>
          </div>

          {/* Scrubber Timeline Bar (Matching Reference Image) */}
          <div className="timeline-scrubber-card">
            <div className="timeline-info-row">
              <span>▲</span>
              <span>00:41:32 / 03:00:28</span>
              <span>770 Tags ▾</span>
            </div>

            {/* Progress Scrubber */}
            <div className="scrubber-track">
              <div className="scrubber-fill" style={{ width: "46%" }}></div>
            </div>

            {/* Tag Timeline Track */}
            <div className="tag-markers-row">
              <span className="tag-pill">▶ Tags</span>
              <div className="tag-bar-track">
                {timelineTags.map((t: any, i: number) => (
                  <div
                    key={t.id}
                    className={`tag-marker ${i % 2 === 0 ? "pass" : "sprint"}`}
                    style={{ left: `${(i + 1) * 22}%`, width: "24px" }}
                    title={t.label}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
