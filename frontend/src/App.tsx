import React from "react";
import { BrowserRouter, Routes, Route, NavLink, Navigate } from "react-router-dom";
import { LivePage } from "./pages/LivePage";
import { WearablePage } from "./pages/WearablePage";
import { TrainingHubPage } from "./pages/TrainingHubPage";
import { MatchUploadPage } from "./pages/MatchUploadPage";
import { CameraNetworkPage } from "./pages/CameraNetworkPage";
import { PlayerMappingPage } from "./pages/PlayerMappingPage";
import { HardwareFlasherPage } from "./pages/HardwareFlasherPage";
import { PlayerTaggingPage } from "./pages/PlayerTaggingPage";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { SettingsPage } from "./pages/SettingsPage";
import {
  IconRecordings,
  IconCalendar,
  IconClips,
  IconDataExplorer,
  IconWearable,
  IconModelHub,
  IconTagPanels,
  IconDownload,
  IconSettings,
} from "./components/Icons";

export const App: React.FC = () => {
  const items = [
    { to: "/recordings", label: "Live match", icon: <IconRecordings size={18} /> },
    { to: "/biometrics", label: "Wearables", icon: <IconWearable size={18} /> },
    { to: "/ingest", label: "Video ingest", icon: <IconClips size={18} /> },
    { to: "/tagging", label: "Tagging", icon: <IconTagPanels size={18} /> },
    { to: "/analytics", label: "Analytics", icon: <IconDownload size={18} /> },
    { to: "/training", label: "Training", icon: <IconModelHub size={18} /> },
    { to: "/cameras", label: "Cameras", icon: <IconDataExplorer size={18} /> },
    { to: "/flasher", label: "ESP32 setup", icon: <IconCalendar size={18} /> },
    { to: "/settings", label: "System", icon: <IconSettings size={18} /> },
  ];
  return (
    <BrowserRouter>
      <div className="app-shell">
        <aside className="sidebar">
          <div>
            <div className="sidebar-logo">
              <div className="logo-mark">44</div>
              <div><div className="logo-text">CPG44</div><span className="logo-caption">Field intelligence</span></div>
            </div>

            <nav className="nav-menu">
              {items.map((item) => <NavLink key={item.to} to={item.to} className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>{item.icon}<span>{item.label}</span></NavLink>)}
            </nav>
          </div>

          <div className="workstation-note">
            <span className="workstation-dot" />
            <div><strong>Local workstation</strong><span>Vision and fusion run here</span></div>
          </div>
        </aside>

        <main className="main-viewport">
          <Routes>
            <Route path="/" element={<Navigate to="/recordings" replace />} />
            <Route path="/recordings" element={<LivePage />} />
            <Route path="/biometrics" element={<WearablePage />} />
            <Route path="/flasher" element={<HardwareFlasherPage />} />
            <Route path="/training" element={<TrainingHubPage />} />
            <Route path="/ingest" element={<MatchUploadPage />} />
            <Route path="/cameras" element={<CameraNetworkPage />} />
            <Route path="/tagging" element={<PlayerTaggingPage />} />
            <Route path="/roster" element={<PlayerMappingPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
};

export default App;
