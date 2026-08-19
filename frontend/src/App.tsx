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
import { SettingsPage } from "./pages/SettingsPage";
import {
  IconRecordings,
  IconCalendar,
  IconClips,
  IconDataExplorer,
  IconWearable,
  IconModelHub,
  IconTagPanels,
  IconSettings,
} from "./components/Icons";

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="app-shell">
        {/* Left Sidebar */}
        <aside className="sidebar">
          <div>
            <div className="sidebar-logo">
              <div className="logo-circle">S</div>
              <div className="logo-text">CPG44</div>
            </div>

            <nav className="nav-menu">
              <NavLink to="/recordings" className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
                <IconRecordings size={18} />
                <span>Recordings</span>
              </NavLink>
              <NavLink to="/biometrics" className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
                <IconWearable size={18} />
                <span>Biometrics</span>
              </NavLink>
              <NavLink to="/flasher" className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
                <IconCalendar size={18} />
                <span>ESP32 Flasher</span>
              </NavLink>
              <NavLink to="/training" className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
                <IconModelHub size={18} />
                <span>Training Hub</span>
              </NavLink>
              <NavLink to="/ingest" className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
                <IconClips size={18} />
                <span>Video Ingest</span>
              </NavLink>
              <NavLink to="/cameras" className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
                <IconDataExplorer size={18} />
                <span>Cameras</span>
              </NavLink>
              <NavLink to="/tagging" className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
                <IconTagPanels size={18} />
                <span>Tag Panels</span>
              </NavLink>
              <NavLink to="/settings" className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
                <IconSettings size={18} />
                <span>Settings</span>
              </NavLink>
            </nav>
          </div>

          <div className="user-profile">
            <div className="avatar">SN</div>
            <div>
              <div style={{ fontSize: "0.8rem", fontWeight: 700 }}>Siddartha</div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Analyst</div>
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
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
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
};

export default App;
