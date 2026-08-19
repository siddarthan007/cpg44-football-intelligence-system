import React from "react";
import { BrowserRouter, Routes, Route, NavLink, Navigate } from "react-router-dom";
import { LivePage } from "./pages/LivePage";
import { WearablePage } from "./pages/WearablePage";
import { MatchUploadPage } from "./pages/MatchUploadPage";
import { CameraNetworkPage } from "./pages/CameraNetworkPage";
import { PlayerMappingPage } from "./pages/PlayerMappingPage";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { SettingsPage } from "./pages/SettingsPage";

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="app-container">
        {/* Top Navigation Bar */}
        <header className="top-navbar">
          <div className="brand-logo">
            <span className="brand-badge">CPG44</span>
            <span>Football Intelligence</span>
          </div>

          <nav className="nav-links">
            <NavLink to="/live" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              ⚽ Live Match
            </NavLink>
            <NavLink to="/upload" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              📤 Video Upload
            </NavLink>
            <NavLink to="/wearable" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              🫀 Wearable Intelligence
            </NavLink>
            <NavLink to="/cameras" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              📡 Camera Network
            </NavLink>
            <NavLink to="/players" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              👥 Squad Mapping
            </NavLink>
            <NavLink to="/analytics" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              📊 Analytics
            </NavLink>
            <NavLink to="/settings" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              ⚙️ Settings
            </NavLink>
          </nav>

          <div className="system-status-pill">
            <div className="pulse-dot"></div>
            <span>RTX 5060 GPU Online</span>
          </div>
        </header>

        {/* Page Routing */}
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Navigate to="/live" replace />} />
            <Route path="/live" element={<LivePage />} />
            <Route path="/upload" element={<MatchUploadPage />} />
            <Route path="/wearable" element={<WearablePage />} />
            <Route path="/cameras" element={<CameraNetworkPage />} />
            <Route path="/players" element={<PlayerMappingPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
};

export default App;
