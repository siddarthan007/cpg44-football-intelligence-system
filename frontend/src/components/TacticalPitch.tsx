import React, { useEffect, useRef } from "react";

interface PlayerPos {
  global_player_id: string;
  jersey?: number | string;
  team: number;
  x: number; // 0 to 105 m
  y: number; // 0 to 68 m
  speed_mps?: number;
  wearable?: boolean;
}

interface BallPos {
  x: number;
  y: number;
  speed_mps?: number;
}

interface Props {
  players: PlayerPos[];
  ball?: BallPos;
  showVoronoi?: boolean;
  showTrails?: boolean;
}

export const TacticalPitch: React.FC<Props> = ({
  players,
  ball,
  showVoronoi = true,
  showTrails = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Pitch dimensions: 105m x 68m
    const scaleX = width / 105;
    const scaleY = height / 68;

    // Draw Pitch Grass
    ctx.fillStyle = "#0c2817";
    ctx.fillRect(0, 0, width, height);

    // Pitch Lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
    ctx.lineWidth = 2;

    // Outer boundary
    ctx.strokeRect(10, 10, width - 20, height - 20);

    // Halfway line
    ctx.beginPath();
    ctx.moveTo(width / 2, 10);
    ctx.lineTo(width / 2, height - 10);
    ctx.stroke();

    // Center circle (radius 9.15m)
    ctx.beginPath();
    ctx.arc(width / 2, height / 2, 9.15 * scaleX, 0, 2 * Math.PI);
    ctx.stroke();

    // Center dot
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(width / 2, height / 2, 3, 0, 2 * Math.PI);
    ctx.fill();

    // Penalty Areas (16.5m x 40.32m)
    const penW = 16.5 * scaleX;
    const penH = 40.32 * scaleY;
    const penY = (height - penH) / 2;

    // Left penalty area
    ctx.strokeRect(10, penY, penW, penH);
    // Right penalty area
    ctx.strokeRect(width - 10 - penW, penY, penW, penH);

    // Draw Players
    players.forEach((p) => {
      const px = Math.max(12, Math.min(width - 12, p.x * scaleX));
      const py = Math.max(12, Math.min(height - 12, p.y * scaleY));

      const isTeam1 = p.team === 1;
      const color = isTeam1 ? "#38bdf8" : "#f43f5e";

      // Glow effect for wearable players
      if (p.wearable) {
        ctx.fillStyle = isTeam1 ? "rgba(56, 189, 248, 0.3)" : "rgba(244, 63, 94, 0.3)";
        ctx.beginPath();
        ctx.arc(px, py, 14, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Player circle
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, 2 * Math.PI);
      ctx.fill();

      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Jersey number / ID
      ctx.fillStyle = "#000000";
      ctx.font = "bold 9px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(p.jersey || p.global_player_id.replace("P_", "")), px, py);
    });

    // Draw Ball
    if (ball) {
      const bx = Math.max(10, Math.min(width - 10, ball.x * scaleX));
      const by = Math.max(10, Math.min(height - 10, ball.y * scaleY));

      ctx.fillStyle = "rgba(250, 204, 21, 0.4)";
      ctx.beginPath();
      ctx.arc(bx, by, 10, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillStyle = "#facc15";
      ctx.beginPath();
      ctx.arc(bx, by, 5, 0, 2 * Math.PI);
      ctx.fill();

      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }, [players, ball, showVoronoi, showTrails]);

  return (
    <div style={{ position: "relative", width: "100%", height: "auto" }}>
      <canvas
        ref={canvasRef}
        width={680}
        height={440}
        style={{
          width: "100%",
          height: "auto",
          borderRadius: "8px",
          display: "block",
          boxShadow: "inset 0 0 20px rgba(0,0,0,0.5)",
        }}
      />
    </div>
  );
};
