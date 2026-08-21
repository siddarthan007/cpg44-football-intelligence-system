import React, { useEffect, useRef } from "react";

interface Node {
  id: number;
  name: string;
  x: number; // 0 to 105 m
  y: number; // 0 to 68 m
  size: number;
  passes: number;
}

interface Link {
  source: number;
  target: number;
  weight: number;
}

interface Props {
  nodes: Node[];
  links: Link[];
  mode: "map" | "network" | "heatmap";
  selectedPlayerId?: number;
  onSelectPlayer?: (id: number) => void;
}

export const PassingPitch: React.FC<Props> = ({
  nodes,
  links,
  mode,
  selectedPlayerId,
  onSelectPlayer,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear & tactile paper pitch background
    ctx.fillStyle = "#fafbfe";
    ctx.fillRect(0, 0, width, height);

    // Draw Vertical Soccer Pitch Outline (FIFA Standard Geometry)
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1.5;

    const pad = 14;
    const pitchW = width - pad * 2;
    const pitchH = height - pad * 2;

    // Pitch Outer border
    ctx.strokeRect(pad, pad, pitchW, pitchH);

    // Halfway Line
    ctx.beginPath();
    ctx.moveTo(pad, height / 2);
    ctx.lineTo(width - pad, height / 2);
    ctx.stroke();

    // Center Circle (radius = 9.15m -> ~17.5% of pitch width)
    ctx.beginPath();
    ctx.arc(width / 2, height / 2, pitchW * 0.175, 0, 2 * Math.PI);
    ctx.stroke();

    // Center Spot
    ctx.fillStyle = "#94a3b8";
    ctx.beginPath();
    ctx.arc(width / 2, height / 2, 2.5, 0, 2 * Math.PI);
    ctx.fill();

    // Penalty boxes (16.5m depth -> ~15.7% of pitch length, 40.32m width -> ~59.3% of pitch width)
    const penW = pitchW * 0.593;
    const penH = pitchH * 0.157;
    const penX = (width - penW) / 2;

    // 6-yard Goal boxes (5.5m depth -> ~5.2% length, 18.32m width -> ~26.9% width)
    const goalW = pitchW * 0.269;
    const goalH = pitchH * 0.052;
    const goalX = (width - goalW) / 2;

    // Top Penalty & Goal Box
    ctx.strokeRect(penX, pad, penW, penH);
    ctx.strokeRect(goalX, pad, goalW, goalH);
    // Top Penalty Spot (11m)
    ctx.beginPath();
    ctx.arc(width / 2, pad + pitchH * 0.105, 2, 0, 2 * Math.PI);
    ctx.fill();
    // Top Penalty Arc (D-Box)
    ctx.beginPath();
    ctx.arc(width / 2, pad + pitchH * 0.105, pitchW * 0.175, 0.65, Math.PI - 0.65);
    ctx.stroke();

    // Bottom Penalty & Goal Box
    ctx.strokeRect(penX, height - pad - penH, penW, penH);
    ctx.strokeRect(goalX, height - pad - goalH, goalW, goalH);
    // Bottom Penalty Spot
    ctx.beginPath();
    ctx.arc(width / 2, height - pad - pitchH * 0.105, 2, 0, 2 * Math.PI);
    ctx.fill();
    // Bottom Penalty Arc
    ctx.beginPath();
    ctx.arc(width / 2, height - pad - pitchH * 0.105, pitchW * 0.175, Math.PI + 0.65, 2 * Math.PI - 0.65);
    ctx.stroke();

    // Corner Arcs
    const cRad = 6;
    ctx.beginPath();
    ctx.arc(pad, pad, cRad, 0, Math.PI / 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(width - pad, pad, cRad, Math.PI / 2, Math.PI);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(pad, height - pad, cRad, 1.5 * Math.PI, 2 * Math.PI);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(width - pad, height - pad, cRad, Math.PI, 1.5 * Math.PI);
    ctx.stroke();

    // Coordinate mapping: Pitch X (0-105m) -> Canvas Y, Pitch Y (0-68m) -> Canvas X
    const mapPos = (x: number, y: number) => {
      const cx = pad + (y / 68) * pitchW;
      const cy = pad + ((105 - x) / 105) * pitchH;
      return { cx, cy };
    };

    // Draw Links (Passing Connections with line width scaled by weight)
    if (mode === "network") {
      links.forEach((l) => {
        const sNode = nodes.find((n) => n.id === l.source);
        const tNode = nodes.find((n) => n.id === l.target);
        if (!sNode || !tNode) return;

        const p1 = mapPos(sNode.x, sNode.y);
        const p2 = mapPos(tNode.x, tNode.y);

        ctx.strokeStyle = l.weight > 15 ? "#0f172a" : "#94a3b8";
        ctx.lineWidth = Math.min(6, Math.max(1.5, l.weight * 0.28));
        ctx.beginPath();
        ctx.moveTo(p1.cx, p1.cy);
        ctx.lineTo(p2.cx, p2.cy);
        ctx.stroke();
      });
    }

    // Draw Heatmap Overlay if selected
    if (mode === "heatmap") {
      nodes.forEach((n) => {
        const p = mapPos(n.x, n.y);
        const rad = 45;
        const grad = ctx.createRadialGradient(p.cx, p.cy, 5, p.cx, p.cy, rad);
        grad.addColorStop(0, "rgba(239, 68, 68, 0.45)");
        grad.addColorStop(0.6, "rgba(245, 158, 11, 0.25)");
        grad.addColorStop(1, "rgba(255, 255, 255, 0)");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(p.cx, p.cy, rad, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // Draw Player Nodes (FIFA Style High-Contrast Tokens with white jersey numbers)
    nodes.forEach((n) => {
      const p = mapPos(n.x, n.y);
      const isSelected = selectedPlayerId === n.id;
      const radius = isSelected ? 15 : Math.max(10, Math.min(14, n.size * 0.42));

      // A simple outline marks the selected player.
      if (isSelected) {
        ctx.strokeStyle = "#1f2933";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(p.cx, p.cy, radius + 6, 0, 2 * Math.PI);
        ctx.stroke();
      }

      // Outer circle
      ctx.fillStyle = isSelected ? "#1d4ed8" : "#2563eb";
      ctx.beginPath();
      ctx.arc(p.cx, p.cy, radius, 0, 2 * Math.PI);
      ctx.fill();

      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Jersey number text
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 10px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(n.id), p.cx, p.cy);

      // Name label under main node
      if (isSelected) {
        ctx.fillStyle = "#1e293b";
        ctx.font = "600 9px -apple-system, BlinkMacSystemFont, sans-serif";
        ctx.fillText(n.name, p.cx, p.cy + radius + 10);
      }
    });
  }, [nodes, links, mode, selectedPlayerId]);

  return (
    <canvas
      ref={canvasRef}
      width={290}
      height={460}
      style={{
        width: "100%",
        height: "100%",
        display: "block",
        cursor: "pointer",
      }}
      onClick={(e) => {
        // Find clicked node
        const rect = e.currentTarget.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;
        const scaleX = 290 / rect.width;
        const scaleY = 460 / rect.height;
        const cx = clickX * scaleX;
        const cy = clickY * scaleY;

        nodes.forEach((n) => {
          const nx = 14 + (n.y / 68) * (290 - 28);
          const ny = 14 + ((105 - n.x) / 105) * (460 - 28);
          const dist = Math.hypot(cx - nx, cy - ny);
          if (dist < 18 && onSelectPlayer) {
            onSelectPlayer(n.id);
          }
        });
      }}
    />
  );
};
