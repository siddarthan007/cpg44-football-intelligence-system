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

    // Clear & background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    // Draw Vertical Soccer Pitch Outline (matching reference image)
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1.5;

    const pad = 12;
    const pitchW = width - pad * 2;
    const pitchH = height - pad * 2;

    // Pitch Outer border
    ctx.strokeRect(pad, pad, pitchW, pitchH);

    // Halfway Line
    ctx.beginPath();
    ctx.moveTo(pad, height / 2);
    ctx.lineTo(width - pad, height / 2);
    ctx.stroke();

    // Center Circle
    ctx.beginPath();
    ctx.arc(width / 2, height / 2, pitchW * 0.18, 0, 2 * Math.PI);
    ctx.stroke();

    // Penalty boxes (top and bottom)
    const penW = pitchW * 0.55;
    const penH = pitchH * 0.16;
    const penX = (width - penW) / 2;

    // Top Goal box
    ctx.strokeRect(penX, pad, penW, penH);
    // Bottom Goal box
    ctx.strokeRect(penX, height - pad - penH, penW, penH);

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
        ctx.lineWidth = Math.min(7, Math.max(1.5, l.weight * 0.3));
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

    // Draw Player Nodes (Royal Blue circles with white jersey numbers, matching reference image)
    nodes.forEach((n) => {
      const p = mapPos(n.x, n.y);
      const isSelected = selectedPlayerId === n.id;
      const radius = isSelected ? 16 : Math.max(10, Math.min(15, n.size * 0.42));

      // Glow for selected
      if (isSelected) {
        ctx.fillStyle = "rgba(37, 99, 235, 0.25)";
        ctx.beginPath();
        ctx.arc(p.cx, p.cy, radius + 6, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Outer circle
      ctx.fillStyle = isSelected || n.id === 27 ? "#2563eb" : "#3b82f6";
      ctx.beginPath();
      ctx.arc(p.cx, p.cy, radius, 0, 2 * Math.PI);
      ctx.fill();

      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Jersey number text
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 10px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(n.id), p.cx, p.cy);

      // Name label under main node
      if (n.id === 27 || isSelected) {
        ctx.fillStyle = "#334155";
        ctx.font = "600 9px sans-serif";
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
          const nx = 12 + (n.y / 68) * (290 - 24);
          const ny = 12 + ((105 - n.x) / 105) * (460 - 24);
          const dist = Math.hypot(cx - nx, cy - ny);
          if (dist < 18 && onSelectPlayer) {
            onSelectPlayer(n.id);
          }
        });
      }}
    />
  );
};
