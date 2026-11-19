import type { Analysis } from './analysis';
import { apiUrl } from './api';

export type SquadRow = { name: string; number: number | null; position: string | null; vest: number | null };
export type Squad = { team_name: string; players: SquadRow[] };

export async function getSquad(): Promise<Squad | null> {
  try { const r = await fetch(apiUrl('/api/v1/squad')); return r.ok ? await r.json() as Squad : null; } catch { return null; }
}

/** Name tracked players from read shirt numbers and vest links.
 *
 * Our side is the one whose read numbers match the squad more often, so the
 * opponents' number 7 is never given our number 7's name. A vest link names
 * its wearer directly; when that player's shirt reads a different number than
 * the squad lists for the vest, the pairing is queued for a human. */
export function applySquad(a: Analysis, squad: Squad | null): Analysis {
  if (!squad?.players.length) return a;
  const byNum = new Map(squad.players.filter(p => p.number != null).map(p => [p.number!, p]));
  const byVest = new Map(squad.players.filter(p => p.vest != null).map(p => [p.vest!, p]));
  const hits = { 1: 0, 2: 0 } as Record<number, number>;
  for (const p of a.players) if (p.number != null && byNum.has(p.number) && (p.team === 1 || p.team === 2)) hits[p.team]++;
  for (const v of a.vests ?? []) { const pl = a.players.find(p => p.id === v.track); if (v.confident && pl && (pl.team === 1 || pl.team === 2)) hits[pl.team] += 2; }
  const ours = hits[1] === hits[2] ? 0 : hits[1] > hits[2] ? 1 : 2;
  const issues: string[] = [];
  const players = a.players.map(p => {
    const viaVest = p.vest != null ? byVest.get(p.vest) : undefined;
    const viaNum = ours && p.team === ours && p.number != null ? byNum.get(p.number) : undefined;
    const who = viaVest ?? viaNum;
    if (viaVest && p.number != null && viaVest.number != null && viaVest.number !== p.number) {
      issues.push(`Vest ${p.vest} belongs to ${viaVest.name} (${viaVest.number}), but the shirt of tracked player ${p.id} reads ${p.number}. Check the vest list or the footage.`);
    }
    if (!who) return p.number != null ? { ...p, label: `${p.number}` } : p;
    return { ...p, label: who.number != null ? `${who.number} ${who.name}` : who.name };
  });
  return { ...a, players, squad_issues: issues };
}
