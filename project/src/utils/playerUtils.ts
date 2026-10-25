import type { Player } from "@/types";
import { players } from "@/data/players";

export function formatHeight(inches: number): string {
  const feet = Math.floor(inches / 12);
  const inch = inches % 12;
  return `${feet}'${inch}"`;
}

const statKeys: (keyof Player)[] = [
  "ppg",
  "rpg",
  "apg",
  "spg",
  "bpg",
  "fgPct",
  "threePct",
  "ftPct",
  "heightInches",
  "weightLbs",
  "age",
  "experienceYears",
];

const ranges: Record<string, { min: number; max: number }> = {};
for (const key of statKeys) {
  const vals = players.map((p) => p[key] as number);
  ranges[key as string] = { min: Math.min(...vals), max: Math.max(...vals) };
}

function normalize(value: number, key: string): number {
  const r = ranges[key];
  if (r.max === r.min) return 1;
  return (value - r.min) / (r.max - r.min);
}

export function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[m][n];
}

export function searchPlayers(query: string): Player[] {
  const q = query.trim().toLowerCase();
  if (q.length < 2) return [];
  return players
    .map((p) => {
      const name = p.name.toLowerCase();
      const team = p.team.toLowerCase();
      let distance: number;
      if (name.includes(q) || team.includes(q)) {
        distance = 0;
      } else {
        const nameParts = name.split(" ");
        const bestPartDist = Math.min(
          ...nameParts.map((part) => levenshtein(q, part))
        );
        distance = Math.min(levenshtein(q, name), bestPartDist);
      }
      return { player: p, distance };
    })
    .filter((r) => r.distance <= 3)
    .sort((a, b) => a.distance - b.distance)
    .map((r) => r.player);
}

export interface SimilarPlayer {
  player: Player;
  similarity: number;
}

export function findSimilarPlayers(target: Player, count = 5): SimilarPlayer[] {
  return players
    .filter((p) => p.id !== target.id)
    .map((p) => {
      let sum = 0;
      for (const key of statKeys) {
        const tv = normalize(target[key] as number, key as string);
        const pv = normalize(p[key] as number, key as string);
        const weight = getStatWeight(key);
        sum += weight * Math.pow(tv - pv, 2);
      }
      const distance = Math.sqrt(sum);
      const similarity = Math.max(0, 1 - distance / Math.sqrt(statKeys.length));
      return { player: p, similarity };
    })
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, count);
}

function getStatWeight(key: keyof Player): number {
  const weights: Partial<Record<keyof Player, number>> = {
    ppg: 2.5,
    rpg: 1.8,
    apg: 1.8,
    spg: 1.2,
    bpg: 1.2,
    fgPct: 1.5,
    threePct: 1.5,
    ftPct: 1.0,
    heightInches: 1.5,
    weightLbs: 1.3,
    age: 1.0,
    experienceYears: 0.8,
  };
  return weights[key] ?? 1.0;
}
