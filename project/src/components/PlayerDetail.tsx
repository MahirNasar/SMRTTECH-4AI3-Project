import { X } from "lucide-react";
import { findSimilarPlayers, formatHeight } from "@/utils/playerUtils";
import type { Player } from "@/types";

interface PlayerDetailProps {
  player: Player;
  onClose: () => void;
  onSelectPlayer: (player: Player) => void;
}

const statCategories: { label: string; key: keyof Player; suffix?: string; pct?: boolean }[] = [
  { label: "Points", key: "ppg" },
  { label: "Rebounds", key: "rpg" },
  { label: "Assists", key: "apg" },
  { label: "Steals", key: "spg" },
  { label: "Blocks", key: "bpg" },
  { label: "FG%", key: "fgPct", pct: true },
  { label: "3PT%", key: "threePct", pct: true },
  { label: "FT%", key: "ftPct", pct: true },
];

function formatStat(value: number, pct?: boolean): string {
  if (pct) return `${(value * 100).toFixed(1)}%`;
  return value.toFixed(1);
}

export function PlayerDetail({ player, onClose, onSelectPlayer }: PlayerDetailProps) {
  const similar = findSimilarPlayers(player, 6);

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/60 backdrop-blur-sm p-4 sm:p-6 animate-fadeIn">
      <div
        className="relative my-4 w-full max-w-3xl rounded-2xl border border-slate-700 bg-slate-900 shadow-2xl animate-slideUp"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute right-4 top-4 z-10 rounded-lg p-2 text-slate-400 hover:bg-slate-800 hover:text-white transition-colors"
        >
          <X className="h-5 w-5" />
        </button>

        <div className="relative overflow-hidden rounded-t-2xl bg-gradient-to-br from-orange-600 via-orange-700 to-slate-900 px-6 pt-8 pb-6 sm:px-8">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,255,255,0.1),transparent_60%)]" />
          <div className="relative flex flex-col sm:flex-row sm:items-center gap-5">
            <div className="flex h-20 w-20 sm:h-24 sm:w-24 shrink-0 items-center justify-center rounded-2xl bg-white/15 backdrop-blur-sm border border-white/20 text-2xl sm:text-3xl font-black text-white">
              {player.teamAbbr}
            </div>
            <div className="min-w-0">
              <p className="text-xs font-medium text-orange-200 uppercase tracking-wider">
                #{player.number} · {player.position}
              </p>
              <h2 className="font-display mt-1 text-3xl sm:text-4xl font-bold text-white leading-none">
                {player.name}
              </h2>
              <p className="mt-1 text-sm text-orange-100">{player.team}</p>
              <div className="mt-3 flex flex-wrap gap-2">
                <span className="rounded-full bg-white/10 px-3 py-1 text-xs font-medium text-white border border-white/15">
                  {formatHeight(player.heightInches)} · {player.weightLbs} lbs
                </span>
                <span className="rounded-full bg-white/10 px-3 py-1 text-xs font-medium text-white border border-white/15">
                  Age {player.age}
                </span>
                <span className="rounded-full bg-white/10 px-3 py-1 text-xs font-medium text-white border border-white/15">
                  {player.experienceYears} yr{player.experienceYears !== 1 ? "s" : ""} exp
                </span>
                <span className="rounded-full bg-white/10 px-3 py-1 text-xs font-medium text-white border border-white/15">
                  {player.conference} Conf
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="px-6 py-6 sm:px-8">
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4">
            Season Averages
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {statCategories.map((stat) => (
              <div
                key={stat.key as string}
                className="rounded-xl border border-slate-700/80 bg-slate-800/50 px-4 py-3 transition-colors hover:border-orange-500/40"
              >
                <p className="text-xs text-slate-400">{stat.label}</p>
                <p className="mt-1 text-xl font-bold text-white tabular-nums">
                  {formatStat(player[stat.key] as number, stat.pct)}
                </p>
              </div>
            ))}
          </div>
        </div>

        <div className="border-t border-slate-800 px-6 py-6 sm:px-8">
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-1">
            Similar Players
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            Based on statistical profile comparison
          </p>
          <div className="space-y-2.5">
            {similar.map((sim, idx) => (
              <button
                key={sim.player.id}
                onClick={() => onSelectPlayer(sim.player)}
                className="group flex w-full items-center gap-4 rounded-xl border border-slate-700/80 bg-slate-800/40 px-4 py-3 text-left transition-all hover:border-orange-500/50 hover:bg-slate-800/80 animate-fadeIn"
                style={{ animationDelay: `${idx * 60}ms` }}
              >
                <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-slate-700 to-slate-800 text-xs font-bold text-slate-300 group-hover:from-orange-500/30 group-hover:to-orange-700/30 group-hover:text-orange-200 transition-all">
                  {sim.player.teamAbbr}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-white truncate group-hover:text-orange-300 transition-colors">
                    {sim.player.name}
                  </p>
                  <p className="text-xs text-slate-400">
                    {sim.player.team} · {sim.player.position}
                  </p>
                </div>
                <div className="flex items-center gap-3 shrink-0">
                  <div className="hidden sm:block w-28 h-2 rounded-full bg-slate-700 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-orange-500 to-orange-400"
                      style={{ width: `${sim.similarity * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-bold text-orange-400 tabular-nums">
                    {(sim.similarity * 100).toFixed(0)}%
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
