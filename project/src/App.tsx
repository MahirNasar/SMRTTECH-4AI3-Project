import { useState } from "react";
import { SearchBar } from "@/components/SearchBar";
import { PlayerDetail } from "@/components/PlayerDetail";
import { players } from "@/data/players";
import { formatHeight } from "@/utils/playerUtils";
import type { Player } from "@/types";

function App() {
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);
  const [browsePlayer, setBrowsePlayer] = useState<Player | null>(null);

  const displayPlayer = browsePlayer ?? selectedPlayer;

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_at_top,rgba(234,88,12,0.08),transparent_55%)]" />
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_at_bottom_left,rgba(30,58,138,0.06),transparent_50%)]" />

      <div
        className="relative z-10 bg-[url('/court-background.svg')] bg-[length:170%_170%] bg-[center_top] bg-no-repeat sm:bg-[length:125%_150%]"
        style={{
          backgroundColor: "#020617",
          backgroundBlendMode: "soft-light",
        }}
      >
        <header className="border-b border-slate-800/80 backdrop-blur-sm bg-slate-950/60 sticky top-0 z-30">
          <div className="mx-auto max-w-6xl px-4 sm:px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-orange-500 to-orange-700">
                <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5 text-white" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M5 12h14M12 5v14" />
                </svg>
              </div>
              <div>
                <h1 className="font-display text-xl font-bold tracking-tight">DoppelPläyer</h1>
                <p className="text-[10px] text-slate-500 -mt-0.5">NBA Player Similarity Search</p>
              </div>
            </div>
            <a
              href="#browse"
              className="text-xs font-medium text-slate-400 hover:text-orange-400 transition-colors"
            >
              Browse All
            </a>
          </div>
        </header>

        <section className="mx-auto max-w-6xl px-4 sm:px-6 pt-16 sm:pt-24 pb-12">
          <div className="text-center max-w-2xl mx-auto mb-10">
            <div className="inline-flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/60 px-3 py-1 text-xs text-slate-400 mb-6 animate-fadeIn">
              <span className="h-1.5 w-1.5 rounded-full bg-orange-500 animate-pulse" />
              {players.length} players loaded
            </div>
            <h2 className="font-display text-5xl sm:text-6xl font-bold tracking-tight leading-none mb-4 animate-fadeIn" style={{ animationDelay: "80ms" }}>
              <span className="typing-title">
                Find Similar{" "}
                <span className="bg-gradient-to-r from-orange-500 to-orange-300 bg-clip-text text-transparent">
                  NBA Players
                </span>
              </span>
            </h2>
            <p className="text-slate-400 text-base sm:text-lg leading-relaxed animate-fadeIn" style={{ animationDelay: "160ms" }}>
              Search any player and discover who has the closest statistical profile.
              Compare points, rebounds, assists, shooting, and physical traits.
            </p>
          </div>

          <div className="flex justify-center animate-fadeIn" style={{ animationDelay: "240ms" }}>
            <SearchBar onSelectPlayer={setSelectedPlayer} />
          </div>

          {selectedPlayer && (
            <div className="mt-10 max-w-4xl mx-auto animate-slideUp">
              <div className="rounded-2xl border border-slate-800 bg-slate-900/60 backdrop-blur-sm overflow-hidden">
                <div className="flex flex-col sm:flex-row items-center gap-5 bg-gradient-to-r from-orange-600/20 to-transparent px-6 py-5">
                  <div className="flex h-16 w-16 items-center justify-center rounded-xl bg-gradient-to-br from-orange-500 to-orange-700 text-lg font-black text-white">
                    {selectedPlayer.teamAbbr}
                  </div>
                  <div className="flex-1 text-center sm:text-left">
                    <h3 className="text-xl font-bold">{selectedPlayer.name}</h3>
                    <p className="text-sm text-slate-400">
                      {selectedPlayer.team} · {selectedPlayer.position} · #{selectedPlayer.number}
                    </p>
                  </div>
                  <button
                    onClick={() => setBrowsePlayer(selectedPlayer)}
                    className="rounded-lg bg-orange-500 px-5 py-2.5 text-sm font-semibold text-white hover:bg-orange-600 active:scale-95 transition-all"
                  >
                    View Profile
                  </button>
                </div>
              </div>
            </div>
          )}
        </section>

        <section id="browse" className="mx-auto max-w-6xl px-4 sm:px-6 pb-20">
          <div className="flex items-baseline justify-between mb-5">
            <h3 className="text-lg font-semibold">All Players</h3>
            <span className="text-xs text-slate-500">{players.length} total</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {players.map((player, idx) => (
              <button
                key={player.id}
                onClick={() => setBrowsePlayer(player)}
                className="group flex items-center gap-3.5 rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-3.5 text-left transition-all hover:border-orange-500/50 hover:bg-slate-800/60 hover:-translate-y-0.5 animate-fadeIn"
                style={{ animationDelay: `${Math.min(idx * 30, 800)}ms` }}
              >
                <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-slate-700 to-slate-800 text-xs font-bold text-slate-300 group-hover:from-orange-500/30 group-hover:to-orange-700/30 group-hover:text-orange-200 transition-all">
                  {player.teamAbbr}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-white truncate group-hover:text-orange-300 transition-colors">
                    {player.name}
                  </p>
                  <p className="text-xs text-slate-400">
                    {player.team} · {player.position} · {player.ppg.toFixed(1)} PPG
                  </p>
                </div>
                <span className="text-xs text-slate-500 group-hover:text-orange-400 transition-colors shrink-0">
                  {formatHeight(player.heightInches)}
                </span>
              </button>
            ))}
          </div>
        </section>
      </div>

      {displayPlayer && (
        <PlayerDetail
          player={displayPlayer}
          onClose={() => {
            setBrowsePlayer(null);
            setSelectedPlayer(null);
          }}
          onSelectPlayer={(p) => setBrowsePlayer(p)}
        />
      )}
    </div>
  );
}

export default App;
