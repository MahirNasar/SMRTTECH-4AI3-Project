import { useState, useEffect, useRef } from "react";
import { Search, X } from "lucide-react";
import { searchPlayers } from "@/utils/playerUtils";
import type { Player } from "@/types";

interface SearchBarProps {
  onSelectPlayer: (player: Player) => void;
}

export function SearchBar({ onSelectPlayer }: SearchBarProps) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Player[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([]);
      setIsOpen(false);
      return;
    }
    const res = searchPlayers(query);
    setResults(res);
    setIsOpen(res.length > 0);
    setHighlightedIndex(-1);
  }, [query]);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  function handleKeyDown(e: React.KeyboardEvent) {
    if (!isOpen || results.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightedIndex((prev) => (prev + 1) % results.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightedIndex((prev) => (prev - 1 + results.length) % results.length);
    } else if (e.key === "Enter" && highlightedIndex >= 0) {
      e.preventDefault();
      selectPlayer(results[highlightedIndex]);
    } else if (e.key === "Escape") {
      setIsOpen(false);
    }
  }

  function selectPlayer(player: Player) {
    onSelectPlayer(player);
    setQuery("");
    setResults([]);
    setIsOpen(false);
  }

  return (
    <div ref={containerRef} className="relative w-full max-w-xl">
      <div className="relative">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => results.length > 0 && setIsOpen(true)}
          placeholder="Search for an NBA player..."
          className="w-full rounded-xl border border-slate-700 bg-slate-800/80 py-3.5 pl-12 pr-10 text-base text-white placeholder-slate-500 outline-none transition-all duration-200 focus:border-orange-500/70 focus:ring-2 focus:ring-orange-500/30"
        />
        {query && (
          <button
            onClick={() => {
              setQuery("");
              setResults([]);
              setIsOpen(false);
            }}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        )}
      </div>

      {isOpen && results.length > 0 && (
        <ul className="absolute z-50 mt-2 w-full overflow-hidden rounded-xl border border-slate-700 bg-slate-800 shadow-2xl shadow-black/50 max-h-80 overflow-y-auto">
          {results.map((player, idx) => (
            <li
              key={player.id}
              onMouseDown={() => selectPlayer(player)}
              onMouseEnter={() => setHighlightedIndex(idx)}
              className={`flex items-center gap-3 px-4 py-3 cursor-pointer transition-colors ${
                highlightedIndex === idx ? "bg-orange-500/20" : "hover:bg-slate-700/50"
              }`}
            >
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-orange-500 to-orange-700 text-sm font-bold text-white">
                {player.teamAbbr}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-white truncate">{player.name}</p>
                <p className="text-xs text-slate-400">
                  {player.team} · {player.position} · #{player.number}
                </p>
              </div>
            </li>
          ))}
        </ul>
      )}

      {query.trim().length >= 2 && results.length === 0 && isOpen && (
        <div className="absolute z-50 mt-2 w-full rounded-xl border border-slate-700 bg-slate-800 px-4 py-6 text-center shadow-2xl shadow-black/50">
          <p className="text-sm text-slate-400">No players found for "{query}"</p>
        </div>
      )}
    </div>
  );
}
