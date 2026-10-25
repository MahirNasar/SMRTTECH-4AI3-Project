export interface Player {
  id: number;
  name: string;
  team: string;
  teamAbbr: string;
  position: string;
  number: number;
  heightInches: number;
  weightLbs: number;
  age: number;
  ppg: number;
  rpg: number;
  apg: number;
  spg: number;
  bpg: number;
  fgPct: number;
  threePct: number;
  ftPct: number;
  experienceYears: number;
  conference: "Eastern" | "Western";
}
