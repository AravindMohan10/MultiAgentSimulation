export type SlimAgent = {
  id: string;
  role: string;
  x: number;
  y: number;
  intent: string | null;
  movementSource: string | null;
  target: [number, number] | null;
  messagesSent: number;
  messagesReceived: number;
  usedFallback: boolean;
};

export type EpisodeFrame = {
  step: number;
  time: number;
  heroPosition: number[] | null;
  agents: SlimAgent[];
};

export type EpisodeObstacle = {
  x: number;
  y: number;
  r: number;
};

export type EpisodeData = {
  episodeId: string;
  worldSize: [number, number];
  captureRadius: number;
  mapTemplate: string | null;
  seed: number | null;
  outcome: string | null;
  steps: number;
  winnerTeam: string | null;
  obstacles: EpisodeObstacle[];
  frames: EpisodeFrame[];
};

export async function loadEpisodeList(): Promise<string[]> {
  const res = await fetch("/api/episodes");
  if (!res.ok) return [];
  const data = (await res.json()) as { episodes: string[] };
  return data.episodes;
}

export async function loadEpisode(id: string): Promise<EpisodeData> {
  const res = await fetch(`/episodes/${id}.json`);
  if (!res.ok) throw new Error(`Episode not found: ${id}`);
  return res.json() as Promise<EpisodeData>;
}
