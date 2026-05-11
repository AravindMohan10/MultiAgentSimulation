/** Map sim world (x,y in [0,W]×[0,H]) to Three.js XZ plane centered at origin. */
export function worldToScene(
  wx: number,
  wy: number,
  worldSize: [number, number]
): [number, number, number] {
  const [ww, wh] = worldSize;
  const x = wx - ww / 2;
  const z = wy - wh / 2;
  return [x, 0, z];
}

export function roleColor(role: string): string {
  if (role === "hero") return "#4a9eff";
  return "#ff5c4a";
}

export function intentLabel(intent: string | null): string {
  if (!intent) return "?";
  return intent.replace(/_/g, " ");
}
