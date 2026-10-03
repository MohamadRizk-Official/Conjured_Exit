// World frame (tracker/paths.py): metres, x FORWARD, y LEFT, z UP (right-handed).
// three.js frame:                  X right,     Y UP,    Z toward the viewer.
//
// Mapping used everywhere in the 3D view (also right-handed, so no mirroring):
//     three.X =  world.x
//     three.Y =  world.z
//     three.Z = -world.y
// Check: world x × world z = -world y  <=>  three X × Y = Z.  OK.
export const w2t = (p) => [p[0], p[2], -p[1]];
export const t2w = (p) => [p[0], -p[2], p[1]];

/** Interpolated world position along a path {points, times} at replay time t (clamped). */
export function positionAt(path, t) {
  const { points, times } = path;
  const n = points.length;
  if (n === 0) return null;
  if (n === 1 || t <= times[0]) return points[0];
  if (t >= times[n - 1]) return points[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t) lo = mid;
    else hi = mid;
  }
  const span = times[hi] - times[lo];
  const u = span > 0 ? (t - times[lo]) / span : 0;
  const a = points[lo];
  const b = points[hi];
  return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
}

export const PATH_COLORS = {
  guide: '#ff9f43',
  spell: '#a29bfe',
};
export const ACTIVE_COLOR = '#ffffff';
