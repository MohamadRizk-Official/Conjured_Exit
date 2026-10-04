// Short audio cues so the presenter knows what the mic did without looking at the screen.
// The AudioContext is created on first use, which always follows a click or key press.
let ctx = null;

function tone(freq, start, dur, type = 'sine', gain = 0.15) {
  const osc = ctx.createOscillator();
  const g = ctx.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  g.gain.setValueAtTime(0, ctx.currentTime + start);
  g.gain.linearRampToValueAtTime(gain, ctx.currentTime + start + 0.01);
  g.gain.linearRampToValueAtTime(0, ctx.currentTime + start + dur);
  osc.connect(g).connect(ctx.destination);
  osc.start(ctx.currentTime + start);
  osc.stop(ctx.currentTime + start + dur + 0.02);
}

const CUES = {
  on: () => tone(660, 0, 0.12), // mic opened
  off: () => tone(440, 0, 0.12), // mic closed
  ok: () => { tone(880, 0, 0.1); tone(1320, 0.1, 0.16); }, // command accepted
  bad: () => tone(180, 0, 0.35, 'square', 0.1), // command rejected / failure
};

export function beep(kind) {
  try {
    ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === 'suspended') ctx.resume();
    CUES[kind]?.();
  } catch {
    /* no audio available: the visual cues still work */
  }
}
