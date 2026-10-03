// Voice commands: browser speech -> the same commands the buttons send.
//
// Phrases are deliberately uncommon so normal talking during the pitch does not
// trigger anything, and no phrase contains another phrase's trigger word.
//
// Voice abort is a controlled landing, not the motor kill: a misheard word must
// never drop the drone. The kill switch stays on Space / Esc / the STOP button.
//
//   abort / halt               -> land            (sent instantly, even mid-sentence)
//   evacuate                   -> alarm
//   begin recording            -> record_start
//   end recording              -> record_stop
//   save as <name>             -> save_as {name}
//   launch <name>              -> cast {name}     (fuzzy-matched to saved paths)
//   exit alpha|bravo blocked   -> exit_blocked {exit}
//   touch down                 -> land
//   guide mode / spell mode    -> set_mode {mode}
//   reset                      -> clear_alarm

export const MIN_CONFIDENCE = 0.6;

const ABORT_RE = /\b(abort|halt)\b/;

export function normalize(text) {
  return (text || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function slugify(words) {
  return normalize(words).replace(/ /g, '_');
}

// True if the (possibly interim) transcript contains an abort word.
export function isAbort(text) {
  return ABORT_RE.test(normalize(text));
}

function levenshtein(a, b) {
  const row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let prev = row[0];
    row[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = row[j];
      row[j] = Math.min(row[j] + 1, row[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = tmp;
    }
  }
  return row[b.length];
}

// 'exit alpha' -> 'exit_alpha'; 'spyral' -> 'spiral' if a path called spiral exists.
export function matchPathName(spoken, paths = []) {
  const want = slugify(spoken);
  const bare = (s) => s.replace(/_/g, '');
  let best = null;
  let bestDist = Infinity;
  for (const p of paths) {
    const name = typeof p === 'string' ? p : p.name;
    if (!name) continue;
    const d = levenshtein(bare(want), bare(name.toLowerCase()));
    if (d < bestDist) {
      best = name;
      bestDist = d;
    }
  }
  const limit = Math.max(2, Math.floor(bare(want).length * 0.3));
  return best !== null && bestDist <= limit ? best : want;
}

// Speech recognisers mishear some phrases the same way every time; accept those too.
const RULES = [
  { re: ABORT_RE, cmd: () => ({ name: 'land', args: {} }) },
  { re: /\bevacuate\b/, cmd: () => ({ name: 'alarm', args: {} }) },
  { re: /\bbegin recording\b/, cmd: () => ({ name: 'record_start', args: {} }) },
  { re: /\b(end|and) recording\b/, cmd: () => ({ name: 'record_stop', args: {} }) },
  { re: /\bsave (?:this )?as (.+)/, cmd: (m) => ({ name: 'save_as', args: { name: slugify(m[1]) } }) },
  { re: /\b(?:launch|lunch) (.+)/, cmd: (m, paths) => ({ name: 'cast', args: { name: matchPathName(m[1], paths) } }) },
  { re: /\bexit (?:alpha|alfa|a)\b.*\bblock/, cmd: () => ({ name: 'exit_blocked', args: { exit: 'A' } }) },
  { re: /\bexit (?:bravo|b|be|bee)\b.*\bblock/, cmd: () => ({ name: 'exit_blocked', args: { exit: 'B' } }) },
  { re: /\btouch ?down\b/, cmd: () => ({ name: 'land', args: {} }) },
  { re: /\b(guide|spell) mode\b/, cmd: (m) => ({ name: 'set_mode', args: { mode: m[1] } }) },
  { re: /\breset\b/, cmd: () => ({ name: 'clear_alarm', args: {} }) },
];

// Returns {name, args} or null. First matching rule wins.
export function parseCommand(text, paths = []) {
  const t = normalize(text);
  if (!t) return null;
  for (const rule of RULES) {
    const m = t.match(rule.re);
    if (m) return rule.cmd(m, paths);
  }
  return null;
}

// Catches commands the backend would accept and then quietly drop (it only logs
// the failure). Returns a reason string, or null if the command can go.
// Landing is never blocked: the page's view of the flight state may be stale.
export function precheck(cmd, state) {
  const s = state || {};
  if (cmd.name === 'cast') {
    const names = (s.paths || []).map((p) => p.name);
    if (!names.includes(cmd.args.name)) {
      return names.length ? `no saved path "${cmd.args.name}"` : 'no saved paths yet';
    }
  }
  if (cmd.name === 'save_as') {
    const rec = s.recording || {};
    if (rec.active) return 'say "end recording" first';
    if (!(rec.n_samples > 1)) return 'nothing recorded yet';
  }
  return null;
}

export function speechSupported() {
  return typeof window !== 'undefined' && !!(window.SpeechRecognition || window.webkitSpeechRecognition);
}

// Wraps the Web Speech API. Callbacks:
//   onTranscript({interim, final})           live text for the top bar
//   onCommand({cmd, heard, confidence})      a command to send
//   onIgnored({heard, confidence, reason})   heard something but did not act
//   onListening(bool)
export class VoiceListener {
  constructor({ getPaths = () => [], onTranscript, onCommand, onIgnored, onListening } = {}) {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    this.rec = new SR();
    this.rec.continuous = true;
    this.rec.interimResults = true;
    this.rec.lang = 'en-US';
    this.listening = false;
    this.abortFired = new Set(); // result indexes that already sent `stop`
    this.getPaths = getPaths;
    this.cb = { onTranscript, onCommand, onIgnored, onListening };

    this.rec.onresult = (e) => this.handleResult(e);
    // Chrome ends the session after silence; keep going while we want to listen.
    this.rec.onend = () => {
      if (this.listening) {
        try { this.rec.start(); } catch { /* already started */ }
      } else {
        this.cb.onListening?.(false);
      }
    };
    this.rec.onerror = (e) => {
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
        this.listening = false;
        this.cb.onIgnored?.({ heard: '', confidence: 0, reason: 'microphone blocked' });
      }
    };
  }

  start() {
    if (this.listening) return;
    this.listening = true;
    this.abortFired.clear();
    try { this.rec.start(); } catch { /* already started */ }
    this.cb.onListening?.(true);
  }

  stop() {
    this.listening = false;
    try { this.rec.stop(); } catch { /* not started */ }
  }

  handleResult(e) {
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      const heard = r[0].transcript;
      const confidence = r[0].confidence;

      // Abort skips the confidence check and does not wait for the sentence to end.
      if (!this.abortFired.has(i) && isAbort(heard)) {
        this.abortFired.add(i);
        this.cb.onCommand?.({ cmd: { name: 'land', args: {} }, heard, confidence });
      }

      if (!r.isFinal) {
        interim += heard;
        continue;
      }
      this.cb.onTranscript?.({ interim: '', final: heard });
      if (this.abortFired.has(i)) continue;

      const cmd = parseCommand(heard, this.getPaths());
      if (!cmd) {
        this.cb.onIgnored?.({ heard, confidence, reason: 'no command' });
      } else if (confidence < MIN_CONFIDENCE) {
        this.cb.onIgnored?.({ heard, confidence, reason: 'low confidence' });
      } else {
        this.cb.onCommand?.({ cmd, heard, confidence });
      }
    }
    if (interim) this.cb.onTranscript?.({ interim, final: null });
  }
}
