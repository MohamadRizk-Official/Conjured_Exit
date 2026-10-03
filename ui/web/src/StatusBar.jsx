import React, { useEffect, useRef, useState } from 'react';

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v || 0));

/**
 * Desk hardware check: stabilizer roll/pitch at 5 Hz while the real drone is idle
 * (ui.live; the simulator fakes it).  Big numbers + an artificial horizon so a
 * person tilting the drone sees it respond.  "age" appears once the last sample
 * is older than 1 s (measured from when the browser received it).
 */
function HwCheck({ hw, source }) {
  const [, force] = useState(0);
  const seenRef = useRef({ ts: null, at: 0 });
  useEffect(() => {
    const id = setInterval(() => force((n) => n + 1), 250);
    return () => clearInterval(id);
  }, []);
  if (!hw || !hw.active) return null;
  if (hw.ts !== seenRef.current.ts) seenRef.current = { ts: hw.ts, at: Date.now() };
  const age = seenRef.current.at ? (Date.now() - seenRef.current.at) / 1000 : null;
  const stale = age !== null && age > 1.0;
  const roll = hw.roll_deg || 0;
  const pitch = hw.pitch_deg || 0;
  const horizon = { transform: `translateY(${clamp(pitch, -45, 45) * 1.4}px) rotate(${-clamp(roll, -60, 60)}deg)` };
  return (
    <div className={`hwcard ${stale ? 'stale' : ''}`}>
      <div className="hw-head">
        <span className="pill-label">Hardware check</span>
        <span className="hw-src">{source === 'live' ? 'LIVE 5 Hz' : 'SIM (fake)'}</span>
      </div>
      <div className="hw-body">
        <div className="horizon" aria-label="artificial horizon">
          <div className="horizon-sky" style={horizon} />
          <div className="horizon-ref" />
        </div>
        <div className="hw-nums">
          <div className="hw-num">
            <span className="hw-k">roll</span>
            <span className="hw-v">{roll.toFixed(1)}°</span>
            <div className="tilt"><div className="tilt-dot" style={{ left: `${50 + clamp(roll, -45, 45) / 0.9}%` }} /></div>
          </div>
          <div className="hw-num">
            <span className="hw-k">pitch</span>
            <span className="hw-v">{pitch.toFixed(1)}°</span>
            <div className="tilt"><div className="tilt-dot" style={{ left: `${50 + clamp(pitch, -45, 45) / 0.9}%` }} /></div>
          </div>
        </div>
      </div>
      {stale && <div className="hw-age">age {age.toFixed(1)} s - no fresh sample</div>}
    </div>
  );
}

function Pill({ ok, warn, label, value }) {
  const cls = ok === undefined ? '' : ok ? 'ok' : warn ? 'warn' : 'bad';
  return (
    <div className={`pill ${cls}`}>
      <span className="pill-label">{label}</span>
      <span className="pill-value">{value}</span>
    </div>
  );
}

export default function StatusBar({ state, connected, lastMsgAt }) {
  const s = state || {};
  const link = s.link || {};
  const tr = s.tracking || {};
  const fl = s.flight || {};
  const rp = s.replay || {};
  const log = s.log || [];
  const logRef = useRef();
  const stale = lastMsgAt && Date.now() - lastMsgAt > 4000;
  const hwActive = !!s.hwcheck?.active;

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log.length, log[log.length - 1]]);

  return (
    <footer className={`status ${hwActive ? 'with-hw' : ''}`}>
      <div className="pills">
        <Pill ok={connected && !stale} warn={connected} label="UI" value={connected ? (stale ? 'stale' : 'live') : 'reconnecting...'} />
        <Pill
          ok={!!link.connected}
          warn={!!link.connecting}
          label={`Link (${(s.source || 'none').toUpperCase()})`}
          value={`${link.connecting ? 'connecting...' : link.connected ? 'connected' : 'down'}  ${link.uri || ''}`.trim()}
        />
        <Pill
          ok={link.battery_v >= 3.6}
          warn={link.battery_v >= 3.4}
          label="Battery"
          value={link.battery_v ? `${link.battery_v.toFixed(2)} V` : '-'}
        />
        <Pill
          ok={!!tr.ok}
          label="Tracking"
          value={`${tr.ok ? 'ok' : 'LOST'}  ${(tr.fps || 0).toFixed(0)} fps  ${(tr.latency_ms || 0).toFixed(0)} ms`}
        />
        <Pill
          ok={fl.state === 'idle' || fl.state === 'flying' || fl.state === 'hover'}
          warn={fl.state === 'takeoff' || fl.state === 'landing'}
          label="Flight"
          value={(fl.state || 'idle').toUpperCase()}
        />
        <Pill ok={!!fl.estimator_converged} label="Estimator" value={fl.estimator_converged ? 'converged' : 'not converged'} />
        <div className="pill replay">
          <span className="pill-label">Replay</span>
          <span className="pill-value">
            {rp.active ? `${rp.t.toFixed(1)} / ${rp.duration.toFixed(1)} s` : s.active_path ? `preview ${s.active_path}` : 'idle'}
          </span>
          <div className="bar">
            <div className="fill" style={{ width: `${Math.round((rp.progress || 0) * 100)}%` }} />
          </div>
        </div>
      </div>
      <pre className="log" ref={logRef}>
        {log.length ? log.join('\n') : 'no log lines yet'}
      </pre>
      {hwActive && <HwCheck hw={s.hwcheck} source={s.source} />}
    </footer>
  );
}
