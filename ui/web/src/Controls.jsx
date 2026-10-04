import React, { useState } from 'react';

function Section({ title, children, className = '' }) {
  return (
    <section className={`panel ${className}`}>
      {title && <h2>{title}</h2>}
      {children}
    </section>
  );
}

export default function Controls({ state, send }) {
  const [name, setName] = useState('');
  const s = state || {};
  const mode = s.mode || 'guide';
  const rec = s.recording || {};
  const alarm = s.alarm || {};
  const blocked = alarm.blocked_exits || [];
  const paths = s.paths || [];
  const flight = s.flight?.state || 'idle';
  const airborne = ['takeoff', 'hover', 'flying', 'landing'].includes(flight);
  const armed = !!s.flight?.armed;

  const saveAs = () => {
    const n = name.trim();
    if (!n) return;
    send('save_as', { name: n });
    setName('');
  };

  return (
    <aside className="controls">
      <button
        className={`btn arm ${armed ? 'on' : 'off'}`}
        onClick={() => send('arm', { on: !armed })}
        title="Motors can only start while armed. Arming clears itself after every landing or stop."
      >
        {armed ? 'ARMED' : 'DISARMED'}
        <small>{armed ? 'ready to fly - click to disarm' : 'click to arm before ALARM / Cast'}</small>
      </button>

      <button
        className={`btn alarm ${armed ? '' : 'disarmed'}`}
        onClick={() => send('alarm')}
        title="Guide mode: take off and fly the open exit route (voice: 'fire' / 'lead me out')"
      >
        <span className="alarm-icon">!</span> ALARM
        <small>{alarm.active ? `active - exit ${alarm.exit}` : 'lead me out'}</small>
      </button>

      <Section title="Mode">
        <div className="segmented">
          <button className={`seg ${mode === 'guide' ? 'on' : ''}`} onClick={() => send('set_mode', { mode: 'guide' })}>
            Guide
          </button>
          <button className={`seg ${mode === 'spell' ? 'on' : ''}`} onClick={() => send('set_mode', { mode: 'spell' })}>
            Spell
          </button>
        </div>
        <p className="hint">{mode === 'guide' ? 'Records the hand-carried drone. True scale.' : 'Records the wand tip. Shape is fitted into the geofence.'}</p>
      </Section>

      <Section title="Record">
        <div className="row">
          <button className={`btn rec ${rec.active ? 'pulse' : ''}`} disabled={rec.active} onClick={() => send('record_start')}>
            {rec.active ? 'Recording...' : 'Start'}
          </button>
          <button className="btn" disabled={!rec.active} onClick={() => send('record_stop')}>
            Stop
          </button>
        </div>
        <div className="row meta">
          <span>{rec.n_samples || 0} samples</span>
          <span className={`tag ${rec.mode}`}>{rec.mode || mode}</span>
        </div>
        <div className="row">
          <input
            className="text"
            placeholder="path name (e.g. Exit A)"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && saveAs()}
          />
          <button className="btn" disabled={!name.trim() || rec.active || !(rec.n_samples > 1)} onClick={saveAs}>
            Save as
          </button>
        </div>
      </Section>

      <Section title={`Paths (${paths.length})`} className="paths">
        {paths.length === 0 && <p className="hint">No stored paths yet. Record one, or start the server with --sim.</p>}
        <ul className="pathlist">
          {paths.map((p) => {
            const active = p.name === s.active_path;
            return (
              <li key={p.name} className={active ? 'active' : ''}>
                <div className="pathinfo">
                  <span className="pathname">{p.name}</span>
                  <span className={`tag ${p.mode}`}>{p.mode}</span>
                  <span className="pathmeta">
                    {p.length_m.toFixed(2)} m / {p.duration_s.toFixed(0)} s
                  </span>
                </div>
                <div className="pathbtns">
                  <button className="btn small" disabled={active} onClick={() => send('select_path', { name: p.name })}>
                    {active ? 'Selected' : 'Select'}
                  </button>
                  <button className={`btn small cast ${armed ? '' : 'disarmed'}`} onClick={() => send('cast', { name: p.name })}>
                    Cast
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      </Section>

      <Section title="Exits">
        <div className="row">
          {['A', 'B'].map((ex) => {
            const isBlocked = blocked.includes(ex);
            return (
              <button
                key={ex}
                className={`btn exit ${isBlocked ? 'blocked' : ''}`}
                onClick={() => send('exit_blocked', { exit: ex })}
                title={`Voice: "Exit ${ex} is blocked"`}
              >
                Exit {ex} {isBlocked ? 'BLOCKED' : 'blocked'}
              </button>
            );
          })}
        </div>
      </Section>

      <Section title="Flight" className="flight">
        <div className="row">
          <button
            className={`btn hover ${armed ? '' : 'disarmed'}`}
            disabled={airborne}
            onClick={() => send('hover', { height_m: 0.6, seconds: 8 })}
            title="Take off, hold 0.6 m for 8 s, land. Needs ARMED."
          >
            HOVER TEST
            <small>0.6 m, 8 s, lands by itself</small>
          </button>
          <button className="btn land" onClick={() => send('land')} disabled={!airborne}>
            LAND
          </button>
          <button className="btn clear" onClick={() => send('clear_alarm')}>
            Clear alarm
          </button>
        </div>
        <button className="btn stop" onClick={() => send('stop')} title="Emergency stop: motors off (also Space / Esc)">
          STOP
          <small>motors off - Space / Esc</small>
        </button>
      </Section>
    </aside>
  );
}
