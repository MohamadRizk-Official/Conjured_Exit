import React, { useEffect } from 'react';
import Scene from './Scene.jsx';
import Controls from './Controls.jsx';
import StatusBar from './StatusBar.jsx';
import { useConfig, usePathGeometry, useSocket } from './useSocket.js';

/** SIM / LIVE source toggle (disabled while a LIVE connect is in progress). */
function SourceToggle({ state, config, send }) {
  const src = state?.source || 'none';
  const connecting = !!state?.link?.connecting;
  const avail = config?.sources || [];
  const err = state?.link?.error;
  return (
    <div className="source-toggle">
      <div className="segmented small">
        {['sim', 'live'].map((s) => {
          const on = src === s;
          const label = s === 'live' && connecting ? 'LIVE - connecting...' : s.toUpperCase();
          return (
            <button
              key={s}
              className={`seg ${s} ${on ? 'on' : ''}`}
              disabled={connecting || !avail.includes(s)}
              title={
                s === 'sim'
                  ? 'Fake drone (ui.sim); start the server with --sim'
                  : 'Real drone link (ui.live); start the server with --live'
              }
              onClick={() => send('set_source', { source: s })}
            >
              {label}
            </button>
          );
        })}
      </div>
      {src === 'none' && !connecting && <span className="src-note">no source</span>}
      {err && !connecting && <span className="src-error" title={err}>{err}</span>}
    </div>
  );
}

export default function App() {
  const { state, connected, lastMsgAt, send } = useSocket();
  const geoms = usePathGeometry(state?.paths);
  const config = useConfig();

  // Safety hotkeys: Space or Escape = emergency stop (unless typing in a field).
  useEffect(() => {
    const onKey = (e) => {
      const tag = (e.target && e.target.tagName) || '';
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (e.code === 'Space' || e.code === 'Escape') {
        e.preventDefault();
        send('stop');
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [send]);

  const alarm = state?.alarm;
  const flight = state?.flight?.state;
  const trackingLost = state && state.tracking && !state.tracking.ok && flight && flight !== 'idle';

  return (
    <div className={`app ${alarm?.active ? 'alarm-active' : ''}`}>
      <main className="view">
        <header className="brand">
          <span className="logo">PATHCASTER</span>
          <span className="mode-badge">{(state?.mode || 'guide').toUpperCase()} MODE</span>
          {state?.active_path && <span className="active-badge">path: {state.active_path}</span>}
          <SourceToggle state={state} config={config} send={send} />
        </header>
        {alarm?.active && (
          <div className="banner alarm-banner">
            ALARM - LEADING OUT VIA EXIT {alarm.exit}
            {alarm.blocked_exits?.length ? <small>blocked: {alarm.blocked_exits.join(', ')}</small> : null}
          </div>
        )}
        {flight === 'estop' && <div className="banner estop-banner">EMERGENCY STOP - motors off</div>}
        {trackingLost && <div className="banner lost-banner">TRACKING LOST</div>}
        {!connected && <div className="banner ws-banner">connecting to backend...</div>}
        <Scene state={state} geoms={geoms} config={config} />
        <div className="legend">
          <span><i style={{ background: '#00e5ff' }} /> drone</span>
          <span><i style={{ background: '#ff6bd6' }} /> wand</span>
          <span><i style={{ background: '#ffd93d' }} /> recording</span>
          <span><i style={{ background: '#ff9f43' }} /> guide path</span>
          <span><i style={{ background: '#a29bfe' }} /> spell path</span>
          <span><i style={{ background: '#ffffff' }} /> active + ghost</span>
          <span><i style={{ background: '#39d98a' }} /> geofence</span>
        </div>
      </main>
      <Controls state={state} send={send} />
      <StatusBar state={state} connected={connected} lastMsgAt={lastMsgAt} />
    </div>
  );
}
