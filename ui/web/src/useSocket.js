import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * WebSocket to /ws with automatic reconnect.  Returns the latest AppState
 * snapshot, the socket status and a `send(name, args)` that pushes a command
 * over the socket (falling back to POST /api/command while reconnecting).
 */
export function useSocket() {
  const [state, setState] = useState(null);
  const [connected, setConnected] = useState(false);
  const [lastMsgAt, setLastMsgAt] = useState(0);
  const wsRef = useRef(null);
  // The server answers commands in order, so each ack/error settles the oldest pending send.
  const pendingRef = useRef([]);

  useEffect(() => {
    let alive = true;
    let retry = 400;
    let timer = null;

    const connect = () => {
      if (!alive) return;
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${proto}://${location.host}/ws`);
      wsRef.current = ws;
      ws.onopen = () => {
        setConnected(true);
        retry = 400;
      };
      ws.onmessage = (ev) => {
        let msg;
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (msg.type === 'state') {
          setState(msg.state);
          setLastMsgAt(Date.now());
        } else if (msg.type === 'ack') {
          pendingRef.current.shift()?.resolve(msg);
        } else if (msg.type === 'error') {
          console.warn('[ws] server error:', msg.error);
          pendingRef.current.shift()?.reject(new Error(msg.error));
        }
      };
      ws.onclose = () => {
        setConnected(false);
        wsRef.current = null;
        for (const p of pendingRef.current.splice(0)) p.reject(new Error('connection lost'));
        if (alive) {
          timer = setTimeout(connect, retry);
          retry = Math.min(retry * 1.7, 3000);
        }
      };
      ws.onerror = () => {
        try {
          ws.close();
        } catch {
          /* ignore */
        }
      };
    };
    connect();

    // while disconnected keep the snapshot fresh over plain HTTP
    const poll = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === 1) return;
      fetch('/api/state')
        .then((r) => (r.ok ? r.json() : null))
        .then((s) => {
          if (s) {
            setState(s);
            setLastMsgAt(Date.now());
          }
        })
        .catch(() => {});
    }, 1000);

    return () => {
      alive = false;
      clearTimeout(timer);
      clearInterval(poll);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // Returns a promise that resolves on the server's ack and rejects on its error.
  // Callers that do not care can ignore it.
  const send = useCallback((name, args = {}, source = 'ui') => {
    const ws = wsRef.current;
    if (ws && ws.readyState === 1) {
      const reply = new Promise((resolve, reject) => pendingRef.current.push({ resolve, reject }));
      reply.catch(() => {}); // unobserved rejections are fine
      ws.send(JSON.stringify({ type: 'command', name, args, source }));
      return reply;
    }
    const reply = fetch('/api/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, args, source }),
    }).then(async (r) => {
      if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail || `HTTP ${r.status}`);
      return r.json();
    });
    reply.catch((e) => console.warn('[cmd] failed', e));
    return reply;
  }, []);

  return { state, connected, lastMsgAt, send };
}

/**
 * Keeps the geometry (points + times) of every stored path, fetched from
 * GET /api/paths/{name} whenever the list in the state changes.
 */
export function usePathGeometry(summaries) {
  const [geoms, setGeoms] = useState({});
  const key = (summaries || []).map((p) => `${p.name}:${p.n_points}:${p.duration_s}`).join('|');

  useEffect(() => {
    let cancelled = false;
    const wanted = summaries || [];
    const names = new Set(wanted.map((p) => p.name));
    Promise.all(
      wanted.map((p) =>
        fetch(`/api/paths/${encodeURIComponent(p.name)}`)
          .then((r) => (r.ok ? r.json() : null))
          .then((d) => (d ? [p.name, { name: p.name, mode: d.mode, points: d.points, times: d.times }] : null))
          .catch(() => null),
      ),
    ).then((entries) => {
      if (cancelled) return;
      const next = {};
      for (const e of entries) if (e) next[e[0]] = e[1];
      // keep nothing for paths that disappeared
      setGeoms(Object.fromEntries(Object.entries(next).filter(([n]) => names.has(n))));
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return geoms;
}

/** GET /api/config once (geofence box etc.). */
export function useConfig() {
  const [cfg, setCfg] = useState(null);
  useEffect(() => {
    let alive = true;
    const load = () =>
      fetch('/api/config')
        .then((r) => (r.ok ? r.json() : null))
        .then((c) => {
          if (alive && c) setCfg(c);
          else if (alive) setTimeout(load, 1500);
        })
        .catch(() => alive && setTimeout(load, 1500));
    load();
    return () => {
      alive = false;
    };
  }, []);
  return cfg;
}
