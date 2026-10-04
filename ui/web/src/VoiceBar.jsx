import React, { useEffect, useRef, useState } from 'react';
import { VoiceListener, parseCommand, precheck, speechSupported } from './voice.js';

const describe = (cmd) => [cmd.name, cmd.args?.name, cmd.args?.exit, cmd.args?.mode].filter(Boolean).join(' ');

/**
 * Voice panel: mic toggle, hold-V push-to-talk, live transcript, last-command
 * chip (green on ack, red on error or a failed precheck) and a typed fallback
 * for a noisy room. Voice commands go through the same `send` as the buttons,
 * tagged source 'voice'.
 */
export default function VoiceBar({ send, state }) {
  const paths = state?.paths || [];
  const [listening, setListening] = useState(false);
  const [hearing, setHearing] = useState(false);
  const [interim, setInterim] = useState('');
  const [final, setFinal] = useState('');
  const [chip, setChip] = useState(null); // {text, status: sent|ack|error|ignored}
  const [typed, setTyped] = useState('');
  const listener = useRef(null);
  const pathsRef = useRef(paths);
  const stateRef = useRef(state);
  const sendRef = useRef(send);
  const pttActive = useRef(false);
  const seq = useRef(0);
  pathsRef.current = paths;
  stateRef.current = state;
  sendRef.current = send;

  const dispatch = (cmd, heard) => {
    const id = ++seq.current;
    const text = `${describe(cmd)} ← "${heard.trim()}"`;
    const problem = precheck(cmd, stateRef.current);
    if (problem) {
      setChip({ id, text: `${text}: ${problem}`, status: 'error' });
      return;
    }
    const settle = (status) => setChip((c) => (c && c.id === id ? { ...c, status } : c));
    setChip({ id, text, status: 'sent' });
    Promise.resolve(sendRef.current(cmd.name, cmd.args, 'voice')).then(
      () => settle('ack'),
      () => settle('error'),
    );
  };

  useEffect(() => {
    if (!speechSupported()) return undefined;
    listener.current = new VoiceListener({
      getPaths: () => pathsRef.current,
      onListening: (on) => {
        setListening(on);
        if (on) setChip((c) => (c && c.status === 'fault' ? null : c));
        else setHearing(false);
      },
      onHearing: setHearing,
      onError: (text) => setChip({ id: ++seq.current, text, status: 'fault' }),
      onTranscript: ({ interim: i, final: f }) => {
        setInterim(i);
        if (f !== null) setFinal(f);
      },
      onCommand: ({ cmd, heard }) => dispatch(cmd, heard),
      onIgnored: ({ heard, confidence, reason }) => {
        if (reason === 'no command') return; // ordinary talking: stay quiet
        const conf = confidence ? `, ${confidence.toFixed(2)}` : '';
        setChip({ id: ++seq.current, text: `ignored "${heard.trim()}" (${reason}${conf})`, status: 'ignored' });
      },
    });
    return () => listener.current?.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Hold V to talk (not while typing in a field).
  useEffect(() => {
    const inField = (e) => ['INPUT', 'TEXTAREA'].includes((e.target && e.target.tagName) || '');
    const down = (e) => {
      if (e.code !== 'KeyV' || e.repeat || inField(e) || !listener.current) return;
      if (listener.current.listening) return; // mic already on
      pttActive.current = true;
      listener.current.start();
    };
    const up = (e) => {
      if (e.code !== 'KeyV' || !pttActive.current) return;
      pttActive.current = false;
      listener.current.stop();
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, []);

  const toggle = () => {
    if (!listener.current) return;
    if (listening) listener.current.stop();
    else listener.current.start();
  };

  const submitTyped = (e) => {
    e.preventDefault();
    const text = typed.trim();
    if (!text) return;
    const cmd = parseCommand(text, paths);
    if (cmd) dispatch(cmd, text);
    else setChip({ id: ++seq.current, text: `unknown "${text}"`, status: 'error' });
    setTyped('');
  };

  const supported = speechSupported();
  return (
    <section className="panel voice">
      <h2>Voice</h2>
      <div className="row">
        <button
          className={`btn mic ${listening ? 'on' : ''} ${hearing ? 'hearing' : ''}`}
          onClick={toggle}
          disabled={!supported}
          title={supported ? 'Click to toggle, or hold V to talk' : 'Speech recognition needs Chrome or Edge'}
        >
          {listening ? (hearing ? '● Hearing you...' : '● Listening - speak now') : supported ? 'Mic off (hold V)' : 'No speech in this browser'}
        </button>
      </div>
      <div className="transcript">
        {final || interim ? (
          <>
            <span className="final">{final}</span> <span className="interim">{interim}</span>
          </>
        ) : (
          <span className="interim">say "evacuate", "begin recording", "launch …"</span>
        )}
      </div>
      {chip && <div className={`voice-chip ${chip.status}`}>{chip.text}</div>}
      <form className="row" onSubmit={submitTyped}>
        <input
          className="text"
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          placeholder="or type a command"
          aria-label="Typed voice command"
        />
      </form>
    </section>
  );
}
