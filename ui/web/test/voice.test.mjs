import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseCommand, isAbort, matchPathName, precheck, VoiceListener } from '../src/voice.js';

const paths = [{ name: 'exit_alpha' }, { name: 'exit_bravo' }, { name: 'spiral' }];
const cmd = (t) => parseCommand(t, paths);

test('every phrase maps to the right command', () => {
  assert.deepEqual(cmd('Abort!'), { name: 'land', args: {} });
  assert.deepEqual(cmd('halt'), { name: 'land', args: {} });
  assert.deepEqual(cmd('Evacuate.'), { name: 'alarm', args: {} });
  assert.deepEqual(cmd('begin recording'), { name: 'record_start', args: {} });
  assert.deepEqual(cmd('End recording'), { name: 'record_stop', args: {} });
  assert.deepEqual(cmd('and recording'), { name: 'record_stop', args: {} });
  assert.deepEqual(cmd('save as exit alpha'), { name: 'save_as', args: { name: 'exit_alpha' } });
  assert.deepEqual(cmd('save this as Spiral'), { name: 'save_as', args: { name: 'spiral' } });
  assert.deepEqual(cmd('launch spiral'), { name: 'cast', args: { name: 'spiral' } });
  assert.deepEqual(cmd('lunch exit bravo'), { name: 'cast', args: { name: 'exit_bravo' } });
  assert.deepEqual(cmd('exit alpha blocked'), { name: 'exit_blocked', args: { exit: 'A' } });
  assert.deepEqual(cmd('exit a is blocked'), { name: 'exit_blocked', args: { exit: 'A' } });
  assert.deepEqual(cmd('exit bravo blocked'), { name: 'exit_blocked', args: { exit: 'B' } });
  assert.deepEqual(cmd('touch down'), { name: 'land', args: {} });
  assert.deepEqual(cmd('Touchdown'), { name: 'land', args: {} });
  assert.deepEqual(cmd('guide mode'), { name: 'set_mode', args: { mode: 'guide' } });
  assert.deepEqual(cmd('spell mode'), { name: 'set_mode', args: { mode: 'spell' } });
  assert.deepEqual(cmd('reset'), { name: 'clear_alarm', args: {} });
});

test('pitch sentences trigger nothing', () => {
  for (const s of [
    "exit signs can't move",
    'if the fire is at the door the sign still points you into it',
    'we stop the recording and it flies the route',
    'this is an emergency situation',
    "let's go to the next part",
    'it will land at the exit',
    'show it a path once with your hands',
  ]) {
    assert.equal(cmd(s), null, s);
  }
});

test('launch fuzzy-matches saved names', () => {
  assert.equal(matchPathName('spyral', paths), 'spiral');
  assert.equal(matchPathName('exit alfa', paths), 'exit_alpha');
  assert.equal(matchPathName('zigzag', paths), 'zigzag'); // unknown: sent as-is, backend errors
});

test('abort is detected in interim text', () => {
  assert.ok(isAbort('okay abort'));
  assert.ok(!isAbort('aboard the drone'));
});

// Fake SpeechRecognition so the listener logic can run in Node.
class FakeSR {
  start() {}
  stop() {}
}
const result = (transcript, confidence, isFinal) => {
  const r = [{ transcript, confidence }];
  r.isFinal = isFinal;
  return r;
};

function makeListener() {
  globalThis.window = { SpeechRecognition: FakeSR };
  const sent = [];
  const ignored = [];
  const l = new VoiceListener({
    getPaths: () => paths,
    onCommand: ({ cmd }) => sent.push(cmd.name),
    onIgnored: ({ reason }) => ignored.push(reason),
  });
  return { l, sent, ignored };
}

test('abort lands: fires once from interim, not again on final', () => {
  const { l, sent } = makeListener();
  const r0 = result('abort', 0, false);
  l.handleResult({ resultIndex: 0, results: [r0] });
  l.handleResult({ resultIndex: 0, results: [result('abort the', 0, false)] });
  l.handleResult({ resultIndex: 0, results: [result('abort the flight', 0.9, true)] });
  assert.deepEqual(sent, ['land']);
});

test('low confidence commands are ignored, abort is not', () => {
  const { l, sent, ignored } = makeListener();
  l.handleResult({ resultIndex: 0, results: [result('evacuate', 0.4, true)] });
  l.handleResult({ resultIndex: 1, results: [null, result('halt', 0.2, true)] });
  assert.deepEqual(sent, ['land']);
  assert.deepEqual(ignored, ['low confidence']);
});

test('interim commands other than abort wait for the final result', () => {
  const { l, sent } = makeListener();
  l.handleResult({ resultIndex: 0, results: [result('evacuate', 0, false)] });
  assert.deepEqual(sent, []);
  l.handleResult({ resultIndex: 0, results: [result('evacuate', 0.85, true)] });
  assert.deepEqual(sent, ['alarm']);
});

test('voice never sends the motor kill', () => {
  for (const s of ['abort', 'halt', 'abort abort', 'halt the drone']) {
    assert.notEqual(cmd(s).name, 'stop', s);
  }
});

test('precheck blocks commands the backend would drop', () => {
  const state = { paths, recording: { active: false, n_samples: 0 } };
  assert.equal(precheck({ name: 'cast', args: { name: 'spiral' } }, state), null);
  assert.match(precheck({ name: 'cast', args: { name: 'zigzag' } }, state), /no saved path "zigzag"/);
  assert.equal(precheck({ name: 'cast', args: { name: 'x' } }, {}), 'no saved paths yet');
  assert.equal(precheck({ name: 'save_as', args: { name: 'a' } }, state), 'nothing recorded yet');
  assert.match(precheck({ name: 'save_as', args: { name: 'a' } }, { recording: { active: true, n_samples: 50 } }), /end recording/);
  assert.equal(precheck({ name: 'save_as', args: { name: 'a' } }, { recording: { active: false, n_samples: 50 } }), null);
  // landing is never blocked, whatever the page thinks the flight state is
  assert.equal(precheck({ name: 'land', args: {} }, { flight: { state: 'idle' } }), null);
});

test('speech errors stop the listener and explain why; routine ones are ignored', () => {
  const { l } = makeListener();
  const errors = [];
  l.cb.onError = (t) => errors.push(t);
  l.listening = true;
  l.rec.onerror({ error: 'no-speech' });
  l.rec.onerror({ error: 'aborted' });
  assert.deepEqual(errors, []);
  assert.equal(l.listening, true);
  l.rec.onerror({ error: 'network' });
  assert.equal(l.listening, false);
  assert.match(errors[0], /unreachable/);
  l.rec.onerror({ error: 'weird' });
  assert.equal(errors[1], 'speech error: weird');
});

test('hearing callbacks follow speech start and end', () => {
  const { l } = makeListener();
  const seen = [];
  l.cb.onHearing = (on) => seen.push(on);
  l.rec.onspeechstart();
  l.rec.onspeechend();
  assert.deepEqual(seen, [true, false]);
});
