import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { URL } from 'node:url';
import { Buffer } from 'node:buffer';
import ts from 'typescript';

const source = await readFile(new URL('../src/services/chatProgress.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { withChatProgress, progressText } = await import('data:text/javascript;base64,' + Buffer.from(compiled).toString('base64'));

test('quota text only appears for an explicit quota status', () => {
  assert.match(progressText({ phase:'quota_wait', provider:'Groq' }), /Waiting for Groq quota/);
  assert.doesNotMatch(progressText(null), /quota/);
  assert.doesNotMatch(progressText({ phase:'processing' }), /quota/);
});

test('polls quota and resumed processing, then stops and aborts polling', async () => {
  let complete;
  let signal;
  let polls = 0;
  const updates = [];
  const work = new Promise(resolve => { complete = resolve; });
  const result = await withChatProgress(() => work, async s => {
    signal = s;
    return { phase: ++polls === 1 ? 'quota_wait' : 'processing' };
  }, value => { updates.push(value.phase); if (updates.length === 2) complete('answer'); }, 1);
  assert.equal(result, 'answer');
  assert.deepEqual(updates, ['quota_wait', 'processing']);
  assert.equal(signal.aborted, true);
});

test('status errors never discard the main answer', async () => {
  let complete;
  const work = new Promise(resolve => { complete = resolve; });
  const result = await withChatProgress(() => work, async () => { throw new Error('offline'); },
    value => { assert.equal(value.phase, 'unavailable'); complete('answer'); }, 1);
  assert.equal(result, 'answer');
});

test('main request failure propagates without starting a polling loop', async () => {
  let polled = false;
  await assert.rejects(withChatProgress(async () => { throw new Error('request failed'); },
    async () => { polled = true; return { phase:'processing' }; }, () => {}, 1), /request failed/);
  assert.equal(polled, false);
});
