import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
import { URL } from 'node:url';
import { Buffer } from 'node:buffer';

const source = (await readFile(new URL('../src/lib/guestSession.ts', import.meta.url), 'utf8'))
  .replace('import { supabase } from "./supabaseClient";', 'const supabase = null;');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { createSessionLoader } = await import('data:text/javascript;base64,' + Buffer.from(compiled).toString('base64'));

test('existing sessions are retained without creating a new identity', async () => {
  const session = { access_token: 'existing' };
  const load = createSessionLoader({ auth: {
    getSession: async () => ({ data: { session }, error: null }),
    signInAnonymously: () => { throw new Error('Must not replace identity'); },
  }});
  assert.equal(await load(), session);
});

test('concurrent requests create one anonymous identity', async () => {
  let created = 0;
  const session = { access_token: 'private' };
  const load = createSessionLoader({ auth: {
    getSession: async () => ({ data: { session: null }, error: null }),
    signInAnonymously: async () => { created++; return { data: { session }, error: null }; },
  }});
  const sessions = await Promise.all([load(), load(), load()]);
  assert.equal(created, 1);
  assert.ok(sessions.every(s => s === session));
});

test('disabled anonymous auth fails explicitly and can retry', async () => {
  let count = 0;
  const load = createSessionLoader({ auth: {
    getSession: async () => ({ data: { session: null }, error: null }),
    signInAnonymously: async () => { count++; return { data: {}, error: { code: 'anonymous_provider_disabled' } }; },
  }});
  await assert.rejects(load(), /Enable anonymous sign-ins/);
  await assert.rejects(load(), /Enable anonymous sign-ins/);
  assert.equal(count, 2);
});

test('restoration failure does not silently create a new library', async () => {
  const load = createSessionLoader({ auth: {
    getSession: async () => ({ data: {}, error: new Error('network') }),
    signInAnonymously: () => { throw new Error('Must not create'); },
  }});
  await assert.rejects(load(), /could not be restored/);
});
