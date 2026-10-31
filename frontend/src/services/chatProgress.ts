export interface ChatProgress {
  phase: 'processing' | 'quota_wait' | 'complete' | 'failed' | 'unavailable';
  provider?: string | null;
  retry_after_seconds?: number;
}

// Status polling is optional: a status outage must never discard an answer.
export async function withChatProgress<T>(
  work: () => Promise<T>, poll: (signal: AbortSignal) => Promise<ChatProgress>,
  update: (value: ChatProgress) => void, interval = 1500,
): Promise<T> {
  const controller = new AbortController();
  let finished = false;
  let failures = 0;
  let timer: ReturnType<typeof setTimeout>;
  async function tick() {
    try {
      const progress = await poll(controller.signal);
      failures = 0;
      if (!finished) update(progress);
    } catch {
      // The first lookup can precede registration of the POST request.
      if (!finished && ++failures >= 2) update({ phase: 'unavailable' });
    }
    if (!finished) timer = setTimeout(() => { void tick(); }, interval);
  }
  timer = setTimeout(() => { void tick(); }, interval);
  try { return await work(); }
  finally { finished = true; clearTimeout(timer); controller.abort(); }
}

export function progressText(progress: ChatProgress | null): string {
  if (progress?.phase === 'quota_wait') return `Waiting for ${progress.provider || 'model-provider'} quota`;
  if (progress?.phase === 'unavailable') return 'Live status is temporarily unavailable. Your question is still running.';
  if (progress?.phase === 'complete') return 'Your answer is ready. Loading the response…';
  if (progress?.phase === 'failed') return 'The request has stopped. Loading the result…';
  return 'Reading your notes and preparing the answer…';
}
