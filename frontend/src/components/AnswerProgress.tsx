import { useEffect, useMemo, useState } from 'react';
import { Clock3, LoaderCircle } from 'lucide-react';
import { progressText, type ChatProgress } from '../services/chatProgress';

export default function AnswerProgress({ progress }: { progress: ChatProgress | null }) {
  const [now, setNow] = useState(Date.now);
  const waiting = progress?.phase === 'quota_wait';
  const deadline = useMemo(() => Date.now() + (progress?.retry_after_seconds || 0) * 1000, [progress]);
  useEffect(() => {
    if (!waiting) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [waiting, deadline]);
  const seconds = Math.max(0, Math.ceil((deadline - now) / 1000));
  return <div className={`max-w-3xl rounded-xl border p-4 text-sm ${waiting ? 'border-amber-400/25 bg-amber-400/5' : 'border-slate-800 bg-slate-900'}`}>
    <p role="status" className={`flex items-center gap-2 ${waiting ? 'text-amber-300' : 'text-slate-300'}`}>
      {waiting ? <Clock3 size={17} aria-hidden="true" /> : <LoaderCircle size={17} className="animate-spin" aria-hidden="true" />}
      {progressText(progress)}
    </p>
    {waiting && <p className="mt-2 text-xs text-slate-400">
      The provider asked us to pause. Your question is still active—no need to send it again.
      {' '}{seconds > 0 ? `Next retry in about ${seconds}s.` : 'Retrying when the provider allows it.'}
      {' '}The provider may extend this wait.
    </p>}
  </div>;
}
