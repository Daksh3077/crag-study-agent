import type { SessionUsage } from "../services/api";

export default function SessionAllowance({ usage, error, onRetry }: {
  usage: SessionUsage | null; error: string; onRetry: () => void;
}) {
  return <div className="text-xs text-slate-400 space-y-1" aria-live="polite">
    <p>{usage
      ? `Guest allowance: ${Math.max(0, usage.questions_limit - usage.questions_used)} of ${usage.questions_limit} questions left · ${Math.max(0, usage.uploads_limit - usage.uploads_used)} of ${usage.uploads_limit} uploads left`
      : error || "Loading your guest allowance…"}</p>
    {error && <button type="button" className="secondary" onClick={onRetry}>Retry allowance check</button>}
    <p>5 questions across all study modes and 2 total uploads per guest session. Refreshing, new conversations, and deleting files do not reset these limits. Accepted requests count even if processing fails.</p>
    {usage && (usage.questions_used >= usage.questions_limit || usage.uploads_used >= usage.uploads_limit) &&
      <p>You can still read your existing documents and answers.</p>}
  </div>;
}
