import { CheckCircle2, RotateCcw, Circle, AlertTriangle, XCircle } from "lucide-react";
import type { PipelineStep } from "../services/api";
const icons = { done: CheckCircle2, skipped: Circle, running: RotateCcw, warning: AlertTriangle, error: XCircle };
const colors = { done: "text-emerald-400", skipped: "text-slate-500", running: "text-brand-400", warning: "text-amber-400", error: "text-rose-400" };
export default function PipelineTrace({ steps }: { steps: PipelineStep[] }) {
  if (!steps.length) return null;
  return <details className="border border-slate-800 rounded-lg p-3 text-xs">
    <summary className="cursor-pointer text-slate-400">How this answer was checked</summary>
    <ol className="mt-3 space-y-2">{steps.map((step, index) => {
      const Icon = icons[step.status] || AlertTriangle;
      return <li key={index} className="flex gap-2"><Icon aria-hidden className={`w-4 h-4 shrink-0 ${colors[step.status] || "text-amber-400"}`} />
        <span><strong>{step.name}</strong>{step.detail && <span className="block text-slate-400">{step.detail}</span>}</span></li>;
    })}</ol>
  </details>;
}
