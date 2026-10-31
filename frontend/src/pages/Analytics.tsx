import { useEffect, useState } from "react";
import AppHeader from "../components/AppHeader";
import { getAnalyticsSummary, type AnalyticsSummary } from "../services/api";
export default function Analytics() {
  const [data, setData] = useState<AnalyticsSummary>();
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    getAnalyticsSummary().then(d => { if (active) setData(d); }).catch(e => { if (active) setError(e.message); });
    return () => { active = false; };
  }, []);
  const cards = data ? [
    ["Documents", data.total_documents], ["Questions", data.total_questions],
    ["Answers checked as supported", data.supported_answers], ["Validation unavailable", data.validation_unavailable],
    ["Average retrieval corrections", data.average_correction_attempts.toFixed(1)],
    ["Average response time", `${(data.average_response_time_ms / 1000).toFixed(1)}s`],
  ] : [];
  return <div className="min-h-screen"><AppHeader /><main className="max-w-5xl mx-auto p-6 py-10">
    <h1 className="text-3xl font-semibold">Your study activity</h1>
    <p className="text-slate-400 mt-2 mb-8">Source-validation results describe the automated checks. They are not a guarantee that an answer is correct.</p>
    {error ? <p role="alert" className="notice">{error}</p> : !data ? <p role="status">Loading analytics…</p> :
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">{cards.map(([label, value]) => <div className="panel" key={label}>
        <p className="text-sm text-slate-400">{label}</p><p className="text-3xl font-semibold mt-3">{value}</p>
      </div>)}</div>}
  </main></div>;
}
