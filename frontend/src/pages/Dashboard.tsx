import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { UploadCloud } from "lucide-react";
import AppHeader from "../components/AppHeader";
import SessionAllowance from "../components/SessionAllowance";
import { useSessionUsage } from "../lib/useSessionUsage";
import { deleteDocument, listDocuments, retryDocument, uploadDocument, type DocumentOut } from "../services/api";

export default function Dashboard() {
  const { usage, usageError, refreshUsage } = useSessionUsage();
  const uploadsBlocked = !usage || usage.uploads_used >= usage.uploads_limit;
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [ocrMode, setOcrMode] = useState<"auto" | "force">("auto");
  const [subject, setSubject] = useState("");
  const [now, setNow] = useState(Date.now);
  const refresh = useCallback(async () => {
    try { setDocuments(await listDocuments()); setError(""); }
    catch (e) { setError(e instanceof Error ? e.message : "Could not load documents."); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);
  const processing = documents.some(d => ["queued", "processing"].includes(d.status));
  const hasQuotaPause = documents.some(d => d.ingestion.pages?.some(p => p.status === "pending" || p.error_code === "rate_limit"));
  useEffect(() => {
    if (!hasQuotaPause) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [hasQuotaPause]);
  useEffect(() => {
    if (!processing) return;
    const timer = window.setInterval(() => { void refresh(); }, 3000);
    return () => window.clearInterval(timer);
  }, [processing, refresh]);
  async function upload(file: File) {
    if (uploadsBlocked) return;
    setUploading(true); setError("");
    try { await uploadDocument({ file, subject, ocr_mode: ocrMode }); await refresh(); }
    catch (e) { setError(e instanceof Error ? e.message : "Upload failed."); }
    finally { await refreshUsage(); setUploading(false); }
  }
  async function action(id: string, type: "retry" | "delete") {
    if (type === "delete" && !window.confirm("Delete this document and its indexed content?")) return;
    try { await (type === "retry" ? retryDocument(id) : deleteDocument(id)); await refresh(); }
    catch (e) { setError(e instanceof Error ? e.message : "Document action failed."); }
  }
  return <div className="min-h-screen"><AppHeader />
    <main className="max-w-5xl mx-auto p-6 sm:py-10 space-y-6">
      <div className="library-heading"><div><p className="eyebrow">YOUR PERSONAL KNOWLEDGE SPACE</p><h1>A library of possibilities.</h1><p className="mt-2 text-sm text-slate-400">Bring your notes. Find your next moment of clarity.</p></div><div className="library-stats"><div><strong>{loading ? '—' : documents.length}</strong><span>Documents</span></div><div><strong>{loading ? '—' : documents.filter(d => d.status === 'ready').length}</strong><span>Ready to study</span></div></div></div>
      <p className="text-xs text-slate-400">Your library is private to this browser session. Keep this browser's site data to retain access; another browser or device starts a separate library.</p>
      <section aria-label="Upload study material" className="panel upload-panel space-y-4">
        <SessionAllowance usage={usage} error={usageError} onRetry={() => { void refreshUsage(); }} />
        <div className="upload-title"><UploadCloud size={25} aria-hidden="true" /><div><h2 className="font-medium">Add to your library</h2><p className="text-xs text-slate-400 mt-1">Typed, scanned, or handwritten. Start with a document.</p></div></div>
        <div className="grid sm:grid-cols-2 gap-4">
          <label className="text-sm">Document type<select className="field mt-1" value={ocrMode} onChange={e => setOcrMode(e.target.value as "auto" | "force")}>
            <option value="auto">Automatic: typed or scanned</option><option value="force">Handwritten / mixed pages: OCR every page</option>
          </select></label>
          <label className="text-sm">Subject (optional)<input className="field mt-1" value={subject} onChange={e => setSubject(e.target.value)} placeholder="e.g. Computer networks" /></label>
        </div>
        <p className="text-xs text-slate-400">For handwritten notes, select OCR every page. Use clear, upright scans with all page edges visible. Dense pages may take longer.</p>
        <label className="upload-zone block rounded-xl border border-dashed p-6 text-center cursor-pointer hover:border-brand-400">
          <span>{uploading ? "Saving your document…" : "Choose a PDF, TXT, JPG, PNG or WEBP"}</span>
          <input aria-label="Choose study document" type="file" className="block mx-auto mt-4 max-w-full text-sm text-slate-400"
            accept=".pdf,.txt,.jpg,.jpeg,.png,.webp" disabled={uploading || uploadsBlocked}
            onChange={e => { const file = e.target.files?.[0]; if (file) void upload(file); e.target.value = ""; }} />
        </label>
      </section>
      {error && <p role="alert" className="notice">{error}</p>}
      {loading && <p role="status" className="text-slate-400">Loading your documents…</p>}
      {!loading && !documents.length && <p className="text-slate-400">Your uploaded documents will appear here.</p>}
      <div className="space-y-4">{documents.map(doc => {
        const busy = ["queued", "processing"].includes(doc.status);
        const pages = doc.ingestion.pages || [];
        const pending = pages.filter(p => p.status === "pending");
        const problems = pages.filter(p => p.status !== "pending" && (p.status === "failed" || p.warning));
        const retryAt = Math.max(0, ...pages.filter(p => p.status === "pending" || p.error_code === "rate_limit")
          .map(p => Date.parse(p.retry_at || "")).filter(Number.isFinite));
        const quotaPaused = !busy && (pending.length > 0 || retryAt > now);
        const retryBlocked = retryAt > now;
        const dailyQuota = pages.find(p => p.quota_kind === "daily" && Date.parse(p.retry_at || "") > now);
        return <article key={doc.id} className="panel space-y-3">
          <div className="flex flex-wrap justify-between gap-3">
            <div><h2 className="font-semibold break-all">{doc.name}</h2><p className="text-sm text-slate-400">{doc.subject || "Study notes"}</p></div>
            <span className={doc.status === "ready" ? "text-emerald-400 text-sm" : "text-amber-300 text-sm"}>{quotaPaused ? "OCR paused" : doc.status.replace(/_/g, " ")}</span>
          </div>
          <p className="text-xs text-slate-400">{doc.processed_pages} / {doc.page_count || "?"} pages processed · {doc.chunk_count} searchable passages</p>
          {busy && <progress aria-label="Document processing progress" className="w-full accent-indigo-500" max={doc.page_count || 1} value={doc.processed_pages} />}
          {busy && doc.ingestion.waiting && <p role="status" className="notice">The OCR provider requested a pause on page {doc.ingestion.waiting.page}. Next retry: {new Date(doc.ingestion.waiting.retry_at).toLocaleTimeString()}. The provider must have quota available to continue. Completed pages are saved.</p>}
          {doc.status === "needs_reindex" && <p className="notice">Re-upload this older document to use the improved extraction and retrieval pipeline.</p>}
          {quotaPaused && <p role="status" className="notice">
            {dailyQuota && <>The OCR model's daily quota{dailyQuota.quota_limit ? ` (limit: ${dailyQuota.quota_limit})` : ""} is exhausted. Retrying every minute will not restore it. </>}
            {pending.length > 0 ? `${pending.length} pages are waiting for OCR. ` : "OCR retries are waiting for provider quota. "}
            {retryBlocked ? `Resume after ${new Date(retryAt).toLocaleString()}. ` : "The saved waiting period has ended. Select Resume to continue. "}
            Completed pages are saved. Answers currently cover only the searchable pages.
          </p>}
          {doc.ingestion.error && !quotaPaused && <p role="alert" className="notice">{doc.ingestion.error}</p>}
          {!!problems.length && <details><summary className="text-amber-300 cursor-pointer text-sm">{problems.length} pages need attention</summary>
            <ul className="mt-2 text-sm space-y-1">{problems.map(p => <li key={p.page}>Page {p.page}: {p.warning || p.error}</li>)}</ul>
          </details>}
          <div className="flex gap-4 text-sm">
            {["ready", "partial"].includes(doc.status) && <Link className="text-brand-400" to={`/workspace?document=${doc.id}`}>Study this document</Link>}
            {["failed", "partial"].includes(doc.status) && <button disabled={retryBlocked} className="text-brand-400 disabled:opacity-50 disabled:cursor-not-allowed" onClick={() => { void action(doc.id, "retry"); }}>Resume pending / unclear pages</button>}
            <button disabled={busy} className="text-slate-400 hover:text-rose-400 disabled:opacity-30" onClick={() => { void action(doc.id, "delete"); }}>Delete</button>
          </div>
        </article>;
      })}</div>
    </main>
  </div>;
}
