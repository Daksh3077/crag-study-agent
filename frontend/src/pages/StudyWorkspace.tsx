import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import AppHeader from "../components/AppHeader";
import AnswerContent from "../components/AnswerContent";
import DocumentList from "../components/DocumentList";
import PipelineTrace from "../components/PipelineTrace";
import AnswerProgress from '../components/AnswerProgress';
import { getConversation, getDocumentSource, listConversations, listDocuments, sendChatMessage,
  type ChatResponse, type ChatProgress, type Conversation, type DocumentOut, type StudyMode } from "../services/api";
interface Turn { role: "user" | "assistant"; content: string; meta?: ChatResponse; }
const modes: { value: StudyMode; label: string }[] = [
  { value: "ask", label: "Ask" }, { value: "quick_revision", label: "Quick revision" },
  { value: "detailed_summary", label: "Detailed summary" }, { value: "exam_answer", label: "Exam answer" },
  { value: "important_questions", label: "Study questions" }, { value: "mcq", label: "MCQs" },
];
const statuses = { supported: "Checked against sources", unsupported: "Could not verify the answer",
  not_checked: "Adaptive answer · not independently checked",
  unavailable: "Source check unavailable", insufficient_context: "Not enough evidence", generation_failed: "Generation failed" };
export default function StudyWorkspace() {
  const [params] = useSearchParams();
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [selected, setSelected] = useState<string[]>(params.get("document") ? [params.get("document")!] : []);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationId, setConversationId] = useState<string>();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [mode, setMode] = useState<StudyMode>("ask");
  const [verification, setVerification] = useState<"adaptive" | "full">("adaptive");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [answerPending, setAnswerPending] = useState(false);
  const [progress, setProgress] = useState<ChatProgress | null>(null);
  const activeRequest = useRef<AbortController | null>(null);
  useEffect(() => () => activeRequest.current?.abort(), []);
  const [error, setError] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const bottom = useRef<HTMLDivElement>(null);
  const preview = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    let active = true;
    Promise.all([listDocuments(), listConversations()]).then(([docs, chats]) => {
      if (active) { setDocuments(docs.filter(d => ["ready", "partial"].includes(d.status))); setConversations(chats); }
    }).catch(e => { if (active) setError(e.message); });
    return () => { active = false; };
  }, []);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth" }); }, [turns, loading]);
  async function send() {
    if (!input.trim() || loading || !documents.length) return;
    const question = input.trim();
    setInput(""); setError(""); setLoading(true);
    setAnswerPending(true); setProgress(null);
    const controller = new AbortController();
    activeRequest.current = controller;
    setTurns(t => [...t, { role: "user", content: question }]);
    try {
      const response = await sendChatMessage({ question, document_ids: selected, mode, conversation_id: conversationId,
        verification_mode: verification }, setProgress, controller.signal);
      setConversationId(response.conversation_id);
      setTurns(t => [...t, { role: "assistant", content: response.answer, meta: response }]);
      listConversations().then(setConversations).catch(() => { /* Answer remains usable if sidebar refresh fails. */ });
    } catch (e) {
      if (controller.signal.aborted) return;
      setError(e instanceof Error ? e.message : "Could not reach the study agent.");
      setInput(question);
      setTurns(t => t.slice(0, -1));
    } finally { setLoading(false); setAnswerPending(false); setProgress(null); activeRequest.current = null; }
  }
  async function openConversation(id: string) {
    if (loading) return;
    if (!id) { setConversationId(undefined); setTurns([]); return; }
    setLoading(true); setError("");
    try {
      const messages = await getConversation(id);
      setConversationId(id);
      setTurns(messages.map(m => ({ role: m.role, content: m.content, meta: m.metadata?.answer ? m.metadata : undefined })));
    } catch (e) { setError(e instanceof Error ? e.message : "Could not load conversation."); }
    finally { setLoading(false); }
  }
  async function openSource(id: string, page?: number) {
    try { const result = await getDocumentSource(id); setSourceUrl(result.url + (page ? `#page=${page}` : "")); preview.current?.showModal(); }
    catch (e) { setError(e instanceof Error ? e.message : "Could not open source."); }
  }
  return <div className="workspace-shell h-screen flex flex-col"><AppHeader />
    <div className="flex flex-1 min-h-0 flex-col md:flex-row">
      <aside className="md:w-64 shrink-0 border-b md:border-b-0 md:border-r border-slate-800 p-4 overflow-y-auto max-h-56 md:max-h-none">
        <details open className="mb-5"><summary className="font-medium text-sm mb-2 cursor-pointer">Source documents</summary>
          <DocumentList documents={documents} selectedIds={selected} onToggle={id => {
            if (!loading) setSelected(s => s.includes(id) ? s.filter(x => x !== id) : [...s, id]);
          }} />
          <p className="text-xs text-slate-500 mt-3">{selected.length ? `${selected.length} documents selected` : "Search all your ready documents"}</p>
        </details>
        <label className="text-xs text-slate-400">Conversation<select aria-label="Conversation" className="field mt-1" disabled={loading} value={conversationId || ""}
          onChange={e => { void openConversation(e.target.value); }}>
          <option value="">New conversation</option>{conversations.map(c => <option key={c.id} value={c.id}>{c.title}</option>)}
        </select></label>
      </aside>
      <main className="flex-1 flex flex-col min-w-0 min-h-0">
        <div className="flex gap-2 p-3 overflow-x-auto border-b border-slate-800">{modes.map(m => <button key={m.value} disabled={loading} aria-pressed={mode === m.value}
          onClick={() => setMode(m.value)} className={`rounded-full px-3 py-2 text-xs whitespace-nowrap ${mode === m.value ? "bg-brand-600" : "bg-slate-900 text-slate-400"}`}>{m.label}</button>)}</div>
        <div className="px-4 py-2 border-b border-slate-800 text-xs text-slate-400">
          <label>Answer checking <select className="bg-slate-900 rounded p-2 ml-2" disabled={loading}
            value={verification} onChange={e => setVerification(e.target.value as "adaptive" | "full")}>
            <option value="adaptive">Adaptive (faster)</option><option value="full">Full source check</option>
          </select></label>
          <p className="mt-1">{verification === "adaptive"
            ? "Strong matches in Ask mode skip the independent answer check. Uncertain matches and other study modes receive a full check."
            : "Every answer is checked against its cited passages; this takes longer."}</p>
        </div>
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
          {!turns.length && <div className="workspace-empty max-w-xl mx-auto"><p className="eyebrow mb-4">YOUR STUDY ROOM</p><h1>Make sense of the details.</h1>
            <p className="text-slate-400 mt-3">Ask a specific question, summarize your document, or generate practice questions. Answers link back to the pages used.</p>
            <div className="flex flex-wrap gap-2 mt-5">{["Summarize this document", "Generate 5 MCQs from these notes"].map(q => <button key={q} className="secondary text-sm" onClick={() => { setInput(q); setMode(q.startsWith("Summarize") ? "detailed_summary" : "mcq"); }}>{q}</button>)}</div>
          </div>}
          {turns.map((turn, i) => <div key={i} className={turn.role === "user" ? "flex justify-end" : ""}>
            <article className={`max-w-3xl rounded-xl p-4 sm:p-5 text-sm ${turn.role === "user" ? "bg-brand-600 whitespace-pre-wrap" : "bg-slate-900 border border-slate-800"}`}>
              {turn.role === "assistant" ? <AnswerContent content={turn.content} /> : turn.content}
              {turn.meta && <div className="mt-4 pt-4 border-t border-slate-800 space-y-3">
                <p className={turn.meta.validation_status === "supported" ? "text-emerald-400 text-xs" : "text-amber-300 text-xs"}>{statuses[turn.meta.validation_status]}</p>
                {turn.meta.warnings.map((warning, j) => <p className="notice" key={j}>{warning}</p>)}
                {!!turn.meta.sources.length && <details><summary className="text-brand-400 text-xs cursor-pointer">Source passages ({turn.meta.sources.length})</summary>
                  <ul className="mt-3 space-y-3">{turn.meta.sources.map(source => <li key={source.citation_id} className="border-l-2 border-slate-700 pl-3">
                    <button className="text-brand-400 text-xs text-left" onClick={() => { void openSource(source.document_id, source.page); }}>[{source.citation_id}] {source.document_name} · page {source.page || "?"}</button>
                    <p className="mt-1 text-xs text-slate-400 whitespace-pre-wrap">{source.snippet}</p>
                  </li>)}</ul>
                </details>}
                <PipelineTrace steps={turn.meta.pipeline} />
                {turn.meta.stage_timings_ms && <details className="text-xs text-slate-500">
                  <summary className="cursor-pointer">Processing times</summary>
                  <ul className="mt-2">{Object.entries(turn.meta.stage_timings_ms).map(([stage, ms]) =>
                    <li key={stage}>{stage}: {(ms / 1000).toFixed(2)}s</li>)}</ul>
                </details>}
                <p className="text-xs text-slate-500">{(turn.meta.response_time_ms / 1000).toFixed(1)}s · {turn.meta.correction_attempts} retrieval corrections · {turn.meta.generation_attempts} answer attempts</p>
              </div>}
            </article>
          </div>)}
          {answerPending && <AnswerProgress progress={progress} />}
          {loading && !answerPending && <p role="status" className="text-sm text-slate-400">Loading your conversation…</p>}
          <div ref={bottom} />
        </div>
        <form className="p-4 border-t border-slate-800" onSubmit={e => { e.preventDefault(); void send(); }}>
          {error && <p role="alert" className="notice mb-3">{error}</p>}
          <div className="flex gap-2"><label className="sr-only" htmlFor="question">Question about your notes</label>
            <textarea id="question" rows={2} className="field resize-none" value={input} maxLength={4000} placeholder="Ask about your notes…"
              onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(); } }} />
            <button className="primary self-end" disabled={loading || !input.trim() || !documents.length}>Send</button>
          </div>
        </form>
      </main>
    </div>
    <dialog ref={preview} className="w-[95vw] h-[90vh] bg-slate-950 text-white rounded-xl p-4 backdrop:bg-black/70">
      <div className="flex justify-between items-center mb-3"><h2>Original source</h2><button className="secondary" onClick={() => preview.current?.close()}>Close</button></div>
      {sourceUrl && <iframe title="Original source document" src={sourceUrl} className="w-full h-[calc(100%-4rem)] bg-white" />}
    </dialog>
  </div>;
}
