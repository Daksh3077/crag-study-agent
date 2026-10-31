import { ensurePrivateSession } from "../lib/guestSession";
import { withChatProgress, type ChatProgress } from './chatProgress';
export type { ChatProgress } from './chatProgress';
const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

export interface SourceCitation {
  citation_id: string; document_id: string; document_name: string;
  page?: number; chunk_index?: number; snippet: string;
}
export interface PipelineStep {
  name: string; status: "done" | "skipped" | "running" | "error" | "warning"; detail?: string;
}
export interface ChatResponse {
  conversation_id: string; answer: string; sources: SourceCitation[];
  relevance_score: number; groundedness_confidence: number | null;
  validation_status: "supported" | "unsupported" | "unavailable" | "insufficient_context" | "generation_failed" | "not_checked";
  stage_timings_ms?: Record<string, number>;
  warnings: string[]; correction_attempts: number; generation_attempts: number;
  used_external_fallback: boolean; pipeline: PipelineStep[]; response_time_ms: number;
}
export type StudyMode = "ask" | "quick_revision" | "detailed_summary" | "exam_answer" | "important_questions" | "mcq";
export interface PageReport { page: number; status: string; method: string; warning?: string; error?: string; retry_at?: string; error_code?: string; quota_kind?: string; quota_limit?: string | null; }
export interface DocumentOut {
  id: string; user_id: string; name: string; subject?: string; file_path: string; created_at: string;
  status: string; page_count: number; processed_pages: number; chunk_count: number;
  ingestion: { pages?: PageReport[]; error?: string; ocr_mode?: string; waiting?: { page: number; retry_at: string } };
}
export interface AnalyticsSummary {
  total_documents: number; total_questions: number; average_confidence: number;
  average_correction_attempts: number; fallback_usage_rate: number; average_response_time_ms: number;
  supported_answers: number; validation_unavailable: number;
}
export interface Conversation { id: string; title: string; created_at: string; }
export interface SavedTurn { role: "user" | "assistant"; content: string; metadata?: ChatResponse; }

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const session = await ensurePrivateSession();
  const token = session.access_token;
  const response = await fetch(BASE_URL + path, {
    ...options,
    headers: { ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      Authorization: `Bearer ${token}`, ...options.headers },
  });
  if (!response.ok) {
    let detail = "";
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : "";
    } catch { /* Use the HTTP status if no JSON body is available. */ }
    throw new Error(detail || `Request failed (${response.status}). Check the backend logs.`);
  }
  return response.json() as Promise<T>;
}
export const sendChatMessage = (params: { question: string; document_ids: string[]; mode: StudyMode; conversation_id?: string; verification_mode?: "adaptive" | "full" }, onProgress?: (progress: ChatProgress) => void, signal?: AbortSignal) => {
  const requestId = crypto.randomUUID();
  const work = () => request<ChatResponse>("/api/chat", { method: "POST", body: JSON.stringify({ ...params, request_id: requestId }), signal });
  return onProgress ? withChatProgress(work,
    pollSignal => request<ChatProgress>(`/api/chat/requests/${requestId}`, { signal: pollSignal }), onProgress) : work();
};
export function uploadDocument(params: { file: File; subject?: string; ocr_mode: "auto" | "force" }) {
  const body = new FormData();
  body.append("file", params.file);
  body.append("ocr_mode", params.ocr_mode);
  if (params.subject) body.append("subject", params.subject);
  return request<DocumentOut>("/api/documents/upload", { method: "POST", body });
}
export const listDocuments = () => request<DocumentOut[]>("/api/documents");
export const deleteDocument = (id: string) => request("/api/documents/" + id, { method: "DELETE" });
export const retryDocument = (id: string) => request("/api/documents/" + id + "/retry", { method: "POST" });
export const getDocumentSource = (id: string) => request<{ url: string }>("/api/documents/" + id + "/source");
export const getAnalyticsSummary = () => request<AnalyticsSummary>("/api/analytics/summary");
export const listConversations = () => request<Conversation[]>("/api/chat/conversations");
export const getConversation = (id: string) => request<SavedTurn[]>("/api/chat/conversations/" + id);
