/**
 * Types of the HTTP API. They mirror the Pydantic schemas in backend/app/schemas/ and the
 * server-sent events of backend/app/api/routers/chat.py; keep the two in sync.
 */

/**
 * Mirrors `ReportStatus` in backend/app/models/report.py. The status tracks the report's index:
 * `ready` means it can be searched. How each datapoint's extraction went is in `extractions`.
 */
export type ReportStatus = "pending" | "parsing" | "chunking" | "embedding" | "extracting" | "ready" | "failed";

/** Mirrors `ExtractionStatus` in backend/app/models/report.py. */
export type ExtractionStatus = "pending" | "extracted" | "not_found" | "error";

/** Outcome of one registry entry on a report. */
export interface Extraction {
  key: string;
  status: ExtractionStatus;
  /** The error behind `error`; null otherwise. */
  message: string | null;
  updated_at: string;
}

export interface Report {
  id: string;
  company: string;
  fiscal_year: number;
  title: string | null;
  filename: string;
  page_count: number;
  status: ReportStatus;
  /** Ingestion progress, 0–100. */
  progress: number;
  /** Units of the current stage done and in total: pages while parsing, chunks while embedding, datapoints while extracting. */
  done: number;
  total: number;
  /** What the worker did last ("Embedded chunks 65-128 of 928 in 2.1 s"); `updated_at` says when. */
  activity: string | null;
  /** When ingestion started; null until it does. */
  started_at: string | null;
  /** Which vectors the report holds; searchable only when it equals `Status.embedding_model`. */
  embedding_model: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
  /** One per registry entry once extraction has started; empty before that. */
  extractions: Extraction[];
}

export interface UploadResult {
  started: Report[];
  rejected: { filename: string; reason: string }[];
}

/** Outcome of checking a quote verbatim against the stored page text. */
export type Verification = "verified" | "fuzzy" | "failed";

export interface Datapoint {
  id: string;
  report_id: string;
  key: string;
  label: string;
  value: string | null;
  numeric_value: number | null;
  unit: string | null;
  period: string | null;
  qualifiers: Record<string, string>;
  quote: string | null;
  chunk_id: string | null;
  page: number | null;
  verification: Verification;
  model: string | null;
  company: string;
  fiscal_year: number;
}

/** One entry of the backend extraction registry (GET /api/datapoints/registry). */
export interface RegistryEntry {
  key: string;
  label: string;
  /** true when a report can have many values (goals), false for a single figure */
  multi: boolean;
}

export interface Citation {
  chunk_id: string;
  quote: string;
  verification: Verification;
  report_id?: string;
  company?: string;
  fiscal_year?: number;
  page?: number;
}

export type AnswerStatus = "answered" | "partial" | "not_found";

/** Why an answer was produced without the model (mirrors `Reason` in backend/app/chat/pipeline.py). */
export type AnswerReason =
  "greeting" | "no_reports" | "reports_processing" | "reports_failed" | "reports_need_reembedding" | "no_evidence";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: AnswerStatus | null;
  citations: Citation[];
  meta: {
    /** The question rewritten to stand alone, as used for retrieval. */
    standalone_question?: string;
    /** Citations removed because their quote was not found in the report text. */
    dropped_citations?: number;
    /** Set when the answer was produced without the model: small talk, nothing to search, no evidence. */
    reason?: AnswerReason;
    /** True when retrieval was skipped altogether (small talk, nothing to search). */
    skipped_retrieval?: boolean;
    [key: string]: unknown;
  };
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface Status {
  llm_provider: string;
  llm_model: string;
  embedding_provider: string;
  /** The configured embedding signature ("openai/text-embedding-3-large"). */
  embedding_model: string;
  reranker_model: string | null;
  /** Indexed reports embedded with another model; they are not searched until re-embedded. */
  reports_to_reembed: number;
}

export interface ChatRequest {
  question: string;
  conversation_id?: string | null;
  /** Restrict retrieval to these reports; null or empty means all indexed reports. */
  report_ids?: string[] | null;
}

export type ChatStage = "loading_models" | "analyzing" | "retrieving" | "generating";

/** Where the question in flight is, with what the backend reports about that stage. */
export interface ChatProgress {
  name: ChatStage;
  /** `loading_models`: the local models being loaded before the first search after a start. */
  models?: string[];
  /** `retrieving`: the question rewritten to stand alone, as it is searched. */
  query?: string;
  /** `generating`: how many retrieved passages the answer is written from. */
  chunks?: number;
}

/** Server-sent events of POST /api/chat, in the order they arrive. */
export type ChatStreamEvent =
  | { event: "conversation"; data: { id: string; title: string } }
  | { event: "stage"; data: ChatProgress }
  | { event: "answer"; data: Message }
  | { event: "error"; data: { message: string } };
