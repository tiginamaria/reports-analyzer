import type {
  ChatRequest,
  ChatStreamEvent,
  Conversation,
  ConversationDetail,
  Datapoint,
  RegistryEntry,
  Report,
  Status,
  UploadResult,
} from "../types";
import { request, requestStream } from "./client";
import { parseSse } from "./sse";

export const api = {
  status: () => request<Status>("/status"),

  reports: () => request<Report[]>("/reports"),
  /** Upload one or many PDFs in a single request; each starts its own ingestion. */
  upload: (files: File[]) => {
    const body = new FormData();
    for (const file of files) body.append("files", file);
    return request<UploadResult>("/reports", { method: "POST", body });
  },
  deleteReport: (id: string) => request<void>(`/reports/${id}`, { method: "DELETE" }),
  pdfUrl: (reportId: string, page?: number) => `/api/reports/${reportId}/pdf${page ? `#page=${page}` : ""}`,
  /** A single page rendered as an image; cheap to show, unlike the whole PDF. */
  pageImageUrl: (reportId: string, page: number) => `/api/reports/${reportId}/pages/${page}`,

  datapoints: () => request<Datapoint[]>("/datapoints"),
  /** The datapoints every report is scanned for; the Data page builds one section per entry. */
  registry: () => request<RegistryEntry[]>("/datapoints/registry"),

  conversations: () => request<Conversation[]>("/conversations"),
  conversation: (id: string) => request<ConversationDetail>(`/conversations/${id}`),
  deleteConversation: (id: string) => request<void>(`/conversations/${id}`, { method: "DELETE" }),
};

/**
 * POST /api/chat and yield its server-sent events as they arrive: one `conversation`,
 * `stage` events while the pipeline runs, then exactly one `answer` or `error`.
 */
export async function* chatStream(body: ChatRequest, signal?: AbortSignal): AsyncGenerator<ChatStreamEvent> {
  const stream = await requestStream("/chat", body, signal);
  for await (const { event, data } of parseSse(stream)) {
    // The event shapes are defined by the backend (app/api/chat.py, app/chat/pipeline.py).
    yield { event, data: JSON.parse(data) as unknown } as ChatStreamEvent;
  }
}
