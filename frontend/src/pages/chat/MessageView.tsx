import { AnswerStatusBadge, VerifyBadge } from "../../components/Badges";
import { plural } from "../../lib/format";
import { reportLabel } from "../../lib/reports";
import type { AnswerReason, Citation, Message } from "../../types";

/** How an answer produced without the model is labelled; `greeting` and `no_evidence` need no label. */
const REASON_LABELS: Partial<Record<AnswerReason, { label: string; title: string }>> = {
  no_reports: { label: "no reports yet", title: "Nothing has been uploaded; the question was not searched" },
  reports_processing: {
    label: "reports still processing",
    title: "No report is ready to search yet; the question was not sent to the model",
  },
  reports_failed: { label: "ingestion failed", title: "Every report failed to ingest; nothing could be searched" },
  reports_need_reembedding: {
    label: "re-embedding needed",
    title: "The reports were indexed with another embedding model; run python -m app.reembed",
  },
};

interface Props {
  message: Message;
  onCite: (citation: Citation) => void;
}

export default function MessageView({ message, onCite }: Props) {
  return message.role === "user" ? (
    <UserMessage message={message} />
  ) : (
    <AssistantMessage message={message} onCite={onCite} />
  );
}

function UserMessage({ message }: { message: Message }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-slate-900 px-4 py-2 text-sm text-white">
        {message.content}
      </div>
    </div>
  );
}

function AssistantMessage({ message, onCite }: Props) {
  const dropped = message.meta.dropped_citations ?? 0;
  const reason = message.meta.reason;
  const reasonLabel = reason ? REASON_LABELS[reason] : undefined;
  const searched = message.meta.skipped_retrieval ? undefined : message.meta.standalone_question;
  return (
    <div className="card p-4 text-sm">
      <div className="mb-2 flex items-center gap-2">
        {reason !== "greeting" && <AnswerStatusBadge status={message.status} />}
        {reasonLabel && (
          <span className="badge bg-amber-100 text-amber-800" title={reasonLabel.title}>
            {reasonLabel.label}
          </span>
        )}
        {dropped > 0 && (
          <span className="badge bg-red-100 text-red-800" title="Quotes that could not be found in the report text">
            {plural(dropped, "citation")} failed verification
          </span>
        )}
        {searched && (
          <span className="truncate text-xs text-slate-400" title={searched}>
            searched: {searched}
          </span>
        )}
      </div>
      <div className="whitespace-pre-wrap text-slate-800">{message.content}</div>
      {message.citations.length > 0 && (
        <ul className="mt-3 space-y-2 border-t border-slate-100 pt-3">
          {message.citations.map((citation, i) => (
            <CitationRow key={`${citation.chunk_id}:${i}`} citation={citation} onOpen={() => onCite(citation)} />
          ))}
        </ul>
      )}
    </div>
  );
}

function CitationRow({ citation, onOpen }: { citation: Citation; onOpen: () => void }) {
  const failed = citation.verification === "failed";
  return (
    <li className="flex items-start gap-2 text-xs">
      <button
        type="button"
        className="shrink-0 rounded-md border border-emerald-200 bg-emerald-50 px-2 py-0.5 font-medium text-emerald-800 hover:bg-emerald-100 disabled:opacity-60"
        onClick={onOpen}
        disabled={!citation.page}
      >
        {reportLabel(citation)} · p. {citation.page ?? "?"}
      </button>
      <VerifyBadge status={citation.verification} />
      <span className={`text-slate-600 italic ${failed ? "line-through" : ""}`}>“{citation.quote}”</span>
    </li>
  );
}
