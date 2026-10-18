import { useCallback, useEffect, useState } from "react";
import { api } from "../../api";
import { Notice } from "../../components/Notice";
import PdfPanel from "../../components/PdfPanel";
import { plural } from "../../lib/format";
import { isIndexed, pdfTargetOf, type PdfTarget } from "../../lib/reports";
import type { ChatProgress, Citation, Conversation, Message, Report } from "../../types";
import ConversationSidebar from "./ConversationSidebar";
import MessageView from "./MessageView";
import { useChat } from "./useChat";

/** The status line of the question in flight, with what the backend reports about the stage. */
function stageLabel({ name, query, chunks }: ChatProgress): string {
  switch (name) {
    case "analyzing":
      return "Understanding the question…";
    case "retrieving":
      return query ? `Searching the reports for “${query}”…` : "Searching the reports…";
    case "generating":
      return chunks ? `Writing the answer from ${plural(chunks, "passage")}…` : "Writing the answer…";
    default:
      return name;
  }
}

const EXAMPLE_QUESTIONS = [
  "How much did Shell spend on climate change adaptation in 2025?",
  "How many FTEs did ABN AMRO have at the end of 2025?",
  "What are Heineken's emission reduction targets?",
  "Compare the net-zero target years of ASML and Shell.",
];

export default function ChatPage() {
  const chat = useChat();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [reports, setReports] = useState<Report[]>([]);
  const [scope, setScope] = useState<string[]>([]);
  const [pdf, setPdf] = useState<PdfTarget | null>(null);

  const loadConversations = useCallback(
    () =>
      api
        .conversations()
        .then(setConversations)
        .catch(() => {}),
    [],
  );

  useEffect(() => {
    void loadConversations();
    api
      .reports()
      .then((all) => setReports(all.filter(isIndexed)))
      .catch(() => {});
  }, [loadConversations]);

  const send = async (question: string) => {
    await chat.send(question, scope);
    await loadConversations();
  };

  const deleteConversation = async (id: string) => {
    await api.deleteConversation(id);
    if (id === chat.conversationId) chat.reset();
    await loadConversations();
  };

  const toggleScope = (id: string) =>
    setScope((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

  return (
    <div className="flex h-full">
      <ConversationSidebar
        conversations={conversations}
        currentId={chat.conversationId}
        onOpen={chat.open}
        onNew={chat.reset}
        onDelete={deleteConversation}
        reports={reports}
        scope={scope}
        onToggleScope={toggleScope}
      />

      <section className="flex min-w-0 flex-1 flex-col">
        <MessageList
          messages={chat.messages}
          stage={chat.stage}
          error={chat.error}
          onCite={(citation) => setPdf(pdfTargetOf(citation))}
        />
        <Composer disabled={chat.busy} onSend={send} />
      </section>

      <PdfPanel target={pdf} onClose={() => setPdf(null)} />
    </div>
  );
}

interface MessageListProps {
  messages: Message[];
  stage: ChatProgress | null;
  error: string | null;
  onCite: (citation: Citation) => void;
}

function MessageList({ messages, stage, error, onCite }: MessageListProps) {
  const [bottom, setBottom] = useState<HTMLDivElement | null>(null);

  useEffect(() => {
    bottom?.scrollIntoView({ behavior: "smooth" });
  }, [bottom, messages, stage]);

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-6 py-4">
      <div className="mx-auto max-w-3xl space-y-4">
        {messages.length === 0 && !stage && <WelcomeCard />}
        {messages.map((message) => (
          <MessageView key={message.id} message={message} onCite={onCite} />
        ))}
        {stage && (
          <div className="flex items-center gap-2 text-sm text-slate-500" role="status">
            <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-600" />
            {stageLabel(stage)}
          </div>
        )}
        {error && <Notice tone="error">{error}</Notice>}
        <div ref={setBottom} />
      </div>
    </div>
  );
}

function WelcomeCard() {
  return (
    <div className="card p-6 text-sm text-slate-600">
      <p className="mb-2 font-medium text-slate-800">Ask about the ingested annual reports.</p>
      <ul className="list-disc space-y-1 pl-5">
        {EXAMPLE_QUESTIONS.map((question) => (
          <li key={question}>{question}</li>
        ))}
      </ul>
      <p className="mt-3 text-xs text-slate-500">
        Every answer cites verbatim passages. Quotes are checked against the PDF text; a failed check is shown and the
        answer is downgraded.
      </p>
    </div>
  );
}

function Composer({ disabled, onSend }: { disabled: boolean; onSend: (question: string) => void }) {
  const [question, setQuestion] = useState("");
  const trimmed = question.trim();

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!trimmed || disabled) return;
    setQuestion("");
    onSend(trimmed);
  };

  return (
    <form onSubmit={submit} className="border-t border-slate-200 bg-white p-4">
      <div className="mx-auto flex max-w-3xl gap-2">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about the reports…"
          aria-label="Question"
          className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-emerald-600 focus:outline-none"
          disabled={disabled}
        />
        <button type="submit" className="btn btn-primary" disabled={disabled || !trimmed}>
          Ask
        </button>
      </div>
    </form>
  );
}
