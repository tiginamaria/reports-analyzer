import { useCallback, useEffect, useRef, useState } from "react";
import { api, chatStream } from "../../api";
import { errorMessage } from "../../lib/format";
import type { ChatProgress, Message } from "../../types";

const LOST_CONNECTION =
  "The connection closed before an answer arrived. Reopen the conversation to check whether it was stored.";

/** The optimistic user turn shown while the backend is still working. */
function localUserMessage(content: string): Message {
  return {
    id: `local-${crypto.randomUUID()}`,
    role: "user",
    content,
    status: null,
    citations: [],
    meta: {},
    created_at: "",
  };
}

/**
 * One conversation and the stream of the question in flight. `send` resolves when the
 * backend has finished (answer or error), so callers can refresh the conversation list.
 */
export function useChat() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [stage, setStage] = useState<ChatProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef<AbortController | null>(null);

  // Leaving the page cancels the request; the backend still stores the answer.
  useEffect(() => () => inFlight.current?.abort(), []);

  // Opening another conversation or starting a new one abandons the question in flight;
  // the backend still finishes and stores its answer.
  const open = useCallback(async (id: string) => {
    inFlight.current?.abort();
    try {
      const detail = await api.conversation(id);
      setConversationId(id);
      setMessages(detail.messages);
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  }, []);

  const reset = useCallback(() => {
    inFlight.current?.abort();
    setConversationId(null);
    setMessages([]);
    setError(null);
  }, []);

  const send = useCallback(
    async (question: string, reportIds: string[]) => {
      if (inFlight.current) return;
      const controller = new AbortController();
      inFlight.current = controller;
      setError(null);
      setMessages((prev) => [...prev, localUserMessage(question)]);
      setStage({ name: "analyzing" });
      let settled = false; // an `answer` or `error` event arrived
      try {
        const stream = chatStream(
          { question, conversation_id: conversationId, report_ids: reportIds.length ? reportIds : null },
          controller.signal,
        );
        for await (const event of stream) {
          switch (event.event) {
            case "conversation":
              setConversationId(event.data.id);
              break;
            case "stage":
              setStage(event.data);
              break;
            case "answer":
              settled = true;
              setMessages((prev) => [...prev, event.data]);
              break;
            case "error":
              settled = true;
              setError(event.data.message);
              break;
          }
        }
        // The backend always ends with an answer or an error; a stream that stops without
        // one was cut off (server restart, dropped proxy connection).
        if (!settled) setError(LOST_CONNECTION);
      } catch (err) {
        if (!controller.signal.aborted) setError(errorMessage(err));
      } finally {
        inFlight.current = null;
        setStage(null);
      }
    },
    [conversationId],
  );

  return { conversationId, messages, stage, error, busy: stage !== null, open, reset, send };
}
