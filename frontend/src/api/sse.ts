export interface SseMessage {
  event: string;
  data: string;
}

/**
 * Silence longer than this means the connection is dead even though it was never closed (a
 * proxy can keep the browser's side open after the server's side went away). The backend
 * sends a ping comment every 15 s while it works, so a live stream is never this quiet.
 */
export const SSE_IDLE_MS = 60_000;

/**
 * Split a byte stream into server-sent events. Only `event:` and `data:` fields are used;
 * comments and ids are ignored. Multi-line data is joined with "\n" as the spec requires.
 * Throws when nothing arrives for `idleMs`.
 */
export async function* parseSse(stream: ReadableStream<Uint8Array>, idleMs = SSE_IDLE_MS): AsyncGenerator<SseMessage> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await readOrTimeout(reader, idleMs);
      if (done) break;
      // sse-starlette ends lines with CRLF; normalise on the whole buffer so a CRLF split
      // across two reads is still joined.
      buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, "\n");
      let end: number;
      while ((end = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        const message = parseBlock(block);
        if (message) yield message;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** `reader.read()`, rejecting when nothing arrives within `ms`; the stream is cancelled then. */
async function readOrTimeout(reader: ReadableStreamDefaultReader<Uint8Array>, ms: number) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`No response from the server for ${ms / 1000} seconds`)), ms);
  });
  try {
    return await Promise.race([reader.read(), timeout]);
  } catch (err) {
    await reader.cancel().catch(() => undefined);
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

function parseBlock(block: string): SseMessage | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice("event:".length).trim();
    else if (line.startsWith("data:")) data.push(line.slice("data:".length).trim());
  }
  return data.length ? { event, data: data.join("\n") } : null;
}
