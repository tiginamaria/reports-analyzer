export interface SseMessage {
  event: string;
  data: string;
}

/**
 * Split a byte stream into server-sent events. Only `event:` and `data:` fields are used;
 * comments and ids are ignored. Multi-line data is joined with "\n" as the spec requires.
 */
export async function* parseSse(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseMessage> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
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

function parseBlock(block: string): SseMessage | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice("event:".length).trim();
    else if (line.startsWith("data:")) data.push(line.slice("data:".length).trim());
  }
  return data.length ? { event, data: data.join("\n") } : null;
}
