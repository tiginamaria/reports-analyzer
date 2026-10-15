/** Thrown for any non-2xx response; `message` is the backend's `detail` when it sends one. */
class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function errorFromResponse(res: Response): Promise<ApiError> {
  let detail = res.statusText || `HTTP ${res.status}`;
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // body was not JSON; keep the status text
  }
  return new ApiError(detail, res.status);
}

/** Fetch a JSON endpoint under `/api`, turning HTTP errors into `ApiError`. */
export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, init);
  if (!res.ok) throw await errorFromResponse(res);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** POST JSON and return the raw streaming response body (used for server-sent events). */
export async function requestStream(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<ReadableStream<Uint8Array>> {
  const res = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw await errorFromResponse(res);
  if (!res.body) throw new ApiError("Empty response body", res.status);
  return res.body;
}
