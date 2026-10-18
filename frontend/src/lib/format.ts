/** "1 file", "3 files". */
export const plural = (count: number, noun: string, pluralNoun = `${noun}s`) =>
  `${count} ${count === 1 ? noun : pluralNoun}`;

/** A user-facing message for anything a promise can reject with. */
export const errorMessage = (err: unknown) => (err instanceof Error ? err.message : String(err));

/** "12 s", "3 min 05 s", "1 h 02 min": a duration for progress lines. */
export function duration(ms: number): string {
  const seconds = Math.max(0, Math.round(ms / 1000));
  if (seconds < 60) return `${seconds} s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min ${String(seconds % 60).padStart(2, "0")} s`;
  return `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, "0")} min`;
}

/** Milliseconds between an ISO timestamp and now (0 when the timestamp is missing). */
export const sinceMs = (iso: string | null | undefined, now = Date.now()) =>
  iso ? Math.max(0, now - new Date(iso).getTime()) : 0;
