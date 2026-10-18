/** "1 file", "3 files". */
export const plural = (count: number, noun: string, pluralNoun = `${noun}s`) =>
  `${count} ${count === 1 ? noun : pluralNoun}`;

/** A user-facing message for anything a promise can reject with. */
export const errorMessage = (err: unknown) => (err instanceof Error ? err.message : String(err));
