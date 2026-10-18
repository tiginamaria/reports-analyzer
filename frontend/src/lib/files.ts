export const isPdf = (file: File) => file.name.toLowerCase().endsWith(".pdf");

export const pdfsOf = (files: Iterable<File> | ArrayLike<File> | null) => Array.from(files ?? []).filter(isPdf);

/** Collect PDF files from a drop event, walking dropped folders recursively. */
export async function pdfsFromDrop(transfer: DataTransfer): Promise<File[]> {
  const entries = Array.from(transfer.items)
    .map((item) => item.webkitGetAsEntry())
    .filter((entry): entry is FileSystemEntry => entry !== null);
  // Browsers without the entries API only expose flat files.
  if (entries.length === 0) return pdfsOf(transfer.files);

  const found: File[] = [];
  for (const entry of entries) await collectPdfs(entry, found);
  return found;
}

async function collectPdfs(entry: FileSystemEntry, into: File[]): Promise<void> {
  if (entry.isFile) {
    const file = await fileOf(entry as FileSystemFileEntry);
    if (isPdf(file)) into.push(file);
  } else if (entry.isDirectory) {
    for (const child of await readDirectory(entry as FileSystemDirectoryEntry)) {
      await collectPdfs(child, into);
    }
  }
}

const fileOf = (entry: FileSystemFileEntry) => new Promise<File>((resolve, reject) => entry.file(resolve, reject));

/** `readEntries` returns results in batches; keep reading until it returns an empty one. */
async function readDirectory(entry: FileSystemDirectoryEntry): Promise<FileSystemEntry[]> {
  const reader = entry.createReader();
  const all: FileSystemEntry[] = [];
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
    if (batch.length === 0) return all;
    all.push(...batch);
  }
}
