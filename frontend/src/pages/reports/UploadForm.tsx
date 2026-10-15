import { useRef, useState } from "react";
import { api } from "../../api";
import { Notice } from "../../components/Notice";
import { pdfsFromDrop, pdfsOf } from "../../lib/files";
import { errorMessage, plural } from "../../lib/format";

/** Pick PDFs (files, a folder, or a drop) and upload them in one request. */
export default function UploadForm({ onUploaded }: { onUploaded: () => void }) {
  const [selected, setSelected] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  const clear = () => {
    setSelected([]);
    // Reset the inputs so choosing the same file again fires `onChange`.
    for (const input of [fileInput, folderInput]) if (input.current) input.current.value = "";
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (selected.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      const { rejected } = await api.upload(selected);
      if (rejected.length > 0) setError(rejected.map((r) => `${r.filename}: ${r.reason}`).join("\n"));
      clear();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
      onUploaded();
    }
  };

  return (
    <form onSubmit={submit} className="space-y-3">
      <DropZone
        selected={selected}
        onDrop={setSelected}
        onPickFiles={() => fileInput.current?.click()}
        onPickFolder={() => folderInput.current?.click()}
      />
      {/* Kept outside the drop zone so a programmatic click does not bubble back into it. */}
      <input
        ref={fileInput}
        type="file"
        accept="application/pdf"
        multiple
        hidden
        onChange={(e) => setSelected(pdfsOf(e.target.files))}
      />
      <input
        ref={folderInput}
        type="file"
        webkitdirectory=""
        hidden
        onChange={(e) => setSelected(pdfsOf(e.target.files))}
      />

      <div className="flex flex-wrap items-center gap-3">
        <button type="submit" className="btn btn-primary" disabled={busy || selected.length === 0}>
          {busy ? "Uploading…" : selected.length > 1 ? `Upload ${plural(selected.length, "file")}` : "Upload"}
        </button>
        {selected.length > 0 && !busy && (
          <button type="button" className="btn" onClick={clear}>
            Clear
          </button>
        )}
        {busy && <span className="text-sm text-slate-600">Uploading {plural(selected.length, "file")}…</span>}
      </div>
      {error && <Notice tone="error">{error}</Notice>}
    </form>
  );
}

interface DropZoneProps {
  selected: File[];
  onDrop: (files: File[]) => void;
  onPickFiles: () => void;
  onPickFolder: () => void;
}

function DropZone({ selected, onDrop, onPickFiles, onPickFolder }: DropZoneProps) {
  const [dragging, setDragging] = useState(false);

  const drop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    onDrop(await pdfsFromDrop(e.dataTransfer));
  };

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Choose PDF files to upload"
      onClick={onPickFiles}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onPickFiles();
        }
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={drop}
      className={`cursor-pointer rounded-lg border-2 border-dashed px-4 py-6 text-center text-sm ${
        dragging ? "border-emerald-600 bg-emerald-50" : "border-slate-300 bg-slate-50 hover:bg-slate-100"
      }`}
    >
      {selected.length === 0 ? (
        <>
          <div className="font-medium text-slate-700">Drop a PDF or a folder here, or click to choose files</div>
          <div className="mt-1 text-xs text-slate-500">
            A folder uploads every PDF inside it.{" "}
            <button
              type="button"
              className="text-emerald-700 underline"
              onClick={(e) => {
                e.stopPropagation();
                onPickFolder();
              }}
            >
              Choose a folder
            </button>
          </div>
        </>
      ) : (
        <>
          <div className="font-medium text-slate-700">{plural(selected.length, "PDF")} selected</div>
          <ul className="mt-1 max-h-24 overflow-y-auto text-xs text-slate-500">
            {selected.map((file) => (
              <li key={`${file.name}:${file.size}`}>{file.name}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
