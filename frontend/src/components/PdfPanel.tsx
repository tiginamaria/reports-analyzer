import { useState } from "react";
import { api } from "../api";
import type { PdfTarget } from "../lib/reports";

/**
 * Shows the cited page of the source report as an image rendered by the backend, so opening
 * a citation costs one page, not the whole PDF. The full PDF stays one click away.
 */
export default function PdfPanel({ target, onClose }: { target: PdfTarget | null; onClose: () => void }) {
  if (!target) return null;
  const src = api.pageImageUrl(target.reportId, target.page);
  const title = `${target.label}, page ${target.page}`;
  return (
    <aside className="flex w-[46%] min-w-[380px] flex-col border-l border-slate-200 bg-white" aria-label={title}>
      <div className="flex items-center justify-between border-b border-slate-200 px-4 py-2 text-sm">
        <span className="truncate font-medium">
          {target.label} · page {target.page}
        </span>
        <div className="flex items-center gap-2">
          <a
            className="text-emerald-700 hover:underline"
            href={api.pdfUrl(target.reportId, target.page)}
            target="_blank"
            rel="noreferrer"
          >
            open full PDF
          </a>
          <button type="button" className="btn !py-0.5" onClick={onClose}>
            close
          </button>
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-auto bg-slate-100 p-3">
        {/* Keyed by src so the loading state resets whenever the page changes. */}
        <PageImage key={src} src={src} alt={title} />
      </div>
    </aside>
  );
}

function PageImage({ src, alt }: { src: string; alt: string }) {
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  return (
    <>
      {state === "loading" && <p className="py-6 text-center text-sm text-slate-500">Rendering page…</p>}
      {state === "error" && <p className="py-6 text-center text-sm text-red-700">Could not render this page.</p>}
      <img
        src={src}
        alt={alt}
        onLoad={() => setState("ready")}
        onError={() => setState("error")}
        className={`mx-auto w-full max-w-[900px] bg-white shadow ${state === "ready" ? "" : "hidden"}`}
      />
    </>
  );
}
