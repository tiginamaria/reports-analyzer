import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import { api } from "../api";
import type { Status } from "../types";

const NAV_LINKS = [
  { to: "/reports", label: "Reports" },
  { to: "/data", label: "Extracted data" },
  { to: "/chat", label: "Chat" },
];

export default function AppHeader() {
  return (
    <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-3">
      <div className="flex items-center gap-8">
        <h1 className="text-lg font-semibold tracking-tight">Annual Report Analyzer</h1>
        <nav className="flex gap-1" aria-label="Main">
          {NAV_LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) =>
                `rounded-md px-3 py-1.5 text-sm font-medium ${
                  isActive ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      </div>
      <BackendStatus />
    </header>
  );
}

type StatusState = { kind: "loading" } | { kind: "ok"; status: Status } | { kind: "unreachable" };

/** Which models the backend is configured with, so a demo always shows what produced the answers. */
function BackendStatus() {
  const [state, setState] = useState<StatusState>({ kind: "loading" });

  useEffect(() => {
    api
      .status()
      .then((status) => setState({ kind: "ok", status }))
      .catch(() => setState({ kind: "unreachable" }));
  }, []);

  if (state.kind === "loading") return null;
  if (state.kind === "unreachable") return <span className="text-xs text-red-600">backend unreachable</span>;

  const { llm_provider, llm_model, embedding_model, reranker_model, reports_to_reembed } = state.status;
  return (
    <div className="text-xs text-slate-500">
      LLM{" "}
      <span className="font-mono">
        {llm_provider}/{llm_model}
      </span>{" "}
      · embeddings <span className="font-mono">{embedding_model}</span>
      {reranker_model && (
        <>
          {" "}
          · reranker <span className="font-mono">{reranker_model}</span>
        </>
      )}
      {reports_to_reembed > 0 && (
        <span
          className="ml-2 text-amber-700"
          title="Indexed with another embedding model, so not searched; run python -m app.reembed"
        >
          · {reports_to_reembed} report{reports_to_reembed === 1 ? "" : "s"} need re-embedding
        </span>
      )}
    </div>
  );
}
