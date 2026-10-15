import type { ReactNode } from "react";

const TONE = {
  error: "bg-red-50 text-red-700",
  info: "bg-slate-100 text-slate-600",
};

/** A short status or error message. Multi-line text keeps its line breaks. */
export function Notice({ tone = "info", children }: { tone?: keyof typeof TONE; children: ReactNode }) {
  return (
    <p
      role={tone === "error" ? "alert" : "status"}
      className={`rounded-md p-3 text-sm whitespace-pre-line ${TONE[tone]}`}
    >
      {children}
    </p>
  );
}
