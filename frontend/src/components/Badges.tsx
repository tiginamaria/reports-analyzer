import type { AnswerStatus, ExtractionStatus, ReportStatus, Verification } from "../types";

const VERIFICATION: Record<Verification, { className: string; label: string }> = {
  verified: { className: "bg-emerald-100 text-emerald-800", label: "verified" },
  fuzzy: { className: "bg-amber-100 text-amber-800", label: "fuzzy match" },
  failed: { className: "bg-red-100 text-red-800", label: "quote not found" },
};

export function VerifyBadge({ status }: { status: Verification }) {
  const { className, label } = VERIFICATION[status];
  return (
    <span className={`badge ${className}`} title={`Verbatim check: ${label}`}>
      {label}
    </span>
  );
}

function reportStatusClass(status: ReportStatus) {
  if (status === "ready") return "bg-emerald-100 text-emerald-800";
  if (status === "failed") return "bg-red-100 text-red-800";
  return "bg-blue-100 text-blue-800";
}

export function ReportStatusBadge({ status }: { status: ReportStatus }) {
  return <span className={`badge ${reportStatusClass(status)}`}>{status}</span>;
}

/** `unknown`: the report was ingested before outcomes were recorded, so it has no row. */
export type ExtractionState = ExtractionStatus | "unknown";

const EXTRACTION: Record<ExtractionState, { className: string; label: string; title: string }> = {
  pending: { className: "bg-blue-100 text-blue-800", label: "extracting…", title: "The extraction is running" },
  extracted: {
    className: "bg-emerald-100 text-emerald-800",
    label: "extracted",
    title: "At least one value was stored",
  },
  not_found: {
    className: "bg-slate-200 text-slate-700",
    label: "not found",
    title: "The model found nothing in the passages retrieved for this datapoint",
  },
  error: {
    className: "bg-red-100 text-red-800",
    label: "extraction failed",
    title: "The extraction did not run to the end",
  },
  unknown: {
    className: "bg-slate-100 text-slate-500",
    label: "outcome not recorded",
    title: "Ingested before outcomes were recorded; extract it to find out",
  },
};

export function ExtractionBadge({ status }: { status: ExtractionState }) {
  const { className, label, title } = EXTRACTION[status];
  return (
    <span className={`badge ${className}`} title={title}>
      {label}
    </span>
  );
}

const ANSWER: Record<AnswerStatus, { className: string; label: string }> = {
  answered: { className: "bg-emerald-100 text-emerald-800", label: "answered" },
  partial: { className: "bg-amber-100 text-amber-800", label: "partial" },
  not_found: { className: "bg-slate-200 text-slate-700", label: "not stated in the report" },
};

export function AnswerStatusBadge({ status }: { status: AnswerStatus | null }) {
  if (!status) return null;
  const { className, label } = ANSWER[status] ?? { className: "bg-slate-100 text-slate-700", label: status };
  return <span className={`badge ${className}`}>{label}</span>;
}
