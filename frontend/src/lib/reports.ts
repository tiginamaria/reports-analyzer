import type { Report } from "../types";

/** Statuses in which a report's index is complete and it can be searched (backend INDEXED_STATUSES). */
const INDEXED_STATUSES: ReadonlySet<Report["status"]> = new Set(["ready", "extracting"]);

export const isIndexed = (report: Report) => INDEXED_STATUSES.has(report.status);

export const isIngesting = (report: Report) => report.status !== "ready" && report.status !== "failed";

/** What `done`/`total` count in each stage; null for stages without units. */
export function progressUnit(status: Report["status"]): string | null {
  switch (status) {
    case "parsing":
      return "pages";
    case "embedding":
      return "chunks";
    case "extracting":
      return "datapoints";
    default:
      return null;
  }
}

/** An indexed report whose vectors come from another embedding model than the one configured now. */
export const needsReembedding = (report: Report, configured: string | undefined) =>
  isIndexed(report) && configured !== undefined && report.embedding_model !== configured;

/** The datapoints whose last extraction attempt failed (as opposed to finding nothing). */
export const failedExtractions = (report: Report) => report.extractions.filter((e) => e.status === "error");

/** Anything that names a report: a Report, a Datapoint or a Citation. */
interface ReportRef {
  company?: string;
  fiscal_year?: number;
}

/** "ABN AMRO 2025": how a report is named throughout the UI. */
export const reportLabel = (ref: ReportRef) => [ref.company ?? "?", ref.fiscal_year ?? ""].join(" ").trim();

/** A page of a source PDF to show in the PdfPanel. */
export interface PdfTarget {
  reportId: string;
  page: number;
  label: string;
}

/** The page a datapoint or citation points to; null when the backend did not record one. */
export function pdfTargetOf(ref: ReportRef & { report_id?: string; page?: number | null }): PdfTarget | null {
  if (!ref.report_id || !ref.page) return null;
  return { reportId: ref.report_id, page: ref.page, label: reportLabel(ref) };
}
