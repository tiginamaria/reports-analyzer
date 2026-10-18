import { api } from "../../api";
import { ReportStatusBadge } from "../../components/Badges";
import { EmptyRow, Table, Td, Tr } from "../../components/Table";
import { plural } from "../../lib/format";
import { failedExtractions, isIngesting } from "../../lib/reports";
import type { ExtractionStatus, Report } from "../../types";

const COLUMNS = ["Company", "Year", "File", "Pages", "Status", ""];

interface Props {
  reports: Report[];
  onDelete: (report: Report) => void;
  /** Extract the report's failed datapoints again. */
  onRetry: (report: Report) => void;
}

export default function ReportsTable({ reports, onDelete, onRetry }: Props) {
  return (
    <Table columns={COLUMNS}>
      {reports.length === 0 && (
        <EmptyRow colSpan={COLUMNS.length}>No reports yet. Upload a PDF to get started.</EmptyRow>
      )}
      {reports.map((report) => (
        <Tr key={report.id}>
          <Td className="font-medium">{report.company || <span className="text-slate-400">detecting…</span>}</Td>
          <Td>{report.fiscal_year || "–"}</Td>
          <Td>
            <a
              className="text-emerald-700 hover:underline"
              href={api.pdfUrl(report.id)}
              target="_blank"
              rel="noreferrer"
            >
              {report.filename}
            </a>
          </Td>
          <Td>{report.page_count || "–"}</Td>
          <Td>
            <div className="flex items-center gap-2">
              <ReportStatusBadge status={report.status} />
              {isIngesting(report) && <ProgressBar percent={report.progress} />}
            </div>
            {report.error && (
              <p className="mt-1 max-w-md truncate text-xs text-red-700" title={report.error}>
                {report.error}
              </p>
            )}
            <ExtractionSummary report={report} onRetry={() => onRetry(report)} />
          </Td>
          <Td className="text-right whitespace-nowrap">
            <button type="button" className="btn !py-0.5 text-red-700" onClick={() => onDelete(report)}>
              delete
            </button>
          </Td>
        </Tr>
      ))}
    </Table>
  );
}

/**
 * How the datapoint extraction went, apart from the index status: a report is `ready` as soon
 * as it can be searched, even when some of its datapoints failed to extract.
 */
function ExtractionSummary({ report, onRetry }: { report: Report; onRetry: () => void }) {
  if (report.extractions.length === 0) return null;
  const count = (status: ExtractionStatus) => report.extractions.filter((e) => e.status === status).length;
  const failed = failedExtractions(report).length;
  const parts = [
    count("pending") > 0 && `${count("pending")} extracting`,
    count("extracted") > 0 && `${count("extracted")} extracted`,
    count("not_found") > 0 && `${count("not_found")} not found`,
  ].filter(Boolean);
  return (
    <p className="mt-1 text-xs text-slate-500">
      Datapoints: {parts.join(" · ")}
      {failed > 0 && (
        <span className="text-red-700">
          {parts.length > 0 && " · "}
          {plural(failed, "extraction")} failed
          {report.status === "ready" && (
            <button type="button" className="ml-2 underline" onClick={onRetry}>
              retry
            </button>
          )}
        </span>
      )}
    </p>
  );
}

function ProgressBar({ percent }: { percent: number }) {
  return (
    <div
      className="h-1.5 w-28 overflow-hidden rounded bg-slate-200"
      role="progressbar"
      aria-valuenow={percent}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className="h-full bg-blue-500 transition-all" style={{ width: `${percent}%` }} />
    </div>
  );
}
