import { api } from "../../api";
import { ReportStatusBadge } from "../../components/Badges";
import { EmptyRow, Table, Td, Tr } from "../../components/Table";
import { isIngesting } from "../../lib/reports";
import type { Report } from "../../types";

const COLUMNS = ["Company", "Year", "File", "Pages", "Status", ""];

export default function ReportsTable({ reports, onDelete }: { reports: Report[]; onDelete: (report: Report) => void }) {
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
