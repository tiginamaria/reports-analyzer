import { useCallback, useEffect, useState } from "react";
import { api } from "../../api";
import { Notice } from "../../components/Notice";
import { errorMessage } from "../../lib/format";
import { isIngesting, reportLabel } from "../../lib/reports";
import type { Report } from "../../types";
import ReportsTable from "./ReportsTable";
import UploadForm from "./UploadForm";

const POLL_INTERVAL_MS = 2500;

export default function ReportsPage() {
  const [reports, setReports] = useState<Report[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(
    () =>
      api.reports().then(
        (list) => {
          setReports(list);
          setError(null);
        },
        (err: unknown) => setError(errorMessage(err)),
      ),
    [],
  );

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Poll only while something is still ingesting; a settled list does not change on its own.
  const ingesting = reports.some(isIngesting);
  useEffect(() => {
    if (!ingesting) return;
    const timer = setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [ingesting, refresh]);

  const remove = async (report: Report) => {
    if (!confirm(`Delete ${reportLabel(report)} (${report.filename}) and all its data?`)) return;
    try {
      await api.deleteReport(report.id);
    } catch (err) {
      setError(errorMessage(err));
    }
    await refresh();
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <section className="card p-5">
        <h2 className="mb-3 text-base font-semibold">Add annual reports</h2>
        <UploadForm onUploaded={refresh} />
        <p className="mt-3 text-xs text-slate-500">
          Ingestion parses the PDF, detects the company and fiscal year from its first pages, chunks and embeds it
          locally, then extracts FTE and sustainability goals with the configured LLM. Large reports take several
          minutes on CPU.
        </p>
      </section>

      {error && <Notice tone="error">{error}</Notice>}

      <section className="card overflow-hidden">
        <ReportsTable reports={reports} onDelete={remove} />
      </section>
    </div>
  );
}
