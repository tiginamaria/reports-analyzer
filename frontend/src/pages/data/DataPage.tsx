import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../../api";
import { Notice } from "../../components/Notice";
import PdfPanel from "../../components/PdfPanel";
import { groupBy } from "../../lib/collections";
import { errorMessage, plural } from "../../lib/format";
import { isIndexed, isIngesting, pdfTargetOf, reportLabel, type PdfTarget } from "../../lib/reports";
import type { Datapoint, RegistryEntry, Report } from "../../types";
import { type EntryOutcome, MultiValueSection, OtherDatapoints, SingleValueSection } from "./DatapointSections";

const ALL_REPORTS = "";
const POLL_INTERVAL_MS = 2500;

/**
 * Pre-extracted datapoints, one section per entry of the backend's extraction registry.
 * Adding a datapoint to `registry.yaml` adds a section here without a frontend change.
 *
 * Every section lists every indexed report: with its datapoints, or with the outcome of the
 * extraction (not found, failed, still running) when there are none, so a failed extraction
 * is visible rather than an empty section.
 */
export default function DataPage() {
  const [registry, setRegistry] = useState<RegistryEntry[]>([]);
  const [reports, setReports] = useState<Report[]>([]);
  const [datapoints, setDatapoints] = useState<Datapoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [reportFilter, setReportFilter] = useState(ALL_REPORTS);
  const [pdf, setPdf] = useState<PdfTarget | null>(null);

  const load = useCallback(
    () =>
      Promise.all([api.registry(), api.reports(), api.datapoints()]).then(
        ([entries, all, points]) => {
          setRegistry(entries);
          setReports(all);
          setDatapoints(points);
          setError(null);
        },
        (err: unknown) => setError(errorMessage(err)),
      ),
    [],
  );

  useEffect(() => {
    void load();
  }, [load]);

  // Poll while a report is still ingesting or extracting, so outcomes and new datapoints
  // appear without leaving the page.
  const busy = reports.some(isIngesting);
  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(() => void load(), POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [busy, load]);

  /** Reports whose index is complete; extraction has at least started for them. */
  const indexed = useMemo(() => reports.filter(isIndexed), [reports]);
  const shown = reportFilter ? indexed.filter((r) => r.id === reportFilter) : indexed;
  const stillIndexing = reports.filter((r) => isIngesting(r) && !isIndexed(r)).length;

  const reportOptions = useMemo(
    () => indexed.map((r): [string, string] => [r.id, reportLabel(r)]).sort(([, a], [, b]) => a.localeCompare(b)),
    [indexed],
  );

  const byKey = useMemo(() => {
    const visible = reportFilter ? datapoints.filter((d) => d.report_id === reportFilter) : datapoints;
    return groupBy(visible, (d) => d.key);
  }, [datapoints, reportFilter]);

  const registryKeys = new Set(registry.map((entry) => entry.key));
  const other = [...byKey].filter(([key]) => !registryKeys.has(key)).flatMap(([, rows]) => rows);

  /** The shown reports without datapoints for `key`, each with the reason. */
  const outcomesOf = (key: string): EntryOutcome[] =>
    shown.flatMap((report) => {
      const outcome = report.extractions.find((e) => e.key === key);
      if (outcome?.status === "extracted") return [];
      return [
        {
          reportId: report.id,
          report: reportLabel(report),
          status: outcome?.status ?? "unknown",
          message: outcome?.message ?? null,
          canRetry: report.status === "ready",
        },
      ];
    });

  const open = (d: Datapoint) => setPdf(pdfTargetOf(d));

  const retry = async (reportId: string, key: string) => {
    try {
      await api.extractAgain(reportId, [key]);
    } catch (err) {
      setError(errorMessage(err));
    }
    await load();
  };

  return (
    <div className="flex h-full">
      <div className="min-w-0 flex-1 overflow-y-auto p-6">
        <div className="mx-auto max-w-5xl space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold">Pre-extracted datapoints</h2>
            <select
              value={reportFilter}
              onChange={(e) => setReportFilter(e.target.value)}
              aria-label="Filter by report"
              className="rounded-md border border-slate-300 bg-white px-2 py-1 text-sm"
            >
              <option value={ALL_REPORTS}>All reports</option>
              {reportOptions.map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          {error && <Notice tone="error">{error}</Notice>}
          {stillIndexing > 0 && (
            <Notice>
              {plural(stillIndexing, "report is", "reports are")} still being indexed; their datapoints appear here once
              extraction starts.
            </Notice>
          )}
          {!error && indexed.length === 0 && stillIndexing === 0 && (
            <p className="card p-6 text-sm text-slate-500">
              Nothing extracted yet. Ingest a report first; extraction runs at the end of ingestion.
            </p>
          )}

          {indexed.length > 0 &&
            registry.map((entry) => {
              const Section = entry.multi ? MultiValueSection : SingleValueSection;
              return (
                <Section
                  key={entry.key}
                  entry={entry}
                  rows={byKey.get(entry.key) ?? []}
                  outcomes={outcomesOf(entry.key)}
                  onOpen={open}
                  onRetry={retry}
                />
              );
            })}

          {other.length > 0 && <OtherDatapoints rows={other} onOpen={open} />}
        </div>
      </div>
      <PdfPanel target={pdf} onClose={() => setPdf(null)} />
    </div>
  );
}
