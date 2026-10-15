import { useEffect, useMemo, useState } from "react";
import { api } from "../../api";
import { Notice } from "../../components/Notice";
import PdfPanel from "../../components/PdfPanel";
import { groupBy } from "../../lib/collections";
import { errorMessage } from "../../lib/format";
import { pdfTargetOf, reportLabel, type PdfTarget } from "../../lib/reports";
import type { Datapoint, RegistryEntry } from "../../types";
import { MultiValueSection, OtherDatapoints, SingleValueSection } from "./DatapointSections";

const ALL_REPORTS = "";

/**
 * Pre-extracted datapoints, one section per entry of the backend's extraction registry.
 * Adding a datapoint to `registry.yaml` adds a section here without a frontend change.
 */
export default function DataPage() {
  const [registry, setRegistry] = useState<RegistryEntry[]>([]);
  const [datapoints, setDatapoints] = useState<Datapoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [reportFilter, setReportFilter] = useState(ALL_REPORTS);
  const [pdf, setPdf] = useState<PdfTarget | null>(null);

  useEffect(() => {
    Promise.all([api.registry(), api.datapoints()])
      .then(([entries, points]) => {
        setRegistry(entries);
        setDatapoints(points);
      })
      .catch((err) => setError(errorMessage(err)));
  }, []);

  /** Every report that has at least one datapoint, for the filter dropdown. */
  const reportOptions = useMemo(() => {
    const byId = new Map(datapoints.map((d) => [d.report_id, reportLabel(d)]));
    return [...byId].sort(([, a], [, b]) => a.localeCompare(b));
  }, [datapoints]);

  const byKey = useMemo(() => {
    const visible = reportFilter ? datapoints.filter((d) => d.report_id === reportFilter) : datapoints;
    return groupBy(visible, (d) => d.key);
  }, [datapoints, reportFilter]);

  const registryKeys = new Set(registry.map((entry) => entry.key));
  const other = [...byKey].filter(([key]) => !registryKeys.has(key)).flatMap(([, rows]) => rows);

  const open = (d: Datapoint) => setPdf(pdfTargetOf(d));

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
          {!error && datapoints.length === 0 && (
            <p className="card p-6 text-sm text-slate-500">
              Nothing extracted yet. Ingest a report first; extraction runs at the end of ingestion.
            </p>
          )}

          {registry.map((entry) => {
            const rows = byKey.get(entry.key);
            if (!rows) return null;
            const Section = entry.multi ? MultiValueSection : SingleValueSection;
            return <Section key={entry.key} entry={entry} rows={rows} onOpen={open} />;
          })}

          {other.length > 0 && <OtherDatapoints rows={other} onOpen={open} />}
        </div>
      </div>
      <PdfPanel target={pdf} onClose={() => setPdf(null)} />
    </div>
  );
}
