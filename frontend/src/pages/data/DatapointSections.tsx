import type { ReactNode } from "react";
import { VerifyBadge } from "../../components/Badges";
import { Table, Td, Tr } from "../../components/Table";
import { reportLabel } from "../../lib/reports";
import type { Datapoint, RegistryEntry } from "../../types";

interface SectionProps {
  entry: RegistryEntry;
  rows: Datapoint[];
  onOpen: (d: Datapoint) => void;
}

/** One figure per report (e.g. FTE): the value is the headline, qualifiers alongside. */
export function SingleValueSection({ entry, rows, onOpen }: SectionProps) {
  return (
    <Section title={entry.label}>
      <Table columns={["Company", "Value", "Period", "Qualifiers", "Source"]}>
        {rows.map((d) => (
          <Tr key={d.id}>
            <Td className="font-medium">{reportLabel(d)}</Td>
            <Td>
              <span className="text-base font-semibold whitespace-nowrap">{d.value ?? "–"}</span>{" "}
              <span className="text-slate-500">{d.unit}</span>
            </Td>
            <Td>{d.period ?? "–"}</Td>
            <Td className="text-xs text-slate-600">
              {Object.entries(d.qualifiers).map(([key, value]) => (
                <div key={key}>
                  <span className="text-slate-400">{key}:</span> {value}
                </div>
              ))}
            </Td>
            <Td>
              <SourceQuote d={d} onOpen={() => onOpen(d)} />
            </Td>
          </Tr>
        ))}
      </Table>
    </Section>
  );
}

/** Many items per report (e.g. sustainability goals): the label is the headline. */
export function MultiValueSection({ entry, rows, onOpen }: SectionProps) {
  return (
    <Section title={entry.label} count={rows.length}>
      <Table columns={["Company", "Item", "Value", "Period", "Source"]}>
        {rows.map((d) => (
          <Tr key={d.id}>
            <Td>{reportLabel(d)}</Td>
            <Td>
              <div className="font-medium">{d.label}</div>
              <div className="text-xs text-slate-500">
                {Object.entries(d.qualifiers)
                  .map(([key, value]) => `${key}: ${value}`)
                  .join(" · ")}
              </div>
            </Td>
            <Td>
              {d.value} {d.unit && <span className="text-slate-500">{d.unit}</span>}
            </Td>
            <Td>{d.period ?? "–"}</Td>
            <Td>
              <SourceQuote d={d} onOpen={() => onOpen(d)} />
            </Td>
          </Tr>
        ))}
      </Table>
    </Section>
  );
}

/** Datapoints whose key is no longer in the registry (extracted by an older configuration). */
export function OtherDatapoints({ rows, onOpen }: Pick<SectionProps, "rows" | "onOpen">) {
  return (
    <section className="card p-4 text-sm">
      <h3 className="mb-2 font-semibold">Other datapoints</h3>
      {rows.map((d) => (
        <div key={d.id} className="border-t border-slate-100 py-2">
          {reportLabel(d)} · {d.label}: <b>{d.value}</b> {d.unit} <SourceQuote d={d} onOpen={() => onOpen(d)} />
        </div>
      ))}
    </section>
  );
}

function Section({ title, count, children }: { title: string; count?: number; children: ReactNode }) {
  return (
    <section className="card overflow-hidden">
      <h3 className="border-b border-slate-200 bg-slate-50 px-4 py-2 text-sm font-semibold">
        {title}
        {count !== undefined && <span className="font-normal text-slate-500"> ({count})</span>}
      </h3>
      {children}
    </section>
  );
}

/** The verification badge, a link to the cited page and the quote itself. */
function SourceQuote({ d, onOpen }: { d: Datapoint; onOpen: () => void }) {
  return (
    <div className="max-w-md">
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <VerifyBadge status={d.verification} />
        {d.page && (
          <button type="button" className="text-xs text-emerald-700 hover:underline" onClick={onOpen}>
            p. {d.page}
          </button>
        )}
      </div>
      {d.quote && (
        <blockquote className="border-l-2 border-slate-200 pl-2 text-xs text-slate-600 italic">“{d.quote}”</blockquote>
      )}
    </div>
  );
}
