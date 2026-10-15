import type { ReactNode } from "react";

/**
 * The one table style used across the app: a muted uppercase header row and
 * hairline-separated body rows. Cells keep the shared padding via `Td`.
 */
export function Table({ columns, children }: { columns: string[]; children: ReactNode }) {
  return (
    // Scroll instead of clipping when the columns cannot fit, e.g. next to the open PDF panel.
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-slate-50 text-left text-xs tracking-wide text-slate-500 uppercase">
          <tr>
            {columns.map((column, i) => (
              <th key={i} className="px-4 py-2 font-medium">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Tr({ children }: { children: ReactNode }) {
  return <tr className="border-t border-slate-100 align-top">{children}</tr>;
}

export function Td({ className = "", children, ...rest }: React.TdHTMLAttributes<HTMLTableCellElement>) {
  return (
    <td className={`px-4 py-2 ${className}`} {...rest}>
      {children}
    </td>
  );
}

export function EmptyRow({ colSpan, children }: { colSpan: number; children: ReactNode }) {
  return (
    <tr>
      <td colSpan={colSpan} className="px-4 py-8 text-center text-slate-500">
        {children}
      </td>
    </tr>
  );
}
