import { reportLabel } from "../../lib/reports";
import type { Conversation, Report } from "../../types";

interface Props {
  conversations: Conversation[];
  currentId: string | null;
  onOpen: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  /** Reports that can be searched; the user may narrow a question to a subset of them. */
  reports: Report[];
  scope: string[];
  onToggleScope: (id: string) => void;
}

export default function ConversationSidebar({
  conversations,
  currentId,
  onOpen,
  onNew,
  onDelete,
  reports,
  scope,
  onToggleScope,
}: Props) {
  return (
    <aside className="flex w-64 flex-col border-r border-slate-200 bg-white">
      <div className="border-b border-slate-200 p-3">
        <button type="button" className="btn w-full justify-center" onClick={onNew}>
          + New conversation
        </button>
      </div>

      <ul className="min-h-0 flex-1 overflow-y-auto p-2 text-sm" aria-label="Conversations">
        {conversations.map((conversation) => (
          <li key={conversation.id} className="group flex items-center">
            <button
              type="button"
              onClick={() => onOpen(conversation.id)}
              aria-current={conversation.id === currentId ? "true" : undefined}
              className={`min-w-0 flex-1 truncate rounded-md px-2 py-1.5 text-left hover:bg-slate-100 ${
                conversation.id === currentId ? "bg-slate-100 font-medium" : ""
              }`}
              title={conversation.title}
            >
              {conversation.title}
            </button>
            <button
              type="button"
              className="hidden px-1 text-slate-400 group-hover:block hover:text-red-600"
              aria-label={`Delete conversation "${conversation.title}"`}
              onClick={() => onDelete(conversation.id)}
            >
              ×
            </button>
          </li>
        ))}
      </ul>

      {/* min-w-0 overrides the fieldset default of min-inline-size: min-content, which would widen the sidebar. */}
      <fieldset className="min-w-0 border-t border-slate-200 p-3 text-xs">
        <legend className="sr-only">Limit to reports</legend>
        <div className="mb-1 font-medium text-slate-600">Limit to reports</div>
        {reports.length === 0 && <div className="text-slate-400">no ready reports</div>}
        {reports.map((report) => (
          <label key={report.id} className="flex cursor-pointer items-center gap-2 py-0.5">
            <input type="checkbox" checked={scope.includes(report.id)} onChange={() => onToggleScope(report.id)} />
            <span className="min-w-0 truncate">
              {reportLabel(report)}
              <span className="text-slate-400"> · {shortFilename(report.filename)}</span>
            </span>
          </label>
        ))}
        <div className="mt-1 text-slate-400">
          Unchecked = all reports; the question's company and year also narrow the search.
        </div>
      </fieldset>
    </aside>
  );
}

const shortFilename = (filename: string) => filename.replace(/\.pdf$/i, "").slice(0, 28);
