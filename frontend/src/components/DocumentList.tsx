import type { DocumentOut } from "../services/api";
export default function DocumentList({ documents, selectedIds, onToggle }: {
  documents: DocumentOut[]; selectedIds: string[]; onToggle: (id: string) => void;
}) {
  if (!documents.length) return <p className="text-sm text-slate-400">No documents ready yet. Upload your notes from Documents.</p>;
  return <ul className="space-y-2">{documents.map(doc => <li key={doc.id}>
    <label className="flex items-start gap-2 p-2 rounded hover:bg-slate-800 cursor-pointer">
      <input className="mt-1 accent-indigo-500" type="checkbox" checked={selectedIds.includes(doc.id)} onChange={() => onToggle(doc.id)} />
      <span className="text-sm break-all">{doc.name}{doc.status === "partial" && <span className="block text-xs text-amber-300">Some pages need attention</span>}</span>
    </label>
  </li>)}</ul>;
}
