interface Props {
  label: string
  value: string
  sub?: string
  color?: "default" | "positive" | "negative" | "muted"
}

const colors = {
  default:  "text-white",
  positive: "text-emerald-400",
  negative: "text-red-400",
  muted:    "text-slate-400",
}

export default function StatCard({ label, value, sub, color = "default" }: Props) {
  return (
    <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-4">
      <p className="text-xs text-slate-500 mb-1">{label}</p>
      <p className={`mono text-xl font-medium ${colors[color]}`}>{value}</p>
      {sub && <p className="mono text-[11px] text-slate-600 mt-0.5">{sub}</p>}
    </div>
  )
}
