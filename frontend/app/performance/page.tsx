import { fetchPerformance } from "@/lib/api"
import { fmtPct, pnlClass } from "@/lib/utils"
import StatCard from "@/components/StatCard"

export default async function PerformancePage() {
  const data = await fetchPerformance().catch(() => null)

  if (!data) return (
    <div className="text-slate-600 text-sm p-8">Données non disponibles.</div>
  )

  const o = data.overall
  const curve: { date: string; pnl: number }[] = data.pnl_curve ?? []

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-lg font-semibold text-white">Performance</h1>
        <p className="text-sm text-slate-500 mt-0.5">Toutes macros — live + shadow</p>
      </div>

      {/* Métriques globales */}
      <div className="grid grid-cols-5 gap-3 mb-6">
        <StatCard label="Trades"        value={String(o.n ?? 0)} />
        <StatCard label="Win rate"      value={o.wr != null ? `${(o.wr * 100).toFixed(1)}%` : "—"} />
        <StatCard label="Return total"  value={fmtPct(o.total_return)}
          color={o.total_return > 0 ? "positive" : o.total_return < 0 ? "negative" : "muted"} />
        <StatCard label="Profit factor" value={o.profit_factor?.toFixed(2) ?? "—"} />
        <StatCard label="Max DD"        value={fmtPct(o.max_dd)} color="negative" />
      </div>

      {/* Courbe P&L + par macro côte à côte */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        {/* Courbe P&L */}
        <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
          <p className="text-xs text-slate-500 mb-4">P&L cumulé</p>
          {curve.length > 1 ? (
            <PnlCurve points={curve} />
          ) : (
            <div className="h-[160px] flex items-center justify-center text-slate-600 text-sm">
              Pas assez de données
            </div>
          )}
        </div>

        {/* Par macro */}
        <div className="bg-[#0d0d14] border border-white/5 rounded-lg overflow-hidden">
          <div className="px-5 py-3 border-b border-white/5">
            <span className="text-xs text-slate-500">Par macro</span>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr>
                {["Macro","Type","N","WR","Return"].map(h => (
                  <th key={h} className="px-4 py-2.5 text-left text-[11px] font-medium text-slate-500 uppercase tracking-wider border-b border-white/5">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.by_macro.filter((m: any) => m.n > 0).map((m: any) => (
                <tr key={m.mac_idx} className="border-b border-white/[0.03] hover:bg-white/[0.015]">
                  <td className="px-4 py-2.5 mono text-slate-300">{m.mac_name}</td>
                  <td className="px-4 py-2.5">
                    <span className={m.type === "live" ? "badge-live" : "badge-shadow"}>
                      {m.type === "live" ? "LIVE" : "SHADOW"}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 mono text-slate-400">{m.n}</td>
                  <td className="px-4 py-2.5 mono text-slate-300">
                    {m.wr != null ? `${(m.wr * 100).toFixed(0)}%` : "—"}
                  </td>
                  <td className={`px-4 py-2.5 mono font-medium ${pnlClass(m.total_return)}`}>
                    {fmtPct(m.total_return)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Par mois */}
      {data.by_month?.length > 0 && (
        <div className="bg-[#0d0d14] border border-white/5 rounded-lg overflow-hidden">
          <div className="px-5 py-3 border-b border-white/5">
            <span className="text-xs text-slate-500">Par mois</span>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr>
                {["Mois","Trades","WR","Return"].map(h => (
                  <th key={h} className="px-5 py-2.5 text-left text-[11px] font-medium text-slate-500 uppercase tracking-wider border-b border-white/5">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.by_month.map((m: any) => (
                <tr key={m.month} className="border-b border-white/[0.03] hover:bg-white/[0.015]">
                  <td className="px-5 py-2.5 mono text-slate-400">{m.month}</td>
                  <td className="px-5 py-2.5 mono text-slate-300">{m.n}</td>
                  <td className="px-5 py-2.5 mono text-slate-300">{(m.wr * 100).toFixed(0)}%</td>
                  <td className={`px-5 py-2.5 mono font-medium ${pnlClass(m.total_return)}`}>
                    {fmtPct(m.total_return)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function PnlCurve({ points }: { points: { date: string; pnl: number }[] }) {
  const values  = points.map(p => p.pnl)
  const min     = Math.min(...values)
  const max     = Math.max(...values)
  const range   = max - min || 1
  const W       = 500
  const H       = 160
  const pad     = 10

  const coords = points.map((p, i) => {
    const x = pad + (i / (points.length - 1)) * (W - pad * 2)
    const y = H - pad - ((p.pnl - min) / range) * (H - pad * 2)
    return `${x},${y}`
  })

  const first = coords[0], last = coords[coords.length - 1]
  const lastX = parseFloat(last.split(",")[0])
  const color = values[values.length - 1] >= 0 ? "#4ade80" : "#f87171"
  const fill  = values[values.length - 1] >= 0 ? "#4ade8018" : "#f8717118"

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.15" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <line x1={pad} y1={H/2} x2={W-pad} y2={H/2} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
      <path
        d={`M${first} L${coords.join(" L")} L${lastX},${H} L${pad},${H} Z`}
        fill="url(#g)"
      />
      <polyline
        points={coords.join(" ")}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  )
}
