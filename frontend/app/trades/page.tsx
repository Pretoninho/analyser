import { fetchTrades } from "@/lib/api"
import { fmtPct, fmtPrice, pnlClass } from "@/lib/utils"
import StatCard from "@/components/StatCard"
import ExitBadge from "@/components/ExitBadge"
import CtxBadges from "@/components/CtxBadges"
import Link from "next/link"

export default async function TradesPage({
  searchParams,
}: {
  searchParams: Promise<{ type?: string; exit?: string; mac?: string }>
}) {
  const sp = await searchParams
  const data = await fetchTrades({
    type: sp.type,
    exit_reason: sp.exit,
    mac_idx: sp.mac ? Number(sp.mac) : undefined,
  }).catch(() => ({ trades: [], total: 0, metrics: {} }))

  const m = data.metrics

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-lg font-semibold text-white">Trades</h1>
          <p className="text-sm text-slate-500 mt-0.5">{data.total} trades</p>
        </div>

        {/* Filtres */}
        <div className="flex gap-2">
          {[
            { label: "Tous",    href: "/trades" },
            { label: "Live",    href: "/trades?type=live" },
            { label: "Shadow",  href: "/trades?type=shadow" },
          ].map(({ label, href }) => (
            <Link key={href} href={href}
              className="px-3 py-1.5 text-xs text-slate-400 bg-white/5 rounded-md hover:bg-white/8 transition-colors">
              {label}
            </Link>
          ))}
          {[
            { label: "TP",  href: `?${sp.type ? `type=${sp.type}&` : ""}exit=TP` },
            { label: "SL",  href: `?${sp.type ? `type=${sp.type}&` : ""}exit=SL` },
            { label: "EOD", href: `?${sp.type ? `type=${sp.type}&` : ""}exit=EOD` },
          ].map(({ label, href }) => (
            <Link key={label} href={href}
              className="px-3 py-1.5 text-xs text-slate-400 bg-white/5 rounded-md hover:bg-white/8 transition-colors">
              {label}
            </Link>
          ))}
        </div>
      </div>

      {/* Métriques */}
      <div className="grid grid-cols-5 gap-3 mb-6">
        <StatCard label="Trades"        value={String(m.n ?? 0)} />
        <StatCard label="Win rate"      value={m.wr != null ? `${(m.wr * 100).toFixed(1)}%` : "—"} />
        <StatCard label="Return total"  value={fmtPct(m.total_return)}
          color={m.total_return > 0 ? "positive" : m.total_return < 0 ? "negative" : "muted"} />
        <StatCard label="Profit factor" value={m.profit_factor != null ? m.profit_factor.toFixed(2) : "—"} />
        <StatCard label="Sharpe"        value={m.sharpe != null ? m.sharpe.toFixed(3) : "—"} />
      </div>

      {/* Table */}
      <div className="bg-[#0d0d14] border border-white/5 rounded-lg overflow-hidden">
        <table className="w-full data-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Macro</th>
              <th>Type</th>
              <th>Direction</th>
              <th>Contexte</th>
              <th className="text-right">Entry</th>
              <th className="text-right">TP</th>
              <th className="text-right">SL</th>
              <th className="text-right">P&L</th>
              <th className="text-right">Exit</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {data.trades.map((t: any, i: number) => (
              <tr key={i}>
                <td className="mono text-slate-400">{t.date}</td>
                <td className="mono text-slate-300">{t.mac_name}</td>
                <td>
                  <span className={t.type === "live" ? "badge-live" : "badge-shadow"}>
                    {t.type === "live" ? "LIVE" : "SHADOW"}
                  </span>
                </td>
                <td className={`mono font-medium ${t.direction === "LONG" ? "text-emerald-400" : "text-red-400"}`}>
                  {t.direction === "LONG" ? "↑" : "↓"} {t.direction}
                </td>
                <td>
                  <CtxBadges lc={t.lc_label} sc={t.sc_label} pc={t.pc_label} />
                </td>
                <td className="text-right mono text-slate-300">{fmtPrice(t.entry_px)}</td>
                <td className="text-right mono text-slate-500">{fmtPrice(t.tp_px)}</td>
                <td className="text-right mono text-slate-500">{fmtPrice(t.sl_px)}</td>
                <td className={`text-right mono font-medium ${pnlClass(t.pnl)}`}>
                  {fmtPct(t.pnl)}
                </td>
                <td className="text-right"><ExitBadge reason={t.exit_reason} /></td>
                <td className="text-right">
                  <Link href={`/trades/${t.date}/${t.mac_idx}`}
                    className="text-[11px] text-slate-500 hover:text-slate-300 transition-colors">
                    Détail →
                  </Link>
                </td>
              </tr>
            ))}
            {data.trades.length === 0 && (
              <tr><td colSpan={11} className="text-center text-slate-600 py-8">Aucun trade</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
