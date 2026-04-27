import Link from "next/link"
import { fetchDaily } from "@/lib/api"
import { fmtPct, fmtPrice, pnlClass, prevDay, nextDay, today } from "@/lib/utils"
import ExitBadge from "@/components/ExitBadge"
import CtxBadges from "@/components/CtxBadges"
import StatCard from "@/components/StatCard"

export default async function DailyPage({ params }: { params: Promise<{ date: string }> }) {
  const { date } = await params
  const data = await fetchDaily(date).catch(() => null)

  const isToday = date === today()
  const prev = prevDay(date)
  const next = nextDay(date)

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-lg font-semibold text-white">Rapport journalier</h1>
          <p className="mono text-sm text-slate-500 mt-0.5">{date}</p>
        </div>
        <div className="flex items-center gap-2">
          <Link href={`/daily/${prev}`} className="px-3 py-1.5 text-xs text-slate-400 bg-white/5 rounded-md hover:bg-white/8 transition-colors">
            ← Précédent
          </Link>
          {!isToday && (
            <Link href={`/daily/${next}`} className="px-3 py-1.5 text-xs text-slate-400 bg-white/5 rounded-md hover:bg-white/8 transition-colors">
              Suivant →
            </Link>
          )}
          <Link href={`/daily/${today()}`} className="px-3 py-1.5 text-xs text-slate-400 bg-white/5 rounded-md hover:bg-white/8 transition-colors">
            Aujourd'hui
          </Link>
        </div>
      </div>

      {!data ? (
        <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-8 text-center text-slate-600 text-sm">
          Aucune donnée pour le {date}
        </div>
      ) : (
        <>
          {/* Métriques du jour */}
          <div className="grid grid-cols-4 gap-3 mb-6">
            <StatCard
              label="P&L live"
              value={fmtPct(data.summary.live_pnl)}
              color={data.summary.live_pnl > 0 ? "positive" : data.summary.live_pnl < 0 ? "negative" : "muted"}
            />
            <StatCard label="Trades live"   value={String(data.summary.live_trades)} />
            <StatCard label="Trades shadow" value={String(data.summary.shadow_trades)} color="muted" />
            <StatCard
              label="P&L shadow"
              value={fmtPct(data.summary.shadow_pnl)}
              color={data.summary.shadow_pnl > 0 ? "positive" : data.summary.shadow_pnl < 0 ? "negative" : "muted"}
            />
          </div>

          {/* Table macros */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg overflow-hidden">
            <div className="px-5 py-3 border-b border-white/5">
              <span className="text-sm font-medium text-slate-300">Macros — {date}</span>
            </div>
            <table className="w-full data-table">
              <thead>
                <tr>
                  <th>Heure</th>
                  <th>Type</th>
                  <th>Contexte</th>
                  <th>Signal</th>
                  <th className="text-right">Entry</th>
                  <th className="text-right">TP</th>
                  <th className="text-right">SL</th>
                  <th className="text-right">P&L</th>
                  <th className="text-right">Exit</th>
                  <th className="text-right"></th>
                </tr>
              </thead>
              <tbody>
                {data.macros.map((m: any) => (
                  <tr key={m.mac_idx}>
                    <td className="mono text-slate-300">{m.mac_name}</td>
                    <td>
                      <span className={m.type === "live" ? "badge-live" : "badge-shadow"}>
                        {m.type === "live" ? "LIVE" : "SHADOW"}
                      </span>
                    </td>
                    <td>
                      {m.sc >= 0
                        ? <CtxBadges lc={m.lc_label} sc={m.sc_label} pc={m.pc_label} muted={!m.would_trade} />
                        : <span className="text-slate-600 text-xs">—</span>
                      }
                    </td>
                    <td>
                      {m.would_trade ? (
                        <span className={`mono text-sm font-medium ${m.direction === "LONG" ? "text-emerald-400" : "text-red-400"}`}>
                          {m.direction === "LONG" ? "↑" : "↓"} {m.direction}
                          <span className="ml-2 text-[10px] text-slate-500 font-normal">Q={fmtPct(m.q_val)}</span>
                        </span>
                      ) : (
                        <span className="mono text-xs text-slate-600">FLAT {m.flat_reason ? `(${m.flat_reason})` : ""}</span>
                      )}
                    </td>
                    <td className="text-right mono text-slate-300">{fmtPrice(m.entry_px)}</td>
                    <td className="text-right mono text-slate-500">{fmtPrice(m.tp_px)}</td>
                    <td className="text-right mono text-slate-500">{fmtPrice(m.sl_px)}</td>
                    <td className={`text-right mono font-medium ${pnlClass(m.pnl)}`}>
                      {m.pnl != null ? fmtPct(m.pnl) : "—"}
                    </td>
                    <td className="text-right">
                      {m.would_trade ? <ExitBadge reason={m.exit_reason} /> : null}
                    </td>
                    <td className="text-right">
                      {m.would_trade && (
                        <Link
                          href={`/trades/${date}/${m.mac_idx}`}
                          className="text-[11px] text-slate-500 hover:text-slate-300 transition-colors"
                        >
                          Détail →
                        </Link>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
