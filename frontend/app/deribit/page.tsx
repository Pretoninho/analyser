import Link from "next/link"

import DeribitSignalActions from "@/components/DeribitSignalActions"
import { fetchDeribitBacktest, fetchDeribitSignal } from "@/lib/api"

function clsAction(action: string) {
  if (action === "LONG") return "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
  if (action === "SHORT") return "bg-rose-500/15 text-rose-300 border border-rose-500/30"
  if (action === "WATCH") return "bg-amber-500/15 text-amber-300 border border-amber-500/30"
  return "bg-white/5 text-slate-400 border border-white/10"
}

function fmtPct(v: number | null | undefined) {
  if (v == null || Number.isNaN(v)) return "-"
  return `${(v * 100).toFixed(2)}%`
}

type PageProps = {
  searchParams?: {
    timeframe?: string
    days?: string
    threshold?: string
  }
}

export default async function DeribitPage({ searchParams }: PageProps) {
  const timeframe = searchParams?.timeframe || "1h"
  const days = Number(searchParams?.days || "90")
  const threshold = Number(searchParams?.threshold || "0.05")

  const [signal, backtest] = await Promise.all([
    fetchDeribitSignal(timeframe, days).catch(() => null),
    fetchDeribitBacktest(timeframe, days, threshold).catch(() => null),
  ])

  if (!signal || !backtest) {
    return <div className="text-slate-600 text-sm p-8">Donnees Deribit non disponibles.</div>
  }

  const rows = backtest.results ?? []
  const signalRows = rows.filter((r) => r.note === "ok")

  return (
    <div>
      <div className="mb-6 flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-lg font-semibold text-white">Deribit Futures Signals</h1>
          <p className="text-sm text-slate-500 mt-0.5">Tous les edges exploites en signal actionnable + backtest.</p>
        </div>
        <div className="flex gap-2 text-xs">
          <Link href={`/deribit?timeframe=1h&days=90&threshold=0.05`} className="px-3 py-2 rounded-md border border-white/10 text-slate-300 hover:bg-white/5">
            1h / 90j
          </Link>
          <Link href={`/deribit?timeframe=4h&days=180&threshold=0.05`} className="px-3 py-2 rounded-md border border-white/10 text-slate-300 hover:bg-white/5">
            4h / 180j
          </Link>
        </div>
      </div>

      <DeribitSignalActions timeframe={timeframe} days={days} />

      <div className="grid grid-cols-5 gap-3 mb-6">
        <Card label="Action" value={signal.signal.action} tone={clsAction(signal.signal.action)} />
        <Card label="Confiance" value={signal.signal.confidence.toFixed(2)} />
        <Card label="Horizon" value={signal.signal.horizon} />
        <Card label="Echeance" value={signal.signal.contract?.tenor || "N/A"} />
        <Card label="Edge total" value={signal.signal.edge_total.toFixed(4)} />
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <section className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
          <h2 className="text-sm font-semibold text-white mb-1">Snapshot signal</h2>
          <p className="text-xs text-slate-500 mb-4">Derniere barre: prix, funding, volatilite, score directionnel.</p>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <KV k="Timestamp" v={signal.latest_ts || "-"} mono />
            <KV k="Close" v={signal.close.toLocaleString(undefined, { maximumFractionDigits: 2 })} mono />
            <KV k="Funding annualise" v={signal.funding_annualized.toFixed(4)} mono />
            <KV k="Vol realisee annualisee" v={signal.realized_vol_annual.toFixed(4)} mono />
            <KV k="Long score" v={signal.signal.long_score.toFixed(4)} mono />
            <KV k="Short score" v={signal.signal.short_score.toFixed(4)} mono />
            <KV k="Net score" v={signal.signal.net_score.toFixed(4)} mono />
            <KV k="Timeframe" v={signal.timeframe} mono />
          </div>
        </section>

        <section className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
          <h2 className="text-sm font-semibold text-white mb-1">Top drivers</h2>
          <p className="text-xs text-slate-500 mb-4">Moteurs dominants du signal courant.</p>

          <div className="space-y-3">
            {signal.drivers.map((d) => (
              <DriverBar key={d.name} label={d.name} score={d.score} />
            ))}
          </div>

          <div className="mt-5 rounded-lg border border-white/10 p-3 bg-white/[0.02]">
            <p className="text-[11px] text-slate-500 mb-2">Interpretation</p>
            <p className="text-xs text-slate-300 leading-5">
              Le moteur combine toutes les briques: funding reversion, carry momentum/stress, dislocation mark/index,
              prime de vol options, skew panic, et term structure kink. Le signal final est base sur le desequilibre LONG/SHORT.
            </p>
            <p className="text-xs text-slate-400 leading-5 mt-2">
              Suggestion d&apos;echeance: {signal.signal.contract?.instrument || "N/A"} / {signal.signal.contract?.tenor || "N/A"}
              {" - "}
              {signal.signal.contract?.why || "Pas de rationale disponible."}
            </p>
          </div>
        </section>
      </div>

      <section className="bg-[#0d0d14] border border-white/5 rounded-lg overflow-hidden">
        <div className="px-5 py-3 border-b border-white/5">
          <h2 className="text-sm font-semibold text-white">Backtest edges</h2>
          <p className="text-xs text-slate-500 mt-1">Hit ratio, corr et lift par edge/horizon (threshold={threshold}).</p>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr>
              {["Edge", "Horizon", "N", "Hit", "Avg active", "Baseline", "Corr", "Lift", "Note"].map((h) => (
                <th key={h} className="px-4 py-2.5 text-left text-[11px] font-medium text-slate-500 uppercase tracking-wider border-b border-white/5">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {signalRows.map((r, i) => (
              <tr key={`${r.edge}-${r.horizon_h}-${i}`} className="border-b border-white/[0.03] hover:bg-white/[0.015]">
                <td className="px-4 py-2.5 mono text-slate-300">{r.edge}</td>
                <td className="px-4 py-2.5 mono text-slate-400">+{r.horizon_h}h</td>
                <td className="px-4 py-2.5 mono text-slate-400">{r.n_signals}</td>
                <td className="px-4 py-2.5 mono text-slate-300">{fmtPct(r.hit_ratio)}</td>
                <td className="px-4 py-2.5 mono text-slate-300">{fmtPct(r.avg_ret_active)}</td>
                <td className="px-4 py-2.5 mono text-slate-400">{fmtPct(r.avg_ret_baseline)}</td>
                <td className="px-4 py-2.5 mono text-slate-400">{r.corr == null ? "-" : r.corr.toFixed(4)}</td>
                <td className="px-4 py-2.5 mono text-slate-400">{r.lift == null ? "-" : r.lift.toFixed(3)}</td>
                <td className="px-4 py-2.5">
                  <span className={r.note === "ok" ? "text-emerald-300" : "text-slate-500"}>{r.note}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}

function Card({
  label,
  value,
  tone,
}: {
  label: string
  value: string
  tone?: string
}) {
  return (
    <div className={`rounded-lg border border-white/10 bg-white/[0.02] px-4 py-3 ${tone || ""}`}>
      <p className="text-[11px] text-slate-500">{label}</p>
      <p className="text-lg text-white font-semibold mt-1 mono">{value}</p>
    </div>
  )
}

function KV({ k, v, mono = false }: { k: string; v: string; mono?: boolean }) {
  return (
    <div className="rounded border border-white/10 px-3 py-2 bg-white/[0.02]">
      <p className="text-[10px] text-slate-500">{k}</p>
      <p className={`${mono ? "mono" : ""} text-slate-300 mt-1`}>{v}</p>
    </div>
  )
}

function DriverBar({ label, score }: { label: string; score: number }) {
  const width = Math.max(4, Math.min(100, score * 100))
  return (
    <div>
      <div className="flex items-center justify-between mb-1 text-[11px]">
        <span className="text-slate-300 mono">{label}</span>
        <span className="text-slate-500 mono">{score.toFixed(4)}</span>
      </div>
      <div className="h-2 rounded bg-white/10 overflow-hidden">
        <div className="h-full bg-gradient-to-r from-cyan-400 to-emerald-400" style={{ width: `${width}%` }} />
      </div>
    </div>
  )
}
