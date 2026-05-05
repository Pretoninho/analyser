"use client"

import { useState, useEffect } from "react"
import { fetchVolSnapshot, VolSnapshot } from "@/lib/api"

// ── Helpers ──────────────────────────────────────────────────────────────────

function clsState(state: string) {
  if (state === "VOL_SHOCK_UP")   return "bg-rose-500/15 border-rose-500/40 text-rose-300"
  if (state === "VOL_CRUSH_DOWN") return "bg-emerald-500/15 border-emerald-500/40 text-emerald-300"
  return "bg-white/5 border-white/10 text-slate-400"
}

function clsRegime(regime: string) {
  if (regime === "RISK_OFF") return "text-rose-400"
  if (regime === "RISK_ON")  return "text-emerald-400"
  return "text-slate-400"
}

function clsAction(action: string) {
  if (action === "LONG")  return "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
  if (action === "SHORT") return "bg-rose-500/15 text-rose-300 border-rose-500/30"
  if (action === "WATCH") return "bg-amber-500/15 text-amber-300 border-amber-500/30"
  return "bg-white/5 text-slate-400 border-white/10"
}

function clsBias(bias: string) {
  if (bias === "SELL_VOL") return "text-rose-300"
  if (bias === "BUY_VOL")  return "text-emerald-300"
  return "text-slate-400"
}

function fmt(v: number | null | undefined, decimals = 2) {
  if (v == null || Number.isNaN(v)) return "—"
  return v.toFixed(decimals)
}

function fmtPct(v: number | null | undefined) {
  if (v == null || Number.isNaN(v)) return "—"
  return `${(v * 100).toFixed(1)}%`
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-[#0d0d14] border border-white/5 rounded-lg px-4 py-3">
      <p className="text-[11px] text-slate-500">{label}</p>
      <p className="text-lg font-semibold text-white mono mt-1">{value}</p>
      {sub && <p className="text-[11px] text-slate-500 mt-0.5">{sub}</p>}
    </div>
  )
}

function DriverBar({ label, score }: { label: string; score: number }) {
  const w = Math.max(4, Math.min(100, score * 100))
  return (
    <div>
      <div className="flex justify-between text-[11px] mb-1">
        <span className="text-slate-300 mono">{label}</span>
        <span className="text-slate-500 mono">{score.toFixed(4)}</span>
      </div>
      <div className="h-1.5 rounded bg-white/10 overflow-hidden">
        <div className="h-full bg-gradient-to-r from-cyan-400 to-indigo-400" style={{ width: `${w}%` }} />
      </div>
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function VolPage() {
  const [data, setData] = useState<VolSnapshot | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]   = useState<string | null>(null)
  const [lastUpdate, setLastUpdate] = useState<string>("")

  const load = async () => {
    try {
      setLoading(true)
      setError(null)
      const snap = await fetchVolSnapshot()
      setData(snap)
      setLastUpdate(new Date().toLocaleTimeString())
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur de chargement")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  if (loading) return (
    <div className="flex items-center justify-center h-64 text-slate-500 text-sm">
      Chargement du signal de volatilité...
    </div>
  )

  if (error) return (
    <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-sm text-red-300">
      {error}
    </div>
  )

  if (!data) return null

  const dvol   = data.dvol
  const signal = data.signal
  const vp     = data.vol_premium

  const hasSignalError = "error" in (signal ?? {})
  const hasDvolError   = "error" in (dvol ?? {})

  return (
    <div className="space-y-6">

      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-lg font-semibold text-white">Vol Signal</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            DVOL Deribit + Signal actionnable + Vol premium
          </p>
        </div>
        <button
          onClick={load}
          className="text-xs text-slate-400 hover:text-slate-200 border border-white/10 rounded px-3 py-1.5 transition-colors"
        >
          Actualiser {lastUpdate && `· ${lastUpdate}`}
        </button>
      </div>

      {/* ── DVOL Banner ─────────────────────────────────────────── */}
      {!hasDvolError && (
        <div className={`rounded-lg border p-4 ${clsState(dvol.dvol_state)}`}>
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div>
              <p className="text-xs opacity-70 mb-1">Deribit Volatility Index</p>
              <p className="text-xl font-bold mono">{dvol.dvol_state.replace(/_/g, " ")}</p>
              <p className={`text-sm font-medium mt-0.5 ${clsRegime(dvol.risk_regime)}`}>
                {dvol.risk_regime.replace(/_/g, " ")}
              </p>
            </div>
            <div className="flex gap-6 text-right">
              <div>
                <p className="text-[11px] opacity-60">DVOL</p>
                <p className="text-lg font-semibold mono">{fmt(dvol.dvol_close)}</p>
              </div>
              <div>
                <p className="text-[11px] opacity-60">Z-Score</p>
                <p className="text-lg font-semibold mono">{dvol.dvol_z > 0 ? "+" : ""}{fmt(dvol.dvol_z)}</p>
              </div>
              <div>
                <p className="text-[11px] opacity-60">ROC 24h</p>
                <p className="text-lg font-semibold mono">{fmtPct(dvol.dvol_roc_24h)}</p>
              </div>
              <div>
                <p className="text-[11px] opacity-60">Intensité</p>
                <p className="text-lg font-semibold mono">{fmt(dvol.intensity)}</p>
              </div>
            </div>
          </div>

          {/* Barre d'intensité */}
          <div className="mt-3">
            <div className="h-1.5 rounded bg-black/20 overflow-hidden">
              <div
                className="h-full bg-current opacity-60 transition-all"
                style={{ width: `${Math.round(dvol.intensity * 100)}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* ── Vol Premium ─────────────────────────────────────────── */}
      {vp && (
        <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
          <h2 className="text-sm font-semibold text-white mb-1">Vol Premium (IV vs Réalisée)</h2>
          <p className="text-xs text-slate-500 mb-4">
            Spread entre IV implicite ATM et vol réalisée annualisée. Positif = options chères → vendre. Négatif = options bon marché → acheter.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <StatCard label="IV ATM" value={fmt(vp.iv_atm, 1) + "%"} />
            <StatCard label="Vol réalisée" value={fmtPct(vp.realized_vol)} />
            <StatCard label="Premium" value={`${vp.premium > 0 ? "+" : ""}${fmtPct(vp.premium)}`} />
            <div className={`bg-[#0d0d14] border rounded-lg px-4 py-3 ${
              vp.bias === "SELL_VOL" ? "border-rose-500/30" :
              vp.bias === "BUY_VOL"  ? "border-emerald-500/30" : "border-white/5"
            }`}>
              <p className="text-[11px] text-slate-500">Biais options</p>
              <p className={`text-lg font-semibold mono mt-1 ${clsBias(vp.bias)}`}>
                {vp.bias.replace("_", " ")}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ── Signal Deribit + Drivers ─────────────────────────────── */}
      {!hasSignalError && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

          {/* Signal actionnable */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-sm font-semibold text-white mb-4">Signal Deribit Futures</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
              <div className={`col-span-2 rounded-lg border px-4 py-3 text-center ${clsAction(signal.signal.action)}`}>
                <p className="text-xs opacity-70 mb-1">Action</p>
                <p className="text-2xl font-bold mono">{signal.signal.action}</p>
                <p className="text-xs opacity-70 mt-1">Confiance {fmt(signal.signal.confidence)}</p>
              </div>
              <StatCard label="Horizon" value={signal.signal.horizon} />
              <StatCard label="Échéance" value={signal.signal.contract?.tenor || "N/A"} />
              <StatCard label="Net score" value={(signal.signal.net_score > 0 ? "+" : "") + fmt(signal.signal.net_score, 4)} />
              <StatCard label="Edge total" value={fmt(signal.signal.edge_total, 4)} />
            </div>
            {signal.signal.contract?.why && (
              <p className="text-[11px] text-slate-500 leading-4">
                {signal.signal.contract.why}
              </p>
            )}
          </div>

          {/* Top Drivers */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-sm font-semibold text-white mb-1">Top Drivers</h2>
            <p className="text-xs text-slate-500 mb-4">Moteurs dominants du signal courant.</p>
            <div className="space-y-3">
              {signal.drivers.map((d) => (
                <DriverBar key={d.name} label={d.name} score={d.score} />
              ))}
            </div>

            {/* Options snapshot */}
            {signal.options && (
              <div className="mt-5 pt-4 border-t border-white/5 grid grid-cols-2 gap-2 text-xs">
                <div className="text-slate-500">IV ATM<span className="float-right text-slate-300 mono">{fmt(signal.options.iv_atm, 1)}%</span></div>
                <div className="text-slate-500">Skew 25d<span className="float-right text-slate-300 mono">{fmt(signal.options.iv_skew_25d, 1)}%</span></div>
                <div className="text-slate-500">Term 1W<span className="float-right text-slate-300 mono">{fmt(signal.options.term_1w, 1)}%</span></div>
                <div className="text-slate-500">Term 1M<span className="float-right text-slate-300 mono">{fmt(signal.options.term_1m, 1)}%</span></div>
                <div className="text-slate-500">Put/Call<span className="float-right text-slate-300 mono">{fmt(signal.options.put_call_ratio)}</span></div>
                <div className="text-slate-500">Term 3M<span className="float-right text-slate-300 mono">{fmt(signal.options.term_3m, 1)}%</span></div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Timestamp ───────────────────────────────────────────── */}
      <p className="text-[11px] text-slate-600 text-right">
        Dernière mise à jour API : {data.timestamp?.slice(0, 19).replace("T", " ")} UTC
      </p>
    </div>
  )
}
