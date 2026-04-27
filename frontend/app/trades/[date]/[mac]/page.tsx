"use client"
import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import dynamic from "next/dynamic"
import Link from "next/link"
import { fetchCandles } from "@/lib/api"
import { fmtPct, fmtPrice, pnlClass } from "@/lib/utils"
import ExitBadge from "@/components/ExitBadge"

const CandleChart = dynamic(() => import("@/components/CandleChart"), { ssr: false })

export default function TradeDetailPage() {
  const { date, mac } = useParams<{ date: string; mac: string }>()
  const [data, setData]   = useState<any>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    fetchCandles(date, Number(mac))
      .then(setData)
      .catch(() => setError(true))
  }, [date, mac])

  if (error) return (
    <div className="text-slate-600 text-sm p-8">
      Impossible de charger les données (Binance API ou date invalide).
    </div>
  )

  if (!data) return (
    <div className="text-slate-600 text-sm p-8 animate-pulse">Chargement...</div>
  )

  const t = data.trade

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <Link href="/trades" className="text-slate-500 hover:text-slate-300 text-sm transition-colors">
          ← Trades
        </Link>
        <div>
          <h1 className="text-lg font-semibold text-white">
            {data.mac_name} ET — {date}
          </h1>
          <p className="mono text-sm text-slate-500 mt-0.5">mac_idx={data.mac_idx}</p>
        </div>
      </div>

      {/* Infos trade */}
      {t ? (
        <div className="grid grid-cols-5 gap-3 mb-6">
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-4">
            <p className="text-xs text-slate-500 mb-1">Direction</p>
            <p className={`mono text-xl font-medium ${t.direction === "LONG" ? "text-emerald-400" : "text-red-400"}`}>
              {t.direction === "LONG" ? "↑" : "↓"} {t.direction}
            </p>
          </div>
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-4">
            <p className="text-xs text-slate-500 mb-1">Entry</p>
            <p className="mono text-xl font-medium text-white">{fmtPrice(t.entry_px)}</p>
          </div>
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-4">
            <p className="text-xs text-slate-500 mb-1">P&L</p>
            <p className={`mono text-xl font-medium ${pnlClass(t.pnl)}`}>{fmtPct(t.pnl)}</p>
          </div>
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-4">
            <p className="text-xs text-slate-500 mb-1">Exit</p>
            <div className="mt-1"><ExitBadge reason={t.exit_reason} /></div>
          </div>
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-4">
            <p className="text-xs text-slate-500 mb-1">Durée</p>
            <p className="mono text-xl font-medium text-white">{t.n_candles} <span className="text-sm text-slate-500">bougies</span></p>
          </div>
        </div>
      ) : (
        <div className="mb-6 p-4 bg-[#0d0d14] border border-white/5 rounded-lg text-slate-600 text-sm">
          Aucun trade exécuté sur cette macro ce jour-là.
        </div>
      )}

      {/* Graphique OHLC */}
      <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-4 mb-4">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs text-slate-500">
            BTCUSDT 1m · fenêtre {data.mac_name} ET ±45 min
          </p>
          {t && (
            <div className="flex gap-3 text-[11px] mono">
              <span className="text-slate-400">Entry <span className="text-white">{fmtPrice(t.entry_px)}</span></span>
              <span className="text-emerald-400">TP <span className="text-white">{fmtPrice(t.tp_px)}</span></span>
              <span className="text-red-400">SL <span className="text-white">{fmtPrice(t.sl_px)}</span></span>
            </div>
          )}
        </div>
        {data.candles.length > 0 ? (
          <CandleChart
            candles={data.candles}
            entryPx={t?.entry_px}
            tpPx={t?.tp_px}
            slPx={t?.sl_px}
          />
        ) : (
          <div className="h-[320px] flex items-center justify-center text-slate-600 text-sm">
            Données OHLC non disponibles (Binance API requise)
          </div>
        )}
      </div>
    </div>
  )
}
