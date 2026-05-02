"use client"

import { FractalSignal } from "@/lib/api"

export default function FractalSignalCard({ signal }: { signal: FractalSignal }) {
  const isPositive = signal.confidence > 0.85

  return (
    <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-4 hover:border-white/10 transition-colors">
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-semibold text-white">{signal.pattern}</span>
            <span className={`text-xs px-2 py-0.5 rounded ${
              isPositive
                ? "bg-emerald-500/15 text-emerald-300"
                : "bg-amber-500/15 text-amber-300"
            }`}>
              {signal.setup}
            </span>
          </div>
          <p className="text-[11px] text-slate-500 mono">{signal.timestamp}</p>
        </div>
        <div className="text-right">
          <p className="text-sm font-semibold text-white">{(signal.confidence * 100).toFixed(1)}%</p>
          <p className="text-[10px] text-slate-500">confiance</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 mb-3">
        <div className="bg-white/[0.03] rounded px-2 py-1.5">
          <p className="text-[10px] text-slate-500">Entry</p>
          <p className="mono text-sm text-slate-300">${signal.entry_price.toFixed(2)}</p>
        </div>
        <div className="bg-white/[0.03] rounded px-2 py-1.5">
          <p className="text-[10px] text-slate-500">Zone</p>
          <p className="mono text-sm text-slate-300">{signal.kz}</p>
        </div>
      </div>

      <div className="text-[10px] text-slate-600">
        <p>Date: <span className="text-slate-400">{signal.day_date}</span></p>
      </div>
    </div>
  )
}
