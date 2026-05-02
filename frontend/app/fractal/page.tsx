"use client"

import { useState, useEffect } from "react"
import FractalSetupToggle from "@/components/FractalSetupToggle"
import FractalSignalCard from "@/components/FractalSignalCard"
import FractalStats from "@/components/FractalStats"
import { fetchFractalSignals, fetchFractalStats, FractalSetupResponse } from "@/lib/api"

type Setup = "strict" | "modere" | "frequent"

export default function FractalPage() {
  const [activeSetup, setActiveSetup] = useState<Setup>("strict")
  const [data, setData] = useState<FractalSetupResponse | null>(null)
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)
        const [setupData, statsData] = await Promise.all([
          fetchFractalSignals(activeSetup),
          fetchFractalStats(),
        ])
        setData(setupData)
        setStats(statsData)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Erreur de chargement")
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [activeSetup])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-lg font-semibold text-white">Fractal Detection</h1>
        <p className="text-sm text-slate-500 mt-0.5">Multi-setup fractal pattern recognition avec logging en base de données</p>
      </div>

      {/* Stats */}
      {stats && !loading && (
        <FractalStats totalSignals={stats.total_signals} bySetup={stats.by_setup} />
      )}

      {/* Setup Toggle */}
      <div className="flex items-center justify-between">
        <FractalSetupToggle active={activeSetup} onChange={setActiveSetup} />
        <div className="text-xs text-slate-500">
          Confiance: <span className="text-slate-300 font-medium">{data?.confidence.toFixed(3)}</span>
        </div>
      </div>

      {/* Signals Grid */}
      <div>
        {loading ? (
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-8 text-center text-slate-500">
            Chargement des signaux...
          </div>
        ) : error ? (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-sm text-red-300">
            {error}
          </div>
        ) : data && data.signals.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {data.signals.map((signal, i) => (
              <FractalSignalCard key={i} signal={signal} />
            ))}
          </div>
        ) : (
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-8 text-center text-slate-500">
            Aucun signal {data?.setup} détecté
          </div>
        )}
      </div>

      {/* Info Box */}
      <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-4">
        <p className="text-xs text-slate-500 mb-2">À propos des setups</p>
        <div className="grid grid-cols-3 gap-4 text-xs text-slate-400">
          <div>
            <p className="font-semibold text-slate-300 mb-1">STRICT</p>
            <p>Weekly + Daily + Killzone + Break & Retest</p>
          </div>
          <div>
            <p className="font-semibold text-slate-300 mb-1">MODÉRÉ</p>
            <p>Daily + Killzone + Break & Retest</p>
          </div>
          <div>
            <p className="font-semibold text-slate-300 mb-1">FRÉQUENT</p>
            <p>Killzone + Break & Retest</p>
          </div>
        </div>
      </div>
    </div>
  )
}
