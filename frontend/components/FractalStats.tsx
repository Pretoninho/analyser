"use client"

import StatCard from "@/components/StatCard"

interface FractalStatsProps {
  totalSignals: number
  bySetup: Record<string, number>
}

export default function FractalStats({ totalSignals, bySetup }: FractalStatsProps) {
  const strict = bySetup["STRICT"] || 0
  const modere = bySetup["MODÉRÉ"] || 0
  const frequent = bySetup["FRÉQUENT"] || 0

  return (
    <div className="grid grid-cols-4 gap-3">
      <StatCard label="Total signaux" value={String(totalSignals)} />
      <StatCard label="STRICT" value={String(strict)} />
      <StatCard label="MODÉRÉ" value={String(modere)} />
      <StatCard label="FRÉQUENT" value={String(frequent)} />
    </div>
  )
}
