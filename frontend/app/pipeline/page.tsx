import Link from "next/link"
import { fetchDaily, fetchPerformance, fetchQTable, fetchTrades } from "@/lib/api"

type DailyMacro = {
  mac_idx: number
  mac_name: string
  type: "live" | "shadow"
  status?: string
  would_trade?: boolean
  direction?: "LONG" | "SHORT" | "FLAT"
  q_val?: number
  flat_reason?: string
  lc_label?: string
  sc_label?: string
  pc_label?: string
  exit_reason?: string
  pnl?: number | null
}

const STEP_COLORS = [
  "#38bdf8",
  "#22c55e",
  "#f59e0b",
  "#fb7185",
] as const

function pct(part: number, total: number) {
  if (total <= 0) return 0
  return (part / total) * 100
}

function fmtPctSigned(n: number) {
  const v = (n * 100).toFixed(2)
  return n >= 0 ? `+${v}%` : `${v}%`
}

export default async function PipelinePage() {
  const today = new Date().toISOString().slice(0, 10)

  const [daily, tradesData, qtableData, perfData] = await Promise.all([
    fetchDaily(today).catch(() => null),
    fetchTrades().catch(() => ({ trades: [], total: 0, metrics: {} })),
    fetchQTable().catch(() => ({ states: [], total: 0 })),
    fetchPerformance().catch(() => null),
  ])

  const macros: DailyMacro[] = Array.isArray(daily?.macros) ? daily.macros : []
  const observedToday = macros.filter((m) => m.status !== "no_data")
  const tradedToday = observedToday.filter((m) => m.would_trade)
  const flatToday = observedToday.filter((m) => !m.would_trade)

  const flatReasonCount = flatToday.reduce<Record<string, number>>((acc, m) => {
    const key = (m.flat_reason || "unknown").toLowerCase()
    acc[key] = (acc[key] || 0) + 1
    return acc
  }, {})

  const trades = Array.isArray(tradesData?.trades) ? tradesData.trades : []
  const exits = trades.reduce((acc: Record<string, number>, t: any) => {
    const key = String(t.exit_reason || "NA").toUpperCase()
    acc[key] = (acc[key] || 0) + 1
    return acc
  }, {})

  const dirs = trades.reduce(
    (acc: { LONG: number; SHORT: number }, t: any) => {
      if (t.direction === "LONG") acc.LONG += 1
      if (t.direction === "SHORT") acc.SHORT += 1
      return acc
    },
    { LONG: 0, SHORT: 0 },
  )

  const states = Array.isArray(qtableData?.states) ? qtableData.states : []
  const qLongStates = states.filter((s: any) => s.best_action === "LONG").length
  const qShortStates = states.filter((s: any) => s.best_action === "SHORT").length

  const topEdges = [...states]
    .map((s: any) => ({
      state: s.state,
      mac_name: s.mac_name,
      best_action: s.best_action,
      q_long: Number(s.q_long || 0),
      q_short: Number(s.q_short || 0),
      edge: Math.max(Math.abs(Number(s.q_long || 0)), Math.abs(Number(s.q_short || 0))),
    }))
    .sort((a, b) => b.edge - a.edge)
    .slice(0, 8)

  const totalMacros = 7
  const stepSeed = totalMacros
  const stepObserved = observedToday.length
  const stepRules = flatToday.filter((m) => (m.flat_reason || "") !== "micro_gate").length
  const stepSignal = tradedToday.length

  return (
    <div>
      <div className="mb-6 flex items-center justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold text-white">Pipeline Decision Explorer</h1>
          <p className="text-sm text-slate-500 mt-0.5">Visualisation du traitement des informations: contexte, filtre et decision.</p>
        </div>
        <Link
          href={`/daily/${today}`}
          className="px-3 py-2 text-xs rounded-md text-cyan-200 bg-cyan-500/10 border border-cyan-400/20 hover:bg-cyan-500/15"
        >
          Voir le rapport du jour
        </Link>
      </div>

      <div className="rounded-xl border border-white/10 p-5 mb-6 bg-[radial-gradient(120%_120%_at_5%_0%,rgba(34,211,238,0.16),transparent_35%),radial-gradient(120%_120%_at_100%_100%,rgba(16,185,129,0.14),transparent_35%),#0b0f16]">
        <div className="grid grid-cols-4 gap-3">
          <MetricCard label="Macros detectees (jour)" value={`${stepObserved}/${totalMacros}`} hint="macros avec donnees du jour" />
          <MetricCard label="Signaux declenches" value={String(stepSignal)} hint="LONG/SHORT apres filtres" />
          <MetricCard label="Trades historiques" value={String(tradesData?.total || 0)} hint="echantillon frontend" />
          <MetricCard
            label="Return global"
            value={perfData?.overall?.total_return != null ? fmtPctSigned(perfData.overall.total_return) : "-"}
            hint="live + shadow"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <section className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
          <h2 className="text-sm font-semibold text-white mb-1">Entonnoir de traitement</h2>
          <p className="text-xs text-slate-500 mb-4">Du setup macro brut a la decision finale.</p>

          <div className="space-y-3">
            <FunnelStep label="1. Setups macro potentiels" value={stepSeed} total={stepSeed} color={STEP_COLORS[0]} />
            <FunnelStep label="2. Donnees observables" value={stepObserved} total={stepSeed} color={STEP_COLORS[1]} />
            <FunnelStep label="3. Passage regles (Q/rules)" value={stepRules} total={stepSeed} color={STEP_COLORS[2]} />
            <FunnelStep label="4. Signal trade final" value={stepSignal} total={stepSeed} color={STEP_COLORS[3]} />
          </div>
        </section>

        <section className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
          <h2 className="text-sm font-semibold text-white mb-1">Raisons de blocage (jour)</h2>
          <p className="text-xs text-slate-500 mb-4">Pourquoi un setup finit en FLAT.</p>

          <div className="space-y-3">
            {Object.keys(flatReasonCount).length === 0 && (
              <div className="text-xs text-slate-600">Aucun blocage detecte aujourd&apos;hui.</div>
            )}
            {(Object.entries(flatReasonCount) as Array<[string, number]>)
              .sort((a, b) => b[1] - a[1])
              .map(([reason, n]) => (
                <ReasonBar key={reason} label={reason} value={n} total={Math.max(1, flatToday.length)} />
              ))}
          </div>
        </section>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <section className="bg-[#0d0d14] border border-white/5 rounded-lg overflow-hidden">
          <div className="px-5 py-3 border-b border-white/5">
            <h2 className="text-sm font-semibold text-white">Etat de decision par macro (jour)</h2>
            <p className="text-xs text-slate-500 mt-1">Contexte London/Sweep/Pool + sortie decisionnelle.</p>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr>
                {[
                  "Macro",
                  "Type",
                  "Contexte",
                  "Decision",
                  "Q",
                  "Exit",
                ].map((h) => (
                  <th key={h} className="px-4 py-2.5 text-left text-[11px] font-medium text-slate-500 uppercase tracking-wider border-b border-white/5">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {macros.map((m) => {
                const isFlat = !m.would_trade
                const decision = m.status === "no_data" ? "NO_DATA" : isFlat ? "FLAT" : m.direction || "-"
                return (
                  <tr key={m.mac_idx} className="border-b border-white/[0.03] hover:bg-white/[0.015]">
                    <td className="px-4 py-2.5 mono text-slate-300">{m.mac_name}</td>
                    <td className="px-4 py-2.5">
                      <span className={m.type === "live" ? "badge-live" : "badge-shadow"}>
                        {(m.type || "shadow").toUpperCase()}
                      </span>
                    </td>
                    <td className="px-4 py-2.5">
                      <div className="flex flex-wrap gap-1.5">
                        <Chip text={m.lc_label || "?"} tone="slate" />
                        <Chip text={m.sc_label || "?"} tone="amber" />
                        <Chip text={m.pc_label || "?"} tone="cyan" />
                      </div>
                    </td>
                    <td className="px-4 py-2.5">
                      <span className={`mono text-xs px-2 py-0.5 rounded ${
                        decision === "LONG"
                          ? "bg-emerald-500/15 text-emerald-300"
                          : decision === "SHORT"
                            ? "bg-rose-500/15 text-rose-300"
                            : "bg-white/5 text-slate-400"
                      }`}>
                        {decision}
                      </span>
                      {isFlat && m.flat_reason && m.status !== "no_data" && (
                        <span className="ml-2 text-[11px] text-slate-500">({m.flat_reason})</span>
                      )}
                    </td>
                    <td className="px-4 py-2.5 mono text-slate-300">
                      {m.q_val != null ? `${(m.q_val * 100).toFixed(3)}%` : "-"}
                    </td>
                    <td className="px-4 py-2.5 mono text-slate-500">{m.exit_reason || "-"}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </section>

        <section className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
          <h2 className="text-sm font-semibold text-white mb-1">Q-Table Signal Map</h2>
          <p className="text-xs text-slate-500 mb-4">Distribution des etats actifs et edge dominant.</p>

          <div className="grid grid-cols-3 gap-2 mb-4">
            <MetricCard label="Etats actifs" value={String(qtableData?.total || 0)} small />
            <MetricCard label="Bias LONG" value={String(qLongStates)} small />
            <MetricCard label="Bias SHORT" value={String(qShortStates)} small />
          </div>

          <div className="space-y-2">
            {topEdges.map((s: any) => (
              <div key={s.state} className="rounded border border-white/10 px-3 py-2 bg-white/[0.02]">
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="mono text-slate-400">state {s.state} · {s.mac_name}</span>
                  <span className="text-slate-300">{s.best_action}</span>
                </div>
                <div className="w-full h-1.5 rounded bg-white/10 overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-cyan-400 to-emerald-400" style={{ width: `${Math.min(100, s.edge * 5000)}%` }} />
                </div>
              </div>
            ))}
            {topEdges.length === 0 && <div className="text-xs text-slate-600">Aucun etat actif disponible.</div>}
          </div>
        </section>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <section className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
          <h2 className="text-sm font-semibold text-white mb-1">Sorties des derniers trades</h2>
          <p className="text-xs text-slate-500 mb-4">Repartition TP / SL / EOD pour lecture execution.</p>
          <div className="space-y-3">
            {(Object.entries(exits) as Array<[string, number]>)
              .sort((a, b) => b[1] - a[1])
              .map(([reason, n]) => (
                <ReasonBar key={reason} label={reason} value={n} total={Math.max(1, trades.length)} />
              ))}
            {Object.keys(exits).length === 0 && <div className="text-xs text-slate-600">Pas de trades disponibles.</div>}
          </div>
        </section>

        <section className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
          <h2 className="text-sm font-semibold text-white mb-1">Direction du flux</h2>
          <p className="text-xs text-slate-500 mb-4">Orientation LONG/SHORT sur les derniers signaux.</p>
          <div className="space-y-3 mb-5">
            <ReasonBar label="LONG" value={dirs.LONG} total={Math.max(1, trades.length)} accent="emerald" />
            <ReasonBar label="SHORT" value={dirs.SHORT} total={Math.max(1, trades.length)} accent="rose" />
          </div>

          <div className="rounded-lg border border-white/10 p-3 bg-white/[0.02]">
            <p className="text-[11px] text-slate-500 mb-2">Lecture rapide</p>
            <p className="text-xs text-slate-300 leading-5">
              Ce panneau combine traitement contextuel (lc/sc/pc), filtre decisionnel (Q et gate), puis resultat d&apos;execution (exit reason).
              Il sert de couche de transparence pour comprendre comment une information brute devient un signal exploitable.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}

function MetricCard({
  label,
  value,
  hint,
  small = false,
}: {
  label: string
  value: string
  hint?: string
  small?: boolean
}) {
  return (
    <div className={`rounded-lg border border-white/10 bg-white/[0.02] ${small ? "px-3 py-2" : "px-4 py-3"}`}>
      <p className="text-[11px] text-slate-500">{label}</p>
      <p className={`${small ? "text-base" : "text-lg"} text-white font-semibold mt-1 mono`}>{value}</p>
      {hint && <p className="text-[10px] text-slate-600 mt-1">{hint}</p>}
    </div>
  )
}

function FunnelStep({
  label,
  value,
  total,
  color,
}: {
  label: string
  value: number
  total: number
  color: string
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1 text-[11px]">
        <span className="text-slate-300">{label}</span>
        <span className="mono text-slate-500">{value} ({pct(value, total).toFixed(0)}%)</span>
      </div>
      <div className="h-2 rounded bg-white/10 overflow-hidden">
        <div
          className="h-full rounded"
          style={{ width: `${pct(value, total)}%`, background: color }}
        />
      </div>
    </div>
  )
}

function Chip({ text, tone }: { text: string; tone: "slate" | "amber" | "cyan" }) {
  const cls =
    tone === "amber"
      ? "bg-amber-500/15 text-amber-200"
      : tone === "cyan"
        ? "bg-cyan-500/15 text-cyan-200"
        : "bg-slate-500/20 text-slate-200"

  return <span className={`text-[10px] px-2 py-0.5 rounded ${cls}`}>{text}</span>
}

function ReasonBar({
  label,
  value,
  total,
  accent = "cyan",
}: {
  label: string
  value: number
  total: number
  accent?: "cyan" | "emerald" | "rose"
}) {
  const color = accent === "emerald" ? "bg-emerald-400" : accent === "rose" ? "bg-rose-400" : "bg-cyan-400"
  return (
    <div>
      <div className="flex items-center justify-between text-[11px] mb-1">
        <span className="text-slate-300 uppercase tracking-wide">{label}</span>
        <span className="mono text-slate-500">{value} ({pct(value, total).toFixed(0)}%)</span>
      </div>
      <div className="h-2 rounded bg-white/10 overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${pct(value, total)}%` }} />
      </div>
    </div>
  )
}
