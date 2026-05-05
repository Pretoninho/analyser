"use client"

import { useState, useEffect, useMemo, useCallback } from "react"
import { fetchVolSnapshot, fetchAdvisor, AdvisorResponse } from "@/lib/api"

// ═══════════════════════════════════════════════════════════════
// BLACK-SCHOLES MATH
// ═══════════════════════════════════════════════════════════════

// Approximation Abramowitz & Stegun (error < 7.5e-8)
function ncdf(x: number): number {
  const a = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
  const k = 1 / (1 + 0.2316419 * Math.abs(x))
  let p = k * a[0]; let kk = k
  for (let i = 1; i < 5; i++) { kk *= k; p += kk * a[i] }
  const n = 1 - (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * x * x) * p
  return x >= 0 ? n : 1 - n
}

function npdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)
}

function d1(S: number, K: number, T: number, r: number, q: number, v: number): number {
  return (Math.log(S / K) + (r - q + 0.5 * v * v) * T) / (v * Math.sqrt(T))
}

function bsPrice(S: number, K: number, T: number, r: number, q: number, v: number, call: boolean): number {
  if (T <= 0) return Math.max(0, call ? S - K : K - S)
  if (v <= 0)  return Math.max(0, call ? S * Math.exp(-q * T) - K * Math.exp(-r * T)
                                       : K * Math.exp(-r * T) - S * Math.exp(-q * T))
  const D1 = d1(S, K, T, r, q, v), D2 = D1 - v * Math.sqrt(T)
  const eq = Math.exp(-q * T), er = Math.exp(-r * T)
  return call
    ? S * eq * ncdf(D1)  - K * er * ncdf(D2)
    : K * er * ncdf(-D2) - S * eq * ncdf(-D1)
}

function bsGreeks(S: number, K: number, T: number, r: number, q: number, v: number, call: boolean) {
  if (T <= 0 || v <= 0) return { delta: call ? 1 : -1, gamma: 0, vega: 0, theta: 0, rho: 0 }
  const D1 = d1(S, K, T, r, q, v), D2 = D1 - v * Math.sqrt(T)
  const sT = Math.sqrt(T), pdf1 = npdf(D1)
  const eq = Math.exp(-q * T), er = Math.exp(-r * T)
  const delta = call ? eq * ncdf(D1)  : -eq * ncdf(-D1)
  const gamma = eq * pdf1 / (S * v * sT)
  const vega  = S * eq * pdf1 * sT / 100           // per 1% IV
  const theta = call
    ? (-S * eq * pdf1 * v / (2 * sT) - r * K * er * ncdf(D2)  + q * S * eq * ncdf(D1))  / 365
    : (-S * eq * pdf1 * v / (2 * sT) + r * K * er * ncdf(-D2) - q * S * eq * ncdf(-D1)) / 365
  const rho = call
    ?  K * T * er * ncdf(D2)  / 100
    : -K * T * er * ncdf(-D2) / 100
  return { delta, gamma, vega, theta, rho }
}

// Newton-Raphson IV solver
function ivSolve(mkt: number, S: number, K: number, T: number, r: number, q: number, call: boolean): number | null {
  if (mkt <= 0 || T <= 0) return null
  let v = 0.5
  for (let i = 0; i < 200; i++) {
    const p = bsPrice(S, K, T, r, q, v, call)
    const vg = bsGreeks(S, K, T, r, q, v, call).vega * 100
    const err = p - mkt
    if (Math.abs(err) < 1e-6) return v
    if (Math.abs(vg) < 1e-10) break
    v -= err / vg
    if (v <= 0.001) v = 0.001
    if (v > 20) v = 20
  }
  return v > 0 ? v : null
}

// Probability ITM (risk-neutral)
function probITM(S: number, K: number, T: number, r: number, q: number, v: number, call: boolean): number {
  if (T <= 0 || v <= 0) return call ? (S > K ? 1 : 0) : (S < K ? 1 : 0)
  const D2 = d1(S, K, T, r, q, v) - v * Math.sqrt(T)
  return call ? ncdf(D2) : ncdf(-D2)
}

// ═══════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════

type LegSide = "long" | "short"
type OptType = "call" | "put"
type Tab = "advisor" | "pricer" | "position" | "probas" | "gestion" | "aide" | "regles"

// ── Gestion — types ───────────────────────────────────────────────────────────

interface GestionLeg {
  id: number
  type: OptType
  action: "BUY" | "SELL"
  strike: number
  entryPremium: number  // prime encaissée (SELL) ou payée (BUY)
}

type AlertLevel = "URGENT" | "WARNING" | "SUCCESS" | "INFO"

interface Alert {
  level: AlertLevel
  label: string
  detail: string
  action: string
}

const ADVISOR_ASSETS = ["BTC", "ETH"]

// ── Kelly Sizing ──────────────────────────────────────────────────────────────

type KellyFraction = 1 | 0.5 | 0.25

interface KellyParams { p: number; b: number; hint: string }

function estimateKellyParams(strategy: string): KellyParams {
  switch (strategy) {
    case "SHORT_PUT":
    case "SHORT_CALL":
      return { p: 0.75, b: 0.50, hint: "25δ OTM · TP 50% prime / SL 2× prime" }
    case "SHORT_STRANGLE":
      return { p: 0.60, b: 0.50, hint: "Double 25δ · TP 50% / SL 2× prime totale" }
    case "IRON_CONDOR":
      return { p: 0.65, b: 0.70, hint: "Condor · risque limité par les ailes" }
    case "LONG_STRADDLE":
    case "LONG_STRANGLE":
      return { p: 0.40, b: 2.50, hint: "Achat vol · gain illimité, perte = prime payée" }
    case "BULL_CALL_SPREAD":
    case "BEAR_PUT_SPREAD":
      return { p: 0.50, b: 1.20, hint: "Spread débit · risque = prime payée" }
    default:
      return { p: 0.50, b: 1.00, hint: "Estimations neutres par défaut" }
  }
}

function kellyCalc(p: number, b: number, fraction: KellyFraction, capital: number) {
  const q = 1 - p
  const fStar = (p * b - q) / b        // Kelly optimal brut
  const fApplied = Math.max(0, fStar) * fraction
  const maxRisk = capital * fApplied
  return { fStar, fApplied, maxRisk, positive: fStar > 0 }
}

// Couleurs de l'advisor selon la stratégie
function clsAdvisor(color: string) {
  if (color === "emerald") return "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
  if (color === "amber")   return "border-amber-500/40 bg-amber-500/10 text-amber-300"
  if (color === "cyan")    return "border-cyan-500/40 bg-cyan-500/10 text-cyan-300"
  if (color === "violet")  return "border-violet-500/40 bg-violet-500/10 text-violet-300"
  return "border-white/10 bg-white/5 text-slate-400"
}

function IvpGauge({ ivp, ivpPct }: { ivp: number | null; ivpPct: number | null }) {
  if (ivp === null || ivpPct === null) return (
    <div className="text-xs text-slate-500 italic">DVOL non disponible pour cet actif</div>
  )
  const pct = ivpPct
  const color = pct > 70 ? "bg-rose-400" : pct < 30 ? "bg-emerald-400" : "bg-amber-400"
  const label = pct > 70 ? "Vol chère → SELL VOL" : pct < 30 ? "Vol bon marché → BUY VOL" : "Vol neutre"
  return (
    <div>
      <div className="flex justify-between text-xs mb-1.5">
        <span className="text-slate-400">IV Percentile (52 semaines)</span>
        <span className="text-white mono font-semibold">{pct.toFixed(0)}%</span>
      </div>
      <div className="h-2 rounded bg-white/10 overflow-hidden mb-1">
        <div className={`h-full rounded transition-all ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <div className="flex justify-between text-[10px] text-slate-500">
        <span>0 — BUY VOL</span>
        <span className={`font-medium ${pct > 70 ? "text-rose-300" : pct < 30 ? "text-emerald-300" : "text-amber-300"}`}>{label}</span>
        <span>100 — SELL VOL</span>
      </div>
    </div>
  )
}

// Guide contextuel pour les alertes Gestion
function alertGuide(a: Alert): { rule: string; explain: string; showRoll: boolean } | null {
  if (a.label === "Stop Loss") return {
    rule: "Règle des 2× la prime",
    explain: "La prime a doublé contre toi — le marché invalide ta thèse. Racheter maintenant pour limiter la perte à 1× la prime initiale. Chaque heure d'attente aggrave l'exposition au gamma.",
    showRoll: false,
  }
  if (a.label === "TP atteint") return {
    rule: "Règle des 50% profit",
    explain: "Tu as encaissé la moitié de la prime maximale. Racheter maintenant : libère le capital et élimine le risque de retournement brutal sur la deuxième moitié.",
    showRoll: false,
  }
  if (a.label === "Zone gamma critique") return {
    rule: "Règle du gamma terminal (< 7 DTE)",
    explain: "Le gamma explose sous 7 DTE. Un gap de 3-5% peut retourner une position gagnante en quelques heures. Fermer immédiatement ou rouler vers l'échéance suivante.",
    showRoll: true,
  }
  if (a.label.startsWith("DTE")) return {
    rule: "Roll à 21 DTE",
    explain: "La zone optimale pour rouler est 14-21 DTE. Si la position est profitable, racheter et ouvrir un nouveau cycle à 30-45 DTE pour recharger le theta.",
    showRoll: true,
  }
  if (a.label === "Delta drift") return {
    rule: "Règle du delta 0.30",
    explain: "Le delta a dérivé de plus de 0.20 depuis l'entrée. Ta position n'est plus au niveau de risque initial. Rouler le strike vers un nouveau 25δ ou acheter un hedge delta.",
    showRoll: true,
  }
  if (a.label.includes("touché") || a.label.includes("menacé")) return {
    rule: "Strike en danger — Roll immédiat",
    explain: "Le spot s'approche de ton strike vendu. La probabilité d'assignation augmente exponentiellement. Rouler vers un strike plus OTM avant que le delta passe > 0.40.",
    showRoll: true,
  }
  return null
}

interface Leg {
  id: number
  type: OptType
  side: LegSide
  strike: number
  qty: number
  premium: number
}

// ═══════════════════════════════════════════════════════════════
// PRESETS
// ═══════════════════════════════════════════════════════════════

function makePreset(name: string, S: number, iv: number, T: number, r: number, q: number): Leg[] {
  const atm = Math.round(S / 1000) * 1000
  const otm = Math.round(S * 1.05 / 1000) * 1000
  const otm2 = Math.round(S * 1.10 / 1000) * 1000
  const itm = Math.round(S * 0.95 / 1000) * 1000
  const itm2 = Math.round(S * 0.90 / 1000) * 1000
  const p = (K: number, c: boolean) => parseFloat(bsPrice(S, K, T, r, q, iv, c).toFixed(2))
  let id = 1
  const L = (type: OptType, side: LegSide, K: number): Leg => ({
    id: id++, type, side, strike: K, qty: 1, premium: p(K, type === "call")
  })
  switch (name) {
    case "straddle":    return [L("call","long",atm), L("put","long",atm)]
    case "strangle":    return [L("call","long",otm), L("put","long",itm)]
    case "iron_condor": return [L("put","short",itm), L("put","long",itm2), L("call","short",otm), L("call","long",otm2)]
    case "bull_spread":  return [L("call","long",atm), L("call","short",otm)]
    case "bear_spread":  return [L("put","long",atm), L("put","short",itm)]
    case "butterfly":    return [L("call","long",itm), L("call","short",atm), L("call","short",atm), L("call","long",otm)]
    default: return []
  }
}

// ═══════════════════════════════════════════════════════════════
// P&L DIAGRAM (SVG)
// ═══════════════════════════════════════════════════════════════

function PnLDiagram({ legs, spot }: { legs: Leg[]; spot: number }) {
  const W = 580, H = 200, PAD = { t: 10, b: 30, l: 50, r: 10 }
  const inner = { w: W - PAD.l - PAD.r, h: H - PAD.t - PAD.b }

  const spots = useMemo(() => {
    const n = 120
    return Array.from({ length: n }, (_, i) => spot * (0.7 + 0.6 * i / (n - 1)))
  }, [spot])

  const payoffs = useMemo(() => spots.map(s => {
    return legs.reduce((sum, leg) => {
      const intrinsic = leg.type === "call" ? Math.max(0, s - leg.strike) : Math.max(0, leg.strike - s)
      const pnl = (intrinsic - leg.premium) * leg.qty * (leg.side === "long" ? 1 : -1)
      return sum + pnl
    }, 0)
  }), [legs, spots])

  if (legs.length === 0) return (
    <div className="flex items-center justify-center h-[200px] text-slate-600 text-sm">
      Ajoutez des legs pour voir le P&L
    </div>
  )

  const minP = Math.min(...payoffs), maxP = Math.max(...payoffs)
  const range = maxP - minP || 1
  const yPad = range * 0.15
  const yMin = minP - yPad, yMax = maxP + yPad

  const toX = (i: number) => PAD.l + (i / (spots.length - 1)) * inner.w
  const toY = (v: number) => PAD.t + ((yMax - v) / (yMax - yMin)) * inner.h
  const zeroY = toY(0)

  // Build path
  const pathPos: string[] = [], pathNeg: string[] = []
  for (let i = 0; i < spots.length; i++) {
    const x = toX(i), y = toY(payoffs[i])
    const cmd = i === 0 ? "M" : "L"
    if (payoffs[i] >= 0) pathPos.push(`${cmd}${x},${y}`)
    else pathNeg.push(`${cmd}${x},${y}`)
  }

  // Breakevens
  const bes: number[] = []
  for (let i = 1; i < payoffs.length; i++) {
    if (payoffs[i - 1] * payoffs[i] < 0) {
      const t = -payoffs[i - 1] / (payoffs[i] - payoffs[i - 1])
      bes.push(spots[i - 1] + t * (spots[i] - spots[i - 1]))
    }
  }

  const spotX = toX(spots.findIndex(s => s >= spot) || 0)
  const fmt = (v: number) => v > 1000 ? `${(v/1000).toFixed(1)}k` : v.toFixed(0)

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }}>
      {/* Zero line */}
      <line x1={PAD.l} y1={zeroY} x2={W - PAD.r} y2={zeroY} stroke="#ffffff18" strokeWidth={1} />
      {/* Payoff paths */}
      {pathPos.length > 0 && <path d={pathPos.join(" ")} fill="none" stroke="#34d399" strokeWidth={2} strokeLinejoin="round" />}
      {pathNeg.length > 0 && <path d={pathNeg.join(" ")} fill="none" stroke="#f87171" strokeWidth={2} strokeLinejoin="round" />}
      {/* Current spot */}
      <line x1={spotX} y1={PAD.t} x2={spotX} y2={H - PAD.b} stroke="#94a3b8" strokeWidth={1} strokeDasharray="4,3" />
      <text x={spotX + 3} y={PAD.t + 10} fontSize={9} fill="#94a3b8">{fmt(spot)}</text>
      {/* Breakevens */}
      {bes.map((be, i) => {
        const bx = PAD.l + ((be - spots[0]) / (spots[spots.length - 1] - spots[0])) * inner.w
        return <g key={i}>
          <circle cx={bx} cy={zeroY} r={3} fill="#fbbf24" />
          <text x={bx} y={H - PAD.b + 12} fontSize={8} fill="#fbbf24" textAnchor="middle">{fmt(be)}</text>
        </g>
      })}
      {/* Y labels */}
      {[yMax, 0, yMin].map((v, i) => (
        <text key={i} x={PAD.l - 4} y={toY(v) + 4} fontSize={8} fill="#64748b" textAnchor="end">
          {v.toFixed(0)}
        </text>
      ))}
      {/* X labels */}
      {[0.75, 1.0, 1.25].map((f, i) => {
        const v = spot * f
        const x = PAD.l + ((v - spots[0]) / (spots[spots.length - 1] - spots[0])) * inner.w
        return <text key={i} x={x} y={H - 2} fontSize={8} fill="#64748b" textAnchor="middle">{fmt(v)}</text>
      })}
    </svg>
  )
}

// ═══════════════════════════════════════════════════════════════
// HELPERS UI
// ═══════════════════════════════════════════════════════════════

const inp = "bg-[#0d0d14] border border-white/10 rounded px-3 py-1.5 text-sm text-white mono w-full focus:outline-none focus:border-indigo-500/60"
const btn = (active?: boolean) => `px-3 py-1.5 text-xs rounded border transition-colors ${active ? "bg-indigo-600 border-indigo-500 text-white" : "border-white/10 text-slate-400 hover:border-white/20 hover:text-slate-300"}`
const label = "text-[11px] text-slate-500 mb-1 block"

function fmt2(v: number, d = 2) { return isFinite(v) ? v.toFixed(d) : "—" }
function fmtSign(v: number, d = 4) { return (v >= 0 ? "+" : "") + fmt2(v, d) }

function GreekRow({ name, value, hint }: { name: string; value: number; hint: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-white/[0.04]">
      <div>
        <span className="text-sm text-slate-300 mono">{name}</span>
        <span className="ml-2 text-[10px] text-slate-600">{hint}</span>
      </div>
      <span className={`text-sm mono font-medium ${Math.abs(value) < 0.0001 ? "text-slate-500" : "text-white"}`}>
        {fmtSign(value, 4)}
      </span>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════
// PAGE
// ═══════════════════════════════════════════════════════════════

export default function OptionsPage() {
  // Global params
  const [spot, setSpot] = useState(78000)
  const [dte,  setDte]  = useState(7)
  const [ivPct, setIvPct] = useState(40)    // % — e.g. 40 = 40%
  const [rPct,  setRPct]  = useState(5)     // % risk-free
  const [qPct,  setQPct]  = useState(0)     // % funding/dividend

  const T = dte / 365
  const iv = ivPct / 100
  const r  = rPct  / 100
  const q  = qPct  / 100

  const [apiLoading, setApiLoading] = useState(true)
  const [tab, setTab] = useState<Tab>("advisor")
  const [advisorAsset, setAdvisorAsset] = useState("BTC")
  const [advisor, setAdvisor] = useState<AdvisorResponse | null>(null)
  const [advisorLoading, setAdvisorLoading] = useState(false)
  const [advisorError, setAdvisorError] = useState<string | null>(null)

  // Kelly sizing — persisté dans localStorage
  const [capital, setCapital] = useState(10000)
  const [kellyFraction, setKellyFraction] = useState<KellyFraction>(0.25)
  const [winRateStr, setWinRateStr] = useState("")   // vide = auto depuis stratégie
  const [ratioStr, setRatioStr] = useState("")       // vide = auto depuis stratégie

  // Charger capital/fraction depuis localStorage au montage
  useEffect(() => {
    const c = localStorage.getItem("kelly_capital")
    const f = localStorage.getItem("kelly_fraction")
    if (c) setCapital(Number(c))
    if (f) setKellyFraction(Number(f) as KellyFraction)
  }, [])

  const updateCapital = (v: number) => { setCapital(v); localStorage.setItem("kelly_capital", String(v)) }
  const updateFraction = (v: KellyFraction) => { setKellyFraction(v); localStorage.setItem("kelly_fraction", String(v)) }

  // Gestion state
  const [gLegs, setGLegs] = useState<GestionLeg[]>([])
  const [gNextId, setGNextId] = useState(1)
  const [gEntryIv, setGEntryIv] = useState(40)        // IV % au moment de l'entrée
  const [gDteEntry, setGDteEntry] = useState(21)       // DTE à l'entrée
  const [gDteLeft, setGDteLeft] = useState(14)         // DTE restant maintenant
  const [gNewLeg, setGNewLeg] = useState<Omit<GestionLeg, "id">>({
    type: "put", action: "SELL", strike: 78000, entryPremium: 0,
  })

  // Roll simulator state
  const [rollType, setRollType] = useState<OptType>("put")
  const [rollStrikeFrom, setRollStrikeFrom] = useState(78000)
  const [rollDteFrom, setRollDteFrom] = useState(21)
  const [rollPremiumReceived, setRollPremiumReceived] = useState(500)
  const [rollStrikeTo, setRollStrikeTo] = useState(75000)
  const [rollDteTo, setRollDteTo] = useState(45)

  // Checklist pré-trade
  const [checks, setChecks] = useState<Record<string, boolean>>({})
  const toggleCheck = (k: string) => setChecks(c => ({ ...c, [k]: !c[k] }))

  // Ouvrir le simulateur de roll depuis une alerte avec pré-remplissage
  const openRollSimulator = useCallback((leg: GestionLeg) => {
    setRollType(leg.type)
    setRollStrikeFrom(leg.strike)
    setRollDteFrom(gDteLeft)
    setRollPremiumReceived(leg.entryPremium)
    const step = leg.type === "put" ? 0.95 : 1.05
    setRollStrikeTo(Math.round(leg.strike * step / 1000) * 1000)
    setRollDteTo(45)
    setTab("regles")
  }, [gDteLeft])

  // Pré-remplir depuis l'Advisor
  const prefillFromAdvisor = useCallback(() => {
    if (!advisor || advisor.strategy === "WAIT") return
    let id = 1
    const legs: GestionLeg[] = advisor.legs.map(l => ({
      id: id++,
      type: l.type.toLowerCase() as OptType,
      action: l.action as "BUY" | "SELL",
      strike: l.strike,
      entryPremium: 0,
    }))
    setGLegs(legs)
    setGNextId(id)
    if (advisor.iv_atm) setGEntryIv(parseFloat(advisor.iv_atm.toFixed(1)))
    setGDteEntry(advisor.dte_days)
    setGDteLeft(Math.round(advisor.dte_days * 0.67))
  }, [advisor])

  // Calcul des alertes
  const gAlerts = useMemo((): Alert[] => {
    if (gLegs.length === 0) return []
    const alerts: Alert[] = []
    const T     = gDteLeft / 365
    const Tentry = gDteEntry / 365
    const sigNow   = iv / 1      // iv global (déjà en décimal)
    const sigEntry = gEntryIv / 100

    // P&L global
    let pnl = 0
    let totalPremium = 0
    for (const leg of gLegs) {
      const sign = leg.action === "SELL" ? 1 : -1
      const currentMark = bsPrice(spot, leg.strike, Math.max(T, 0.001), r, q, sigNow, leg.type === "call")
      const entryMark   = leg.entryPremium
      pnl += sign * (entryMark - currentMark)
      totalPremium += leg.entryPremium
    }

    // TP / SL
    const tp = totalPremium * 0.50
    const sl = totalPremium * 2.00
    if (pnl >= tp) {
      alerts.push({ level: "SUCCESS", label: "TP atteint", detail: `P&L +${pnl.toFixed(2)} ≥ 50% prime (${tp.toFixed(2)})`, action: "Racheter la position maintenant" })
    } else if (pnl <= -sl) {
      alerts.push({ level: "URGENT", label: "Stop Loss", detail: `Perte −${Math.abs(pnl).toFixed(2)} ≥ 2× prime (${sl.toFixed(2)})`, action: "Fermer la position immédiatement" })
    }

    // DTE critique
    if (gDteLeft <= 7) {
      alerts.push({ level: "WARNING", label: "Zone gamma critique", detail: `DTE = ${gDteLeft}j — gamma élevé, risque de gap`, action: "Fermer ou rouler à l'échéance suivante" })
    } else if (gDteLeft <= 14) {
      alerts.push({ level: "INFO", label: "DTE court", detail: `${gDteLeft}j restants — surveiller la gestion`, action: "Préparer le roll si position profitable" })
    }

    // Strikes en danger + delta drift
    for (const leg of gLegs) {
      if (leg.strike <= 0) continue
      const distPct = Math.abs(spot - leg.strike) / spot * 100
      if (leg.action === "SELL") {
        if (distPct < 3) {
          alerts.push({ level: "URGENT", label: `Strike ${leg.type.toUpperCase()} ${leg.strike.toLocaleString()} touché`, detail: `Spot à ${distPct.toFixed(1)}% du strike vendu`, action: "Rouler le strike immédiatement" })
        } else if (distPct < 8) {
          alerts.push({ level: "WARNING", label: `${leg.type.toUpperCase()} ${leg.strike.toLocaleString()} menacé`, detail: `Spot à ${distPct.toFixed(1)}% du strike vendu`, action: "Surveiller, préparer le roll" })
        }
      }

      // Delta drift
      const deltaEntry = bsGreeks(spot, leg.strike, Tentry, r, q, sigEntry, leg.type === "call").delta
      const deltaNow   = bsGreeks(spot, leg.strike, Math.max(T, 0.001), r, q, sigNow,  leg.type === "call").delta
      const drift = Math.abs(deltaNow - deltaEntry)
      if (drift > 0.20) {
        alerts.push({ level: "WARNING", label: "Delta drift", detail: `${leg.type.toUpperCase()} ${leg.strike} : δ ${deltaEntry.toFixed(2)} → ${deltaNow.toFixed(2)} (drift ${drift.toFixed(2)})`, action: "Rouler ou hedger le delta" })
      }
    }

    if (alerts.length === 0) {
      alerts.push({ level: "INFO", label: "Position saine", detail: `P&L ${pnl >= 0 ? "+" : ""}${pnl.toFixed(2)} · DTE ${gDteLeft}j · pas de seuil atteint`, action: "Continuer à surveiller" })
    }

    return alerts.sort((a, b) => {
      const order = { URGENT: 0, WARNING: 1, SUCCESS: 2, INFO: 3 }
      return order[a.level] - order[b.level]
    })
  }, [gLegs, spot, iv, r, q, gEntryIv, gDteEntry, gDteLeft])

  // Pricer state
  const [strike, setStrike] = useState(78000)
  const [optType, setOptType] = useState<OptType>("call")
  const [mktPriceStr, setMktPriceStr] = useState("")
  const [solvedIV, setSolvedIV] = useState<number | null>(null)

  // Position state
  const [legs, setLegs] = useState<Leg[]>([])
  const [nextId, setNextId] = useState(1)
  const [newLeg, setNewLeg] = useState<Omit<Leg, "id">>({
    type: "call", side: "long", strike: 78000, qty: 1, premium: 0
  })

  // Auto-fill from live API + charger l'advisor initial
  useEffect(() => {
    fetchVolSnapshot().then(snap => {
      if (snap.signal && !("error" in snap.signal)) {
        setSpot(Math.round(snap.signal.close))
        setStrike(Math.round(snap.signal.close))
        setNewLeg(l => ({ ...l, strike: Math.round(snap.signal.close) }))
        const ivAtm = snap.signal.options?.iv_atm
        if (ivAtm) setIvPct(parseFloat(ivAtm.toFixed(1)))
        const fund = snap.signal.funding_annualized
        if (fund) setQPct(parseFloat((fund * 100).toFixed(2)))
      }
    }).catch(() => {}).finally(() => setApiLoading(false))
  }, [])

  const loadAdvisor = useCallback((asset: string) => {
    setAdvisorLoading(true)
    setAdvisorError(null)
    fetchAdvisor(asset).then(data => {
      setAdvisor(data)
      // Pré-remplir les paramètres globaux depuis l'advisor
      if (data.spot) {
        setSpot(Math.round(data.spot))
        setStrike(Math.round(data.spot))
        setNewLeg(l => ({ ...l, strike: Math.round(data.spot!) }))
      }
      if (data.iv_atm) setIvPct(parseFloat(data.iv_atm.toFixed(1)))
    }).catch(e => setAdvisorError(e instanceof Error ? e.message : "Erreur"))
      .finally(() => setAdvisorLoading(false))
  }, [])

  useEffect(() => { loadAdvisor(advisorAsset) }, [advisorAsset, loadAdvisor])

  // IV Solver
  const handleSolveIV = useCallback(() => {
    const mkt = parseFloat(mktPriceStr)
    if (!isFinite(mkt) || mkt <= 0) return
    const solved = ivSolve(mkt, spot, strike, T, r, q, optType === "call")
    setSolvedIV(solved)
  }, [mktPriceStr, spot, strike, T, r, q, optType])

  const price  = useMemo(() => bsPrice(spot, strike, T, r, q, iv, optType === "call"), [spot, strike, T, r, q, iv, optType])
  const greeks = useMemo(() => bsGreeks(spot, strike, T, r, q, iv, optType === "call"), [spot, strike, T, r, q, iv, optType])

  // Position metrics
  const posMetrics = useMemo(() => {
    if (legs.length === 0) return null
    const spots = Array.from({ length: 300 }, (_, i) => spot * (0.5 + i / 300))
    const payoffs = spots.map(s =>
      legs.reduce((sum, leg) => {
        const intr = leg.type === "call" ? Math.max(0, s - leg.strike) : Math.max(0, leg.strike - s)
        return sum + (intr - leg.premium) * leg.qty * (leg.side === "long" ? 1 : -1)
      }, 0)
    )
    const maxP = Math.max(...payoffs), minP = Math.min(...payoffs)
    const bes: number[] = []
    for (let i = 1; i < payoffs.length; i++) {
      if (payoffs[i - 1] * payoffs[i] < 0) {
        const t = -payoffs[i - 1] / (payoffs[i] - payoffs[i - 1])
        bes.push(spots[i - 1] + t * (spots[i] - spots[i - 1]))
      }
    }
    const totalPremium = legs.reduce((s, l) => s + l.premium * l.qty * (l.side === "long" ? -1 : 1), 0)
    const deltaAgg = legs.reduce((s, l) => {
      const g = bsGreeks(spot, l.strike, T, r, q, iv, l.type === "call")
      return s + g.delta * l.qty * (l.side === "long" ? 1 : -1)
    }, 0)
    return { maxP, minP, bes, totalPremium, deltaAgg }
  }, [legs, spot, T, r, q, iv])

  // Probas
  const em1 = spot * (Math.exp(iv * Math.sqrt(T)) - 1)
  const em2 = spot * (Math.exp(2 * iv * Math.sqrt(T)) - 1)
  const probStrikesData = useMemo(() => {
    const strikes = [-20, -15, -10, -5, 0, 5, 10, 15, 20].map(pct => spot * (1 + pct / 100))
    return strikes.map(K => ({
      K: Math.round(K),
      pct: ((K - spot) / spot * 100).toFixed(0),
      probCall: probITM(spot, K, T, r, q, iv, true),
      probPut:  probITM(spot, K, T, r, q, iv, false),
    }))
  }, [spot, T, r, q, iv])

  const addLeg = () => {
    const premium = newLeg.premium > 0 ? newLeg.premium
      : parseFloat(bsPrice(spot, newLeg.strike, T, r, q, iv, newLeg.type === "call").toFixed(2))
    setLegs(l => [...l, { ...newLeg, id: nextId, premium }])
    setNextId(n => n + 1)
  }

  const applyPreset = (name: string) => {
    setLegs(makePreset(name, spot, iv, T, r, q))
    setNextId(10)
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-lg font-semibold text-white">Options Calculator</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Black-Scholes · Greeks · Position Builder · Probabilités
            {apiLoading && <span className="ml-2 text-indigo-400">· chargement données live...</span>}
          </p>
        </div>
      </div>

      {/* Global params */}
      <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-4">
        <p className="text-[11px] text-slate-500 mb-3 uppercase tracking-wider">Paramètres globaux</p>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
          <div>
            <label className={label}>Spot (S)</label>
            <input className={inp} type="number" value={spot} onChange={e => setSpot(+e.target.value)} />
          </div>
          <div>
            <label className={label}>DTE (jours)</label>
            <input className={inp} type="number" value={dte} min={1} onChange={e => setDte(+e.target.value)} />
          </div>
          <div>
            <label className={label}>IV (%)</label>
            <input className={inp} type="number" value={ivPct} step={0.5} onChange={e => setIvPct(+e.target.value)} />
          </div>
          <div>
            <label className={label}>Taux r (%)</label>
            <input className={inp} type="number" value={rPct} step={0.1} onChange={e => setRPct(+e.target.value)} />
          </div>
          <div>
            <label className={label}>Funding q (%)</label>
            <input className={inp} type="number" value={qPct} step={0.01} onChange={e => setQPct(+e.target.value)} />
          </div>
        </div>
        <p className="text-[10px] text-slate-600 mt-2">
          T = {T.toFixed(5)} ans · IV décimal = {iv.toFixed(4)} · pré-rempli depuis Deribit live
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2">
        {(["advisor", "pricer", "position", "probas", "gestion", "aide", "regles"] as Tab[]).map(t => (
          <button key={t} onClick={() => setTab(t)} className={btn(tab === t)}>
            {t === "advisor" ? "Advisor" : t === "pricer" ? "Pricer" : t === "position" ? "Position" : t === "probas" ? "Probas" : t === "gestion" ? "Gestion" : t === "aide" ? "? Aide" : "⚠ Règles"}
          </button>
        ))}
      </div>

      {/* ── TAB ADVISOR ── */}
      {tab === "advisor" && (
        <div className="space-y-4">

          {/* Sélecteur d'actif */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500">Actif :</span>
            {ADVISOR_ASSETS.map(a => (
              <button key={a} onClick={() => setAdvisorAsset(a)} className={btn(advisorAsset === a)}>
                {a}
              </button>
            ))}
            <button
              onClick={() => loadAdvisor(advisorAsset)}
              className="ml-auto text-xs text-slate-400 hover:text-slate-200 border border-white/10 rounded px-3 py-1.5 transition-colors"
            >
              Actualiser
            </button>
          </div>

          {advisorLoading && (
            <div className="flex items-center justify-center h-40 text-slate-500 text-sm">
              Analyse en cours…
            </div>
          )}

          {advisorError && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-sm text-red-300">
              {advisorError}
            </div>
          )}

          {!advisorLoading && advisor && (
            <div className="space-y-4">

              {/* IVP gauge */}
              <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
                <h2 className="text-sm font-semibold text-white mb-3">IV Percentile — {advisor.asset}</h2>
                <IvpGauge ivp={advisor.ivp} ivpPct={advisor.ivp_pct} />
              </div>

              {/* Contexte de marché */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="bg-[#0d0d14] border border-white/5 rounded-lg px-4 py-3">
                  <p className="text-[11px] text-slate-500 mb-1">Régime vol</p>
                  <p className={`text-sm font-semibold mono ${
                    advisor.vol_regime === "SELL_VOL" ? "text-rose-300" :
                    advisor.vol_regime === "BUY_VOL"  ? "text-emerald-300" : "text-slate-400"
                  }`}>{advisor.vol_regime.replace("_", " ")}</p>
                </div>
                <div className="bg-[#0d0d14] border border-white/5 rounded-lg px-4 py-3">
                  <p className="text-[11px] text-slate-500 mb-1">Biais directionnel</p>
                  <p className={`text-sm font-semibold mono ${
                    advisor.directional_bias === "BULLISH" ? "text-emerald-300" :
                    advisor.directional_bias === "BEARISH" ? "text-rose-300" : "text-slate-400"
                  }`}>{advisor.directional_bias}</p>
                </div>
                <div className="bg-[#0d0d14] border border-white/5 rounded-lg px-4 py-3">
                  <p className="text-[11px] text-slate-500 mb-1">DVOL / Signal</p>
                  <p className="text-sm mono text-slate-300">
                    {advisor.dvol_close ? advisor.dvol_close.toFixed(1) : "—"}
                    <span className="text-slate-500 ml-1 text-xs">· {advisor.signal_action}</span>
                  </p>
                </div>
              </div>

              {/* Score de timing */}
              {advisor.timing && (
                <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="text-sm font-semibold text-white">Score de timing</h2>
                    <div className={`px-3 py-1 rounded text-xs font-semibold mono border ${
                      advisor.timing.color === "emerald" ? "bg-emerald-500/15 border-emerald-500/40 text-emerald-300" :
                      advisor.timing.color === "cyan"    ? "bg-cyan-500/15 border-cyan-500/40 text-cyan-300" :
                      advisor.timing.color === "amber"   ? "bg-amber-500/15 border-amber-500/40 text-amber-300" :
                                                           "bg-rose-500/15 border-rose-500/40 text-rose-300"
                    }`}>
                      {advisor.timing.label}
                    </div>
                  </div>

                  {/* Barre de score globale */}
                  <div className="mb-4">
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-slate-500">Score global</span>
                      <span className="text-white mono font-semibold">{advisor.timing.score} / 100</span>
                    </div>
                    <div className="h-2 rounded bg-white/10 overflow-hidden">
                      <div
                        className={`h-full rounded transition-all ${
                          advisor.timing.score >= 75 ? "bg-emerald-400" :
                          advisor.timing.score >= 55 ? "bg-cyan-400" :
                          advisor.timing.score >= 35 ? "bg-amber-400" : "bg-rose-400"
                        }`}
                        style={{ width: `${advisor.timing.score}%` }}
                      />
                    </div>
                  </div>

                  {/* Détail par composante */}
                  <div className="space-y-2">
                    {advisor.timing.details.map((d, i) => (
                      <div key={i}>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-slate-400">{d.label}</span>
                          <span className="text-slate-500 mono">{d.pts}/{d.max} pts</span>
                        </div>
                        <div className="h-1 rounded bg-white/10 overflow-hidden mb-0.5">
                          <div
                            className="h-full bg-indigo-400/60 rounded"
                            style={{ width: `${(d.pts / d.max) * 100}%` }}
                          />
                        </div>
                        <p className="text-[10px] text-slate-600">{d.note}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Recommandation */}
              <div className={`rounded-lg border p-5 ${clsAdvisor(advisor.color)}`}>
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p className="text-xs opacity-70 mb-1">Stratégie recommandée</p>
                    <p className="text-2xl font-bold mono">{advisor.strategy_label}</p>
                    <p className="text-xs opacity-60 mt-1">
                      Risque {advisor.risk_profile} · DTE cible {advisor.dte_days} jours
                    </p>
                  </div>
                  <div className="text-right text-xs opacity-70">
                    {advisor.spot && <p>Spot : <span className="mono">{advisor.spot.toLocaleString()}</span></p>}
                    {advisor.iv_atm && <p>IV ATM : <span className="mono">{advisor.iv_atm.toFixed(1)}%</span></p>}
                    {advisor.skew_25d != null && <p>Skew 25d : <span className="mono">{advisor.skew_25d > 0 ? "+" : ""}{advisor.skew_25d.toFixed(1)}%</span></p>}
                  </div>
                </div>

                {/* Legs */}
                {advisor.legs.length > 0 && (
                  <div className="rounded-lg overflow-hidden border border-white/10 mb-4">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-white/10">
                          {["Action", "Type", "Strike", "DTE", "Delta cible"].map(h => (
                            <th key={h} className="px-3 py-2 text-left text-[10px] opacity-60 font-medium">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {advisor.legs.map((leg, i) => (
                          <tr key={i} className="border-b border-white/[0.06]">
                            <td className={`px-3 py-2 font-semibold mono ${leg.action === "BUY" ? "text-emerald-300" : "text-rose-300"}`}>
                              {leg.action}
                            </td>
                            <td className="px-3 py-2 mono opacity-80">{leg.type}</td>
                            <td className="px-3 py-2 mono font-medium">{leg.strike.toLocaleString()}</td>
                            <td className="px-3 py-2 mono opacity-70">{leg.dte}j</td>
                            <td className="px-3 py-2 mono opacity-70">{leg.delta}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* Rationale */}
                <p className="text-xs opacity-70 leading-5">{advisor.rationale}</p>
              </div>

              {/* ── Kelly Sizing ── */}
              {advisor.strategy !== "WAIT" && (() => {
                const estimated = estimateKellyParams(advisor.strategy)
                const p = winRateStr ? Math.min(0.99, Math.max(0.01, parseFloat(winRateStr) / 100)) : estimated.p
                const b = ratioStr  ? Math.max(0.01, parseFloat(ratioStr)) : estimated.b
                const { fStar, fApplied, maxRisk, positive } = kellyCalc(p, b, kellyFraction, capital)
                const fractionLabel = kellyFraction === 1 ? "Kelly plein" : kellyFraction === 0.5 ? "Demi-Kelly" : "Quart Kelly"

                return (
                  <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-sm font-semibold text-white">Sizing Kelly</h2>
                      <span className="text-[10px] text-slate-500 mono">{fractionLabel}</span>
                    </div>

                    {/* Inputs */}
                    <div className="grid grid-cols-2 gap-3 mb-4">
                      <div>
                        <label className={label}>Capital ($)</label>
                        <input className={inp} type="number" value={capital} min={100}
                          onChange={e => updateCapital(+e.target.value)} />
                      </div>
                      <div>
                        <label className={label}>Fraction Kelly</label>
                        <div className="flex gap-1">
                          {([0.25, 0.5, 1] as KellyFraction[]).map(f => (
                            <button key={f} onClick={() => updateFraction(f)}
                              className={`${btn(kellyFraction === f)} flex-1 text-[11px]`}>
                              {f === 0.25 ? "1/4" : f === 0.5 ? "1/2" : "Full"}
                            </button>
                          ))}
                        </div>
                      </div>
                      <div>
                        <label className={label}>Win rate % (vide = auto {(estimated.p * 100).toFixed(0)}%)</label>
                        <input className={inp} type="number" value={winRateStr} placeholder={`${(estimated.p * 100).toFixed(0)}`}
                          min={1} max={99} onChange={e => setWinRateStr(e.target.value)} />
                      </div>
                      <div>
                        <label className={label}>Ratio G/P (vide = auto {estimated.b.toFixed(2)})</label>
                        <input className={inp} type="number" value={ratioStr} placeholder={`${estimated.b.toFixed(2)}`}
                          min={0.01} step={0.1} onChange={e => setRatioStr(e.target.value)} />
                      </div>
                    </div>

                    <p className="text-[10px] text-slate-600 mb-3">{estimated.hint}</p>

                    {/* Résultat */}
                    {!positive ? (
                      <div className="bg-rose-500/10 border border-rose-500/30 rounded-lg px-4 py-3">
                        <p className="text-sm text-rose-300 font-semibold">Kelly négatif — EV négatif</p>
                        <p className="text-xs text-rose-400/70 mt-0.5">
                          Cette stratégie dans ce contexte n'a pas d'espérance positive estimée. Ne pas trader.
                        </p>
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                        <div className="bg-white/[0.03] rounded-lg px-4 py-3">
                          <p className="text-[11px] text-slate-500">Kelly brut</p>
                          <p className="text-lg font-semibold mono text-white">{(fStar * 100).toFixed(1)}%</p>
                        </div>
                        <div className="bg-white/[0.03] rounded-lg px-4 py-3">
                          <p className="text-[11px] text-slate-500">{fractionLabel}</p>
                          <p className="text-lg font-semibold mono text-indigo-300">{(fApplied * 100).toFixed(2)}%</p>
                        </div>
                        <div className={`rounded-lg px-4 py-3 border ${
                          maxRisk > capital * 0.03 ? "bg-amber-500/10 border-amber-500/30" : "bg-emerald-500/10 border-emerald-500/30"
                        }`}>
                          <p className="text-[11px] text-slate-500">Max risque</p>
                          <p className={`text-lg font-semibold mono ${maxRisk > capital * 0.03 ? "text-amber-300" : "text-emerald-300"}`}>
                            ${maxRisk.toFixed(0)}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })()}

              {/* ── TP/SL sur prime ── */}
              {advisor.strategy !== "WAIT" && advisor.iv_atm && advisor.spot && (() => {
                const estimated = estimateKellyParams(advisor.strategy)
                const p = winRateStr ? Math.min(0.99, Math.max(0.01, parseFloat(winRateStr) / 100)) : estimated.p
                const b = ratioStr   ? Math.max(0.01, parseFloat(ratioStr)) : estimated.b
                const { maxRisk, positive } = kellyCalc(p, b, kellyFraction, capital)

                const isSeller = ["SHORT_PUT","SHORT_CALL","SHORT_STRANGLE","IRON_CONDOR"].includes(advisor.strategy)
                const isBuyer  = ["LONG_STRADDLE","LONG_STRANGLE","BULL_CALL_SPREAD","BEAR_PUT_SPREAD"].includes(advisor.strategy)

                // Estimation prime basée sur IV ATM et DTE (approximation BS ATM)
                const iv = (advisor.iv_atm ?? 40) / 100
                const T  = (advisor.dte_days ?? 21) / 365
                const premiumEstimate = advisor.spot * iv * Math.sqrt(T / (2 * Math.PI))
                const legs = advisor.legs.length

                const premiumPerLeg   = premiumEstimate
                const totalPremium    = premiumPerLeg * (isSeller ? legs : 1)

                // Règles TP/SL
                const tp = isSeller ? totalPremium * 0.50 : totalPremium * 2.0
                const sl = isSeller ? totalPremium * 2.00 : totalPremium * 0.50

                // Validation Kelly
                const kellyOk = positive && sl <= maxRisk
                const contractsOk = positive && maxRisk > 0
                  ? Math.floor(maxRisk / sl)
                  : 0

                return (
                  <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-sm font-semibold text-white">Seuils TP / SL</h2>
                      <span className="text-[10px] text-slate-500">
                        {isSeller ? "Vendeur de vol" : "Acheteur de vol"}
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-3 mb-4">
                      <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg px-4 py-3">
                        <p className="text-[11px] text-slate-500">
                          Take Profit — {isSeller ? "50% prime" : "2× prime"}
                        </p>
                        <p className="text-lg font-semibold mono text-emerald-300">
                          +${tp.toFixed(0)}
                        </p>
                        <p className="text-[10px] text-slate-500 mt-0.5">
                          {isSeller ? "Racheter la position" : "Vendre la position"}
                        </p>
                      </div>
                      <div className="bg-rose-500/10 border border-rose-500/30 rounded-lg px-4 py-3">
                        <p className="text-[11px] text-slate-500">
                          Stop Loss — {isSeller ? "2× prime" : "50% prime"}
                        </p>
                        <p className="text-lg font-semibold mono text-rose-300">
                          −${sl.toFixed(0)}
                        </p>
                        <p className="text-[10px] text-slate-500 mt-0.5">
                          Fermer immédiatement
                        </p>
                      </div>
                      <div className={`rounded-lg px-4 py-3 border ${
                        kellyOk
                          ? "bg-indigo-500/10 border-indigo-500/30"
                          : "bg-rose-500/10 border-rose-500/30"
                      }`}>
                        <p className="text-[11px] text-slate-500">Contrats Kelly</p>
                        <p className={`text-lg font-semibold mono ${kellyOk ? "text-indigo-300" : "text-rose-300"}`}>
                          {contractsOk} contrat{contractsOk > 1 ? "s" : ""}
                        </p>
                        <p className="text-[10px] text-slate-500 mt-0.5">
                          SL ≤ ${maxRisk.toFixed(0)} max
                        </p>
                      </div>
                    </div>

                    {!kellyOk && positive && (
                      <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg px-4 py-2 text-xs text-amber-300">
                        ⚠ SL estimé (${sl.toFixed(0)}) dépasse le max Kelly (${maxRisk.toFixed(0)}).
                        {" "}Réduire la taille à 1 contrat minimum ou ajuster le capital.
                      </div>
                    )}

                    <p className="text-[10px] text-slate-600 mt-3">
                      Prime estimée ≈ ${premiumEstimate.toFixed(0)}/leg (BS ATM simplifié · à affiner avec le prix marché réel)
                    </p>
                  </div>
                )
              })()}

              {/* Disclaimer */}
              <p className="text-[10px] text-slate-600 text-center">
                Recommandation algorithmique basée sur IVP + signal directionnel. Pas un conseil financier.
                Dernière analyse : {advisor.timestamp?.slice(0, 19).replace("T", " ")} UTC
              </p>
            </div>
          )}
        </div>
      )}

      {/* ── TAB PRICER ── */}
      {tab === "pricer" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Inputs */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5 space-y-4">
            <h2 className="text-sm font-semibold text-white">Black-Scholes</h2>

            <div className="flex gap-2">
              {(["call", "put"] as OptType[]).map(t => (
                <button key={t} onClick={() => setOptType(t)} className={`${btn(optType === t)} flex-1`}>
                  {t.toUpperCase()}
                </button>
              ))}
            </div>

            <div>
              <label className={label}>Strike (K)</label>
              <input className={inp} type="number" value={strike} onChange={e => setStrike(+e.target.value)} />
            </div>

            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="bg-white/[0.03] rounded px-3 py-2">
                <p className="text-[10px] text-slate-500">Prix théorique</p>
                <p className="text-xl font-bold text-white mono">{fmt2(price, 2)}</p>
              </div>
              <div className="bg-white/[0.03] rounded px-3 py-2">
                <p className="text-[10px] text-slate-500">Moneyness</p>
                <p className="text-base font-semibold mono text-slate-300">
                  {((spot / strike - 1) * 100).toFixed(2)}%
                </p>
              </div>
            </div>

            {/* IV Solver */}
            <div className="border-t border-white/5 pt-4">
              <p className="text-[11px] text-slate-400 mb-2">IV Solver — entre un prix de marché</p>
              <div className="flex gap-2">
                <input
                  className={inp}
                  type="number"
                  placeholder="Prix marché"
                  value={mktPriceStr}
                  onChange={e => { setMktPriceStr(e.target.value); setSolvedIV(null) }}
                />
                <button onClick={handleSolveIV} className={btn(false) + " whitespace-nowrap"}>
                  Solve IV
                </button>
              </div>
              {solvedIV !== null && (
                <p className="mt-2 text-sm text-indigo-300 mono">
                  IV implicite : <strong>{(solvedIV * 100).toFixed(2)}%</strong>
                  {" "}(vs ATM {ivPct.toFixed(1)}% — spread {((solvedIV - iv) * 100).toFixed(1)}pts)
                </p>
              )}
            </div>
          </div>

          {/* Greeks */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-sm font-semibold text-white mb-4">Greeks</h2>
            <GreekRow name="Delta" value={greeks.delta} hint="Δ prix / Δ spot" />
            <GreekRow name="Gamma" value={greeks.gamma} hint="Δ delta / Δ spot" />
            <GreekRow name="Vega"  value={greeks.vega}  hint="Δ prix / +1% IV" />
            <GreekRow name="Theta" value={greeks.theta} hint="Δ prix / jour" />
            <GreekRow name="Rho"   value={greeks.rho}   hint="Δ prix / +1% taux" />

            <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
              <div className="bg-white/[0.02] rounded px-3 py-2">
                <p className="text-slate-500">Prob ITM (call)</p>
                <p className="text-white mono">{(probITM(spot, strike, T, r, q, iv, true) * 100).toFixed(1)}%</p>
              </div>
              <div className="bg-white/[0.02] rounded px-3 py-2">
                <p className="text-slate-500">Prob ITM (put)</p>
                <p className="text-white mono">{(probITM(spot, strike, T, r, q, iv, false) * 100).toFixed(1)}%</p>
              </div>
              <div className="bg-white/[0.02] rounded px-3 py-2">
                <p className="text-slate-500">Breakeven call</p>
                <p className="text-white mono">{(strike + price).toFixed(0)}</p>
              </div>
              <div className="bg-white/[0.02] rounded px-3 py-2">
                <p className="text-slate-500">Breakeven put</p>
                <p className="text-white mono">{(strike - price).toFixed(0)}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB POSITION ── */}
      {tab === "position" && (
        <div className="space-y-4">
          {/* Presets */}
          <div className="flex gap-2 flex-wrap">
            <span className="text-[11px] text-slate-500 self-center">Presets :</span>
            {[["straddle","Straddle"],["strangle","Strangle"],["iron_condor","Iron Condor"],["bull_spread","Bull Spread"],["bear_spread","Bear Spread"],["butterfly","Butterfly"]].map(([k,l]) => (
              <button key={k} onClick={() => applyPreset(k)} className={btn(false)}>{l}</button>
            ))}
            <button onClick={() => setLegs([])} className="ml-auto px-3 py-1.5 text-xs rounded border border-rose-500/30 text-rose-400 hover:bg-rose-500/10">
              Reset
            </button>
          </div>

          {/* Add leg */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-4">
            <p className="text-[11px] text-slate-500 mb-3">Ajouter un leg</p>
            <div className="flex gap-2 flex-wrap items-end">
              <div className="flex gap-1">
                {(["long","short"] as LegSide[]).map(s => (
                  <button key={s} onClick={() => setNewLeg(l => ({...l, side: s}))} className={btn(newLeg.side === s)}>
                    {s.toUpperCase()}
                  </button>
                ))}
              </div>
              <div className="flex gap-1">
                {(["call","put"] as OptType[]).map(t => (
                  <button key={t} onClick={() => setNewLeg(l => ({...l, type: t}))} className={btn(newLeg.type === t)}>
                    {t.toUpperCase()}
                  </button>
                ))}
              </div>
              <div className="w-28">
                <label className={label}>Strike</label>
                <input className={inp} type="number" value={newLeg.strike}
                  onChange={e => setNewLeg(l => ({...l, strike: +e.target.value}))} />
              </div>
              <div className="w-16">
                <label className={label}>Qty</label>
                <input className={inp} type="number" value={newLeg.qty} min={1}
                  onChange={e => setNewLeg(l => ({...l, qty: +e.target.value}))} />
              </div>
              <div className="w-28">
                <label className={label}>Prime (0=BS auto)</label>
                <input className={inp} type="number" value={newLeg.premium} step={0.01}
                  onChange={e => setNewLeg(l => ({...l, premium: +e.target.value}))} />
              </div>
              <button onClick={addLeg} className="px-4 py-1.5 text-xs rounded bg-indigo-600 text-white hover:bg-indigo-500">
                + Ajouter
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Legs table */}
            <div className="bg-[#0d0d14] border border-white/5 rounded-lg overflow-hidden">
              <div className="px-4 py-2 border-b border-white/5 text-[11px] text-slate-500 uppercase">Legs</div>
              {legs.length === 0
                ? <p className="text-slate-600 text-sm p-4">Aucun leg</p>
                : legs.map(leg => (
                  <div key={leg.id} className="flex items-center justify-between px-4 py-2 border-b border-white/[0.03] text-xs">
                    <span className={leg.side === "long" ? "text-emerald-400" : "text-rose-400"}>
                      {leg.side.toUpperCase()}
                    </span>
                    <span className="text-slate-300 mono">{leg.type.toUpperCase()}</span>
                    <span className="text-white mono">{leg.strike.toLocaleString()}</span>
                    <span className="text-slate-400">×{leg.qty}</span>
                    <span className="text-slate-400 mono">{leg.premium.toFixed(2)}</span>
                    <button onClick={() => setLegs(l => l.filter(x => x.id !== leg.id))}
                      className="text-slate-600 hover:text-rose-400 ml-2">✕</button>
                  </div>
                ))
              }
              {/* Metrics */}
              {posMetrics && (
                <div className="px-4 py-3 border-t border-white/5 grid grid-cols-2 gap-2 text-xs">
                  <div><span className="text-slate-500">Max Profit</span><span className="float-right text-emerald-300 mono">{posMetrics.maxP > 9999 ? "Illimité" : posMetrics.maxP.toFixed(0)}</span></div>
                  <div><span className="text-slate-500">Max Loss</span><span className="float-right text-rose-300 mono">{posMetrics.minP < -9999 ? "Illimité" : posMetrics.minP.toFixed(0)}</span></div>
                  <div><span className="text-slate-500">Prime nette</span><span className="float-right text-white mono">{fmtSign(posMetrics.totalPremium, 2)}</span></div>
                  <div><span className="text-slate-500">Delta global</span><span className="float-right text-white mono">{fmtSign(posMetrics.deltaAgg, 3)}</span></div>
                  {posMetrics.bes.length > 0 && (
                    <div className="col-span-2 text-amber-300">
                      BE: {posMetrics.bes.map(b => b.toFixed(0)).join(" · ")}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* P&L Diagram */}
            <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-4">
              <p className="text-[11px] text-slate-500 mb-2 uppercase">P&L à expiry</p>
              <PnLDiagram legs={legs} spot={spot} />
              <p className="text-[10px] text-slate-600 mt-1">
                — — spot actuel · ● breakeven · <span className="text-emerald-400">profit</span> · <span className="text-rose-400">perte</span>
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB PROBAS ── */}
      {tab === "probas" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Expected Move */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-sm font-semibold text-white mb-4">Expected Move</h2>
            <p className="text-xs text-slate-500 mb-4">
              Mouvement attendu à 1σ et 2σ sur {dte} jours (IV={ivPct}%)
            </p>
            {[
              { label: "1σ (68%)", move: em1, factor: 1 },
              { label: "2σ (95%)", move: em2, factor: 2 },
            ].map(({ label, move }) => (
              <div key={label} className="mb-4">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">{label}</span>
                  <span className="text-white mono">±{move.toFixed(0)} ({(move/spot*100).toFixed(1)}%)</span>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-rose-300 mono">{(spot - move).toFixed(0)}</span>
                  <div className="flex-1 h-2 rounded bg-white/10 relative overflow-hidden">
                    <div className="absolute inset-0 bg-gradient-to-r from-rose-500/30 via-slate-700 to-emerald-500/30 rounded" />
                  </div>
                  <span className="text-emerald-300 mono">{(spot + move).toFixed(0)}</span>
                </div>
              </div>
            ))}
            <div className="border-t border-white/5 pt-3 mt-3 text-xs grid grid-cols-2 gap-2">
              <div className="bg-white/[0.02] rounded px-3 py-2">
                <p className="text-slate-500">Vol quotidienne</p>
                <p className="text-white mono">{(iv / Math.sqrt(365) * 100).toFixed(2)}%</p>
              </div>
              <div className="bg-white/[0.02] rounded px-3 py-2">
                <p className="text-slate-500">Move 1j (1σ)</p>
                <p className="text-white mono">{(spot * iv / Math.sqrt(365)).toFixed(0)}</p>
              </div>
            </div>
          </div>

          {/* Prob table */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg overflow-hidden">
            <div className="px-5 py-3 border-b border-white/5">
              <h2 className="text-sm font-semibold text-white">Prob ITM par strike</h2>
              <p className="text-xs text-slate-500 mt-0.5">Probabilité risk-neutral à expiry ({dte}j)</p>
            </div>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-white/5">
                  {["Strike", "Δ%", "Prob Call ITM", "POP Put vendu", "Prob Put ITM"].map(h => (
                    <th key={h} className="px-3 py-2 text-left text-[10px] text-slate-500 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {probStrikesData.map(row => {
                  const isAtm = Math.abs(row.K - spot) < spot * 0.01
                  return (
                    <tr key={row.K} className={`border-b border-white/[0.03] ${isAtm ? "bg-white/[0.03]" : ""}`}>
                      <td className={`px-3 py-1.5 mono ${isAtm ? "text-indigo-300 font-semibold" : "text-slate-300"}`}>
                        {row.K.toLocaleString()}
                      </td>
                      <td className="px-3 py-1.5 mono text-slate-500">{row.pct}%</td>
                      <td className="px-3 py-1.5 mono text-emerald-300">{(row.probCall * 100).toFixed(1)}%</td>
                      <td className="px-3 py-1.5 mono text-slate-300">{((1 - row.probPut) * 100).toFixed(1)}%</td>
                      <td className="px-3 py-1.5 mono text-rose-300">{(row.probPut * 100).toFixed(1)}%</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── TAB GESTION ── */}
      {tab === "gestion" && (
        <div className="space-y-4">

          {/* Paramètres de la position ouverte */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-white">Position ouverte</h2>
              <button onClick={prefillFromAdvisor} className={btn(false) + " text-indigo-300 border-indigo-500/30"}>
                Pré-remplir depuis l'Advisor
              </button>
            </div>

            <div className="grid grid-cols-3 gap-3 mb-4">
              <div>
                <label className={label}>IV à l'entrée (%)</label>
                <input className={inp} type="number" value={gEntryIv} step={0.5}
                  onChange={e => setGEntryIv(+e.target.value)} />
              </div>
              <div>
                <label className={label}>DTE à l'entrée (j)</label>
                <input className={inp} type="number" value={gDteEntry} min={1}
                  onChange={e => setGDteEntry(+e.target.value)} />
              </div>
              <div>
                <label className={label}>DTE restant (j)</label>
                <input className={inp} type="number" value={gDteLeft} min={0}
                  onChange={e => setGDteLeft(+e.target.value)} />
              </div>
            </div>

            {/* Ajout de legs */}
            <p className="text-[11px] text-slate-500 mb-2">Legs de la position</p>
            <div className="flex gap-2 flex-wrap items-end mb-3">
              <div className="flex gap-1">
                {(["SELL","BUY"] as const).map(a => (
                  <button key={a} onClick={() => setGNewLeg(l => ({...l, action: a}))}
                    className={btn(gNewLeg.action === a)}>
                    {a}
                  </button>
                ))}
              </div>
              <div className="flex gap-1">
                {(["call","put"] as OptType[]).map(t => (
                  <button key={t} onClick={() => setGNewLeg(l => ({...l, type: t}))}
                    className={btn(gNewLeg.type === t)}>
                    {t.toUpperCase()}
                  </button>
                ))}
              </div>
              <div className="w-28">
                <label className={label}>Strike</label>
                <input className={inp} type="number" value={gNewLeg.strike}
                  onChange={e => setGNewLeg(l => ({...l, strike: +e.target.value}))} />
              </div>
              <div className="w-32">
                <label className={label}>Prime entrée ($)</label>
                <input className={inp} type="number" value={gNewLeg.entryPremium} step={1}
                  onChange={e => setGNewLeg(l => ({...l, entryPremium: +e.target.value}))} />
              </div>
              <button onClick={() => {
                setGLegs(l => [...l, { ...gNewLeg, id: gNextId }])
                setGNextId(n => n + 1)
              }} className="px-4 py-1.5 text-xs rounded bg-indigo-600 text-white hover:bg-indigo-500">
                + Ajouter
              </button>
            </div>

            {/* Liste des legs */}
            {gLegs.length > 0 && (
              <div className="rounded-lg overflow-hidden border border-white/5">
                {gLegs.map(leg => (
                  <div key={leg.id} className="flex items-center justify-between px-4 py-2 border-b border-white/[0.03] text-xs">
                    <span className={leg.action === "SELL" ? "text-rose-300 mono" : "text-emerald-300 mono"}>{leg.action}</span>
                    <span className="text-slate-300 mono">{leg.type.toUpperCase()}</span>
                    <span className="text-white mono">{leg.strike.toLocaleString()}</span>
                    <span className="text-slate-500 mono">prime: ${leg.entryPremium}</span>
                    <span className="text-slate-500 mono">δ entrée: {bsGreeks(spot, leg.strike, gDteEntry/365, r, q, gEntryIv/100, leg.type === "call").delta.toFixed(2)}</span>
                    <span className="text-slate-400 mono">δ maintenant: {bsGreeks(spot, leg.strike, Math.max(gDteLeft/365, 0.001), r, q, iv, leg.type === "call").delta.toFixed(2)}</span>
                    <button onClick={() => setGLegs(l => l.filter(x => x.id !== leg.id))}
                      className="text-slate-600 hover:text-rose-400">✕</button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Alertes */}
          {gLegs.length > 0 && (
            <div className="space-y-2">
              <h2 className="text-sm font-semibold text-white">Alertes d'ajustement</h2>
              {gAlerts.map((a, i) => {
                const colors: Record<AlertLevel, string> = {
                  URGENT:  "bg-rose-500/15 border-rose-500/40",
                  WARNING: "bg-amber-500/15 border-amber-500/40",
                  SUCCESS: "bg-emerald-500/15 border-emerald-500/40",
                  INFO:    "bg-white/5 border-white/10",
                }
                const icons: Record<AlertLevel, string> = {
                  URGENT: "🔴", WARNING: "🟡", SUCCESS: "🟢", INFO: "⚪",
                }
                const textColors: Record<AlertLevel, string> = {
                  URGENT: "text-rose-300", WARNING: "text-amber-300", SUCCESS: "text-emerald-300", INFO: "text-slate-400",
                }
                const guide = alertGuide(a)
                return (
                  <div key={i} className={`rounded-lg border p-4 ${colors[a.level]}`}>
                    <div className="flex items-start justify-between">
                      <div>
                        <p className={`text-sm font-semibold ${textColors[a.level]}`}>
                          {icons[a.level]} {a.label}
                        </p>
                        <p className="text-xs text-slate-400 mt-0.5">{a.detail}</p>
                      </div>
                      <p className="text-xs text-slate-500 text-right ml-4 shrink-0">{a.action}</p>
                    </div>
                    {guide && (
                      <div className="mt-3 pt-3 border-t border-white/10">
                        <p className="text-[11px] font-semibold text-slate-300 mb-0.5">📖 {guide.rule}</p>
                        <p className="text-[11px] text-slate-500 leading-5">{guide.explain}</p>
                        {guide.showRoll && gLegs.length > 0 && (
                          <button
                            onClick={() => openRollSimulator(gLegs.find(l => a.detail.includes(String(l.strike))) ?? gLegs[0])}
                            className="mt-2 text-[11px] px-3 py-1 rounded border border-indigo-500/40 text-indigo-300 hover:bg-indigo-500/10 transition-colors"
                          >
                            → Simuler un roll
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}

          {gLegs.length === 0 && (
            <div className="flex items-center justify-center h-32 text-slate-600 text-sm">
              Entrez les legs de votre position ou utilisez "Pré-remplir depuis l'Advisor"
            </div>
          )}
        </div>
      )}

      {/* ── TAB RÈGLES ── */}
      {tab === "regles" && (
        <div className="space-y-5 max-w-3xl">

          {/* Checklist pré-trade */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-base font-semibold text-white mb-1">Checklist avant d'ouvrir une position</h2>
            <p className="text-xs text-slate-500 mb-4">Cocher chaque condition avant d'entrer. Si une case manque → attendre.</p>
            <div className="space-y-2">
              {[
                { key: "ivp",    label: "IVP > 50% — vol chère (favorable vendeur)",  hint: advisor?.ivp_pct != null ? `Actuel : ${advisor.ivp_pct.toFixed(0)}% — ${advisor.ivp_pct > 50 ? "✓ OK" : "✗ Vol pas encore chère"}` : "Vérifier dans l'Advisor" },
                { key: "timing", label: "Score de timing > 60",                        hint: advisor?.timing?.score != null ? `Actuel : ${advisor.timing.score}/100 — ${advisor.timing.score > 60 ? "✓ OK" : "✗ Attendre"}` : "Vérifier dans l'Advisor" },
                { key: "dte",    label: "DTE cible 30-45 jours",                       hint: advisor?.dte_days != null ? `Advisor suggère : ${advisor.dte_days}j — ${advisor.dte_days >= 25 && advisor.dte_days <= 50 ? "✓ OK" : "✗ Hors zone optimale"}` : "30-45j recommandé" },
                { key: "kelly",  label: "Sizing Kelly calculé et capital vérifié",    hint: `Capital : ${capital.toLocaleString()}$ · Max risque Kelly à vérifier dans l'Advisor` },
                { key: "strike", label: "Strike ≥ 1σ OTM du spot actuel",            hint: `Expected move ≈ ±${(spot * iv * Math.sqrt(T)).toFixed(0)} · Strike OTM doit dépasser ${(spot - spot * iv * Math.sqrt(T)).toFixed(0)} (put) ou ${(spot + spot * iv * Math.sqrt(T)).toFixed(0)} (call)` },
                { key: "risk",   label: "Risque max ≤ 2% du capital total",          hint: `Max ${(capital * 0.02).toFixed(0)}$ de perte max sur ${capital.toLocaleString()}$ de capital` },
              ].map(item => (
                <label key={item.key} className="flex items-start gap-3 py-2 border-b border-white/[0.04] cursor-pointer group">
                  <div className="mt-0.5 shrink-0" onClick={() => toggleCheck(item.key)}>
                    <div className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${
                      checks[item.key] ? "bg-emerald-500 border-emerald-500" : "border-white/20 group-hover:border-white/40"
                    }`}>
                      {checks[item.key] && <span className="text-white text-[10px] font-bold">✓</span>}
                    </div>
                  </div>
                  <div onClick={() => toggleCheck(item.key)}>
                    <p className={`text-sm transition-colors ${checks[item.key] ? "text-emerald-300 line-through opacity-60" : "text-slate-200"}`}>{item.label}</p>
                    <p className="text-[10px] text-slate-500">{item.hint}</p>
                  </div>
                </label>
              ))}
            </div>
            {(() => {
              const total = 6
              const done = Object.values(checks).filter(Boolean).length
              const allOk = done === total
              return (
                <div className={`mt-4 rounded-lg px-4 py-3 border ${allOk ? "bg-emerald-500/10 border-emerald-500/30" : "bg-amber-500/10 border-amber-500/30"}`}>
                  <p className={`text-sm font-semibold ${allOk ? "text-emerald-300" : "text-amber-300"}`}>
                    {allOk ? "✓ Toutes conditions remplies — position autorisée" : `${done}/${total} conditions — ${total - done} manquante${total - done > 1 ? "s" : ""}`}
                  </p>
                  {!allOk && <p className="text-xs text-slate-500 mt-0.5">Compléter les cases manquantes ou attendre des conditions plus favorables.</p>}
                </div>
              )
            })()}
            <button onClick={() => setChecks({})} className="mt-3 text-xs text-slate-600 hover:text-slate-400">Réinitialiser la checklist</button>
          </div>

          {/* Simulateur de Roll */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-base font-semibold text-white mb-1">Simulateur de Roll</h2>
            <p className="text-xs text-slate-500 mb-4">
              Un roll = racheter l'option actuelle + vendre une nouvelle. Valide seulement si le crédit net est positif ou nul.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div>
                <p className="text-[11px] text-slate-400 uppercase tracking-wider mb-3">Position actuelle (à racheter)</p>
                <div className="space-y-3">
                  <div>
                    <label className={label}>Type</label>
                    <div className="flex gap-1">
                      {(["put", "call"] as OptType[]).map(t => (
                        <button key={t} onClick={() => setRollType(t)} className={`${btn(rollType === t)} flex-1`}>{t.toUpperCase()}</button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label className={label}>Strike actuel</label>
                    <input className={inp} type="number" value={rollStrikeFrom} onChange={e => setRollStrikeFrom(+e.target.value)} />
                  </div>
                  <div>
                    <label className={label}>DTE restant (jours)</label>
                    <input className={inp} type="number" value={rollDteFrom} min={0} onChange={e => setRollDteFrom(+e.target.value)} />
                  </div>
                  <div>
                    <label className={label}>Prime reçue à l'ouverture ($)</label>
                    <input className={inp} type="number" value={rollPremiumReceived} min={0} onChange={e => setRollPremiumReceived(+e.target.value)} />
                  </div>
                </div>
              </div>
              <div>
                <p className="text-[11px] text-slate-400 uppercase tracking-wider mb-3">Nouvelle position (à vendre)</p>
                <div className="space-y-3">
                  <div>
                    <label className={label}>Strike cible</label>
                    <input className={inp} type="number" value={rollStrikeTo} onChange={e => setRollStrikeTo(+e.target.value)} />
                  </div>
                  <div>
                    <label className={label}>DTE cible (jours)</label>
                    <input className={inp} type="number" value={rollDteTo} min={1} onChange={e => setRollDteTo(+e.target.value)} />
                  </div>
                  <div className="pt-6 space-y-1">
                    <p className="text-[10px] text-slate-500">Même type que la position actuelle : <span className="text-slate-300 font-semibold">{rollType.toUpperCase()}</span></p>
                    <p className="text-[10px] text-slate-600">IV utilisée : {ivPct.toFixed(1)}% · Spot : {spot.toLocaleString()} (paramètres globaux)</p>
                  </div>
                </div>
              </div>
            </div>
            {(() => {
              const buyBack   = bsPrice(spot, rollStrikeFrom, Math.max(rollDteFrom / 365, 0.001), r, q, iv, rollType === "call")
              const newSell   = bsPrice(spot, rollStrikeTo, Math.max(rollDteTo / 365, 0.001), r, q, iv, rollType === "call")
              const netCredit = newSell - buyBack
              const closePnl  = rollPremiumReceived - buyBack
              const isCredit  = netCredit >= -0.01
              return (
                <div className="mt-5 space-y-3">
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div className="bg-rose-500/10 border border-rose-500/20 rounded-lg px-4 py-3">
                      <p className="text-[11px] text-slate-500">Coût rachat (BS)</p>
                      <p className="text-lg font-semibold mono text-rose-300">−${buyBack.toFixed(2)}</p>
                      <p className="text-[10px] text-slate-600">Prix théorique de clôture</p>
                    </div>
                    <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg px-4 py-3">
                      <p className="text-[11px] text-slate-500">Revente nouvelle (BS)</p>
                      <p className="text-lg font-semibold mono text-emerald-300">+${newSell.toFixed(2)}</p>
                      <p className="text-[10px] text-slate-600">Prime de la nouvelle position</p>
                    </div>
                    <div className={`rounded-lg px-4 py-3 border ${isCredit ? "bg-emerald-500/10 border-emerald-500/40" : "bg-rose-500/10 border-rose-500/40"}`}>
                      <p className="text-[11px] text-slate-500">Crédit net du roll</p>
                      <p className={`text-lg font-semibold mono ${isCredit ? "text-emerald-300" : "text-rose-300"}`}>
                        {netCredit >= 0 ? "+" : ""}{netCredit.toFixed(2)}
                      </p>
                      <p className={`text-[10px] font-semibold mt-0.5 ${isCredit ? "text-emerald-400" : "text-rose-400"}`}>
                        {isCredit ? "✓ Roll valide" : "✗ Roll à éviter (débit)"}
                      </p>
                    </div>
                  </div>
                  <div className="bg-white/[0.02] rounded-lg px-4 py-3 text-xs">
                    <span className="text-slate-500">P&L réalisé sur la partie close : </span>
                    <span className={`mono font-semibold ${closePnl >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
                      {closePnl >= 0 ? "+" : ""}{closePnl.toFixed(2)}$
                    </span>
                    <span className="text-slate-600 ml-2">(prime reçue {rollPremiumReceived}$ − rachat {buyBack.toFixed(2)}$)</span>
                  </div>
                </div>
              )
            })()}
            <div className="mt-4 bg-white/[0.02] border border-white/5 rounded-lg px-4 py-3">
              <p className="text-[11px] font-semibold text-slate-300 mb-1">📖 Règle d'or du roll</p>
              <p className="text-xs text-slate-500 leading-5">Ne rouler que si le crédit net est positif ou nul. Payer pour rouler (débit) = reporter le problème en augmentant le risque total engagé. Mieux vaut couper une perte définie que d'ouvrir une perte indéfinie plus grande.</p>
            </div>
          </div>

          {/* Règles non-négociables */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-base font-semibold text-white mb-4">5 règles non-négociables</h2>
            <div className="space-y-3">
              {[
                {
                  num: "1", color: "bg-rose-500",
                  rule: "Couper à 2× la prime",
                  detail: "Si la perte latente atteint 2 fois la prime encaissée, fermer sans exception. Racheter l'option au prix actuel. La perte nette est alors limitée à 1× la prime initiale.",
                  example: "Prime encaissée : 500$. Prime actuelle : 1 000$ → racheter, perte nette = −500$. Sans règle : prime peut exploser à 5 000$, 10 000$."
                },
                {
                  num: "2", color: "bg-amber-500",
                  rule: "Prendre profit à 50% de la prime",
                  detail: "Racheter quand la prime a perdu 50% de sa valeur. Libère le capital, réduit le temps d'exposition au risque gamma, et évite le retournement sur la deuxième moitié.",
                  example: "Prime encaissée : 500$. Racheter à 250$ → profit +250$ réalisé, position fermée, capital libéré."
                },
                {
                  num: "3", color: "bg-orange-500",
                  rule: "Fermer ou rouler avant 21 DTE",
                  detail: "Le gamma s'accélère exponentiellement sous 21 DTE. Un mouvement de 2% peut générer plus de perte en 1 jour qu'en 2 semaines. Rouler à 30-45 DTE si encore en perte.",
                  example: "Position à 15 DTE, profitable → racheter et vendre un nouveau cycle à 45 DTE (recharge theta)."
                },
                {
                  num: "4", color: "bg-violet-500",
                  rule: "Ne jamais ajouter sur une position perdante",
                  detail: "L'averaging down multiplie le risque non-linéairement. Avec les options, doubler une position en perte peut transformer une perte limitée en perte catastrophique.",
                  example: "Short Put en perte → ne pas vendre un 2ème put au même strike ou plus bas. Fermer la position, puis réévaluer."
                },
                {
                  num: "5", color: "bg-cyan-500",
                  rule: "Max 30% du capital engagé simultanément",
                  detail: "BTC et ETH sont corrélés > 0.85. Vendre des options sur les deux est une seule position, pas deux. Le total des pertes max possibles ne doit pas dépasser 30% du capital.",
                  example: `Capital : ${capital.toLocaleString()}$. Max risque cumulé options : ${(capital * 0.30).toFixed(0)}$. Au-delà, réduire ou couvrir.`
                },
              ].map(r => (
                <div key={r.num} className="flex gap-4 p-4 border border-white/5 rounded-lg">
                  <div className={`w-7 h-7 rounded-full ${r.color} flex items-center justify-center text-xs font-bold text-white shrink-0 mt-0.5`}>
                    {r.num}
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-white mb-1">{r.rule}</p>
                    <p className="text-xs text-slate-400 leading-5 mb-1.5">{r.detail}</p>
                    <p className="text-[11px] text-slate-600 italic border-l-2 border-white/10 pl-2">{r.example}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>
      )}

      {/* ── TAB RÈGLES ── */}
      {tab === "regles" && (
        <div className="space-y-5 max-w-3xl">

          {/* Checklist pré-trade */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-base font-semibold text-white mb-1">Checklist avant d'ouvrir une position</h2>
            <p className="text-xs text-slate-500 mb-4">Cocher chaque condition avant d'entrer. Si une case manque → attendre.</p>
            <div className="space-y-2">
              {[
                { key: "ivp",    label: "IVP > 50% — vol chère (favorable vendeur)",  hint: advisor?.ivp_pct != null ? `Actuel : ${advisor.ivp_pct.toFixed(0)}% — ${advisor.ivp_pct > 50 ? "✓ OK" : "✗ Vol pas encore chère"}` : "Vérifier dans l'Advisor" },
                { key: "timing", label: "Score de timing > 60",                        hint: advisor?.timing?.score != null ? `Actuel : ${advisor.timing.score}/100 — ${advisor.timing.score > 60 ? "✓ OK" : "✗ Attendre"}` : "Vérifier dans l'Advisor" },
                { key: "dte",    label: "DTE cible 30-45 jours",                       hint: advisor?.dte_days != null ? `Advisor suggère : ${advisor.dte_days}j — ${advisor.dte_days >= 25 && advisor.dte_days <= 50 ? "✓ OK" : "✗ Hors zone optimale"}` : "30-45j recommandé" },
                { key: "kelly",  label: "Sizing Kelly calculé et capital vérifié",     hint: `Capital : ${capital.toLocaleString()}$ · Max risque Kelly à vérifier dans l'Advisor` },
                { key: "strike", label: "Strike ≥ 1σ OTM du spot actuel",             hint: `Expected move ≈ ±${(spot * iv * Math.sqrt(T)).toFixed(0)} · Put OTM recommandé sous ${(spot - spot * iv * Math.sqrt(T)).toFixed(0)}` },
                { key: "risk",   label: "Risque max ≤ 2% du capital total",           hint: `Max ${(capital * 0.02).toFixed(0)}$ de perte max sur ${capital.toLocaleString()}$ de capital` },
              ].map(item => (
                <div key={item.key} className="flex items-start gap-3 py-2 border-b border-white/[0.04] cursor-pointer group" onClick={() => toggleCheck(item.key)}>
                  <div className="mt-0.5 shrink-0">
                    <div className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${
                      checks[item.key] ? "bg-emerald-500 border-emerald-500" : "border-white/20 group-hover:border-white/40"
                    }`}>
                      {checks[item.key] && <span className="text-white text-[10px] font-bold">✓</span>}
                    </div>
                  </div>
                  <div>
                    <p className={`text-sm transition-colors ${checks[item.key] ? "text-emerald-300 line-through opacity-60" : "text-slate-200"}`}>{item.label}</p>
                    <p className="text-[10px] text-slate-500">{item.hint}</p>
                  </div>
                </div>
              ))}
            </div>
            {(() => {
              const total = 6
              const done = Object.values(checks).filter(Boolean).length
              const allOk = done === total
              return (
                <div className={`mt-4 rounded-lg px-4 py-3 border ${allOk ? "bg-emerald-500/10 border-emerald-500/30" : "bg-amber-500/10 border-amber-500/30"}`}>
                  <p className={`text-sm font-semibold ${allOk ? "text-emerald-300" : "text-amber-300"}`}>
                    {allOk ? "✓ Toutes conditions remplies — position autorisée" : `${done}/${total} conditions — ${total - done} manquante${total - done > 1 ? "s" : ""}`}
                  </p>
                  {!allOk && <p className="text-xs text-slate-500 mt-0.5">Compléter les cases manquantes ou attendre des conditions plus favorables.</p>}
                </div>
              )
            })()}
            <button onClick={() => setChecks({})} className="mt-3 text-xs text-slate-600 hover:text-slate-400">Réinitialiser</button>
          </div>

          {/* Simulateur de Roll */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-base font-semibold text-white mb-1">Simulateur de Roll</h2>
            <p className="text-xs text-slate-500 mb-4">
              Un roll = racheter l'option actuelle + vendre une nouvelle. Valide seulement si le crédit net est positif ou nul.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div>
                <p className="text-[11px] text-slate-400 uppercase tracking-wider mb-3">Position actuelle (à racheter)</p>
                <div className="space-y-3">
                  <div>
                    <label className={label}>Type</label>
                    <div className="flex gap-1">
                      {(["put", "call"] as OptType[]).map(t => (
                        <button key={t} onClick={() => setRollType(t)} className={`${btn(rollType === t)} flex-1`}>{t.toUpperCase()}</button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label className={label}>Strike actuel</label>
                    <input className={inp} type="number" value={rollStrikeFrom} onChange={e => setRollStrikeFrom(+e.target.value)} />
                  </div>
                  <div>
                    <label className={label}>DTE restant (jours)</label>
                    <input className={inp} type="number" value={rollDteFrom} min={0} onChange={e => setRollDteFrom(+e.target.value)} />
                  </div>
                  <div>
                    <label className={label}>Prime reçue à l'ouverture ($)</label>
                    <input className={inp} type="number" value={rollPremiumReceived} min={0} onChange={e => setRollPremiumReceived(+e.target.value)} />
                  </div>
                </div>
              </div>
              <div>
                <p className="text-[11px] text-slate-400 uppercase tracking-wider mb-3">Nouvelle position (à vendre)</p>
                <div className="space-y-3">
                  <div>
                    <label className={label}>Strike cible</label>
                    <input className={inp} type="number" value={rollStrikeTo} onChange={e => setRollStrikeTo(+e.target.value)} />
                  </div>
                  <div>
                    <label className={label}>DTE cible (jours)</label>
                    <input className={inp} type="number" value={rollDteTo} min={1} onChange={e => setRollDteTo(+e.target.value)} />
                  </div>
                  <div className="pt-6 space-y-1">
                    <p className="text-[10px] text-slate-500">Même type que la position actuelle : <span className="text-slate-300 font-semibold">{rollType.toUpperCase()}</span></p>
                    <p className="text-[10px] text-slate-600">IV utilisée : {ivPct.toFixed(1)}% · Spot : {spot.toLocaleString()} (paramètres globaux)</p>
                  </div>
                </div>
              </div>
            </div>
            {(() => {
              const buyBack   = bsPrice(spot, rollStrikeFrom, Math.max(rollDteFrom / 365, 0.001), r, q, iv, rollType === "call")
              const newSell   = bsPrice(spot, rollStrikeTo,   Math.max(rollDteTo / 365, 0.001), r, q, iv, rollType === "call")
              const netCredit = newSell - buyBack
              const closePnl  = rollPremiumReceived - buyBack
              const isCredit  = netCredit >= -0.01
              return (
                <div className="mt-5 space-y-3">
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div className="bg-rose-500/10 border border-rose-500/20 rounded-lg px-4 py-3">
                      <p className="text-[11px] text-slate-500">Coût rachat (BS)</p>
                      <p className="text-lg font-semibold mono text-rose-300">−${buyBack.toFixed(2)}</p>
                      <p className="text-[10px] text-slate-600">Prix théorique de clôture</p>
                    </div>
                    <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg px-4 py-3">
                      <p className="text-[11px] text-slate-500">Revente nouvelle (BS)</p>
                      <p className="text-lg font-semibold mono text-emerald-300">+${newSell.toFixed(2)}</p>
                      <p className="text-[10px] text-slate-600">Prime de la nouvelle position</p>
                    </div>
                    <div className={`rounded-lg px-4 py-3 border ${isCredit ? "bg-emerald-500/10 border-emerald-500/40" : "bg-rose-500/10 border-rose-500/40"}`}>
                      <p className="text-[11px] text-slate-500">Crédit net du roll</p>
                      <p className={`text-lg font-semibold mono ${isCredit ? "text-emerald-300" : "text-rose-300"}`}>
                        {netCredit >= 0 ? "+" : ""}{netCredit.toFixed(2)}
                      </p>
                      <p className={`text-[10px] font-semibold mt-0.5 ${isCredit ? "text-emerald-400" : "text-rose-400"}`}>
                        {isCredit ? "✓ Roll valide" : "✗ Roll à éviter (débit)"}
                      </p>
                    </div>
                  </div>
                  <div className="bg-white/[0.02] rounded-lg px-4 py-3 text-xs">
                    <span className="text-slate-500">P&L réalisé sur la partie close : </span>
                    <span className={`mono font-semibold ${closePnl >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
                      {closePnl >= 0 ? "+" : ""}{closePnl.toFixed(2)}$
                    </span>
                    <span className="text-slate-600 ml-2">(prime reçue {rollPremiumReceived}$ − rachat {buyBack.toFixed(2)}$)</span>
                  </div>
                </div>
              )
            })()}
            <div className="mt-4 bg-white/[0.02] border border-white/5 rounded-lg px-4 py-3">
              <p className="text-[11px] font-semibold text-slate-300 mb-1">📖 Règle d'or du roll</p>
              <p className="text-xs text-slate-500 leading-5">Ne rouler que si le crédit net est positif ou nul. Payer pour rouler (débit) = reporter le problème en augmentant le risque total. Mieux vaut couper une perte définie que d'ouvrir une perte indéfinie plus grande.</p>
            </div>
          </div>

          {/* Règles non-négociables */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-base font-semibold text-white mb-4">5 règles non-négociables</h2>
            <div className="space-y-3">
              {[
                {
                  num: "1", color: "bg-rose-500",
                  rule: "Couper à 2× la prime",
                  detail: "Si la perte latente atteint 2 fois la prime encaissée, fermer sans exception. Racheter l'option au prix actuel. La perte nette est alors limitée à 1× la prime initiale.",
                  example: "Prime encaissée : 500$. Prime actuelle : 1 000$ → racheter, perte nette = −500$. Sans règle : prime peut exploser à 5 000$, 10 000$."
                },
                {
                  num: "2", color: "bg-amber-500",
                  rule: "Prendre profit à 50% de la prime",
                  detail: "Racheter quand la prime a perdu 50% de sa valeur. Libère le capital, réduit le temps d'exposition au risque gamma, et évite le retournement sur la deuxième moitié.",
                  example: "Prime encaissée : 500$. Racheter à 250$ → profit +250$ réalisé, position fermée, capital libéré."
                },
                {
                  num: "3", color: "bg-orange-500",
                  rule: "Fermer ou rouler avant 21 DTE",
                  detail: "Le gamma s'accélère exponentiellement sous 21 DTE. Un mouvement de 2% peut générer plus de perte en 1 jour qu'en 2 semaines. Rouler à 30-45 DTE si encore en perte.",
                  example: "Position à 15 DTE, profitable → racheter et vendre un nouveau cycle à 45 DTE (recharge theta)."
                },
                {
                  num: "4", color: "bg-violet-500",
                  rule: "Ne jamais ajouter sur une position perdante",
                  detail: "L'averaging down multiplie le risque non-linéairement avec les options. Doubler une position en perte peut transformer une perte limitée en perte catastrophique.",
                  example: "Short Put en perte → ne pas vendre un 2ème put au même strike ou plus bas. Fermer, puis réévaluer."
                },
                {
                  num: "5", color: "bg-cyan-500",
                  rule: "Max 30% du capital engagé simultanément",
                  detail: "BTC et ETH sont corrélés > 0.85. Vendre des options sur les deux est une seule position, pas deux. Le total des pertes max possibles ne doit pas dépasser 30% du capital.",
                  example: `Capital : ${capital.toLocaleString()}$. Max risque cumulé options : ${(capital * 0.30).toFixed(0)}$. Au-delà, réduire ou couvrir.`
                },
              ].map(r => (
                <div key={r.num} className="flex gap-4 p-4 border border-white/5 rounded-lg">
                  <div className={`w-7 h-7 rounded-full ${r.color} flex items-center justify-center text-xs font-bold text-white shrink-0 mt-0.5`}>
                    {r.num}
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-white mb-1">{r.rule}</p>
                    <p className="text-xs text-slate-400 leading-5 mb-1.5">{r.detail}</p>
                    <p className="text-[11px] text-slate-600 italic border-l-2 border-white/10 pl-2">{r.example}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>
      )}

      {/* ── TAB AIDE ── */}
      {tab === "aide" && (
        <div className="space-y-5 max-w-3xl">

          {/* Intro */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-base font-semibold text-white mb-2">Qu'est-ce qu'une option ?</h2>
            <p className="text-sm text-slate-400 leading-6">
              Une option est un <span className="text-white">contrat qui donne le droit</span> (pas l'obligation) d'acheter ou de vendre un actif
              à un prix fixé (le <span className="text-indigo-300">strike</span>) avant une date d'expiration.
              Tu paies ou encaisses une <span className="text-indigo-300">prime</span> pour ce droit.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4">
              <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-3">
                <p className="text-xs font-semibold text-emerald-300 mb-1">CALL — droit d'acheter</p>
                <p className="text-xs text-slate-400">Profitable si le prix monte au-dessus du strike. L'acheteur paie une prime. Le vendeur l'encaisse.</p>
              </div>
              <div className="bg-rose-500/10 border border-rose-500/20 rounded-lg p-3">
                <p className="text-xs font-semibold text-rose-300 mb-1">PUT — droit de vendre</p>
                <p className="text-xs text-slate-400">Profitable si le prix baisse en dessous du strike. L'acheteur se protège. Le vendeur encaisse la prime.</p>
              </div>
            </div>
          </div>

          {/* 2 grandes approches */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-base font-semibold text-white mb-3">Les 2 approches fondamentales</h2>
            <div className="space-y-3">
              <div className="border border-white/5 rounded-lg p-4">
                <p className="text-sm font-semibold text-emerald-300 mb-1">Vendeur de volatilité (seller)</p>
                <p className="text-xs text-slate-400 leading-5">
                  Tu encaisses la prime immédiatement. Tu gagnes si le marché reste dans une plage de prix.
                  Avantage : le temps joue pour toi (theta positif).
                  Risque : une forte variation de prix peut dépasser ta prime.
                  <span className="text-slate-300"> → Stratégies : Short Put, Short Strangle, Iron Condor</span>
                </p>
              </div>
              <div className="border border-white/5 rounded-lg p-4">
                <p className="text-sm font-semibold text-violet-300 mb-1">Acheteur de volatilité (buyer)</p>
                <p className="text-xs text-slate-400 leading-5">
                  Tu paies une prime. Tu gagnes si le marché fait un grand mouvement dans n'importe quelle direction.
                  Avantage : perte limitée à la prime payée.
                  Risque : le temps joue contre toi (theta négatif).
                  <span className="text-slate-300"> → Stratégies : Long Straddle, Long Strangle, Spreads</span>
                </p>
              </div>
            </div>
          </div>

          {/* Greeks */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-base font-semibold text-white mb-3">Les Greeks — en une phrase</h2>
            <div className="space-y-2">
              {[
                { name: "Delta (Δ)", color: "text-cyan-300",   desc: "De combien varie le prix de l'option si le spot bouge de 1$. Ex: delta 0.25 = l'option gagne 0.25$ si BTC monte de 1$." },
                { name: "Gamma (Γ)", color: "text-indigo-300", desc: "Vitesse du delta. Un gamma élevé (proche de l'ATM) = le delta change rapidement. Dangereux près de l'expiry." },
                { name: "Vega (ν)",  color: "text-violet-300", desc: "Exposition à la volatilité implicite. +1% de vol = vega en $ de gain/perte. Crucial pour choisir achat vs vente de vol." },
                { name: "Theta (θ)", color: "text-amber-300",  desc: "Érosion temporelle quotidienne. Négatif pour l'acheteur (perd de la valeur chaque jour), positif pour le vendeur." },
                { name: "Rho (ρ)",   color: "text-slate-400",  desc: "Sensibilité aux taux d'intérêt. Peu significatif pour les options crypto à court terme." },
              ].map(g => (
                <div key={g.name} className="flex gap-3 py-2 border-b border-white/[0.04]">
                  <span className={`text-sm font-semibold mono w-24 shrink-0 ${g.color}`}>{g.name}</span>
                  <span className="text-xs text-slate-400 leading-5">{g.desc}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Flux de travail de l'outil */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-base font-semibold text-white mb-4">Mode d'emploi — flux de travail</h2>
            <div className="space-y-3">
              {[
                { step: "1", tab: "Advisor", color: "bg-indigo-500",
                  title: "Lire le régime de marché",
                  desc: "Ouvre l'onglet Advisor. Choisis BTC ou ETH. L'outil analyse l'IVP (rang de la vol sur 52 semaines), le biais directionnel (signal futures), et propose une stratégie." },
                { step: "2", tab: "Advisor", color: "bg-cyan-600",
                  title: "Vérifier le score de timing",
                  desc: "En dessous de la stratégie, le score de timing (0-100) indique si c'est le bon moment. En dessous de 55 → attendre. Au-dessus de 75 → conditions optimales pour entrer." },
                { step: "3", tab: "Advisor", color: "bg-violet-600",
                  title: "Calibrer le sizing Kelly",
                  desc: "Entre ton capital et choisis la fraction Kelly (¼ recommandé). L'outil calcule le max risque autorisé en $ et le nombre de contrats cohérent avec tes règles." },
                { step: "4", tab: "Pricer", color: "bg-emerald-600",
                  title: "Vérifier le prix avec le Pricer",
                  desc: "Dans l'onglet Pricer, entre le strike suggéré par l'Advisor. Vérifie le prix théorique Black-Scholes, les Greeks, et utilise l'IV Solver si tu as le prix du marché." },
                { step: "5", tab: "Position", color: "bg-amber-600",
                  title: "Visualiser le P&L avec Position Builder",
                  desc: "Construis les legs de ta position (ou utilise un preset Strangle/Condor). Le diagramme P&L montre ton max gain, max perte, et breakevens à expiry." },
                { step: "6", tab: "Gestion", color: "bg-rose-600",
                  title: "Surveiller avec le tab Gestion",
                  desc: "Une fois en position, va dans Gestion. Clique 'Pré-remplir depuis l'Advisor', entre le DTE restant. Les alertes te signalent : TP atteint, SL proche, strike menacé, delta drifté." },
              ].map(s => (
                <div key={s.step} className="flex gap-4">
                  <div className={`w-7 h-7 rounded-full ${s.color} flex items-center justify-center text-xs font-bold text-white shrink-0 mt-0.5`}>
                    {s.step}
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-0.5">
                      <p className="text-sm font-semibold text-white">{s.title}</p>
                      <span className="text-[10px] text-slate-500 border border-white/10 rounded px-1.5 py-0.5">{s.tab}</span>
                    </div>
                    <p className="text-xs text-slate-400 leading-5">{s.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Règles de base */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-base font-semibold text-white mb-3">Règles de base (vendeur de vol)</h2>
            <div className="space-y-2">
              {[
                { rule: "Ne jamais risquer > 2% du capital par trade", why: "Kelly criterion — protège du risque de ruine" },
                { rule: "Vendre à 21-30 jours avant expiry (DTE)", why: "Zone optimale de décroissance du theta" },
                { rule: "Prendre profit à 50% de la prime encaissée", why: "Libère le capital, réduit le temps en risque" },
                { rule: "Couper si la prime double contre toi (×2)", why: "Stop loss automatique avant que la perte devienne incontrôlable" },
                { rule: "Fermer ou rouler si DTE < 7 jours", why: "Gamma explose près de l'expiry — risque de gap brutal" },
                { rule: "Ne pas trader si le timing score < 55", why: "Conditions défavorables = EV négatif ou neutre" },
              ].map((r, i) => (
                <div key={i} className="flex gap-3 py-2 border-b border-white/[0.04]">
                  <span className="text-emerald-400 text-sm shrink-0">✓</span>
                  <div>
                    <p className="text-xs text-slate-200">{r.rule}</p>
                    <p className="text-[10px] text-slate-500 mt-0.5">{r.why}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Glossaire */}
          <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5">
            <h2 className="text-base font-semibold text-white mb-3">Glossaire rapide</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2">
              {[
                { term: "IVP",            def: "IV Percentile — rang de la vol implicite actuelle vs les 52 dernières semaines. > 70% = vol chère." },
                { term: "DVOL",           def: "Deribit Volatility Index — équivalent du VIX pour le crypto. Mesure la vol implicite 30j." },
                { term: "Vol premium",    def: "Différence entre IV implicite et vol réalisée. Positif = options surévaluées → favorable au vendeur." },
                { term: "Term structure", def: "Courbe de la vol par échéance. Backwardation (1W > 1M) = stress court terme." },
                { term: "Skew 25d",      def: "Différence IV put 25δ - call 25δ. Positif = marché craint la baisse, puts chers." },
                { term: "ATM",            def: "At The Money — strike égal au prix actuel du spot." },
                { term: "OTM",            def: "Out of The Money — strike au-delà du spot (call plus haut, put plus bas)." },
                { term: "ITM",            def: "In The Money — option qui aurait une valeur intrinsèque si exercée maintenant." },
                { term: "Roll",           def: "Fermer une option et en ouvrir une nouvelle à un strike ou une échéance différente." },
                { term: "Strangle",       def: "Vente (ou achat) simultané d'un call et d'un put OTM de même expiry." },
                { term: "Iron Condor",    def: "Strangle avec ailes achetées pour limiter la perte maximale. Risque limité des deux côtés." },
                { term: "Theta decay",    def: "Érosion de la valeur temps chaque jour. S'accélère dans les dernières semaines." },
              ].map(g => (
                <div key={g.term} className="py-1.5 border-b border-white/[0.04]">
                  <span className="text-xs font-semibold text-indigo-300 mono">{g.term}</span>
                  <span className="text-xs text-slate-500"> — {g.def}</span>
                </div>
              ))}
            </div>
          </div>

        </div>
      )}
    </div>
  )
}
