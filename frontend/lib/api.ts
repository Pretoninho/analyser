const RAW_BASE = process.env.NEXT_PUBLIC_API_URL
const BASE = RAW_BASE?.replace(/\/$/, "")

function getBaseUrl(): string {
  if (BASE) return BASE
  if (process.env.NODE_ENV !== "production") return "http://localhost:8000"
  throw new Error("NEXT_PUBLIC_API_URL is not set in production")
}

function apiUrl(path: string): string {
  return `${getBaseUrl()}${path}`
}

export async function fetchDaily(date: string) {
  const res = await fetch(apiUrl(`/api/daily/${date}`), { cache: "no-store" })
  if (!res.ok) throw new Error(`daily ${date}: ${res.status}`)
  return res.json()
}

export async function fetchTrades(params?: {
  type?: string
  mac_idx?: number
  exit_reason?: string
  from?: string
  to?: string
}) {
  const q = new URLSearchParams()
  if (params?.type)        q.set("type", params.type)
  if (params?.mac_idx)     q.set("mac_idx", String(params.mac_idx))
  if (params?.exit_reason) q.set("exit_reason", params.exit_reason)
  if (params?.from)        q.set("from", params.from)
  if (params?.to)          q.set("to", params.to)
  const res = await fetch(apiUrl(`/api/trades?${q}`), { cache: "no-store" })
  if (!res.ok) throw new Error(`trades: ${res.status}`)
  return res.json()
}

export async function fetchCandles(date: string, macIdx: number) {
  const res = await fetch(apiUrl(`/api/candles/${date}/${macIdx}`), { cache: "no-store" })
  if (!res.ok) throw new Error(`candles: ${res.status}`)
  return res.json()
}

export async function fetchPerformance(type?: string) {
  const q = type ? `?type=${type}` : ""
  const res = await fetch(apiUrl(`/api/performance${q}`), { cache: "no-store" })
  if (!res.ok) throw new Error(`performance: ${res.status}`)
  return res.json()
}

export async function fetchQTable(macIdx?: number) {
  const q = macIdx != null ? `?mac_idx=${macIdx}` : ""
  const res = await fetch(apiUrl(`/api/qtable${q}`), { cache: "no-store" })
  if (!res.ok) throw new Error(`qtable: ${res.status}`)
  return res.json()
}

export type SignalRunResponse = {
  exit_code: number
  stdout: string
  stderr: string
}

async function postRun(path: "/api/live/run" | "/api/shadow/run") {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
    cache: "no-store",
  })
  if (!res.ok) throw new Error(`${path}: ${res.status}`)
  return res.json() as Promise<SignalRunResponse>
}

export async function runLiveSignal() {
  return postRun("/api/live/run")
}

export async function runShadowSignal() {
  return postRun("/api/shadow/run")
}

export type DeribitSignalResponse = {
  asset: string
  timeframe: string
  days: number
  latest_ts: string | null
  close: number
  funding_annualized: number
  realized_vol_annual: number
  signal: {
    action: "LONG" | "SHORT" | "FLAT" | "WATCH"
    horizon: string
    confidence: number
    net_score: number
    long_score: number
    short_score: number
    edge_total: number
    contract?: {
      instrument: string
      tenor: string
      why: string
    }
  }
  edges: Record<string, number>
  drivers: Array<{ name: string; score: number }>
  snapshot: Record<string, unknown>
  options: Record<string, unknown>
}

export type DeribitBacktestResponse = {
  asset: string
  timeframe: string
  days: number
  threshold: number
  total_bars: number
  results: Array<{
    edge: string
    horizon_h: number
    n_signals: number
    hit_ratio: number | null
    avg_ret_active: number | null
    avg_ret_baseline: number | null
    corr: number | null
    lift: number | null
    note: string
  }>
}

export async function fetchDeribitSignal(timeframe = "1h", days = 90) {
  const q = new URLSearchParams({ timeframe, days: String(days) })
  const res = await fetch(apiUrl(`/api/deribit/signal?${q.toString()}`), { cache: "no-store" })
  if (!res.ok) throw new Error(`deribit/signal: ${res.status}`)
  return res.json() as Promise<DeribitSignalResponse>
}

export async function fetchDeribitBacktest(timeframe = "1h", days = 90, threshold = 0.05) {
  const q = new URLSearchParams({ timeframe, days: String(days), threshold: String(threshold) })
  const res = await fetch(apiUrl(`/api/deribit/backtest?${q.toString()}`), { cache: "no-store" })
  if (!res.ok) throw new Error(`deribit/backtest: ${res.status}`)
  return res.json() as Promise<DeribitBacktestResponse>
}

export async function notifyDeribitSignal(timeframe = "1h", days = 90) {
  const q = new URLSearchParams({ timeframe, days: String(days) })
  const res = await fetch(apiUrl(`/api/deribit/futures/notify?${q.toString()}`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
    cache: "no-store",
  })
  if (!res.ok) throw new Error(`deribit/notify: ${res.status}`)
  return res.json() as Promise<{
    status: string
    timeframe: string
    days: number
    action: string
    confidence: number
    timestamp: string
  }>
}

export type FractalSignal = {
  timestamp: string
  setup: string
  day_date: string
  kz: string
  pattern: string
  entry_price: number
  confidence: number
  levels: Record<string, number>
}

export type FractalSetupResponse = {
  setup: string
  symbol: string
  count: number
  confidence: number
  signals: FractalSignal[]
}

export async function fetchFractalSignals(setup: "strict" | "modere" | "frequent", symbol?: string) {
  const q = symbol ? `?symbol=${symbol}` : ""
  const res = await fetch(apiUrl(`/api/fractal/${setup}${q}`), { cache: "no-store" })
  if (!res.ok) throw new Error(`fractal/${setup}: ${res.status}`)
  return res.json() as Promise<FractalSetupResponse>
}

export async function fetchFractalStats(symbol?: string) {
  const q = symbol ? `?symbol=${symbol}` : ""
  const res = await fetch(apiUrl(`/api/fractal/stats${q}`), { cache: "no-store" })
  if (!res.ok) throw new Error(`fractal/stats: ${res.status}`)
  return res.json() as Promise<{
    total_signals: number
    by_setup: Record<string, number>
    by_pattern: Record<string, number>
    uptime: string
  }>
}

export async function fetchFractalHealth() {
  const res = await fetch(apiUrl("/api/fractal/health"), { cache: "no-store" })
  if (!res.ok) throw new Error(`fractal/health: ${res.status}`)
  return res.json() as Promise<{
    status: string
    orchestrator: string
    symbol: string
    available_symbols: string[]
  }>
}
