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

// ── Vol Signal ────────────────────────────────────────────────────────────────

export type VolSnapshot = {
  dvol: {
    asset: string
    dvol_close: number
    dvol_z: number
    dvol_roc_24h: number
    dvol_ret_std_24h: number
    dvol_state: "VOL_SHOCK_UP" | "VOL_CRUSH_DOWN" | "NEUTRAL"
    risk_regime: "RISK_OFF" | "RISK_ON" | "BALANCED"
    intensity: number
    latest_ts: string
    error?: string
  }
  signal: {
    asset: string
    close: number
    funding_annualized: number
    realized_vol_annual: number
    latest_ts: string
    signal: {
      action: "LONG" | "SHORT" | "FLAT" | "WATCH"
      horizon: string
      confidence: number
      net_score: number
      long_score: number
      short_score: number
      edge_total: number
      contract?: { instrument: string; tenor: string; why: string }
    }
    edges: Record<string, number>
    drivers: Array<{ name: string; score: number }>
    options: {
      iv_atm?: number
      iv_skew_25d?: number
      put_call_ratio?: number
      term_1w?: number
      term_1m?: number
      term_3m?: number
    }
    error?: string
  }
  vol_premium: {
    iv_atm: number
    realized_vol: number
    premium: number
    bias: "SELL_VOL" | "BUY_VOL" | "NEUTRAL"
  } | null
  timestamp: string
}

export async function fetchVolSnapshot(asset = "BTC", timeframe = "1h", days = 60) {
  const q = new URLSearchParams({ asset, timeframe, days: String(days) })
  const res = await fetch(apiUrl(`/api/vol/snapshot?${q}`), { cache: "no-store" })
  if (!res.ok) throw new Error(`vol/snapshot: ${res.status}`)
  return res.json() as Promise<VolSnapshot>
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

// ── Options Advisor ───────────────────────────────────────────────────────────

export type AdvisorLeg = {
  action: "BUY" | "SELL"
  type: "CALL" | "PUT"
  strike: number
  dte: number
  delta: string
}

export type AdvisorTiming = {
  score: number
  label: string
  color: string
  details: Array<{ label: string; pts: number; max: number; note: string }>
}

export type AdvisorResponse = {
  asset: string
  timestamp: string
  dvol_asset_supported: boolean
  ivp: number | null
  ivp_pct: number | null
  vol_regime: "SELL_VOL" | "BUY_VOL" | "NEUTRAL"
  directional_bias: "BULLISH" | "BEARISH" | "NEUTRAL"
  signal_action: "LONG" | "SHORT" | "WATCH" | "FLAT"
  dvol_state: string
  dvol_close: number | null
  dvol_z: number | null
  spot: number | null
  iv_atm: number | null
  skew_25d: number | null
  strategy: string
  strategy_label: string
  risk_profile: string
  color: string
  dte_days: number
  legs: AdvisorLeg[]
  rationale: string
  timing: AdvisorTiming | null
}

export async function fetchAdvisor(asset = "BTC", timeframe = "1h", days = 60) {
  const q = new URLSearchParams({ asset, timeframe, days: String(days) })
  const res = await fetch(apiUrl(`/api/options/advisor?${q}`), { cache: "no-store" })
  if (!res.ok) throw new Error(`options/advisor: ${res.status}`)
  return res.json() as Promise<AdvisorResponse>
}
