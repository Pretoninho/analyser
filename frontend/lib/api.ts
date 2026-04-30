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
