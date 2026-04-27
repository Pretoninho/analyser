export function fmt(n: number | null | undefined, decimals = 2): string {
  if (n == null) return "—"
  return n.toFixed(decimals)
}

export function fmtPct(n: number | null | undefined): string {
  if (n == null) return "—"
  const s = (n * 100).toFixed(2)
  return n >= 0 ? `+${s}%` : `${s}%`
}

export function fmtPrice(n: number | null | undefined): string {
  if (n == null) return "—"
  return n.toLocaleString("fr-FR", { minimumFractionDigits: 0, maximumFractionDigits: 2 })
}

export function today(): string {
  return new Date().toISOString().slice(0, 10)
}

export function prevDay(d: string): string {
  const dt = new Date(d)
  dt.setDate(dt.getDate() - 1)
  return dt.toISOString().slice(0, 10)
}

export function nextDay(d: string): string {
  const dt = new Date(d)
  dt.setDate(dt.getDate() + 1)
  return dt.toISOString().slice(0, 10)
}

export function pnlClass(n: number | null | undefined): string {
  if (n == null) return "text-slate-500"
  if (n > 0) return "text-emerald-400"
  if (n < 0) return "text-red-400"
  return "text-slate-400"
}

export function exitBadge(reason: string): { label: string; cls: string } {
  switch (reason?.toUpperCase()) {
    case "TP":  return { label: "TP",  cls: "bg-emerald-950 text-emerald-400" }
    case "SL":  return { label: "SL",  cls: "bg-red-950 text-red-400" }
    case "EOD": return { label: "EOD", cls: "bg-slate-800 text-slate-400" }
    default:    return { label: reason ?? "—", cls: "bg-slate-800 text-slate-500" }
  }
}
