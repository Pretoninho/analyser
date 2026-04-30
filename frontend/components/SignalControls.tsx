"use client"

import { useMemo, useState } from "react"
import { runLiveSignal, runShadowSignal, type SignalRunResponse } from "@/lib/api"

type RunKind = "live" | "shadow"

function shortText(text: string, maxLen = 1400) {
  if (!text) return ""
  return text.length > maxLen ? `${text.slice(0, maxLen)}\n...` : text
}

export default function SignalControls() {
  const [running, setRunning] = useState<RunKind | null>(null)
  const [result, setResult] = useState<SignalRunResponse | null>(null)
  const [error, setError] = useState("")
  const [lastRun, setLastRun] = useState<RunKind | null>(null)

  const runLabel = useMemo(() => {
    if (lastRun === "live") return "LIVE"
    if (lastRun === "shadow") return "SHADOW"
    return "-"
  }, [lastRun])

  async function run(kind: RunKind) {
    try {
      setRunning(kind)
      setError("")
      setResult(null)
      setLastRun(kind)

      const out = kind === "live" ? await runLiveSignal() : await runShadowSignal()
      setResult(out)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur inconnue")
    } finally {
      setRunning(null)
    }
  }

  return (
    <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-5 mb-6">
      <div className="flex items-start justify-between gap-3 mb-4">
        <div>
          <h2 className="text-sm font-semibold text-white">Signal Runner</h2>
          <p className="text-xs text-slate-500 mt-1">
            Declenche manuellement les scripts live/shadow via l&apos;API Railway.
          </p>
        </div>
        <span className="text-[11px] px-2 py-1 rounded bg-white/5 text-slate-400">
          last: {runLabel}
        </span>
      </div>

      <div className="flex flex-wrap gap-2 mb-4">
        <button
          type="button"
          onClick={() => run("live")}
          disabled={running !== null}
          className="px-3 py-2 text-xs rounded-md bg-emerald-500/20 text-emerald-300 border border-emerald-400/20 hover:bg-emerald-500/30 disabled:opacity-50"
        >
          {running === "live" ? "RUNNING LIVE..." : "RUN LIVE NOW"}
        </button>

        <button
          type="button"
          onClick={() => run("shadow")}
          disabled={running !== null}
          className="px-3 py-2 text-xs rounded-md bg-sky-500/20 text-sky-300 border border-sky-400/20 hover:bg-sky-500/30 disabled:opacity-50"
        >
          {running === "shadow" ? "RUNNING SHADOW..." : "RUN SHADOW NOW"}
        </button>
      </div>

      {error && (
        <div className="text-xs text-rose-300 bg-rose-500/10 border border-rose-500/20 rounded p-3 mb-3">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-3">
          <div className="text-xs text-slate-300">
            exit_code: <span className="mono">{result.exit_code}</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="rounded border border-white/10 bg-black/20 p-3">
              <p className="text-[11px] text-slate-500 mb-2">stdout</p>
              <pre className="text-[11px] leading-5 text-slate-300 whitespace-pre-wrap break-words">
                {shortText(result.stdout) || "(empty)"}
              </pre>
            </div>

            <div className="rounded border border-white/10 bg-black/20 p-3">
              <p className="text-[11px] text-slate-500 mb-2">stderr</p>
              <pre className="text-[11px] leading-5 text-slate-300 whitespace-pre-wrap break-words">
                {shortText(result.stderr) || "(empty)"}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
