"use client"

import { useState } from "react"
import { notifyDeribitSignal } from "@/lib/api"

export default function DeribitSignalActions({
  timeframe,
  days,
}: {
  timeframe: string
  days: number
}) {
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string>("")
  const [isErr, setIsErr] = useState(false)

  async function sendDiscord() {
    setBusy(true)
    setMsg("")
    setIsErr(false)
    try {
      const out = await notifyDeribitSignal(timeframe, days)
      setMsg(`Discord envoye: action=${out.action} conf=${(out.confidence ?? 0).toFixed(2)}`)
    } catch (e) {
      setIsErr(true)
      setMsg(e instanceof Error ? e.message : "Erreur inconnue")
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-4 mb-5">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div>
          <p className="text-sm text-white font-semibold">Actions Signal Deribit</p>
          <p className="text-xs text-slate-500 mt-0.5">Envoyer le signal courant sur Discord</p>
        </div>
        <button
          type="button"
          onClick={sendDiscord}
          disabled={busy}
          className="px-3 py-2 rounded-md text-xs font-medium border border-cyan-400/30 text-cyan-200 bg-cyan-500/10 hover:bg-cyan-500/15 disabled:opacity-60"
        >
          {busy ? "Envoi..." : "Notifier Discord"}
        </button>
      </div>
      {msg && (
        <p className={`mt-3 text-xs ${isErr ? "text-rose-300" : "text-emerald-300"}`}>{msg}</p>
      )}
    </div>
  )
}
