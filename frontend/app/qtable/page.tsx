import { fetchQTable } from "@/lib/api"

const ACTION_COLOR: Record<string, string> = {
  LONG:  "text-emerald-400 bg-emerald-950",
  SHORT: "text-red-400 bg-red-950",
  FLAT:  "text-slate-500 bg-slate-800",
}

const MACROS = [
  { idx: 1, name: "08:50" }, { idx: 2, name: "09:50" },
  { idx: 3, name: "10:50" }, { idx: 4, name: "11:50" },
  { idx: 5, name: "12:50" }, { idx: 6, name: "13:50" },
  { idx: 7, name: "14:50" },
]

export default async function QTablePage({
  searchParams,
}: {
  searchParams: Promise<{ mac?: string }>
}) {
  const sp  = await searchParams
  const mac = sp.mac ? Number(sp.mac) : undefined
  const data = await fetchQTable(mac).catch(() => ({ states: [], total: 0 }))

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-lg font-semibold text-white">Q-Table</h1>
          <p className="text-sm text-slate-500 mt-0.5">{data.total} états actifs</p>
        </div>

        {/* Filtre macro */}
        <div className="flex gap-1.5 flex-wrap">
          <a href="/qtable" className="px-3 py-1.5 text-xs text-slate-400 bg-white/5 rounded-md hover:bg-white/8 transition-colors">
            Toutes
          </a>
          {MACROS.map(m => (
            <a key={m.idx} href={`/qtable?mac=${m.idx}`}
              className={`px-3 py-1.5 text-xs rounded-md transition-colors mono ${
                mac === m.idx
                  ? "bg-white/10 text-white"
                  : "text-slate-500 bg-white/5 hover:bg-white/8"
              }`}>
              {m.name}
            </a>
          ))}
        </div>
      </div>

      {data.total === 0 ? (
        <div className="bg-[#0d0d14] border border-white/5 rounded-lg p-8 text-center text-slate-600 text-sm">
          Q-table non disponible (stats_agent.pkl incompatible avec l'environnement actuel).<br />
          <span className="text-slate-700">Elle sera accessible sur Railway avec l'environnement correct.</span>
        </div>
      ) : (
        <div className="bg-[#0d0d14] border border-white/5 rounded-lg overflow-hidden">
          <table className="w-full data-table">
            <thead>
              <tr>
                <th>Macro</th>
                <th>London</th>
                <th>Sweep</th>
                <th>Pool</th>
                <th className="text-right">Q Long</th>
                <th className="text-right">Q Short</th>
                <th className="text-right">N</th>
                <th className="text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {data.states.map((s: any) => (
                <tr key={s.state}>
                  <td className="mono text-slate-300">{s.mac_name}</td>
                  <td className="mono text-xs text-slate-400">{s.lc_label}</td>
                  <td className="mono text-xs text-slate-400">{s.sc_label}</td>
                  <td className="mono text-xs text-slate-400">{s.pc_label}</td>
                  <td className={`text-right mono text-xs font-medium ${s.q_long > 0 ? "text-emerald-400" : "text-red-400"}`}>
                    {(s.q_long * 100).toFixed(3)}%
                  </td>
                  <td className={`text-right mono text-xs font-medium ${s.q_short > 0 ? "text-emerald-400" : "text-red-400"}`}>
                    {(s.q_short * 100).toFixed(3)}%
                  </td>
                  <td className="text-right mono text-slate-500">{s.n}</td>
                  <td className="text-right">
                    <span className={`mono text-[10px] px-2 py-0.5 rounded ${ACTION_COLOR[s.best_action] ?? ""}`}>
                      {s.best_action}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
