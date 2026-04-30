"use client"
import Link from "next/link"
import { usePathname } from "next/navigation"

const links = [
  { href: "/daily",       label: "Rapport journalier" },
  { href: "/pipeline",    label: "Pipeline" },
  { href: "/trades",      label: "Trades" },
  { href: "/performance", label: "Performance" },
  { href: "/qtable",      label: "Q-Table" },
]

export default function Sidebar() {
  const path = usePathname()

  return (
    <aside className="fixed left-0 top-0 h-full w-[210px] bg-[#0d0d14] border-r border-white/5 flex flex-col py-6 px-4">
      {/* Logo */}
      <div className="px-2 mb-8">
        <span className="mono text-sm font-medium text-white tracking-wider">Pi*</span>
        <span className="ml-2 text-[10px] text-slate-500 uppercase tracking-widest">BTCUSDT</span>
      </div>

      {/* Nav */}
      <nav className="flex flex-col gap-0.5 text-sm">
        {links.map(({ href, label }) => {
          const active = path.startsWith(href)
          return (
            <Link
              key={href}
              href={href}
              className={`px-3 py-2 rounded-md transition-colors ${
                active
                  ? "text-white bg-white/8"
                  : "text-slate-500 hover:text-slate-300 hover:bg-white/4"
              }`}
            >
              {label}
            </Link>
          )
        })}
      </nav>

      {/* Status */}
      <div className="mt-auto px-3 py-3 rounded-md bg-[#0d1a12] border border-[#1a3525]">
        <div className="flex items-center gap-2 mb-1">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-[11px] text-emerald-400 font-medium">Live actif</span>
        </div>
        <p className="mono text-[10px] text-slate-500">09:50 ET — mac=2</p>
      </div>
    </aside>
  )
}
