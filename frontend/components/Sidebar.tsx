"use client"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { useState } from "react"

const links = [
  { href: "/fractal", label: "Fractal" },
  { href: "/vol",     label: "Vol Signal" },
  { href: "/options", label: "Options" },
]

export default function Sidebar() {
  const path = usePathname()
  const [open, setOpen] = useState(false)

  const navLinks = (
    <nav className="flex flex-col gap-0.5 text-sm">
      {links.map(({ href, label }) => {
        const active = path.startsWith(href)
        return (
          <Link
            key={href}
            href={href}
            onClick={() => setOpen(false)}
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
  )

  const statusBlock = (
    <div className="px-3 py-3 rounded-md bg-[#0d1a12] border border-[#1a3525]">
      <div className="flex items-center gap-2 mb-1">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        <span className="text-[11px] text-emerald-400 font-medium">Live actif</span>
      </div>
      <p className="mono text-[10px] text-slate-500">Pi* · BTC/ETH</p>
    </div>
  )

  return (
    <>
      {/* ── Desktop sidebar ─────────────────────────────────────── */}
      <aside className="hidden md:flex fixed left-0 top-0 h-full w-[210px] bg-[#0d0d14] border-r border-white/5 flex-col py-6 px-4 z-40">
        <div className="px-2 mb-8">
          <span className="mono text-sm font-medium text-white tracking-wider">Pi*</span>
          <span className="ml-2 text-[10px] text-slate-500 uppercase tracking-widest">
            {process.env.NEXT_PUBLIC_TRADING_SYMBOL || "BTCUSDT"}
          </span>
        </div>
        {navLinks}
        <div className="mt-auto">{statusBlock}</div>
      </aside>

      {/* ── Mobile top bar ──────────────────────────────────────── */}
      <div className="md:hidden fixed top-0 left-0 right-0 z-50 bg-[#0d0d14] border-b border-white/5 flex items-center justify-between px-4 py-3">
        <span className="mono text-sm font-medium text-white">Pi*</span>
        <button
          onClick={() => setOpen(o => !o)}
          className="text-slate-400 hover:text-white p-1"
          aria-label="Menu"
        >
          {open ? (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          ) : (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          )}
        </button>
      </div>

      {/* ── Mobile drawer ───────────────────────────────────────── */}
      {open && (
        <div className="md:hidden fixed inset-0 z-40" onClick={() => setOpen(false)}>
          <div
            className="absolute top-12 left-0 right-0 bg-[#0d0d14] border-b border-white/5 px-4 py-4 space-y-1"
            onClick={e => e.stopPropagation()}
          >
            {navLinks}
            <div className="pt-3">{statusBlock}</div>
          </div>
        </div>
      )}
    </>
  )
}
