"use client"

type Setup = "strict" | "modere" | "frequent"

interface SetupToggleProps {
  active: Setup
  onChange: (setup: Setup) => void
}

const setups: { id: Setup; label: string; description: string }[] = [
  { id: "strict", label: "STRICT", description: "W+D+KZ+BR" },
  { id: "modere", label: "MODÉRÉ", description: "D+KZ+BR" },
  { id: "frequent", label: "FRÉQUENT", description: "KZ+BR" },
]

export default function SetupToggle({ active, onChange }: SetupToggleProps) {
  return (
    <div className="flex gap-2">
      {setups.map(({ id, label, description }) => (
        <button
          key={id}
          onClick={() => onChange(id)}
          className={`px-4 py-2 rounded-md text-sm transition-colors ${
            active === id
              ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/50"
              : "bg-white/5 text-slate-400 border border-white/10 hover:bg-white/8"
          }`}
          title={description}
        >
          {label}
        </button>
      ))}
    </div>
  )
}
