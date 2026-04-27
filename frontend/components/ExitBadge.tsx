import { exitBadge } from "@/lib/utils"

export default function ExitBadge({ reason }: { reason: string }) {
  const { label, cls } = exitBadge(reason)
  return (
    <span className={`mono text-[10px] px-2 py-0.5 rounded ${cls}`}>
      {label}
    </span>
  )
}
