interface Props {
  lc: string
  sc: string
  pc: string
  muted?: boolean
}

export default function CtxBadges({ lc, sc, pc, muted }: Props) {
  const cls = muted
    ? "text-[10px] text-slate-600 bg-white/[0.03] px-1.5 py-0.5 rounded mono"
    : "text-[10px] text-slate-400 bg-white/5 px-1.5 py-0.5 rounded mono"
  return (
    <div className="flex gap-1.5 flex-wrap">
      <span className={cls}>{lc}</span>
      <span className={cls}>{sc}</span>
      <span className={cls}>{pc}</span>
    </div>
  )
}
