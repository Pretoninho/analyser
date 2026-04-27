"use client"
import { useEffect, useRef } from "react"

interface Candle {
  time: number
  open: number
  high: number
  low: number
  close: number
}

interface Props {
  candles: Candle[]
  entryPx?: number | null
  tpPx?: number | null
  slPx?: number | null
  macStart?: number   // minutes ET depuis minuit
  preStart?: number
}

export default function CandleChart({ candles, entryPx, tpPx, slPx }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current || !candles.length) return

    let chart: any
    let cleanup = false

    import("lightweight-charts").then(({ createChart, CrosshairMode, LineStyle }) => {
      if (cleanup || !ref.current) return

      chart = createChart(ref.current, {
        layout: {
          background: { color: "#0d0d14" },
          textColor:  "#64748b",
        },
        grid: {
          vertLines: { color: "rgba(255,255,255,0.03)" },
          horzLines: { color: "rgba(255,255,255,0.03)" },
        },
        crosshair: { mode: CrosshairMode.Normal },
        rightPriceScale: { borderColor: "rgba(255,255,255,0.06)" },
        timeScale: {
          borderColor:      "rgba(255,255,255,0.06)",
          timeVisible:      true,
          secondsVisible:   false,
        },
        width:  ref.current.clientWidth,
        height: 320,
      })

      const series = chart.addCandlestickSeries({
        upColor:        "#4ade80",
        downColor:      "#f87171",
        borderUpColor:  "#4ade80",
        borderDownColor:"#f87171",
        wickUpColor:    "#4ade80",
        wickDownColor:  "#f87171",
      })
      series.setData(candles)

      // Lignes Entry / TP / SL
      if (entryPx) {
        series.createPriceLine({ price: entryPx, color: "#94a3b8", lineWidth: 1, lineStyle: LineStyle.Dashed, title: "Entry" })
      }
      if (tpPx) {
        series.createPriceLine({ price: tpPx, color: "#4ade80", lineWidth: 1, lineStyle: LineStyle.Dotted, title: "TP" })
      }
      if (slPx) {
        series.createPriceLine({ price: slPx, color: "#f87171", lineWidth: 1, lineStyle: LineStyle.Dotted, title: "SL" })
      }

      chart.timeScale().fitContent()

      const obs = new ResizeObserver(() => {
        if (ref.current) chart.resize(ref.current.clientWidth, 320)
      })
      obs.observe(ref.current)
      return () => obs.disconnect()
    })

    return () => {
      cleanup = true
      chart?.remove()
    }
  }, [candles, entryPx, tpPx, slPx])

  return <div ref={ref} className="w-full rounded-lg overflow-hidden" />
}
