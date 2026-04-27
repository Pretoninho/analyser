import { redirect } from "next/navigation"

export default function DailyIndex() {
  const today = new Date().toISOString().slice(0, 10)
  redirect(`/daily/${today}`)
}
