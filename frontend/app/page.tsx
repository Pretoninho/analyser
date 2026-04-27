import { redirect } from "next/navigation"
import { today } from "@/lib/utils"

export default function Home() {
  redirect(`/daily/${today()}`)
}
