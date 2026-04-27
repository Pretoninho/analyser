import type { Metadata } from "next"
import "./globals.css"
import Sidebar from "@/components/Sidebar"

export const metadata: Metadata = {
  title: "Pi* — BTC Trading",
  description: "Pi* ICT trading dashboard",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr" className="h-full">
      <body className="min-h-full">
        <Sidebar />
        <main className="ml-[210px] p-8 min-h-screen">
          {children}
        </main>
      </body>
    </html>
  )
}
