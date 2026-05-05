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
      <body className="min-h-full bg-[#080810]">
        <Sidebar />
        {/* Desktop : marge gauche pour la sidebar. Mobile : marge haute pour la top bar */}
        <main className="md:ml-[210px] pt-14 md:pt-0 p-4 md:p-8 min-h-screen">
          {children}
        </main>
      </body>
    </html>
  )
}
