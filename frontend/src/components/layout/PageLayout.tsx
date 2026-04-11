import { Outlet } from 'react-router-dom'
import { useState } from 'react'
import Sidebar from './Sidebar'

export default function PageLayout() {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <div className="flex min-h-screen">
      <Sidebar mobileOpen={mobileOpen} onClose={() => setMobileOpen(false)} />

      {/* Hamburger button — only visible on mobile */}
      <button
        onClick={() => setMobileOpen(true)}
        className="fixed top-4 left-4 z-30 lg:hidden w-10 h-10 rounded-xl bg-white/[0.06] border border-white/[0.08] backdrop-blur-xl flex items-center justify-center text-white/60 hover:text-white hover:bg-violet-600/30 hover:border-violet-500/30 transition-all shadow-lg"
        aria-label="Open menu"
      >
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M3 5H17M3 10H17M3 15H17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      </button>

      <main className="flex-1 lg:ml-64 min-h-screen flex flex-col">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 pt-16 lg:pt-10 animate-fade-in flex-1 w-full">
          <Outlet />
        </div>
        <footer className="border-t border-white/5 py-8 px-8 text-center bg-black/20">
          <p className="text-white/20 text-xs font-medium tracking-wider">
            © {new Date().getFullYear()} ASIMAN IMANZADE DATA SCIENCE EDUCATION PLATFORM · ALL RIGHTS RESERVED
          </p>
        </footer>
      </main>
    </div>
  )
}
