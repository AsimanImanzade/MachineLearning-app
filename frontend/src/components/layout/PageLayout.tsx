import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'

export default function PageLayout() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 ml-64 min-h-screen flex flex-col">
        <div className="max-w-6xl mx-auto px-8 py-10 animate-fade-in flex-1">
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
