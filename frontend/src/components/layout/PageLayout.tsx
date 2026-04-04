import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'

export default function PageLayout() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 ml-64 min-h-screen">
        <div className="max-w-6xl mx-auto px-8 py-10 animate-fade-in">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
