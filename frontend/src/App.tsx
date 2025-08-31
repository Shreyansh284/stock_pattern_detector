import { Routes, Route, NavLink } from 'react-router-dom'
import Home from './pages/Home'
import Detect from './pages/Detect'

export default function App() {
    return (
        <div className="min-h-screen flex flex-col">
            <header className="border-b bg-white/70 backdrop-blur supports-[backdrop-filter]:bg-white/60 sticky top-0 z-10">
                <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
                    <div className="font-semibold text-lg">Stock Pattern Detector</div>
                    <nav className="flex gap-4 text-sm">
                        <NavLink className={({ isActive }) => `px-3 py-2 rounded-md ${isActive ? 'bg-slate-900 text-white' : 'hover:bg-slate-100'}`} to="/">Home</NavLink>
                        <NavLink className={({ isActive }) => `px-3 py-2 rounded-md ${isActive ? 'bg-slate-900 text-white' : 'hover:bg-slate-100'}`} to="/detect">Detect</NavLink>
                    </nav>
                </div>
            </header>
            <main className="flex-1">
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/detect" element={<Detect />} />
                </Routes>
            </main>
            <footer className="border-t py-4 text-center text-xs text-slate-500 bg-white">Built with React + Vite + Tailwind</footer>
        </div>
    )
}
