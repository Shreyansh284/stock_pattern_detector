import { Routes, Route, NavLink } from 'react-router-dom'
import Home from './pages/Home'
import Detect from './pages/Detect'

export default function App() {
    return (
        <div className="min-h-screen flex flex-col">
            <header className="border-b bg-white/70 backdrop-blur supports-[backdrop-filter]:bg-white/60 sticky top-0 z-10">
                <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="h-8 w-8 rounded-md bg-slate-900 text-white grid place-items-center font-bold">SP</div>
                        <div className="font-semibold text-lg tracking-tight">Stock Pattern Detector</div>
                    </div>
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
            <footer className="border-t py-6 text-center text-xs text-slate-500 bg-white">© {new Date().getFullYear()} Stock Pattern Detector</footer>
        </div>
    )
}
