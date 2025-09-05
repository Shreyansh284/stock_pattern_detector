import { Link } from 'react-router-dom'

export default function Home() {
    return (
        <section className="min-h-[60vh] grid place-items-center bg-gradient-to-b from-slate-50 to-white">
            <div className="text-center">
                <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-slate-900">Choose Data Source</h1>
                <p className="mt-2 text-slate-600">Analyze live market data or past CSV data from your STOCK_DATA folder.</p>
                <div className="mt-6 flex items-center justify-center gap-3">
                    <Link to="/dashboard?source=live" className="inline-flex items-center justify-center rounded-md px-6 py-3 text-sm font-medium shadow-sm bg-slate-900 text-white hover:bg-slate-800">Live Data</Link>
                    <Link to="/dashboard?source=past" className="inline-flex items-center justify-center rounded-md px-6 py-3 text-sm font-medium border border-slate-300 text-slate-700 hover:bg-slate-50">Past Data</Link>
                </div>
            </div>
        </section>
    )
}
