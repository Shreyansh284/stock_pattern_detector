import { Link } from 'react-router-dom'

export default function Home() {
    return (
        <section className="bg-gradient-to-b from-slate-50 to-white">
            <div className="max-w-7xl mx-auto px-6 py-16 grid md:grid-cols-2 gap-8 items-center">
                <div>
                    <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-slate-900">Find patterns. Gain edge.</h1>
                    <p className="mt-3 text-slate-600">Scan stocks across multiple timeframes to spot Double Tops/Bottoms, Cup & Handle, and Head & Shoulders. Beautiful, interactive charts powered by Plotly.</p>
                    <div className="mt-6 flex gap-3">
                        <Link to="/detect" className="inline-flex items-center justify-center rounded-md px-4 py-2.5 text-sm font-medium shadow-sm bg-slate-900 text-white hover:bg-slate-800">Get started</Link>
                        <a href="https://plotly.com/" target="_blank" className="inline-flex items-center justify-center rounded-md px-4 py-2.5 text-sm font-medium border border-slate-300 text-slate-700 hover:bg-slate-50">Learn more</a>
                    </div>
                </div>
                <div>
                    <div className="border rounded-xl bg-white shadow-sm p-4">
                        <div className="h-64 bg-[radial-gradient(circle_at_top_left,#cbd5e1,transparent_60%),radial-gradient(circle_at_bottom_right,#e2e8f0,transparent_60%)] rounded-lg" />
                    </div>
                </div>
            </div>
        </section>
    )
}
