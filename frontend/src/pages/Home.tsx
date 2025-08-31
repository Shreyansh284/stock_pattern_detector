import { Link } from 'react-router-dom'
import Plot from 'react-plotly.js'

export default function Home() {
    return (
        <>
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
                            <Plot
                                data={[
                                    {
                                        x: ['2025-01-01', '2025-02-01', '2025-03-01', '2025-04-01', '2025-05-01'],
                                        y: [100, 120, 110, 130, 125],
                                        type: 'scatter',
                                        mode: 'lines+markers',
                                        marker: { color: '#2563EB' },
                                    },
                                ]}
                                layout={{
                                    title: 'Sample Stock Price',
                                    autosize: true,
                                    margin: { t: 40, r: 20, l: 40, b: 40 },
                                }}
                                useResizeHandler={true}
                                className="w-full h-64"
                                config={{ displayModeBar: false }}
                            />
                        </div>
                    </div>
                </div>
            </section>
            {/* Features section */}
            <section className="bg-white">
                <div className="max-w-7xl mx-auto px-6 py-12">
                    <h2 className="text-2xl font-semibold text-slate-900 mb-4">Key Features</h2>
                    <ul className="list-disc list-inside space-y-2 text-slate-700">
                        <li>📈 Detect Double Tops/Bottoms, Cup & Handle, Head & Shoulders</li>
                        <li>⏱️ Scan across multiple predefined timeframes or custom date ranges</li>
                        <li>💾 Persist your last selections for quick reuse</li>
                        <li>📊 Export interactive charts as PNG images</li>
                        <li>⚡ Fast, client-side filtering and validation for a smooth experience</li>
                    </ul>
                </div>
            </section>
        </>
    )
}
