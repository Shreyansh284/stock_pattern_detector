import { Link } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { fetchTickerTape, type TickerItem } from '../lib/api'

type Ticker = {
    symbol: string
    price: number
    change: number // percentage
    data: number[]
}

function Sparkline({ points, color = '#22c55e' }: { points: number[]; color?: string }) {
    const w = 100
    const h = 32
    const min = Math.min(...points)
    const max = Math.max(...points)
    const range = Math.max(1e-6, max - min)
    const path = points
        .map((v, i) => {
            const x = (i / (points.length - 1)) * w
            const y = h - ((v - min) / range) * (h - 2) - 1
            return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
        })
        .join(' ')
    const rising = points[points.length - 1] >= points[0]
    const stroke = color || (rising ? '#22c55e' : '#ef4444')
    return (
        <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} className="overflow-visible">
            <path d={path} fill="none" stroke={stroke} strokeWidth={2} />
        </svg>
    )
}

function MiniSparkline({ points, upColor = '#10b981', downColor = '#ef4444' }: { points: number[]; upColor?: string; downColor?: string }) {
    const w = 60
    const h = 18
    if (!points || points.length < 2) {
        return <div style={{ width: w, height: h }} className="bg-slate-100 rounded" />
    }
    const min = Math.min(...points)
    const max = Math.max(...points)
    const range = Math.max(1e-6, max - min)
    const path = points
        .map((v, i) => {
            const x = (i / (points.length - 1)) * w
            const y = h - ((v - min) / range) * (h - 2) - 1
            return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
        })
        .join(' ')
    const rising = points[points.length - 1] >= points[0]
    const stroke = rising ? upColor : downColor
    return (
        <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} className="overflow-visible">
            <path d={path} fill="none" stroke={stroke} strokeWidth={1.75} />
        </svg>
    )
}

export default function Home() {
    const [tickers, setTickers] = useState<TickerItem[]>([])
    const [loading, setLoading] = useState<boolean>(true)
    useEffect(() => {
        let mounted = true
        const CACHE_KEY = 'ticker_tape_cache_v1'
        const CACHE_TTL = 60_000 // 60s
        // Try cache first for instant paint
        try {
            const raw = localStorage.getItem(CACHE_KEY)
            if (raw) {
                const cached = JSON.parse(raw) as { ts: number; items: TickerItem[] }
                if (cached && Array.isArray(cached.items) && (Date.now() - cached.ts) < CACHE_TTL) {
                    setTickers(cached.items)
                    setLoading(false)
                }
            }
        } catch { }
        const load = () => {
            fetchTickerTape(20)
                .then((items) => { if (mounted) { setTickers(items); setLoading(false); try { localStorage.setItem(CACHE_KEY, JSON.stringify({ ts: Date.now(), items })) } catch { } } })
                .catch(() => {
                    // Fallback demo items
                    const fallback: TickerItem[] = [
                        { symbol: 'RELIANCE.NS', display_symbol: 'RELIANCE', price: 2891.2, change_pct: 0.82, volume: 0, avg_volume: 0, price_spike: true, volume_spike: false, sparkline: [100, 102, 99, 104, 108, 107, 110, 112, 111, 115] },
                        { symbol: 'TCS.NS', display_symbol: 'TCS', price: 3945.5, change_pct: -0.14, volume: 0, avg_volume: 0, price_spike: false, volume_spike: false, sparkline: [110, 109, 111, 112, 110, 108, 109, 107, 106, 105] },
                        { symbol: 'INFY.NS', display_symbol: 'INFY', price: 1682.1, change_pct: 1.12, volume: 0, avg_volume: 0, price_spike: true, volume_spike: false, sparkline: [80, 82, 84, 83, 85, 86, 88, 90, 92, 93] },
                        { symbol: 'HDFCBANK.NS', display_symbol: 'HDFCBANK', price: 1525.3, change_pct: 0.35, volume: 0, avg_volume: 0, price_spike: false, volume_spike: false, sparkline: [70, 69, 70, 71, 72, 73, 72, 74, 75, 76] },
                    ]
                    if (mounted) { setTickers(fallback); setLoading(false); try { localStorage.setItem(CACHE_KEY, JSON.stringify({ ts: Date.now(), items: fallback })) } catch { } }
                })
        }
        // Only hit network immediately if cache is stale
        try {
            const raw = localStorage.getItem(CACHE_KEY)
            const cached = raw ? (JSON.parse(raw) as { ts: number; items: TickerItem[] }) : null
            const fresh = cached && (Date.now() - cached.ts) < CACHE_TTL
            if (!fresh) load()
        } catch { load() }
        const id = window.setInterval(load, 60000)
        return () => { mounted = false; window.clearInterval(id) }
    }, [])
    return (
        <>
            {/* Top ticker tape */}
            <div className="w-full border-b bg-white/80 backdrop-blur supports-[backdrop-filter]:bg-white/60">
                <div className="ticker">
                    <div className="ticker__track" aria-hidden={false}>
                        <div className="ticker__group flex gap-8 py-2 text-xs">
                            {loading || tickers.length === 0 ? (
                                Array.from({ length: 10 }).map((_, i) => (
                                    <div key={`skeleton-a-${i}`} className="flex items-center gap-2 whitespace-nowrap">
                                        <span className="h-3 w-16 bg-slate-200 rounded animate-pulse" />
                                        <span className="h-3 w-10 bg-slate-100 rounded animate-pulse" />
                                        <span className="h-3 w-8 bg-slate-100 rounded animate-pulse" />
                                        <span className="h-3 w-[60px] bg-slate-100 rounded animate-pulse" />
                                    </div>
                                ))
                            ) : (
                                tickers.map((t, i) => (
                                    <div key={`a-${i}`} className="flex items-center gap-2 whitespace-nowrap">
                                        <span className="font-semibold text-slate-700">{t.display_symbol || t.symbol}</span>
                                        <span className="font-mono text-slate-600">{t.price.toFixed(2)}</span>
                                        <span className={`font-mono ${t.change_pct >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>{t.change_pct >= 0 ? '+' : ''}{t.change_pct.toFixed(2)}%</span>
                                        <MiniSparkline points={t.sparkline || []} />
                                        {t.price_spike && <span className="text-emerald-600" title="Price spike">•</span>}
                                        {t.volume_spike && <span className="text-indigo-600" title="Volume spike">▲</span>}
                                    </div>
                                ))
                            )}
                        </div>
                        <div className="ticker__group flex gap-8 py-2 text-xs" aria-hidden>
                            {loading || tickers.length === 0 ? (
                                Array.from({ length: 10 }).map((_, i) => (
                                    <div key={`skeleton-b-${i}`} className="flex items-center gap-2 whitespace-nowrap">
                                        <span className="h-3 w-16 bg-slate-200 rounded animate-pulse" />
                                        <span className="h-3 w-10 bg-slate-100 rounded animate-pulse" />
                                        <span className="h-3 w-8 bg-slate-100 rounded animate-pulse" />
                                        <span className="h-3 w-[60px] bg-slate-100 rounded animate-pulse" />
                                    </div>
                                ))
                            ) : (
                                tickers.map((t, i) => (
                                    <div key={`b-${i}`} className="flex items-center gap-2 whitespace-nowrap">
                                        <span className="font-semibold text-slate-700">{t.display_symbol || t.symbol}</span>
                                        <span className="font-mono text-slate-600">{t.price.toFixed(2)}</span>
                                        <span className={`font-mono ${t.change_pct >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>{t.change_pct >= 0 ? '+' : ''}{t.change_pct.toFixed(2)}%</span>
                                        <MiniSparkline points={t.sparkline || []} />
                                        {t.price_spike && <span className="text-emerald-600" title="Price spike">•</span>}
                                        {t.volume_spike && <span className="text-indigo-600" title="Volume spike">▲</span>}
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Hero */}
            <section className="relative overflow-hidden bg-gradient-to-b from-slate-50 via-white to-white">
                <div className="absolute inset-0 bg-grid pointer-events-none" />
                <div className="max-w-7xl mx-auto px-6 py-16 grid md:grid-cols-2 gap-10 items-center relative">
                    <div>
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border text-xs text-slate-600 bg-white/70">
                            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" /> Real-time & historical pattern scans
                        </div>
                        <h1 className="mt-4 text-3xl md:text-5xl font-semibold tracking-tight text-slate-900">Spot market patterns with precision.</h1>
                        <p className="mt-3 text-slate-600 md:text-lg">Detect Double Tops/Bottoms, Cup & Handle, and Head & Shoulders across timeframes. Interactive charts, clear validation, and targets.</p>
                        <div className="mt-6 flex gap-3">
                            <Link to="/dashboard?source=live" className="inline-flex items-center justify-center rounded-md px-5 py-2.5 text-sm font-medium shadow-sm bg-slate-900 text-white hover:bg-slate-800">Live Data</Link>
                            <Link to="/dashboard?source=past" className="inline-flex items-center justify-center rounded-md px-5 py-2.5 text-sm font-medium border border-slate-300 text-slate-700 hover:bg-slate-50">Past Data</Link>
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        {(loading && tickers.length === 0) ? (
                            Array.from({ length: 4 }).map((_, i) => (
                                <div key={`hero-skel-${i}`} className="rounded-xl border bg-white/80 backdrop-blur p-4 shadow-sm">
                                    <div className="flex items-baseline justify-between">
                                        <div className="h-4 w-24 bg-slate-200 rounded animate-pulse" />
                                        <div className="h-3 w-10 bg-slate-100 rounded animate-pulse" />
                                    </div>
                                    <div className="mt-2 h-4 w-20 bg-slate-100 rounded animate-pulse" />
                                    <div className="mt-4 h-8 w-full bg-slate-100 rounded animate-pulse" />
                                </div>
                            ))
                        ) : tickers.length > 0 ? (
                            tickers.slice(0, 4).map((t) => (
                                <div key={t.symbol} className="rounded-xl border bg-white/80 backdrop-blur p-4 shadow-sm">
                                    <div className="flex items-baseline justify-between">
                                        <div className="font-medium text-slate-800">{t.display_symbol || t.symbol}</div>
                                        <div className={`text-xs ${t.change_pct >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>{t.change_pct >= 0 ? '+' : ''}{t.change_pct}%</div>
                                    </div>
                                    <div className="mt-1 font-mono text-slate-700">{t.price.toFixed(2)}</div>
                                    <div className="mt-3">
                                        <Sparkline points={t.sparkline || []} color={t.change_pct >= 0 ? '#10b981' : '#ef4444'} />
                                    </div>
                                </div>
                            ))
                        ) : (
                            ([
                                { symbol: 'RELIANCE', price: 2891.2, change: 0.82, data: [100, 102, 99, 104, 108, 107, 110, 112, 111, 115] },
                                { symbol: 'TCS', price: 3945.5, change: -0.14, data: [110, 109, 111, 112, 110, 108, 109, 107, 106, 105] },
                                { symbol: 'INFY', price: 1682.1, change: 1.12, data: [80, 82, 84, 83, 85, 86, 88, 90, 92, 93] },
                                { symbol: 'HDFCBANK', price: 1525.3, change: 0.35, data: [70, 69, 70, 71, 72, 73, 72, 74, 75, 76] },
                            ] as Ticker[]).map((t) => (
                                <div key={t.symbol} className="rounded-xl border bg-white/80 backdrop-blur p-4 shadow-sm">
                                    <div className="flex items-baseline justify-between">
                                        <div className="font-medium text-slate-800">{t.symbol}</div>
                                        <div className={`text-xs ${t.change >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>{t.change >= 0 ? '+' : ''}{t.change}%</div>
                                    </div>
                                    <div className="mt-1 font-mono text-slate-700">{t.price.toFixed(2)}</div>
                                    <div className="mt-3">
                                        <Sparkline points={t.data} color={t.change >= 0 ? '#10b981' : '#ef4444'} />
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </section>

            {/* Feature highlights */}
            <section className="bg-white border-t">
                <div className="max-w-7xl mx-auto px-6 py-12 grid md:grid-cols-3 gap-6">
                    {[
                        { title: 'Multiple patterns', desc: 'Double Top/Bottom, Cup & Handle, Head & Shoulders with clear rule-based validation.' },
                        { title: 'Flexible sources', desc: 'Switch between live Yahoo Finance data and your CSV library for backtesting.' },
                        { title: 'Beautiful charts', desc: 'Interactive Plotly charts with measured targets and validation breakdowns.' },
                    ].map((f) => (
                        <div key={f.title} className="rounded-xl border bg-white p-5 shadow-sm">
                            <div className="text-slate-900 font-semibold">{f.title}</div>
                            <p className="text-slate-600 mt-1 text-sm">{f.desc}</p>
                        </div>
                    ))}
                </div>
            </section>
        </>
    )
}
