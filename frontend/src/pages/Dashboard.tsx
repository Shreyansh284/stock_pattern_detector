import { useState, useMemo, useEffect } from 'react'
import Button from '../components/Button'
import Select from '../components/Select'
import HtmlPanel from '../components/HtmlPanel'
import { detectAll, fetchChartTypes, type Chart, startDetectAll, getDetectAllProgress, getDetectAllResult } from '../lib/api'

type StockPatternResult = {
    stock: string
    patterns: string[]
    pattern_counts: Record<string, number>
    count: number
    current_price: number
    current_volume: number
    charts: Array<Chart & { pattern: string }>
}

type DetectAllResponse = {
    results: StockPatternResult[]
}

export default function Dashboard() {
    const [startDate, setStartDate] = useState<string>('')
    const [endDate, setEndDate] = useState<string>('')
    const [data, setData] = useState<DetectAllResponse | null>(null)
    const [loading, setLoading] = useState(false)
    const [filterLoading, setFilterLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [activeTab, setActiveTab] = useState<'table' | 'chart'>('table')
    const [patternFilter, setPatternFilter] = useState<string | undefined>(undefined)
    const [stockFilter, setStockFilter] = useState<string | undefined>(undefined)
    const itemsPerPage = 4
    const [pages, setPages] = useState<Record<string, number>>({})
    const [chartTypes, setChartTypes] = useState<string[]>(['candle', 'line', 'ohlc'])
    const [chartType, setChartType] = useState<string>('candle')
    const [progress, setProgress] = useState<number>(0)
    const [progressMsg, setProgressMsg] = useState<string>('')
    const [jobId, setJobId] = useState<string | null>(null)

    useEffect(() => {
        // load available chart types from backend (optional)
        fetchChartTypes().then(setChartTypes).catch(() => { })
    }, [])

    // Simple skeletons
    const SkeletonTable = () => (
        <div className="bg-white border rounded-xl overflow-hidden animate-pulse">
            <div className="h-10 bg-slate-100" />
            <div className="divide-y">
                {Array.from({ length: 6 }).map((_, i) => (
                    <div key={i} className="grid grid-cols-5 gap-4 px-4 py-3">
                        <div className="h-4 bg-slate-200 rounded" />
                        <div className="h-4 bg-slate-200 rounded" />
                        <div className="h-4 bg-slate-200 rounded" />
                        <div className="h-4 bg-slate-200 rounded" />
                        <div className="h-4 bg-slate-200 rounded" />
                    </div>
                ))}
            </div>
        </div>
    )

    const SkeletonCharts = () => (
        <div className="space-y-6">
            {Array.from({ length: 2 }).map((_, c) => (
                <div key={c} className="border rounded-xl bg-white shadow-sm">
                    <div className="px-4 py-3 border-b flex items-center justify-between">
                        <div className="h-4 w-48 bg-slate-200 rounded animate-pulse" />
                        <div className="h-4 w-24 bg-slate-200 rounded animate-pulse" />
                    </div>
                    <div className="p-4 space-y-4">
                        {Array.from({ length: 2 }).map((_, i) => (
                            <div key={i}>
                                <div className="mb-2 h-3 w-64 bg-slate-200 rounded animate-pulse" />
                                <div className="w-full h-[980px] bg-slate-100 rounded-lg border animate-pulse" />
                            </div>
                        ))}
                    </div>
                    <div className="px-4 py-2 border-t flex justify-between">
                        <div className="h-8 w-24 bg-slate-200 rounded animate-pulse" />
                        <div className="h-4 w-24 bg-slate-200 rounded animate-pulse self-center" />
                        <div className="h-8 w-24 bg-slate-200 rounded animate-pulse" />
                    </div>
                </div>
            ))}
        </div>
    )

    const runDetectAll = async () => {
        setError(null)
        if (!startDate || !endDate) {
            setError('Please select start and end date.')
            return
        }
        if (new Date(startDate) >= new Date(endDate)) {
            setError('Start date must be before end date.')
            return
        }
        setLoading(true)
        setProgress(0)
        setProgressMsg('Queued')
        setData(null)
        try {
            const { job_id } = await startDetectAll({ start_date: startDate, end_date: endDate, chart_type: chartType as any })
            setJobId(job_id)
            // Poll progress
            await new Promise<void>((resolve, reject) => {
                const poll = async () => {
                    try {
                        const p = await getDetectAllProgress(job_id)
                        setProgress(p.percent)
                        setProgressMsg(p.message || `${p.current}/${p.total}`)
                        if (p.status === 'done') {
                            resolve()
                            return
                        }
                        if (p.status === 'error') {
                            reject(new Error(p.message || 'Error'))
                            return
                        }
                        setTimeout(poll, 800)
                    } catch (err: any) {
                        reject(err)
                    }
                }
                poll()
            })
            const res = await getDetectAllResult(job_id)
            setData(res)
            setProgress(100)
            setProgressMsg('Completed')
        } catch (e: any) {
            console.error('Detection error:', e)
            setError(e?.response?.data?.detail || e?.message || 'Detection failed')
        } finally {
            setLoading(false)
        }
    }

    const results = data?.results || []

    const availablePatterns = useMemo(() => {
        try {
            const pats = new Set<string>()
            results.forEach(r => {
                if (r?.patterns && Array.isArray(r.patterns)) {
                    r.patterns.forEach(p => pats.add(p))
                }
            })
            return Array.from(pats)
        } catch (e) {
            console.error('Error computing available patterns:', e)
            return []
        }
    }, [results])

    const filteredResults = useMemo(() => {
        try {
            let res = results
            if (patternFilter) {
                res = res.filter(r => r?.patterns && Array.isArray(r.patterns) && r.patterns.includes(patternFilter))
            }
            if (stockFilter) {
                res = res.filter(r => r?.stock === stockFilter)
            }
            return res
        } catch (e) {
            console.error('Error filtering results:', e)
            return []
        }
    }, [results, patternFilter, stockFilter])

    // Handle filter loading state separately
    useEffect(() => {
        setFilterLoading(true)
        // Reset pagination when filters change
        setPages({})
        const timer = setTimeout(() => setFilterLoading(false), 200)
        return () => clearTimeout(timer)
    }, [patternFilter, stockFilter])

    // Set default stock filter for chart view
    useEffect(() => {
        if (activeTab === 'chart' && results.length > 0 && !stockFilter) {
            // Find first stock with patterns
            const stockWithPatterns = results.find(r => r.count > 0)
            if (stockWithPatterns) {
                setStockFilter(stockWithPatterns.stock)
            }
        }
        // Clear stock filter when switching back to table view
        if (activeTab === 'table' && stockFilter) {
            setStockFilter(undefined)
        }
    }, [activeTab, results])

    const totalPatterns = useMemo(() => {
        try {
            return results.reduce((sum, r) => sum + (r?.count || 0), 0)
        } catch (e) {
            console.error('Error calculating total patterns:', e)
            return 0
        }
    }, [results])

    return (
        <section className="max-w-7xl mx-auto p-6">
            <div className="mb-6">
                <h1 className="text-2xl font-semibold tracking-tight text-slate-900">Pattern Detection Dashboard</h1>
                <p className="text-slate-600 mt-1">Run detection across all stocks within a date range and explore results.</p>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-6">
                <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Start Date</label>
                    <input
                        type="date"
                        className="w-full rounded-md border-slate-300 focus:ring-2 focus:ring-slate-400 focus:border-slate-400"
                        value={startDate}
                        onChange={e => setStartDate(e.target.value)}
                    />
                </div>
                <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">End Date</label>
                    <input
                        type="date"
                        className="w-full rounded-md border-slate-300 focus:ring-2 focus:ring-slate-400 focus:border-slate-400"
                        value={endDate}
                        onChange={e => setEndDate(e.target.value)}
                    />
                </div>
                <div>
                    <Select
                        label="Chart Type"
                        value={chartType}
                        onChange={(v) => setChartType(v ?? 'candle')}
                        options={chartTypes.map(ct => ({ value: ct, label: ct.charAt(0).toUpperCase() + ct.slice(1) }))}
                        placeholder="Candle"
                    />
                </div>
                <div className="flex items-end">
                    <Button onClick={runDetectAll} disabled={loading}>
                        {loading ? 'Detecting…' : 'Run Detection'}
                    </Button>
                </div>
            </div>

            {error && <div className="mb-4 text-red-600">{error}</div>}

            {loading && (
                <>
                    <div className="mb-4">
                        <div className="rounded-lg bg-slate-800 text-white px-4 py-3 text-sm shadow">
                            {progressMsg || 'Working...'}
                        </div>
                        <div className="mt-2 h-2 w-full rounded bg-slate-300 overflow-hidden">
                            <div
                                className="h-full rounded bg-blue-500 transition-all duration-300"
                                style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
                            />
                        </div>
                    </div>
                    {activeTab === 'table' ? <SkeletonTable /> : <SkeletonCharts />}
                </>
            )}

            {data && !loading && (
                <>
                    <div className="flex items-center gap-4 mb-4">
                        <button
                            className={`px-4 py-2 rounded ${activeTab === 'table' ? 'bg-slate-900 text-white' : 'bg-white'}`}
                            onClick={() => setActiveTab('table')}
                        >Table View</button>
                        <button
                            className={`px-4 py-2 rounded ${activeTab === 'chart' ? 'bg-slate-900 text-white' : 'bg-white'}`}
                            onClick={() => setActiveTab('chart')}
                        >Chart View</button>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
                        {filterLoading ? (
                            <div className="col-span-3 flex justify-center">
                                <div className="text-slate-500">Applying filters...</div>
                            </div>
                        ) : (
                            <>
                                <Select
                                    label="Filter Pattern"
                                    value={patternFilter}
                                    onChange={setPatternFilter}
                                    options={availablePatterns.map(p => ({ value: p }))}
                                    placeholder="All patterns"
                                />
                                <Select
                                    label="Filter Stock"
                                    value={stockFilter}
                                    onChange={(v) => setStockFilter(v)}
                                    options={results.map(r => ({ value: r.stock }))}
                                    placeholder="All stocks"
                                />
                            </>
                        )}
                    </div>

                    {totalPatterns === 0 ? (
                        <div className="border rounded-xl bg-white shadow-sm p-10 text-center text-slate-500">
                            No patterns detected across all stocks.
                        </div>
                    ) : activeTab === 'table' ? (
                        <table className="min-w-full bg-white border rounded-xl divide-y">
                            <thead>
                                <tr>
                                    <th className="px-4 py-2 text-left text-sm font-medium">Stock</th>
                                    <th className="px-4 py-2 text-left text-sm font-medium">Price</th>
                                    <th className="px-4 py-2 text-left text-sm font-medium">Volume</th>
                                    <th className="px-4 py-2 text-left text-sm font-medium">Total Patterns</th>
                                    <th className="px-4 py-2 text-left text-sm font-medium">Pattern Counts</th>
                                    <th className="px-4 py-2 text-left text-sm font-medium">Strength</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y">
                                {filteredResults.map(r => (
                                    <tr key={r.stock}>
                                        <td className="px-4 py-2 text-sm font-medium">{r.stock}</td>
                                        <td className="px-4 py-2 text-sm">
                                            ${r.current_price > 0 ? r.current_price.toFixed(2) : 'N/A'}
                                        </td>
                                        <td className="px-4 py-2 text-sm">
                                            {r.current_volume > 0 ? r.current_volume.toLocaleString() : 'N/A'}
                                        </td>
                                        <td className="px-4 py-2 text-sm">{r.count}</td>
                                        <td className="px-4 py-2 text-sm">
                                            {r.count > 0 ? (() => {
                                                // Build per-pattern strong/weak counts from charts; fall back to totals
                                                const byPattern: Record<string, { strong: number; weak: number; total: number }> = {}
                                                if (Array.isArray(r.charts)) {
                                                    r.charts.forEach((c: any) => {
                                                        const pat = c?.pattern || 'Unknown'
                                                        if (!byPattern[pat]) byPattern[pat] = { strong: 0, weak: 0, total: 0 }
                                                        byPattern[pat].total += 1
                                                        if (c?.strength === 'strong') byPattern[pat].strong += 1
                                                        else if (c?.strength === 'weak') byPattern[pat].weak += 1
                                                    })
                                                }
                                                const entries = Object.keys(byPattern)
                                                    .sort()
                                                    .map(k => {
                                                        const v = byPattern[k]
                                                        // If no strength info, just show total
                                                        if (v.strong + v.weak === 0) return `${k}: ${v.total}`
                                                        return `${k}: ${v.total} (${v.strong} strong / ${v.weak} weak)`
                                                    })
                                                return entries.length > 0
                                                    ? entries.join(', ')
                                                    : Object.entries(r.pattern_counts).map(([p, c]) => `${p}: ${c}`).join(', ')
                                            })() : 'No patterns detected'}
                                        </td>
                                        <td className="px-4 py-2 text-sm">
                                            {/* strength counts per stock: derive from charts */}
                                            {Array.isArray(r.charts) && r.charts.length > 0 ? (() => {
                                                const strong = r.charts.filter((c: any) => c?.strength === 'strong').length
                                                const weak = r.charts.filter((c: any) => c?.strength === 'weak').length
                                                return strong + weak > 0 ? `Strong: ${strong}, Weak: ${weak}` : '—'
                                            })() : '—'}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    ) : (
                        <div className="space-y-6">
                            {filterLoading ? (
                                <SkeletonCharts />
                            ) : (
                                filteredResults.map(r => {
                                    if (!r || !r.charts || !Array.isArray(r.charts)) return null

                                    // Filter charts by pattern if pattern filter is active
                                    let chartsToFilter = r.charts
                                    if (patternFilter) {
                                        chartsToFilter = r.charts.filter(c => c.pattern === patternFilter)
                                    }

                                    // If no charts match the pattern filter, skip this stock
                                    if (chartsToFilter.length === 0) return null

                                    const page = pages[r.stock] || 0
                                    const start = page * itemsPerPage
                                    const end = start + itemsPerPage
                                    const chartsToShow = chartsToFilter.slice(start, end)
                                    const totalPages = Math.ceil(chartsToFilter.length / itemsPerPage)

                                    return (
                                        <div key={r.stock} className="border rounded-xl bg-white shadow-sm">
                                            <div className="px-4 py-3 border-b">
                                                <div className="text-sm font-medium text-slate-700">
                                                    {r.stock} - {chartsToFilter.length} chart(s) {patternFilter ? `for ${patternFilter}` : 'total'}
                                                </div>
                                            </div>
                                            <div className="p-4 space-y-4">
                                                {chartsToShow.map((c, idx) => {
                                                    if (!c || !c.html) return null
                                                    const exp = (c as any)?.explanation as any
                                                    const rules: any[] = Array.isArray(exp?.rules) ? exp.rules : []
                                                    const target = exp?.target
                                                    return (
                                                        <div key={start + idx} className="border rounded-lg overflow-hidden">
                                                            <div className="mb-2 text-xs text-slate-500">
                                                                Pattern: {c.pattern || 'Unknown'} | Timeframe: {c.timeframe || 'Unknown'}{c && (c as any).strength ? ` | Strength: ${(c as any).strength}` : ''}
                                                            </div>
                                                            <HtmlPanel html={c.html} id={`dash-chart-${r.stock}-${start + idx}`} />
                                                            {(exp && (rules.length > 0 || target)) && (
                                                                <details className="bg-slate-50 border-t">
                                                                    <summary className="px-4 py-3 cursor-pointer text-sm font-medium text-slate-700 select-none">Details</summary>
                                                                    <div className="px-4 pb-4 pt-1 grid grid-cols-1 lg:grid-cols-2 gap-4 text-sm">
                                                                        <div>
                                                                            <div className="font-semibold mb-2">Validation rules</div>
                                                                            <ul className="space-y-1 list-disc list-inside">
                                                                                {rules.map((rul: any, i: number) => (
                                                                                    <li key={i} className={rul.passed ? 'text-green-700' : 'text-amber-700'}>
                                                                                        {rul.name}: <span className="font-mono">{rul.value}</span> (expected {rul.expected})
                                                                                        {rul.notes ? <span className="text-slate-500"> — {rul.notes}</span> : null}
                                                                                    </li>
                                                                                ))}
                                                                            </ul>
                                                                        </div>
                                                                        <div>
                                                                            <div className="font-semibold mb-2">Target calculation</div>
                                                                            {target ? (
                                                                                <div>
                                                                                    <div className="text-slate-700 mb-1">{target.formula}</div>
                                                                                    {Array.isArray(target.steps) && target.steps.length > 0 && (
                                                                                        <ol className="list-decimal list-inside space-y-1">
                                                                                            {target.steps.map((s: string, i: number) => (
                                                                                                <li key={i} className="font-mono text-slate-700">{s}</li>
                                                                                            ))}
                                                                                        </ol>
                                                                                    )}
                                                                                    {typeof target.target_price === 'number' && (
                                                                                        <div className="mt-2 text-sm font-medium">Target price: <span className="font-mono">{target.target_price.toFixed(2)}</span></div>
                                                                                    )}
                                                                                </div>
                                                                            ) : (
                                                                                <div className="text-slate-500">No target available</div>
                                                                            )}
                                                                        </div>
                                                                    </div>
                                                                </details>
                                                            )}
                                                        </div>
                                                    )
                                                })}
                                            </div>
                                            {chartsToFilter.length > itemsPerPage && (
                                                <div className="px-4 py-2 flex justify-between border-t">
                                                    <button
                                                        onClick={() => setPages(prev => ({ ...prev, [r.stock]: Math.max(page - 1, 0) }))}
                                                        disabled={page === 0}
                                                        className="px-3 py-1 bg-slate-100 rounded disabled:opacity-50 text-sm"
                                                    >Previous</button>
                                                    <div className="text-sm text-slate-500 self-center">
                                                        Page {page + 1} of {totalPages}
                                                    </div>
                                                    <button
                                                        onClick={() => setPages(prev => ({ ...prev, [r.stock]: Math.min(page + 1, totalPages - 1) }))}
                                                        disabled={page >= totalPages - 1}
                                                        className="px-3 py-1 bg-slate-100 rounded disabled:opacity-50 text-sm"
                                                    >Next</button>
                                                </div>
                                            )}
                                        </div>
                                    )
                                }).filter(Boolean)
                            )}
                        </div>
                    )}
                </>
            )}
        </section>
    )
}
