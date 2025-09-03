import { useState, useMemo, useEffect } from 'react'
import Button from '../components/Button'
import Select from '../components/Select'
import HtmlPanel from '../components/HtmlPanel'
import { detectAll, type Chart } from '../lib/api'

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
        try {
            const res = await detectAll({ start_date: startDate, end_date: endDate })
            if (!res || !res.results) {
                throw new Error('Invalid response from server')
            }
            setData(res)
        } catch (e: any) {
            console.error('Detection error:', e)
            setError(e?.response?.data?.error || e?.message || 'Detection failed')
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
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
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
                <div className="flex items-end">
                    <Button onClick={runDetectAll} disabled={loading}>
                        {loading ? 'Detecting…' : 'Run Detection'}
                    </Button>
                </div>
            </div>

            {error && <div className="mb-4 text-red-600">{error}</div>}

            {data && (
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
                                    onChange={setStockFilter}
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
                                            {r.count > 0 ?
                                                Object.entries(r.pattern_counts).map(([p, c]) => `${p}: ${c}`).join(', ') :
                                                'No patterns detected'
                                            }
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    ) : (
                        <div className="space-y-6">
                            {filterLoading ? (
                                <div className="text-center py-8">
                                    <div className="text-slate-500">Loading charts...</div>
                                </div>
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
                                                    return (
                                                        <div key={start + idx}>
                                                            <div className="mb-2 text-xs text-slate-500">
                                                                Pattern: {c.pattern || 'Unknown'} | Timeframe: {c.timeframe || 'Unknown'}
                                                            </div>
                                                            <HtmlPanel html={c.html} id={`dash-chart-${r.stock}-${start + idx}`} />
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
