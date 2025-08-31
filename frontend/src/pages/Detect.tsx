import { useEffect, useMemo, useState, useCallback } from 'react'
import Button from '../components/Button'
import Select from '../components/Select'
import HtmlPanel from '../components/HtmlPanel'
import { useDetectStore } from '../store/useDetectStore'

export default function Detect() {
    const s = useDetectStore()
    const [q, setQ] = useState('')
    const filteredStocks = useMemo(() => {
        if (!q.trim()) return s.stocks
        const qq = q.toLowerCase()
        return s.stocks.filter(x => x.toLowerCase().includes(qq))
    }, [q, s.stocks])

    useEffect(() => {
        if (!s.stocks.length) s.init()
    }, [])

    const validDateRange = !s.useDateRange || (s.startDate && s.endDate && s.startDate <= s.endDate)
    const canRun = !!s.selectedStock && !!s.selectedPattern && !!s.selectedChartType && validDateRange &&
        (s.useDateRange ? (!!s.startDate && !!s.endDate) : !!s.selectedTimeframe)

    const setPreset = useCallback((period: 'YTD'|'6M'|'1Y') => {
        const today = new Date()
        const end = today.toISOString().slice(0,10)
        let start: string
        if (period === 'YTD') {
            start = new Date(today.getFullYear(), 0, 1).toISOString().slice(0,10)
        } else if (period === '6M') {
            const d = new Date(today)
            d.setMonth(d.getMonth() - 6)
            start = d.toISOString().slice(0,10)
        } else {
            const d = new Date(today)
            d.setFullYear(d.getFullYear() - 1)
            start = d.toISOString().slice(0,10)
        }
        s.setUseDateRange(true)
        s.setStartDate(start)
        s.setEndDate(end)
    }, [s])

    const downloadChart = useCallback(async (idx: number) => {
        const iframe = document.getElementById(`chart-iframe-${idx}`) as HTMLIFrameElement | null
        const win = iframe?.contentWindow as any
        if (win?.Plotly) {
            const plot = win.document.querySelector('.js-plotly-plot')
            await win.Plotly.downloadImage(plot, { format: 'png' })
        }
    }, [])

    return (
        <section className="max-w-7xl mx-auto p-6">
            <div className="mb-6">
                <h1 className="text-2xl font-semibold tracking-tight text-slate-900">Detect Chart Patterns</h1>
                <p className="text-slate-600 mt-1">Pick a symbol, pattern and timeframes. We’ll render interactive charts inline.</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <aside className="lg:col-span-1 space-y-4">
                    <div className="rounded-xl border bg-white shadow-sm">
                        <div className="p-4 border-b">
                            <div className="text-sm font-medium text-slate-800">Symbol</div>
                            <input
                                type="text"
                                placeholder="Search symbols…"
                                value={q}
                                onChange={(e) => setQ(e.target.value)}
                                className="mt-2 w-full rounded-md border-slate-300 focus:ring-2 focus:ring-slate-400 focus:border-slate-400"
                            />
                            <div className="mt-3 max-h-56 overflow-auto border rounded-md divide-y">
                                {filteredStocks.map(sym => (
                                    <button
                                        key={sym}
                                        className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-50 ${s.selectedStock === sym ? 'bg-slate-100 font-medium' : ''}`}
                                        onClick={() => s.setStock(sym)}
                                    >{sym}</button>
                                ))}
                                {filteredStocks.length === 0 && (
                                    <div className="px-3 py-6 text-sm text-slate-500">No matches</div>
                                )}
                            </div>
                        </div>

                        <div className="p-4 border-b">
                            <Select
                                label="Pattern"
                                value={s.selectedPattern}
                                onChange={s.setPattern}
                                options={s.patterns.map(v => ({ value: v }))}
                                placeholder="Pick a pattern"
                            />
                        </div>

                        <div className="p-4 border-b">
                            <Select
                                label="Mode"
                                value={s.selectedMode}
                                onChange={s.setMode}
                                options={s.modes.map(v => ({ value: v }))}
                                placeholder="Pick mode"
                            />
                        </div>

                        <div className="p-4 space-y-4">
                            <div className="flex items-center gap-3">
                                <input
                                    id="toggle-date"
                                    type="checkbox"
                                    className="h-4 w-4"
                                    checked={s.useDateRange}
                                    onChange={(e) => s.setUseDateRange(e.target.checked)}
                                />
                                <label htmlFor="toggle-date" className="text-sm text-slate-700">Use custom date range</label>
                            </div>
                            {!s.useDateRange ? (
                                <Select
                                    label="Timeframe"
                                    value={s.selectedTimeframe}
                                    onChange={s.setTimeframe}
                                    options={s.timeframes.map(v => ({ value: v }))}
                                    placeholder="Pick timeframe"
                                />
                            ) : (
                                <>
                                    <div className="flex gap-2 mb-2">
                                        {['YTD','6M','1Y'].map(p => (
                                            <button
                                                key={p}
                                                type="button"
                                                onClick={() => setPreset(p as any)}
                                                className="px-2 py-1 text-sm bg-slate-100 rounded"
                                            >{p}</button>
                                        ))}
                                    </div>
                                    <div className="grid grid-cols-1 gap-3">
                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-1">Start date</label>
                                            <input
                                                type="date"
                                                className="w-full rounded-md border-slate-300 focus:ring-2 focus:ring-slate-400 focus:border-slate-400"
                                                value={s.startDate ?? ''}
                                                onChange={(e) => s.setStartDate(e.target.value)}
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-1">End date</label>
                                            <input
                                                type="date"
                                                className="w-full rounded-md border-slate-300 focus:ring-2 focus:ring-slate-400 focus:border-slate-400"
                                                value={s.endDate ?? ''}
                                                onChange={(e) => s.setEndDate(e.target.value)}
                                            />
                                        </div>
                                    </div>
                                </>
                            )}
                            <Select
                                label="Chart type"
                                value={s.selectedChartType}
                                onChange={s.setChartType}
                                options={s.chartTypes.map(v => ({ value: v }))}
                                placeholder="Pick chart type"
                            />
                            <div className="mt-4">
                                <Button disabled={!canRun || s.loading} onClick={s.runDetect}>
                                    {s.loading ? 'Detecting…' : 'Run Detection'}
                                </Button>
                            </div>
                        </div>
                    </div>
                </aside>

                <div className="lg:col-span-3 space-y-6">
                    {s.error && (
                        <div className="p-3 rounded-md bg-red-50 border border-red-200 text-red-700 text-sm">
                            {s.error}
                        </div>
                    )}
                    {s.loading ? (
                        [1,2,3].map(i => (
                            <div key={i} className="border rounded-xl bg-white shadow-sm p-4 animate-pulse">
                                <div className="h-4 bg-slate-200 rounded w-1/3 mb-4"></div>
                                <div className="h-64 bg-slate-200 rounded"></div>
                            </div>
                        ))
                    ) : s.charts.length === 0 ? (
                        <div className="border rounded-xl bg-white shadow-sm p-10 text-center text-slate-500">
                            No pattern detected or no charts available. Try other inputs.
                        </div>
                    ) : (
                        s.charts.map((c, idx) => (
                            <div key={`${c.timeframe}-${idx}`} className="border rounded-xl bg-white shadow-sm">
                                <div className="px-4 py-3 border-b flex items-center justify-between">
                                    <div className="text-sm font-medium text-slate-700">Timeframe: {c.timeframe}</div>
                                    <Button type="button" onClick={() => downloadChart(idx)} className="text-sm">
                                        Download
                                    </Button>
                                </div>
                                <div className="p-4">
                                    <HtmlPanel html={c.html} id={`chart-iframe-${idx}`} />
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </section>
    )
}
