import { useEffect, useMemo, useState, useCallback } from 'react'
import Button from '../components/Button'
import Select from '../components/Select'
import HtmlPanel from '../components/HtmlPanel'
import { useDetectStore } from '../store/useDetectStore'

export default function Detect() {
    const s = useDetectStore()
    const [q, setQ] = useState('')
    // If dataSource is 'past', show only local CSV symbols from StockData
    const localCsvSymbols = [
        '360ONE', '3MINDIA', 'AARTIIND', 'AAVAS', 'ABB', 'ABBOTINDIA', 'ABCAPITAL', 'ABFRL', 'ACC', 'ACE', 'ACI', 'ADANIENSOL', 'ADANIENT', 'ADANIGREEN', 'ADANIPORTS', 'ADANIPOWER', 'AETHER', 'AFFLE', 'AIAENG', 'AJANTPHARM', 'ALKEM', 'ALKYLAMINE', 'ALLCARGO', 'ALOKINDS', 'AMBER', 'AMBUJACEM', 'ANANDRATHI', 'ANGELONE', 'ANURAS', 'APARINDS', 'APLAPOLLO', 'APLLTD', 'APOLLOHOSP', 'APOLLOTYRE', 'APTUS', 'ARE&M', 'ASAHIINDIA', 'ASHOKLEY', 'ASIANPAINT', 'ASTERDM', 'ASTRAL', 'ASTRAZEN', 'ATGL', 'ATUL', 'AUBANK', 'AUROPHARMA', 'AVANTIFEED', 'AWL', 'AXISBANK', 'BAJAJ-AUTO', 'BAJAJFINSV', 'BAJAJHLDNG', 'BAJFINANCE', 'BALAMINES', 'BALKRISIND', 'BALRAMCHIN', 'BANDHANBNK', 'BANKBARODA', 'BANKINDIA', 'BATAINDIA', 'BAYERCROP', 'BBTC', 'BDL', 'BEL', 'BEML', 'BERGEPAINT', 'BHARATFORG', 'BHARTIARTL', 'BHEL', 'BIKAJI', 'BIOCON', 'BIRLACORPN', 'BLS', 'BLUEDART', 'BLUESTARCO', 'BORORENEW', 'BOSCHLTD', 'BPCL', 'BRIGADE', 'BRITANNIA', 'BSE', 'BSOFT', 'CAMPUS', 'CAMS', 'CANBK', 'CANFINHOME', 'CAPLIPOINT', 'CARBORUNIV', 'CASTROLIND', 'CCL', 'CDSL', 'CEATLTD', 'CELLO', 'CENTRALBK', 'CENTURYPLY', 'CENTURYTEX', 'CERA', 'CESC', 'CGCL', 'CGPOWER', 'CHALET', 'CHAMBLFERT', 'CHEMPLASTS', 'CHENNPETRO', 'CHOLAFIN', 'CHOLAHLDNG', 'CIEINDIA', 'CIPLA', 'CLEAN', 'COALINDIA', 'COCHINSHIP', 'COFORGE', 'COLPAL', 'CONCOR', 'CONCORDBIO', 'COROMANDEL', 'CRAFTSMAN', 'CREDITACC', 'CRISIL', 'CROMPTON', 'CSBBANK', 'CUB', 'CUMMINSIND', 'CYIENT', 'DABUR', 'DALBHARAT', 'DATAPATTNS', 'DCMSHRIRAM', 'DEEPAKFERT', 'DEEPAKNTR', 'DELHIVERY', 'DEVYANI', 'DIVISLAB', 'DIXON', 'DLF', 'DMART', 'DOMS', 'DRREDDY', 'EASEMYTRIP', 'ECLERX', 'EICHERMOT', 'EIDPARRY', 'EIHOTEL', 'ELECON', 'ELGIEQUIP', 'EMAMILTD', 'ENDURANCE', 'ENGINERSIN', 'EPL', 'EQUITASBNK', 'ERIS', 'ESCORTS', 'EXIDEIND', 'FACT', 'FDC', 'FEDERALBNK', 'FINCABLES', 'FINEORG', 'FINPIPE', 'FIVESTAR', 'FLUOROCHEM'
        // ...add all other symbols as needed
    ];
    const filteredStocks = useMemo(() => {
        let stocks = s.stocks;
        if (s.dataSource === 'past') {
            stocks = localCsvSymbols;
        }
        if (!q.trim()) return stocks;
        const qq = q.toLowerCase();
        return stocks.filter(x => x.toLowerCase().includes(qq));
    }, [q, s.stocks, s.dataSource]);

    useEffect(() => {
        if (!s.stocks.length) s.init()
    }, [])

    const validDateRange = !s.useDateRange || (s.startDate && s.endDate && s.startDate <= s.endDate)
    const canRun = !!s.selectedStock && !!s.selectedPattern && !!s.selectedChartType && validDateRange &&
        (s.useDateRange ? (!!s.startDate && !!s.endDate) : !!s.selectedTimeframe)

    const setPreset = useCallback((period: 'YTD' | '6M' | '1Y') => {
        const today = new Date()
        const end = today.toISOString().slice(0, 10)
        let start: string
        if (period === 'YTD') {
            start = new Date(today.getFullYear(), 0, 1).toISOString().slice(0, 10)
        } else if (period === '6M') {
            const d = new Date(today)
            d.setMonth(d.getMonth() - 6)
            start = d.toISOString().slice(0, 10)
        } else {
            const d = new Date(today)
            d.setFullYear(d.getFullYear() - 1)
            start = d.toISOString().slice(0, 10)
        }
        s.setUseDateRange(true)
        s.setStartDate(start)
        s.setEndDate(end)
    }, [s])

    const downloadChart = useCallback(async (elementId: string) => {
        const iframe = document.getElementById(elementId) as HTMLIFrameElement | null
        const win = iframe?.contentWindow as any
        if (win?.Plotly) {
            const plot = win.document.querySelector('.js-plotly-plot')
            await win.Plotly.downloadImage(plot, { format: 'png' })
        }
    }, [])

    return (
        <section className="max-w-8xl mx-9 pt-4 pb-4">
            <div className="mb-6">
                <h1 className="text-2xl font-semibold tracking-tight text-slate-900">Detect Chart Patterns</h1>
                <p className="text-slate-600 mt-1">Pick a symbol, pattern and timeframes. We’ll render interactive charts inline.</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-y-6 gap-x-3">
                <aside className="lg:col-span-1 space-y-4">
                    <div className="rounded-xl border bg-white shadow-sm">
                        <div className="p-4 border-b">
                            <div className="text-sm font-medium text-slate-800">Symbol</div>
                            <input
                                type="text"
                                placeholder="Search symbols…"
                                value={s.selectedStock ? s.selectedStock : q}
                                onChange={(e) => {
                                    setQ(e.target.value);
                                    if (s.selectedStock) s.setStock(undefined);
                                }}
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
                                label="Data Source"
                                value={s.dataSource}
                                onChange={(v) => s.setDataSource((v as any) ?? 'live')}
                                options={[{ value: 'live', label: 'Live' }, { value: 'past', label: 'Past' }]}
                                placeholder="Live"
                            />
                        </div>

                        <div className="p-4 border-b">
                            <Select
                                label="Pattern"
                                value={s.selectedPattern}
                                onChange={s.setPattern}
                                options={[{ value: 'All', label: 'All' }, ...s.patterns.map(v => ({ value: v }))]}
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
                                        {['YTD', '6M', '1Y'].map(p => (
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
                        [1, 2, 3].map(i => (
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
                        <>
                            {Array.isArray(s.strongCharts) && s.strongCharts.length > 0 && (
                                <div className="space-y-4">
                                    {s.strongCharts.map((c, idx) => (
                                        <details key={`strong-${c.timeframe}-${idx}`} className="border rounded-xl bg-white shadow-sm" closed>
                                            <summary className="px-4 py-3 border-b flex items-center justify-between cursor-pointer select-none">
                                                <span className="text-sm font-medium text-slate-700">Timeframe: {c.timeframe}</span>
                                                <Button type="button" onClick={() => downloadChart(`chart-iframe-strong-${idx}`)} className="text-sm">Download</Button>
                                            </summary>
                                            <div className="p-4">
                                                <HtmlPanel html={c.html} id={`chart-iframe-strong-${idx}`} />
                                                {/* Explanation details */}
                                                {c.explanation && (() => {
                                                    const exp = c.explanation;
                                                    const rules = Array.isArray(exp?.rules) ? exp.rules : [];
                                                    const target = exp?.target;
                                                    return (rules.length > 0 || target) ? (
                                                        <details className="bg-slate-50 border-t mt-4">
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
                                                    ) : null;
                                                })()}
                                            </div>
                                        </details>
                                    ))}
                                </div>
                            )}
                            {Array.isArray(s.weakCharts) && s.weakCharts.length > 0 && (
                                <div className="space-y-4">
                                    <div className="text-sm font-semibold text-amber-700">Weak patterns</div>
                                    {s.weakCharts.map((c, idx) => (
                                        <details key={`weak-${c.timeframe}-${idx}`} className="border rounded-xl bg-white shadow-sm" closed>
                                            <summary className="px-4 py-3 border-b flex items-center justify-between cursor-pointer select-none">
                                                <span className="text-sm font-medium text-slate-700">Timeframe: {c.timeframe}</span>
                                                <Button type="button" onClick={() => downloadChart(`chart-iframe-weak-${idx}`)} className="text-sm">Download</Button>
                                            </summary>
                                            <div className="p-4">
                                                <HtmlPanel html={c.html} id={`chart-iframe-weak-${idx}`} />
                                                {/* Explanation details */}
                                                {c.explanation && (() => {
                                                    const exp = c.explanation;
                                                    const rules = Array.isArray(exp?.rules) ? exp.rules : [];
                                                    const target = exp?.target;
                                                    return (rules.length > 0 || target) ? (
                                                        <details className="bg-slate-50 border-t mt-4">
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
                                                    ) : null;
                                                })()}
                                            </div>
                                        </details>
                                    ))}
                                </div>
                            )}
                            {/* Fallback single list if grouping absent */}
                            {(!s.strongCharts?.length && !s.weakCharts?.length) && (
                                s.charts.map((c, idx) => (
                                    <details key={`${c.timeframe}-${idx}`} className="border rounded-xl bg-white shadow-sm" closed>
                                        <summary className="px-4 py-3 border-b flex items-center justify-between cursor-pointer select-none">
                                            <span className="text-sm font-medium text-slate-700">Timeframe: {c.timeframe}</span>
                                            <Button type="button" onClick={() => downloadChart(`chart-iframe-${idx}`)} className="text-sm">Download</Button>
                                        </summary>
                                        <div className="p-4">
                                            <HtmlPanel html={c.html} id={`chart-iframe-${idx}`} />
                                            {/* Explanation details */}
                                            {c.explanation && (() => {
                                                const exp = c.explanation;
                                                const rules = Array.isArray(exp?.rules) ? exp.rules : [];
                                                const target = exp?.target;
                                                return (rules.length > 0 || target) ? (
                                                    <details className="bg-slate-50 border-t mt-4">
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
                                                ) : null;
                                            })()}
                                        </div>
                                    </details>
                                ))
                            )}
                            {/* // )} */}
                        </>
                    )}
                </div>
            </div>
        </section>
    )
}
