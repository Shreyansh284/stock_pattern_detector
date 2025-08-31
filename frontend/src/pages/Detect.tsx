import { useEffect } from 'react'
import Button from '../components/Button'
import Select from '../components/Select'
import CheckboxGroup from '../components/CheckboxGroup'
import HtmlPanel from '../components/HtmlPanel'
import { useDetectStore } from '../store/useDetectStore'

export default function Detect() {
    const s = useDetectStore()

    useEffect(() => {
        if (!s.stocks.length) s.init()
    }, [])

    const canRun = !!s.selectedStock && !!s.selectedPattern && s.selectedTimeframes.length > 0

    return (
        <section className="max-w-6xl mx-auto p-4 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Select
                    label="Stock"
                    value={s.selectedStock}
                    onChange={s.setStock}
                    options={s.stocks.map(v => ({ value: v }))}
                    placeholder="Pick a stock"
                />
                <Select
                    label="Pattern"
                    value={s.selectedPattern}
                    onChange={s.setPattern}
                    options={s.patterns.map(v => ({ value: v }))}
                    placeholder="Pick a pattern"
                />
                <div className="flex items-end">
                    <Button disabled={!canRun || s.loading} onClick={s.runDetect}>
                        {s.loading ? 'Detecting…' : 'Run Detection'}
                    </Button>
                </div>
            </div>

            <CheckboxGroup
                label="Timeframes"
                options={s.timeframes}
                values={s.selectedTimeframes}
                onToggle={s.toggleTimeframe}
            />

            {s.error && (
                <div className="p-3 rounded-md bg-red-50 border border-red-200 text-red-700 text-sm">
                    {s.error}
                </div>
            )}

            <div className="space-y-6">
                {s.charts.map((c, idx) => (
                    <div key={`${c.timeframe}-${idx}`} className="space-y-2">
                        <div className="text-sm font-medium text-slate-700">Timeframe: {c.timeframe}</div>
                        <HtmlPanel html={c.html} />
                    </div>
                ))}
            </div>
        </section>
    )
}
