type Opt = { value: string; label?: string }

export default function Select({ label, value, onChange, options, placeholder }: {
    label: string
    value?: string
    onChange: (v?: string) => void
    options: Opt[]
    placeholder?: string
}) {
    return (
        <label className="block">
            <div className="text-sm mb-1 text-slate-600">{label}</div>
            <select
                className="w-full rounded-md border-slate-300 shadow-sm focus:ring-2 focus:ring-slate-400 focus:border-slate-400"
                value={value ?? ''}
                onChange={(e) => onChange(e.target.value || undefined)}
            >
                <option value="">{placeholder ?? 'Select...'}</option>
                {options.map((o) => (
                    <option key={o.value} value={o.value}>{o.label ?? o.value}</option>
                ))}
            </select>
        </label>
    )
}
