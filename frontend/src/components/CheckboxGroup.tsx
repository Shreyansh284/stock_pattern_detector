export default function CheckboxGroup({ label, options, values, onToggle }: {
    label: string
    options: string[]
    values: string[]
    onToggle: (v: string) => void
}) {
    return (
        <fieldset className="block">
            <legend className="text-sm mb-2 text-slate-600">{label}</legend>
            <div className="flex flex-wrap gap-2">
                {options.map(o => (
                    <label key={o} className="flex items-center gap-2 px-3 py-2 rounded-md border border-slate-300 bg-white cursor-pointer">
                        <input
                            type="checkbox"
                            className="accent-slate-900"
                            checked={values.includes(o)}
                            onChange={() => onToggle(o)}
                        />
                        <span className="text-sm">{o}</span>
                    </label>
                ))}
            </div>
        </fieldset>
    )
}
