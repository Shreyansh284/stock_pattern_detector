export default function Button({ children, onClick, disabled, type = 'button' }: {
    children: React.ReactNode
    onClick?: () => void
    disabled?: boolean
    type?: 'button' | 'submit' | 'reset'
}) {
    return (
        <button
            type={type}
            onClick={onClick}
            disabled={disabled}
            className={`inline-flex items-center justify-center rounded-md px-4 py-2.5 text-sm font-medium shadow-sm transition-colors border
        ${disabled ? 'bg-slate-200 text-slate-500 cursor-not-allowed' : 'bg-slate-900 text-white hover:bg-slate-800 border-slate-900'}`}
        >
            {children}
        </button>
    )
}
