export default function HtmlPanel({ html, height = 640 }: { html: string; height?: number }) {
    return (
        <div className="w-full overflow-hidden rounded-lg border bg-white">
            <iframe
                title="Plotly Chart"
                srcDoc={html}
                className="w-full"
                style={{ height }}
            />
        </div>
    )
}
