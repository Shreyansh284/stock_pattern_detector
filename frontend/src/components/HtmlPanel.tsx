export default function HtmlPanel({ html, height = 640, id }: { html: string; height?: number; id?: string }) {
    return (
        <div className="w-full overflow-hidden rounded-lg border bg-white">
            <iframe
                id={id}
                title="Plotly Chart"
                srcDoc={html}
                className="w-full"
                style={{ height }}
            />
        </div>
    )
}
