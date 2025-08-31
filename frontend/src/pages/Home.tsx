export default function Home() {
    return (
        <section className="max-w-6xl mx-auto p-4">
            <div className="prose max-w-none">
                <h1>Welcome</h1>
                <p>
                    Use the Detect tab to choose a stock, pattern and timeframe(s). We'll call your FastAPI backend and render interactive Plotly charts inline.
                </p>
            </div>
        </section>
    )
}
