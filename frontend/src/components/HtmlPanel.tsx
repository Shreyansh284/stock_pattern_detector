import { useEffect, useRef } from 'react'

export default function HtmlPanel({ html, height, id, autoHeight = true }: { html: string; height?: number; id?: string; autoHeight?: boolean }) {
    const iframeRef = useRef<HTMLIFrameElement | null>(null)

    useEffect(() => {
        const iframe = iframeRef.current
        if (!iframe || !autoHeight || height) return

        const setHeight = () => {
            try {
                const doc = iframe.contentDocument || iframe.contentWindow?.document
                if (!doc) return
                const body = doc.body
                const htmlEl = doc.documentElement
                const newHeight = Math.max(
                    body?.scrollHeight || 0,
                    body?.offsetHeight || 0,
                    htmlEl?.clientHeight || 0,
                    htmlEl?.scrollHeight || 0,
                    htmlEl?.offsetHeight || 0
                )
                if (newHeight && iframe.style.height !== `${newHeight}px`) {
                    iframe.style.height = `${newHeight}px`
                }
            } catch { }
        }

        const onLoad = () => {
            setHeight()
            try {
                const doc = iframe.contentDocument || iframe.contentWindow?.document
                const target = doc?.documentElement || doc?.body
                if (!target) return
                const RZ: any = (window as any).ResizeObserver
                if (typeof RZ === 'function') {
                    const ro = new RZ(() => setHeight())
                    ro.observe(target)
                        ; (iframe as any)._ro = ro
                }
            } catch { }
        }

        iframe.addEventListener('load', onLoad)
        // Also try once in case it's already loaded
        setTimeout(setHeight, 50)

        return () => {
            iframe.removeEventListener('load', onLoad)
            const ro = (iframe as any)._ro
            if (ro && ro.disconnect) ro.disconnect()
        }
    }, [html, autoHeight, height])

    return (
        <div className="w-full overflow-hidden rounded-lg border bg-white">
            <iframe
                ref={iframeRef}
                id={id}
                title="Plotly Chart"
                srcDoc={html}
                className="w-full block"
                style={{ height: height ?? undefined }}
            />
        </div>
    )
}
