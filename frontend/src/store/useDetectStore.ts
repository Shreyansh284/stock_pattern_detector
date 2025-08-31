import { create } from 'zustand'
import { detect, fetchPatterns, fetchStocks, fetchTimeframes, type Chart } from '../lib/api'

type State = {
  stocks: string[]
  patterns: string[]
  timeframes: string[]
  selectedStock?: string
  selectedPattern?: string
  selectedTimeframes: string[]
  charts: Chart[]
  loading: boolean
  error?: string
}

type Actions = {
  init: () => Promise<void>
  setStock: (v?: string) => void
  setPattern: (v?: string) => void
  toggleTimeframe: (v: string) => void
  runDetect: () => Promise<void>
  clearCharts: () => void
}

export const useDetectStore = create<State & Actions>((set, get) => ({
  stocks: [],
  patterns: [],
  timeframes: [],
  selectedTimeframes: [],
  charts: [],
  loading: false,

  init: async () => {
    set({ loading: true, error: undefined })
    try {
      const [stocks, patterns, timeframes] = await Promise.all([
        fetchStocks(),
        fetchPatterns(),
        fetchTimeframes(),
      ])
      set({ stocks, patterns, timeframes })
    } catch (e: any) {
      set({ error: e?.message ?? 'Failed to load metadata' })
    } finally {
      set({ loading: false })
    }
  },
  setStock: (v) => set({ selectedStock: v }),
  setPattern: (v) => set({ selectedPattern: v }),
  toggleTimeframe: (v) => {
    const cur = get().selectedTimeframes
    set({ selectedTimeframes: cur.includes(v) ? cur.filter(t => t !== v) : [...cur, v] })
  },
  runDetect: async () => {
    const { selectedStock, selectedPattern, selectedTimeframes } = get()
    if (!selectedStock || !selectedPattern || selectedTimeframes.length === 0) {
      set({ error: 'Please select stock, pattern and at least one timeframe.' })
      return
    }
    set({ loading: true, error: undefined, charts: [] })
    try {
      const res = await detect({ stock: selectedStock, pattern: selectedPattern, timeframes: selectedTimeframes })
      set({ charts: res.charts })
    } catch (e: any) {
      set({ error: e?.message ?? 'Detection failed' })
    } finally {
      set({ loading: false })
    }
  },
  clearCharts: () => set({ charts: [] }),
}))
