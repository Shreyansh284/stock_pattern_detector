import { create } from 'zustand'
import { detect, fetchPatterns, fetchStocks, fetchTimeframes, fetchChartTypes, type Chart } from '../lib/api'

type State = {
  stocks: string[]
  patterns: string[]
  timeframes: string[]
  chartTypes: string[]
  selectedStock?: string
  selectedPattern?: string
  selectedTimeframe?: string
  selectedChartType?: string
  charts: Chart[]
  loading: boolean
  error?: string
}

type Actions = {
  init: () => Promise<void>
  setStock: (v?: string) => void
  setPattern: (v?: string) => void
  setTimeframe: (v?: string) => void
  setChartType: (v?: string) => void
  runDetect: () => Promise<void>
  clearCharts: () => void
}

export const useDetectStore = create<State & Actions>((set, get) => ({
  stocks: [],
  patterns: [],
  timeframes: [],
  chartTypes: [],
  charts: [],
  loading: false,

  init: async () => {
    set({ loading: true, error: undefined })
    try {
      const [stocks, patterns, timeframes, chartTypes] = await Promise.all([
        fetchStocks(),
        fetchPatterns(),
        fetchTimeframes(),
        fetchChartTypes(),
      ])
      set({ stocks, patterns, timeframes, chartTypes })
    } catch (e: any) {
      set({ error: e?.message ?? 'Failed to load metadata' })
    } finally {
      set({ loading: false })
    }
  },
  setStock: (v) => set({ selectedStock: v }),
  setPattern: (v) => set({ selectedPattern: v }),
  setTimeframe: (v) => set({ selectedTimeframe: v }),
  setChartType: (v) => set({ selectedChartType: v }),
  runDetect: async () => {
    const { selectedStock, selectedPattern, selectedTimeframe, selectedChartType } = get()
    if (!selectedStock || !selectedPattern || !selectedTimeframe || !selectedChartType) {
      set({ error: 'Please select stock, pattern, timeframe and chart type.' })
      return
    }
    set({ loading: true, error: undefined, charts: [] })
    try {
  const res = await detect({ stock: selectedStock, pattern: selectedPattern, timeframe: selectedTimeframe, chart_type: selectedChartType })
  set({ charts: res.charts })
    } catch (e: any) {
      set({ error: e?.message ?? 'Detection failed' })
    } finally {
      set({ loading: false })
    }
  },
  clearCharts: () => set({ charts: [] }),
}))
