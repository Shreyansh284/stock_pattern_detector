import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { detect, fetchPatterns, fetchStocks, fetchTimeframes, fetchChartTypes, fetchModes, type Chart } from '../lib/api'

type State = {
  stocks: string[]
  patterns: string[]
  timeframes: string[]
  chartTypes: string[]
  modes: string[]
  selectedStock?: string
  selectedPattern?: string
  selectedTimeframe?: string
  selectedChartType?: string
  selectedMode?: string
  // Date range mode
  useDateRange: boolean
  startDate?: string
  endDate?: string
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
  setMode: (v?: string) => void
  setUseDateRange: (v: boolean) => void
  setStartDate: (v?: string) => void
  setEndDate: (v?: string) => void
  runDetect: () => Promise<void>
  clearCharts: () => void
}

export const useDetectStore = create<State & Actions>()(
  persist(
    (set, get) => ({
  stocks: [],
  patterns: [],
  timeframes: [],
  chartTypes: [],
  modes: [],
  charts: [],
  loading: false,
  useDateRange: false,

  init: async () => {
    set({ loading: true, error: undefined })
    try {
      const [stocks, patterns, timeframes, chartTypes, modes] = await Promise.all([
        fetchStocks(),
        fetchPatterns(),
        fetchTimeframes(),
        fetchChartTypes(),
        fetchModes(),
      ])
      set({ stocks, patterns, timeframes, chartTypes, modes, selectedMode: modes?.[0] })
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
  setMode: (v) => set({ selectedMode: v }),
  setUseDateRange: (v) => set({ useDateRange: v }),
  setStartDate: (v) => set({ startDate: v }),
  setEndDate: (v) => set({ endDate: v }),
  runDetect: async () => {
  const { selectedStock, selectedPattern, selectedTimeframe, selectedChartType, useDateRange, startDate, endDate, selectedMode } = get()
  if (!selectedStock || !selectedPattern || !selectedChartType || !selectedMode) {
      set({ error: 'Please select stock, pattern and chart type.' })
      return
    }
    if (useDateRange) {
      if (!startDate || !endDate) {
        set({ error: 'Please select start and end date.' })
        return
      }
    } else {
      if (!selectedTimeframe) {
        set({ error: 'Please select a timeframe or switch to date range.' })
        return
      }
    }
    set({ loading: true, error: undefined, charts: [] })
    try {
      const base = { stock: selectedStock, pattern: selectedPattern, chart_type: selectedChartType, mode: selectedMode }
      const payload = useDateRange
        ? { ...base, start_date: startDate, end_date: endDate }
        : { ...base, timeframe: selectedTimeframe }
      const res = await detect(payload as any)
  set({ charts: res.charts })
    } catch (e: any) {
      set({ error: e?.message ?? 'Detection failed' })
    } finally {
      set({ loading: false })
    }
  },
  clearCharts: () => set({ charts: [] }),
    }),
    {
      name: 'detect-store',
      partialize: state => ({
        selectedStock: state.selectedStock,
        selectedPattern: state.selectedPattern,
        selectedTimeframe: state.selectedTimeframe,
        selectedChartType: state.selectedChartType,
        selectedMode: state.selectedMode,
        useDateRange: state.useDateRange,
        startDate: state.startDate,
        endDate: state.endDate,
      }),
    }
  )
)
