import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { DetectAllResponse } from '../lib/api'

export type StrengthFilter = 'all' | 'strong' | 'weak'

type DashboardState = {
  data: DetectAllResponse | null
  patternFilter?: string
  stockFilter?: string
  strengthFilter: StrengthFilter
}

type DashboardActions = {
  setData: (data: DetectAllResponse | null) => void
  setPatternFilter: (v?: string) => void
  setStockFilter: (v?: string) => void
  setStrengthFilter: (v: StrengthFilter) => void
  clear: () => void
}

export const useDashboardStore = create<DashboardState & DashboardActions>()(
  persist(
    (set) => ({
      data: null,
      patternFilter: undefined,
      stockFilter: undefined,
      strengthFilter: 'all',
      setData: (data) => set({ data }),
      setPatternFilter: (v) => set({ patternFilter: v }),
      setStockFilter: (v) => set({ stockFilter: v }),
      setStrengthFilter: (v) => set({ strengthFilter: v }),
      clear: () => set({ data: null })
    }),
    {
      name: 'dashboard-store',
      // Persist only filters; do not persist data (HTML can be large)
      partialize: (state) => ({
        patternFilter: state.patternFilter,
        stockFilter: state.stockFilter,
        strengthFilter: state.strengthFilter,
      }),
      version: 1,
    }
  )
)
