import axios from 'axios'

export const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export type DetectRequest = {
  stock: string
  pattern: string
  chart_type: string
  timeframe?: string
  start_date?: string
  end_date?: string
  mode?: string
}

export type Chart = { timeframe: string; html: string; pattern?: string }

export async function fetchStocks() {
  const { data } = await axios.get<string[]>(`${API_BASE}/stocks`)
  return data
}

export async function fetchPatterns() {
  const { data } = await axios.get<string[]>(`${API_BASE}/patterns`)
  return data
}

export async function fetchTimeframes() {
  const { data } = await axios.get<string[]>(`${API_BASE}/timeframes`)
  return data
}

export async function fetchChartTypes() {
  const { data } = await axios.get<string[]>(`${API_BASE}/chart-types`)
  return data
}

export async function fetchModes() {
  const { data } = await axios.get<string[]>(`${API_BASE}/modes`)
  return data
}

export async function detect(req: DetectRequest) {
  const { data } = await axios.post<{ charts: Chart[] }>(`${API_BASE}/detect`, req)
  return data
}
// Request & response types for detecting patterns across all stocks
export type StockPatternResult = {
  stock: string
  patterns: string[]
  pattern_counts: Record<string, number>
  count: number
  current_price: number
  current_volume: number
  charts: Array<Chart & { pattern: string }>
}
export type DetectAllRequest = { start_date: string; end_date: string }
export type DetectAllResponse = { results: StockPatternResult[] }
/**
 * Run pattern detection across all stocks in a given date range.
 */
export async function detectAll(req: DetectAllRequest) {
  const { data } = await axios.post<DetectAllResponse>(`${API_BASE}/detect-all`, req)
  return data
}
