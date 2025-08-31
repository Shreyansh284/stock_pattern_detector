import axios from 'axios'

export const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export type DetectRequest = {
  stock: string
  pattern: string
  timeframe: string
  chart_type: string
}

export type Chart = { timeframe: string; html: string }

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

export async function detect(req: DetectRequest) {
  const { data } = await axios.post<{ charts: Chart[] }>(`${API_BASE}/detect`, req)
  return data
}
