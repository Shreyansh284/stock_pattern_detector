import axios from 'axios'

export const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

// Simple in-memory cache for static endpoints to avoid repeated round-trips
type CacheEntry<T> = { ts: number; data: T }
const _cache = new Map<string, CacheEntry<any>>()
const TTL_STATIC = 5 * 60 * 1000 // 5 minutes
function getCached<T>(key: string, ttl = TTL_STATIC): T | undefined {
  const e = _cache.get(key)
  if (e && (Date.now() - e.ts) < ttl) return e.data as T
  return undefined
}
function setCached<T>(key: string, data: T) {
  _cache.set(key, { ts: Date.now(), data })
}

export type DetectRequest = {
  stock: string
  pattern: string
  chart_type: string
  timeframe?: string
  start_date?: string
  end_date?: string
  mode?: string
  data_source?: 'live' | 'past'
  stock_data_dir?: string
}

export type Chart = { timeframe: string; html: string; pattern?: string; strength?: 'strong' | 'weak'; explanation?: any }

export async function fetchStocks() {
  const k = 'stocks'
  const c = getCached<string[]>(k)
  if (c) return c
  const { data } = await axios.get<string[]>(`${API_BASE}/stocks`)
  setCached(k, data)
  return data
}

export async function fetchPatterns() {
  const k = 'patterns'
  const c = getCached<string[]>(k)
  if (c) return c
  const { data } = await axios.get<string[]>(`${API_BASE}/patterns`)
  setCached(k, data)
  return data
}

export async function fetchTimeframes() {
  const k = 'timeframes'
  const c = getCached<string[]>(k)
  if (c) return c
  const { data } = await axios.get<string[]>(`${API_BASE}/timeframes`)
  setCached(k, data)
  return data
}

export async function fetchChartTypes() {
  const k = 'chart-types'
  const c = getCached<string[]>(k)
  if (c) return c
  const { data } = await axios.get<string[]>(`${API_BASE}/chart-types`)
  setCached(k, data)
  return data
}

export async function fetchModes() {
  const k = 'modes'
  const c = getCached<string[]>(k)
  if (c) return c
  const { data } = await axios.get<string[]>(`${API_BASE}/modes`)
  setCached(k, data)
  return data
}

export async function detect(req: DetectRequest) {
  const { data } = await axios.post<{ charts: Chart[]; strong_charts?: Chart[]; weak_charts?: Chart[] }>(`${API_BASE}/detect`, req)
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
export type DetectAllRequest = { start_date: string; end_date: string; chart_type?: 'candle' | 'line' | 'ohlc'; mode?: 'lenient' | 'strict'; data_source?: 'live' | 'past'; stock_data_dir?: string; stock_limit?: number }
export type DetectAllResponse = { results: StockPatternResult[] }
/**
 * Run pattern detection across all stocks in a given date range.
 */
export async function detectAll(req: DetectAllRequest) {
  const { data } = await axios.post<DetectAllResponse>(`${API_BASE}/detect-all`, req)
  return data
}

// Background job endpoints for accurate progress
export async function startDetectAll(req: DetectAllRequest) {
  const { data } = await axios.post<{ job_id: string }>(`${API_BASE}/detect-all-start`, req)
  return data
}
export type DetectAllProgress = { status: string; current: number; total: number; symbol?: string; message?: string; percent: number }
export async function getDetectAllProgress(job_id: string) {
  const { data } = await axios.get<DetectAllProgress>(`${API_BASE}/detect-all-progress`, { params: { job_id } })
  return data
}
export async function getDetectAllResult(job_id: string) {
  const { data } = await axios.get<DetectAllResponse>(`${API_BASE}/detect-all-result`, { params: { job_id } })
  return data
}

// Home ticker tape
export type TickerItem = { symbol: string; display_symbol?: string; price: number; change_pct: number; volume: number; avg_volume: number; price_spike: boolean; volume_spike: boolean; sparkline: number[] }
export async function fetchTickerTape(count = 20) {
  const { data } = await axios.get<{ tickers: TickerItem[] }>(`${API_BASE}/ticker-tape`, { params: { count } })
  return data.tickers
}
