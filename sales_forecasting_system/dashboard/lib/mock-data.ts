// lib/mock-data.ts
// Realistic mock data that mirrors the real API responses
// Used when the FastAPI backend is not connected (Vercel static demo)

export const MODELS = [
  {
    name: 'SARIMA',
    description: 'Seasonal ARIMA with automatic order selection via pmdarima.',
    type: 'Statistical',
    color: '#60a5fa',
  },
  {
    name: 'Prophet',
    description: 'Facebook Prophet with yearly/weekly seasonality and US holidays.',
    type: 'Statistical / ML',
    color: '#818cf8',
  },
  {
    name: 'XGBoost',
    description: 'Gradient-boosted trees using lag + rolling + calendar features; recursive forecasting.',
    type: 'Machine Learning',
    color: '#34d399',
  },
  {
    name: 'LSTM',
    description: 'Multi-layer LSTM with dropout; sequence-to-one; recursive multi-step.',
    type: 'Deep Learning',
    color: '#fb923c',
  },
]

export const US_STATES = [
  'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
  'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
  'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
  'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
  'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
  'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
  'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
  'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
  'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
  'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming',
]

const modelNames = ['SARIMA', 'Prophet', 'XGBoost', 'LSTM']
const seed = (n: number) => ((Math.sin(n) * 43758.5453) % 1 + 1) / 2

export function generateForecast(state: string, weeks: number = 8) {
  const base = 8000 + state.length * 800
  const dates: string[] = []
  const values: number[] = []
  const now = new Date('2024-01-29')
  for (let i = 1; i <= weeks; i++) {
    const d = new Date(now)
    d.setDate(d.getDate() + i * 7)
    dates.push(d.toISOString().split('T')[0])
    const noise = seed(state.charCodeAt(0) * i) * 2000 - 1000
    values.push(Math.round(base + noise + i * 120))
  }
  const modelIdx = state.charCodeAt(0) % 4
  return {
    state,
    model_used: modelNames[modelIdx],
    weeks,
    forecast: dates.map((date, i) => ({ date, sales: values[i] })),
    generated_at: new Date().toISOString(),
  }
}

export function generateHistorical(state: string, points: number = 52) {
  const base = 8000 + state.length * 800
  const data = []
  const start = new Date('2023-01-02')
  for (let i = 0; i < points; i++) {
    const d = new Date(start)
    d.setDate(d.getDate() + i * 7)
    const noise = seed(state.charCodeAt(0) * 31 + i) * 3000 - 1500
    const trend = i * 80
    const seasonal = Math.sin((i / 52) * 2 * Math.PI) * 1500
    data.push({
      date: d.toISOString().split('T')[0],
      sales: Math.max(1000, Math.round(base + noise + trend + seasonal)),
    })
  }
  return data
}

export function generateMetrics() {
  const rows: {
    state: string; model: string; rmse: number; mae: number;
    mape: number; smape: number; r2: number;
  }[] = []
  const top10 = US_STATES.slice(0, 10)
  top10.forEach((state) => {
    modelNames.forEach((model, mi) => {
      const s = state.charCodeAt(0) + mi * 7
      rows.push({
        state,
        model,
        rmse: Math.round(500 + seed(s) * 1500),
        mae: Math.round(300 + seed(s + 1) * 900),
        mape: parseFloat((3 + seed(s + 2) * 12).toFixed(2)),
        smape: parseFloat((4 + seed(s + 3) * 10).toFixed(2)),
        r2: parseFloat((0.7 + seed(s + 4) * 0.28).toFixed(3)),
      })
    })
  })
  return rows
}

export function generateBestModels() {
  const out: Record<string, string> = {}
  US_STATES.forEach((state) => {
    out[state] = modelNames[state.charCodeAt(0) % 4]
  })
  return out
}

export const BEST_MODELS = generateBestModels()

export function modelDistribution() {
  const counts: Record<string, number> = { SARIMA: 0, Prophet: 0, XGBoost: 0, LSTM: 0 }
  Object.values(BEST_MODELS).forEach((m) => counts[m]++)
  return Object.entries(counts).map(([name, value]) => ({ name, value }))
}

export function kpiStats() {
  const metrics = generateMetrics()
  const avgRmse = Math.round(metrics.reduce((a, b) => a + b.rmse, 0) / metrics.length)
  const avgMape = parseFloat((metrics.reduce((a, b) => a + b.mape, 0) / metrics.length).toFixed(2))
  const avgR2 = parseFloat((metrics.reduce((a, b) => a + b.r2, 0) / metrics.length).toFixed(3))
  return { avgRmse, avgMape, avgR2, totalStates: US_STATES.length }
}
