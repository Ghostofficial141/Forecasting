'use client'
import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { generateMetrics } from '@/lib/mock-data'
import { Activity } from 'lucide-react'

type Metric = 'rmse' | 'mae' | 'mape' | 'r2'

const MODEL_COLORS: Record<string, string> = {
  SARIMA: '#60a5fa',
  Prophet: '#818cf8',
  XGBoost: '#34d399',
  LSTM: '#fb923c',
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-surface-elevated border border-surface-border rounded-xl p-3 shadow-xl text-sm">
      <p className="text-slate-400 mb-1">{label}</p>
      {payload.map((p: any) => (
        <p key={p.name} className="font-semibold" style={{ color: p.fill }}>
          {p.name}: {typeof p.value === 'number' && p.value > 10
            ? Math.round(p.value).toLocaleString()
            : p.value?.toFixed(3)}
        </p>
      ))}
    </div>
  )
}

export function MetricsTable() {
  const [mounted, setMounted] = useState(false)
  const [metric, setMetric] = useState<Metric>('rmse')

  useEffect(() => {
    setMounted(true)
  }, [])

  const all = generateMetrics()

  // Aggregate by model
  const byModel: Record<string, number[]> = {}
  all.forEach((row) => {
    if (!byModel[row.model]) byModel[row.model] = []
    byModel[row.model].push(row[metric])
  })
  const chartData = Object.entries(byModel).map(([model, vals]) => ({
    model,
    value: parseFloat((vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(3)),
  }))

  const metricLabels: Record<Metric, string> = {
    rmse: 'RMSE', mae: 'MAE', mape: 'MAPE (%)', r2: 'R²',
  }

  if (!mounted) {
    return (
      <div className="card p-6 min-h-[500px] flex flex-col items-center justify-center animate-fade-in">
        <Activity size={24} className="text-slate-500 animate-pulse mb-2" />
        <span className="text-slate-400 text-sm">Loading Model Metrics...</span>
      </div>
    )
  }

  return (
    <div className="card p-6 animate-fade-in">
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Activity size={18} className="text-brand-400" />
          Model Performance Metrics
        </h2>
        <div className="flex gap-1">
          {(['rmse', 'mae', 'mape', 'r2'] as Metric[]).map((m) => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
                metric === m
                  ? 'bg-brand-600 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-surface-elevated'
              }`}
            >
              {metricLabels[m]}
            </button>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2e3250" vertical={false} />
          <XAxis dataKey="model" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(59,130,246,0.05)' }} />
          <Bar dataKey="value" radius={[6, 6, 0, 0]} maxBarSize={56}>
            {chartData.map((d) => (
              <Cell key={d.model} fill={MODEL_COLORS[d.model] ?? '#60a5fa'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Raw table */}
      <div className="mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-surface-border">
              {['State', 'Model', 'RMSE', 'MAE', 'MAPE', 'R²'].map((h) => (
                <th key={h} className="text-left text-slate-400 font-medium pb-2 pr-4">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {all.slice(0, 12).map((row, i) => (
              <tr
                key={i}
                className="border-b border-surface-border/50 hover:bg-surface-elevated/50 transition-colors"
              >
                <td className="py-2 pr-4 text-slate-200">{row.state}</td>
                <td className="py-2 pr-4">
                  <span
                    className="badge"
                    style={{
                      backgroundColor: `${MODEL_COLORS[row.model]}20`,
                      color: MODEL_COLORS[row.model],
                    }}
                  >
                    {row.model}
                  </span>
                </td>
                <td className="py-2 pr-4 text-slate-300">{row.rmse.toLocaleString()}</td>
                <td className="py-2 pr-4 text-slate-300">{row.mae.toLocaleString()}</td>
                <td className="py-2 pr-4 text-slate-300">{row.mape}%</td>
                <td className="py-2 text-slate-300">{row.r2}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-slate-500 mt-3">Showing top 12 of {all.length} records</p>
      </div>
    </div>
  )
}
