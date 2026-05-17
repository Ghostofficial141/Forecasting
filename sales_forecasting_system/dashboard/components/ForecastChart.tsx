'use client'
import { useState, useEffect } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { generateHistorical, generateForecast, US_STATES } from '@/lib/mock-data'
import { TrendingUp } from 'lucide-react'

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  const isForecast = payload[0]?.payload?.isForecast
  return (
    <div className="bg-surface-elevated border border-surface-border rounded-xl p-3 shadow-xl text-sm">
      <p className="text-slate-400 mb-1">{label}</p>
      <p className="font-bold text-white">
        ${payload[0]?.value?.toLocaleString()}
      </p>
      {isForecast && <p className="text-xs text-brand-400 mt-1">📈 Forecast</p>}
    </div>
  )
}

export function ForecastChart() {
  const [mounted, setMounted] = useState(false)
  const [state, setState] = useState('California')
  const [weeks, setWeeks] = useState(8)

  useEffect(() => {
    setMounted(true)
  }, [])

  const historical = generateHistorical(state, 26)
  const forecast = generateForecast(state, weeks)

  const histData = historical.map((d) => ({ ...d, isForecast: false }))
  const foreData = forecast.forecast.map((d) => ({ ...d, isForecast: true }))
  const combined = [...histData, ...foreData]
  const splitDate = historical[historical.length - 1]?.date

  if (!mounted) {
    return (
      <div className="card p-6 min-h-[420px] flex flex-col items-center justify-center animate-fade-in">
        <TrendingUp size={24} className="text-slate-500 animate-pulse mb-2" />
        <span className="text-slate-400 text-sm">Loading Sales Forecast...</span>
      </div>
    )
  }

  return (
    <div className="card p-6 animate-fade-in">
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
        <div>
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <TrendingUp size={18} className="text-brand-400" />
            Sales Forecast
          </h2>
          <p className="text-sm text-slate-400 mt-0.5">
            Historical (26 wks) + {weeks}-week forecast · Model:{' '}
            <span className="text-brand-400 font-medium">{forecast.model_used}</span>
          </p>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <select
            id="state-select"
            value={state}
            onChange={(e) => setState(e.target.value)}
            className="bg-surface-elevated border border-surface-border text-white text-sm rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-500"
          >
            {US_STATES.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <select
            id="weeks-select"
            value={weeks}
            onChange={(e) => setWeeks(Number(e.target.value))}
            className="bg-surface-elevated border border-surface-border text-white text-sm rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-500"
          >
            {[4, 8, 12, 16, 26].map((w) => (
              <option key={w} value={w}>{w} weeks</option>
            ))}
          </select>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <AreaChart data={combined} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
          <defs>
            <linearGradient id="historicalGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="forecastGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#818cf8" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#2e3250" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fill: '#64748b', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: '#64748b', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            x={splitDate}
            stroke="#3b82f6"
            strokeDasharray="4 2"
            label={{ value: 'Now', fill: '#60a5fa', fontSize: 11, position: 'top' }}
          />
          <Area
            type="monotone"
            dataKey="sales"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#historicalGrad)"
            dot={false}
            activeDot={{ r: 5, fill: '#60a5fa' }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
