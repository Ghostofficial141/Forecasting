'use client'
import { useState, useEffect } from 'react'
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { modelDistribution, MODELS } from '@/lib/mock-data'
import { BarChart3 } from 'lucide-react'

const COLORS = MODELS.map((m) => m.color)

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-surface-elevated border border-surface-border rounded-xl p-3 shadow-xl text-sm">
      <p className="font-semibold text-white">{payload[0]?.name}</p>
      <p className="text-slate-400">{payload[0]?.value} states</p>
    </div>
  )
}

export function ModelDistributionChart() {
  const [mounted, setMounted] = useState(false)
  const data = modelDistribution()

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <div className="card p-6 min-h-[350px] flex flex-col items-center justify-center">
        <BarChart3 size={24} className="text-slate-500 animate-pulse mb-2" />
        <span className="text-slate-400 text-sm">Loading Best Model Distribution...</span>
      </div>
    )
  }

  return (
    <div className="card p-6">
      <h2 className="text-lg font-semibold text-white flex items-center gap-2 mb-5">
        <BarChart3 size={18} className="text-brand-400" />
        Best Model Distribution
      </h2>
      <p className="text-sm text-slate-400 mb-4 -mt-3">Best model per US state</p>

      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={90}
            paddingAngle={4}
            dataKey="value"
          >
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          <Legend
            formatter={(value) => (
              <span style={{ color: '#94a3b8', fontSize: 12 }}>{value}</span>
            )}
          />
        </PieChart>
      </ResponsiveContainer>

      <div className="grid grid-cols-2 gap-2 mt-2">
        {data.map((d, i) => (
          <div key={d.name} className="flex items-center gap-2 text-sm">
            <div
              className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
              style={{ backgroundColor: COLORS[i % COLORS.length] }}
            />
            <span className="text-slate-300">{d.name}</span>
            <span className="text-slate-500 ml-auto">{d.value} states</span>
          </div>
        ))}
      </div>
    </div>
  )
}

