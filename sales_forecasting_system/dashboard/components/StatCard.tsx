'use client'
import { LucideIcon } from 'lucide-react'

interface StatCardProps {
  label: string
  value: string | number
  sub?: string
  icon: LucideIcon
  iconColor?: string
  trend?: { value: string; up: boolean }
  delay?: number
}

export function StatCard({ label, value, sub, icon: Icon, iconColor = 'text-brand-400', trend, delay = 0 }: StatCardProps) {
  return (
    <div
      className="stat-card"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-start justify-between">
        <div className={`p-2.5 rounded-xl bg-surface-elevated ${iconColor}`}>
          <Icon size={20} />
        </div>
        {trend && (
          <span
            className={`badge ${trend.up ? 'bg-emerald-500/15 text-emerald-400' : 'bg-red-500/15 text-red-400'}`}
          >
            {trend.up ? '▲' : '▼'} {trend.value}
          </span>
        )}
      </div>
      <div>
        <p className="text-2xl font-bold text-white">{value}</p>
        <p className="text-sm text-slate-400 mt-0.5">{label}</p>
        {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
      </div>
    </div>
  )
}
