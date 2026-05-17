'use client'
import { BEST_MODELS, MODELS } from '@/lib/mock-data'
import { MapPin } from 'lucide-react'

const MODEL_COLORS: Record<string, string> = {
  SARIMA: '#60a5fa',
  Prophet: '#818cf8',
  XGBoost: '#34d399',
  LSTM: '#fb923c',
}

export function BestModelsPanel() {
  const entries = Object.entries(BEST_MODELS).slice(0, 20)

  return (
    <div className="card p-6">
      <h2 className="text-lg font-semibold text-white flex items-center gap-2 mb-1">
        <MapPin size={18} className="text-brand-400" />
        Best Model per State
      </h2>
      <p className="text-sm text-slate-400 mb-5">Top 20 states shown</p>

      <div className="space-y-2">
        {entries.map(([state, model]) => (
          <div
            key={state}
            className="flex items-center justify-between py-2 px-3 rounded-xl hover:bg-surface-elevated transition-colors duration-150"
          >
            <span className="text-sm text-slate-300">{state}</span>
            <span
              className="badge"
              style={{
                backgroundColor: `${MODEL_COLORS[model] ?? '#60a5fa'}20`,
                color: MODEL_COLORS[model] ?? '#60a5fa',
              }}
            >
              {model}
            </span>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="mt-4 pt-4 border-t border-surface-border flex flex-wrap gap-3">
        {MODELS.map((m) => (
          <div key={m.name} className="flex items-center gap-1.5 text-xs text-slate-400">
            <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: m.color }} />
            {m.name}
          </div>
        ))}
      </div>
    </div>
  )
}
