'use client'
import { MODELS } from '@/lib/mock-data'

const typeColor: Record<string, string> = {
  Statistical: 'bg-blue-500/15 text-blue-400',
  'Statistical / ML': 'bg-indigo-500/15 text-indigo-400',
  'Machine Learning': 'bg-emerald-500/15 text-emerald-400',
  'Deep Learning': 'bg-orange-500/15 text-orange-400',
}

export function ModelsGrid() {
  return (
    <div className="card p-6">
      <h2 className="text-lg font-semibold text-white mb-5">Forecasting Models</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {MODELS.map((m, i) => (
          <div
            key={m.name}
            className="card-elevated p-4 flex flex-col gap-2 hover:border-brand-500/40 transition-all duration-300 animate-slide-up"
            style={{ animationDelay: `${i * 80}ms` }}
          >
            <div className="flex items-center gap-3">
              <div
                className="w-3 h-3 rounded-full flex-shrink-0"
                style={{ backgroundColor: m.color }}
              />
              <span className="font-semibold text-white">{m.name}</span>
              <span className={`badge ml-auto ${typeColor[m.type] ?? 'bg-slate-500/20 text-slate-300'}`}>
                {m.type}
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">{m.description}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
