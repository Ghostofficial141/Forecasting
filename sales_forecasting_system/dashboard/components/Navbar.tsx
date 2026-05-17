'use client'
import { TrendingUp, Github, BookOpen } from 'lucide-react'

export function Navbar() {
  return (
    <header className="sticky top-0 z-50 border-b border-surface-border bg-surface/80 backdrop-blur-xl">
      <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-brand-600/30">
            <TrendingUp size={18} className="text-white" />
          </div>
          <div>
            <span className="font-bold text-white text-base">SalesForecast</span>
            <span className="ml-1.5 text-xs badge bg-brand-500/20 text-brand-400">v1.0</span>
          </div>
        </div>

        {/* Nav links */}
        <nav className="hidden md:flex items-center gap-1">
          {['Overview', 'Forecast', 'Metrics', 'Models'].map((item) => (
            <a
              key={item}
              href={`#${item.toLowerCase()}`}
              className="btn-ghost text-sm"
            >
              {item}
            </a>
          ))}
        </nav>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <a
            href="https://github.com/Ghostofficial141/Forecasting"
            target="_blank"
            rel="noreferrer"
            className="btn-ghost text-sm flex items-center gap-1.5"
          >
            <Github size={15} />
            <span className="hidden sm:inline">GitHub</span>
          </a>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="btn-primary text-sm flex items-center gap-1.5"
          >
            <BookOpen size={15} />
            <span className="hidden sm:inline">API Docs</span>
          </a>
        </div>
      </div>
    </header>
  )
}
