'use client'
import { Navbar } from '@/components/Navbar'
import { StatCard } from '@/components/StatCard'
import { ForecastChart } from '@/components/ForecastChart'
import { MetricsTable } from '@/components/MetricsTable'
import { ModelDistributionChart } from '@/components/ModelDistributionChart'
import { BestModelsPanel } from '@/components/BestModelsPanel'
import { ModelsGrid } from '@/components/ModelsGrid'
import { kpiStats } from '@/lib/mock-data'
import {
  MapPin, Cpu, BarChart2, Percent,
  TrendingUp, Zap,
} from 'lucide-react'

export default function Home() {
  const { avgRmse, avgMape, avgR2, totalStates } = kpiStats()

  return (
    <div className="min-h-screen bg-surface bg-gradient-mesh">
      <Navbar />

      {/* Hero */}
      <section id="overview" className="max-w-screen-2xl mx-auto px-4 sm:px-6 py-12">
        <div className="mb-8 animate-fade-in">
          <div className="inline-flex items-center gap-2 badge bg-brand-500/15 text-brand-400 mb-4 py-1.5 px-4 rounded-full text-xs font-semibold">
            <Zap size={12} />
            Production-Ready MLOps Platform
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white leading-tight">
            Sales Forecasting{' '}
            <span className="gradient-text">Dashboard</span>
          </h1>
          <p className="mt-3 text-slate-400 max-w-2xl text-sm sm:text-base">
            End-to-end time-series forecasting using{' '}
            <span className="text-blue-400 font-medium">SARIMA</span>,{' '}
            <span className="text-indigo-400 font-medium">Prophet</span>,{' '}
            <span className="text-emerald-400 font-medium">XGBoost</span>, and{' '}
            <span className="text-orange-400 font-medium">LSTM</span> — with automatic
            model selection per US state.
          </p>
        </div>

        {/* KPI Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard
            label="US States Covered"
            value={totalStates}
            icon={MapPin}
            iconColor="text-brand-400"
            trend={{ value: '100%', up: true }}
            delay={0}
          />
          <StatCard
            label="Avg RMSE"
            value={`${avgRmse.toLocaleString()}`}
            sub="Root Mean Squared Error"
            icon={BarChart2}
            iconColor="text-indigo-400"
            delay={80}
          />
          <StatCard
            label="Avg MAPE"
            value={`${avgMape}%`}
            sub="Mean Absolute % Error"
            icon={Percent}
            iconColor="text-emerald-400"
            trend={{ value: 'Low', up: true }}
            delay={160}
          />
          <StatCard
            label="Avg R²"
            value={avgR2}
            sub="Coefficient of determination"
            icon={TrendingUp}
            iconColor="text-orange-400"
            trend={{ value: 'Strong fit', up: true }}
            delay={240}
          />
        </div>

        {/* Pipeline Badges */}
        <div id="models" className="flex flex-wrap gap-2 mb-12">
          {[
            { label: 'Data Ingestion', color: 'bg-blue-500/15 text-blue-400' },
            { label: 'Data Validation', color: 'bg-blue-500/15 text-blue-400' },
            { label: 'Preprocessing', color: 'bg-indigo-500/15 text-indigo-400' },
            { label: 'Feature Engineering', color: 'bg-indigo-500/15 text-indigo-400' },
            { label: 'Model Training', color: 'bg-emerald-500/15 text-emerald-400' },
            { label: 'Model Evaluation', color: 'bg-orange-500/15 text-orange-400' },
            { label: 'Model Selection', color: 'bg-pink-500/15 text-pink-400' },
          ].map((step, i) => (
            <span key={i} className={`badge py-1.5 px-4 text-xs font-semibold ${step.color}`}>
              {i + 1}. {step.label}
            </span>
          ))}
        </div>

        {/* Forecast Chart */}
        <div id="forecast" className="mb-8">
          <ForecastChart />
        </div>

        {/* Middle Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2">
            <ModelDistributionChart />
          </div>
          <div>
            <BestModelsPanel />
          </div>
        </div>

        {/* Metrics */}
        <div id="metrics" className="mb-8">
          <MetricsTable />
        </div>

        {/* Models Grid */}
        <ModelsGrid />
      </section>

      {/* Footer */}
      <footer className="border-t border-surface-border mt-16">
        <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-slate-500 text-sm">
            <Cpu size={14} />
            Sales Forecasting System — SARIMA · Prophet · XGBoost · LSTM
          </div>
          <p className="text-slate-600 text-xs">
            Built for Vercel · FastAPI backend · Production MLOps
          </p>
        </div>
      </footer>
    </div>
  )
}
