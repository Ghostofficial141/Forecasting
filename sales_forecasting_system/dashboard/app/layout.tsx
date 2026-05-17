import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })

export const metadata: Metadata = {
  title: 'Sales Forecasting Dashboard',
  description:
    'Production-ready time-series forecasting platform using SARIMA, Prophet, XGBoost & LSTM.',
  keywords: ['sales forecasting', 'ML dashboard', 'time series', 'LSTM', 'XGBoost'],
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="bg-surface text-white antialiased">{children}</body>
    </html>
  )
}
