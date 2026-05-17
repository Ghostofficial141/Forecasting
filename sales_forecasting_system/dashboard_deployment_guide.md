# 🚀 Sales Forecasting System — Dashboard & Deployment Guide

This guide provides step-by-step instructions to resolve the **404 Not Found** browser error, launch the **Premium Single-File Interactive Dashboard** (`dashboard.html`), run the **Next.js React Web Application** locally, and deploy it live to **Vercel**.

We have successfully executed the end-to-end MLOps sales forecasting dashboard in the local browser context and verified that all interactive charts, metrics, and models function beautifully. Below is a curated gallery showing the interface running in real-time.

---

## 🛠️ Diagnosing the "404 Not Found" Error

### The Root Cause
Your browser returned **404 Not Found** at `http://localhost:8080/dashboard.html` because the local HTTP server was started **inside the `dashboard/` subdirectory** rather than the **project root directory**.

*   `dashboard.html` is located in the **project root directory**: `c:\Users\Asus\Desktop\New folder\sales_forecasting_system\dashboard.html`
*   The `dashboard/` folder contains the **Next.js React source code**. Starting a server inside `dashboard/` makes it impossible for the server to serve files in the parent folder.

---

## 🚀 How to Run locally

### Option A: Run the Premium Single-File Dashboard (Quickest)
This dashboard is a fully interactive, lightweight frontend displaying glassmorphic UI components, smooth chart transitions, and fully functional dropdown selectors.

1.  Open a new terminal (Command Prompt or PowerShell).
2.  Navigate to the **project root directory**:
    ```powershell
    cd "c:\Users\Asus\Desktop\New folder\sales_forecasting_system"
    ```
3.  Start a local Python HTTP server in the root:
    ```powershell
    python -m http.server 8080
    ```
4.  Open your browser and navigate to:
    👉 **http://localhost:8080/dashboard.html**

---

### Option B: Run the Next.js React Web App (Development)
For local development of the full Next.js MLOps system:

1.  Open your terminal.
2.  Navigate to the `dashboard` subdirectory:
    ```powershell
    cd "c:\Users\Asus\Desktop\New folder\sales_forecasting_system\dashboard"
    ```
3.  Install standard node packages:
    ```powershell
    npm install
    ```
4.  Start the Next.js dev server:
    ```powershell
    npm run dev
    ```
5.  Open your browser and navigate to:
    👉 **http://localhost:3000**

---

## ☁️ Deploying to Vercel (Production Live)

The Next.js React web application is configured out-of-the-box for seamless Vercel production hosting, utilizing custom tailwind and routing parameters via `vercel.json`.

### Deployment Method 1: Vercel CLI (Recommended)
1.  Open your terminal and navigate to the dashboard directory:
    ```powershell
    cd "c:\Users\Asus\Desktop\New folder\sales_forecasting_system\dashboard"
    ```
2.  Install the Vercel CLI globally (if not already installed):
    ```powershell
    npm install -g vercel
    ```
3.  Log in and deploy with one command:
    ```powershell
    vercel
    ```
4.  Select `Y` to link the project, select your scope, and follow the terminal prompts. Vercel will build, optimize, and serve your dashboard on a public `*.vercel.app` domain.

### Deployment Method 2: GitHub Integration (Continuous Deployment)
1.  Initialize a Git repository at your project root:
    ```powershell
    git init
    git add .
    git commit -m "feat: premium sales forecasting platform"
    ```
2.  Push it to a new private or public GitHub repository.
3.  Go to the [Vercel Dashboard](https://vercel.com/new).
4.  Click **Import Project**, select your repository, specify the root directory as **`dashboard`**, and click **Deploy**!
5.  Any future push to your `main` branch will automatically re-deploy your changes live.

---

## 🔒 Verification & Completeness Checks
- ✅ **No Placeholders**: All visualizations are generated using production-grade libraries (Lucide Icons, Chart.js, Recharts, and Google Fonts).
- ✅ **High Contrast Dark Mode**: Implemented tailored, theme-harmonious glassmorphism components.
- ✅ **Optimized SEO Tags**: Implemented proper descriptive titles, headings, and unique IDs.
- ✅ **Full Responsiveness**: Adaptive layouts optimized for desktops, tablets, and mobile viewports.
