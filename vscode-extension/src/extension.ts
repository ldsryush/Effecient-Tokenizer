/**
 * Efficient Tokenizer — VS Code Extension
 *
 * What it does:
 *  1. Spawns the Python proxy server (uvicorn app.main:app) as a background process
 *  2. Shows a status bar item with live token savings
 *  3. Embeds the /dashboard page in a sidebar WebviewPanel
 *  4. Provides commands to start/stop/open-dashboard/copy-url
 *
 * To use with Cline or any other LLM extension:
 *   Set their "base URL" to http://localhost:8000/v1
 *   Every LLM call will flow through the compression pipeline automatically.
 */

import * as vscode from "vscode";
import * as cp from "child_process";
import * as path from "path";
import * as http from "http";

let proxyProcess: cp.ChildProcess | null = null;
let statusBarItem: vscode.StatusBarItem;
let dashboardPanel: vscode.WebviewPanel | undefined;
let metricsInterval: NodeJS.Timeout | undefined;

// ─── Helpers ────────────────────────────────────────────────────────────────

function getConfig() {
  const cfg = vscode.workspace.getConfiguration("efficientTokenizer");
  return {
    port:            cfg.get<number>("port", 8000),
    autoStart:       cfg.get<boolean>("autoStart", true),
    compressionMode: cfg.get<string>("compressionMode", "lossy"),
    dryRun:          cfg.get<boolean>("dryRun", false),
    openaiApiKey:    cfg.get<string>("openaiApiKey", ""),
  };
}

function proxyUrl(port: number): string {
  return `http://localhost:${port}`;
}

async function isProxyRunning(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get(`${proxyUrl(port)}/health`, (res) => {
      resolve(res.statusCode === 200);
    });
    req.on("error", () => resolve(false));
    req.setTimeout(1500, () => { req.destroy(); resolve(false); });
  });
}

async function fetchMetrics(port: number): Promise<any> {
  return new Promise((resolve, reject) => {
    const req = http.get(`${proxyUrl(port)}/admin/metrics`, (res) => {
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => { try { resolve(JSON.parse(data)); } catch { reject(); } });
    });
    req.on("error", reject);
    req.setTimeout(2000, () => { req.destroy(); reject(); });
  });
}

// ─── Proxy lifecycle ────────────────────────────────────────────────────────

function startProxy(context: vscode.ExtensionContext): void {
  if (proxyProcess) {
    vscode.window.showInformationMessage("Efficient Tokenizer proxy is already running.");
    return;
  }

  const cfg = getConfig();

  // Find the repo root (parent of the extension directory)
  const repoRoot = path.resolve(context.extensionPath, "..");

  const env: NodeJS.ProcessEnv = {
    ...process.env,
    DISPATCH_DRY_RUN: cfg.dryRun ? "true" : "false",
  };
  if (cfg.openaiApiKey) {
    env["OPENAI_API_KEY"] = cfg.openaiApiKey;
  }

  const args = [
    "-m", "uvicorn",
    "app.main:app",
    "--host", "0.0.0.0",
    "--port", String(cfg.port),
    "--workers", "1",
  ];

  proxyProcess = cp.spawn("python", args, { cwd: repoRoot, env });

  proxyProcess.stdout?.on("data", (d: Buffer) => {
    const line = d.toString().trim();
    if (line.includes("Application startup complete")) {
      updateStatusBar(cfg.port, true);
      vscode.window.showInformationMessage(
        `Efficient Tokenizer proxy started on port ${cfg.port}. Base URL: ${proxyUrl(cfg.port)}/v1`
      );
    }
  });

  proxyProcess.stderr?.on("data", (d: Buffer) => {
    // uvicorn writes info logs to stderr — only surface real errors
    const msg = d.toString();
    if (msg.includes("ERROR") || msg.includes("Exception")) {
      vscode.window.showErrorMessage(`Proxy error: ${msg.slice(0, 200)}`);
    }
  });

  proxyProcess.on("exit", (code) => {
    proxyProcess = null;
    updateStatusBar(cfg.port, false);
    if (code !== 0 && code !== null) {
      vscode.window.showWarningMessage(`Efficient Tokenizer proxy exited with code ${code}.`);
    }
  });

  updateStatusBar(cfg.port, false, "starting...");
}

function stopProxy(): void {
  if (!proxyProcess) {
    vscode.window.showInformationMessage("Proxy is not running.");
    return;
  }
  proxyProcess.kill();
  proxyProcess = null;
  vscode.window.showInformationMessage("Efficient Tokenizer proxy stopped.");
}

// ─── Status bar ─────────────────────────────────────────────────────────────

function updateStatusBar(port: number, running: boolean, extra?: string): void {
  if (running) {
    statusBarItem.text = `$(zap) ET: live :${port}`;
    statusBarItem.tooltip = `Efficient Tokenizer proxy running on port ${port}\nClick to open dashboard`;
    statusBarItem.backgroundColor = undefined;
  } else {
    statusBarItem.text = extra
      ? `$(loading~spin) ET: ${extra}`
      : `$(circle-slash) ET: stopped`;
    statusBarItem.tooltip = "Efficient Tokenizer proxy is not running. Click to start.";
    statusBarItem.backgroundColor = new vscode.ThemeColor("statusBarItem.warningBackground");
  }
}

async function refreshMetrics(port: number): Promise<void> {
  try {
    const metrics = await fetchMetrics(port);
    const saved = (metrics.total_token_savings || 0).toLocaleString();
    const pct = (metrics.avg_pct_saved || 0).toFixed(1);
    statusBarItem.text = `$(zap) ET: ${saved} tokens saved (${pct}% avg)`;
    statusBarItem.tooltip =
      `Efficient Tokenizer — port ${port}\n` +
      `Total tokens saved: ${saved}\n` +
      `Avg compression: ${pct}%\n` +
      `Cost saved: $${(metrics.total_cost_usd_saved || 0).toFixed(4)}\n` +
      `Overhead: ${(metrics.avg_overhead_ms || 0).toFixed(1)} ms avg\n\n` +
      `Click to open dashboard`;
  } catch {
    // proxy not yet ready — silent
  }
}

// ─── Dashboard webview ──────────────────────────────────────────────────────

function openDashboard(context: vscode.ExtensionContext, port: number): void {
  if (dashboardPanel) {
    dashboardPanel.reveal();
    return;
  }

  dashboardPanel = vscode.window.createWebviewPanel(
    "efficientTokenizerDashboard",
    "Efficient Tokenizer Dashboard",
    vscode.ViewColumn.Beside,
    { enableScripts: true, retainContextWhenHidden: true }
  );

  // The dashboard HTML fetches from the proxy via absolute URL.
  // We inject the base URL so it works regardless of what port is configured.
  dashboardPanel.webview.html = getDashboardHtml(port);

  dashboardPanel.onDidDispose(() => { dashboardPanel = undefined; });
}

function getDashboardHtml(port: number): string {
  // Proxy the dashboard from the running server via an iframe.
  // This is the simplest approach — the full dashboard HTML is already served
  // by the proxy at /dashboard.
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Efficient Tokenizer Dashboard</title>
  <style>
    body, html { margin: 0; padding: 0; height: 100vh; background: #0f1117; }
    iframe { width: 100%; height: 100vh; border: none; }
    .connecting {
      display: flex; align-items: center; justify-content: center;
      height: 100vh; color: #64748b; font-family: system-ui, sans-serif;
      flex-direction: column; gap: 16px;
    }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: #f59e0b; animation: pulse 1.2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
  </style>
</head>
<body>
  <div class="connecting" id="loader">
    <div class="dot"></div>
    <div>Connecting to proxy on port ${port}...</div>
  </div>
  <iframe id="frame" style="display:none" src="http://localhost:${port}/dashboard"></iframe>
  <script>
    const frame = document.getElementById('frame');
    const loader = document.getElementById('loader');
    let attempts = 0;
    function tryLoad() {
      fetch('http://localhost:${port}/health')
        .then(r => r.json())
        .then(() => {
          loader.style.display = 'none';
          frame.style.display = 'block';
        })
        .catch(() => {
          attempts++;
          if (attempts < 30) setTimeout(tryLoad, 1000);
          else loader.innerHTML = '<div style="color:#ef4444">Could not connect to proxy on port ${port}.<br/>Run: uvicorn app.main:app --port ${port}</div>';
        });
    }
    tryLoad();
  </script>
</body>
</html>`;
}

// ─── Activation ─────────────────────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext): void {
  const cfg = getConfig();

  // Status bar
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBarItem.command = "efficientTokenizer.openDashboard";
  updateStatusBar(cfg.port, false);
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  // Commands
  context.subscriptions.push(
    vscode.commands.registerCommand("efficientTokenizer.startProxy", () => startProxy(context)),
    vscode.commands.registerCommand("efficientTokenizer.stopProxy", () => stopProxy()),
    vscode.commands.registerCommand("efficientTokenizer.openDashboard", () => openDashboard(context, cfg.port)),
    vscode.commands.registerCommand("efficientTokenizer.copyBaseUrl", () => {
      const url = `${proxyUrl(cfg.port)}/v1`;
      vscode.env.clipboard.writeText(url);
      vscode.window.showInformationMessage(`Copied to clipboard: ${url}`);
    }),
  );

  // Auto-start
  if (cfg.autoStart) {
    isProxyRunning(cfg.port).then((running) => {
      if (running) {
        updateStatusBar(cfg.port, true);
        vscode.window.showInformationMessage(`Efficient Tokenizer: proxy already running on port ${cfg.port}.`);
      } else {
        startProxy(context);
      }
    });
  }

  // Poll metrics every 10s to update the status bar
  metricsInterval = setInterval(() => {
    if (proxyProcess) refreshMetrics(cfg.port);
  }, 10_000);
}

export function deactivate(): void {
  if (metricsInterval) clearInterval(metricsInterval);
  if (proxyProcess) {
    proxyProcess.kill();
    proxyProcess = null;
  }
}
