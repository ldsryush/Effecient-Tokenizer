/**
 * Efficient Tokenizer — VS Code Extension
 *
 * Features:
 *  1. Status bar with live savings (polls /admin/metrics every 5s)
 *  2. Sidebar webview: session report, compression replay, settings
 *  3. Optional proxy auto-start and graceful degradation when offline
 */

import * as vscode from "vscode";
import * as cp from "child_process";
import * as path from "path";
import * as http from "http";

let proxyProcess: cp.ChildProcess | null = null;
let statusBarItem: vscode.StatusBarItem;
let metricsInterval: NodeJS.Timeout | undefined;
let viewProvider: EfficientTokenizerViewProvider | undefined;
let sessionStartTs = Date.now() / 1000;

// ─── Helpers ────────────────────────────────────────────────────────────────

function getConfig() {
  const cfg = vscode.workspace.getConfiguration("efficientTokenizer");
  return {
    port: cfg.get<number>("port", 8000),
    autoStart: cfg.get<boolean>("autoStart", true),
    compressionMode: cfg.get<string>("compressionMode", "lossy"),
    dryRun: cfg.get<boolean>("dryRun", false),
    openaiApiKey: cfg.get<string>("openaiApiKey", ""),
    relevanceThreshold: cfg.get<number>("relevanceThreshold", 0.15),
    dedupThreshold: cfg.get<number>("dedupThreshold", 0.92),
  };
}

function proxyUrl(port: number): string {
  return `http://localhost:${port}`;
}

async function httpGetJson(url: string, timeoutMs = 2000): Promise<any> {
  return new Promise((resolve, reject) => {
    const req = http.get(url, (res) => {
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        try {
          resolve(JSON.parse(data));
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on("error", reject);
    req.setTimeout(timeoutMs, () => {
      req.destroy();
      reject(new Error("timeout"));
    });
  });
}

async function isProxyRunning(port: number): Promise<boolean> {
  try {
    const res = await httpGetJson(`${proxyUrl(port)}/health`, 1200);
    return res?.status === "ok";
  } catch {
    return false;
  }
}

// ─── Proxy lifecycle ────────────────────────────────────────────────────────

function startProxy(context: vscode.ExtensionContext): void {
  if (proxyProcess) {
    vscode.window.showInformationMessage("Efficient Tokenizer proxy is already running.");
    return;
  }

  const cfg = getConfig();
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
      updateStatusBarOnline(cfg.port, 0, 0);
    }
  });

  proxyProcess.stderr?.on("data", (d: Buffer) => {
    const msg = d.toString();
    if (msg.includes("ERROR") || msg.includes("Exception")) {
      vscode.window.showErrorMessage(`Proxy error: ${msg.slice(0, 200)}`);
    }
  });

  proxyProcess.on("exit", () => {
    proxyProcess = null;
    setStatusBarOffline();
  });

  setStatusBarStarting();
}

function stopProxy(): void {
  if (!proxyProcess) {
    vscode.window.showInformationMessage("Proxy is not running.");
    return;
  }
  proxyProcess.kill();
  proxyProcess = null;
  setStatusBarOffline();
}

// ─── Status bar ─────────────────────────────────────────────────────────────

function setStatusBarStarting(): void {
  statusBarItem.text = `$(loading~spin) ET: starting`;
  statusBarItem.tooltip = "Efficient Tokenizer is starting...";
  statusBarItem.backgroundColor = undefined;
  statusBarItem.color = new vscode.ThemeColor("disabledForeground");
}

function setStatusBarOffline(): void {
  statusBarItem.text = `$(circle-slash) ET: offline`;
  statusBarItem.tooltip =
    "Efficient Tokenizer proxy is not running.\n" +
    "Run: Efficient Tokenizer: Start Proxy Server";
  statusBarItem.backgroundColor = undefined;
  statusBarItem.color = new vscode.ThemeColor("disabledForeground");
}

function updateStatusBarOnline(port: number, savedTokens: number, avgPct: number): void {
  const pct = Math.max(0, avgPct);
  const pctLabel = pct.toFixed(1);
  const savedLabel = savedTokens.toLocaleString();

  statusBarItem.text = `$(zap) ET: ${savedLabel} saved (${pctLabel}% avg)`;
  statusBarItem.tooltip =
    `Efficient Tokenizer — port ${port}\n` +
    `Tokens saved: ${savedLabel}\n` +
    `Avg compression: ${pctLabel}%\n` +
    `Click to open the sidebar report`;

  if (pct >= 30) {
    statusBarItem.color = new vscode.ThemeColor("charts.green");
  } else if (pct >= 15) {
    statusBarItem.color = new vscode.ThemeColor("charts.yellow");
  } else {
    statusBarItem.color = new vscode.ThemeColor("disabledForeground");
  }
  statusBarItem.backgroundColor = undefined;
}

// ─── Sidebar view ───────────────────────────────────────────────────────────

class EfficientTokenizerViewProvider implements vscode.WebviewViewProvider {
  private view?: vscode.WebviewView;
  private context: vscode.ExtensionContext;

  constructor(context: vscode.ExtensionContext) {
    this.context = context;
  }

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = { enableScripts: true };
    view.webview.html = getSidebarHtml(view.webview, this.context);

    view.webview.onDidReceiveMessage(async (msg) => {
      if (msg?.type === "saveSettings") {
        await applySettings(msg.payload);
        refreshAll(this.context);
      }
      if (msg?.type === "restoreTurn") {
        await sendToChatInput(String(msg.payload || ""));
      }
      if (msg?.type === "openProxyUrl") {
        const cfg = getConfig();
        vscode.env.openExternal(vscode.Uri.parse(`${proxyUrl(cfg.port)}/dashboard`));
      }
    });
  }

  postMessage(type: string, payload: any): void {
    if (!this.view) return;
    this.view.webview.postMessage({ type, payload });
  }
}

function getSidebarHtml(webview: vscode.Webview, context: vscode.ExtensionContext): string {
  const nonce = String(Date.now());
  const cfg = getConfig();
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} https:; style-src ${webview.cspSource} 'unsafe-inline' https:; script-src 'nonce-${nonce}';" />
  <title>Efficient Tokenizer</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Fira+Code:wght@400;600&display=swap');
    :root {
      --bg: #f5f3ef;
      --bg-2: #e8e1d7;
      --ink: #1d1b16;
      --ink-soft: #5a5349;
      --accent: #0e7c86;
      --accent-2: #e08e45;
      --card: #ffffff;
      --line: #d9cfc3;
      --muted: #b2a89c;
    }
    body {
      margin: 0;
      padding: 16px;
      font-family: 'Space Grotesk', system-ui, sans-serif;
      color: var(--ink);
      background: radial-gradient(1200px 800px at 20% -10%, #fef7e7, transparent),
                  radial-gradient(900px 600px at 110% 10%, #f3e7ff, transparent),
                  var(--bg);
    }
    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
    }
    .title {
      font-size: 20px;
      font-weight: 700;
    }
    .badge {
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      background: var(--bg-2);
      color: var(--ink-soft);
    }
    .tabs {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-bottom: 12px;
    }
    .tab {
      text-align: center;
      padding: 8px 10px;
      border-radius: 10px;
      background: var(--card);
      border: 1px solid var(--line);
      font-weight: 600;
      cursor: pointer;
    }
    .tab.active {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    .panel { display: none; }
    .panel.active { display: block; }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      margin-bottom: 12px;
      box-shadow: 0 8px 18px rgba(13, 12, 10, 0.05);
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }
    .metric {
      background: #faf7f1;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
    }
    .metric h4 { margin: 0 0 6px 0; font-size: 12px; color: var(--ink-soft); }
    .metric .value { font-size: 18px; font-weight: 700; }
    .stage-list { display: grid; gap: 6px; }
    .stage-row { display: flex; justify-content: space-between; font-size: 13px; }
    .row-muted { color: var(--ink-soft); }
    .replay-item { border: 1px solid var(--line); border-radius: 10px; padding: 8px; margin-bottom: 8px; background: #fbfaf8; }
    .replay-header { display: flex; justify-content: space-between; font-weight: 600; cursor: pointer; }
    .replay-body { margin-top: 6px; display: none; }
    .replay-body pre { background: #111; color: #e5e7eb; padding: 8px; border-radius: 8px; font-family: 'Fira Code', monospace; font-size: 12px; overflow: auto; }
    .btn {
      border: none; padding: 6px 10px; border-radius: 8px; cursor: pointer; font-weight: 600;
    }
    .btn.accent { background: var(--accent-2); color: #fff; }
    .settings-grid { display: grid; gap: 10px; }
    .settings-grid label { font-size: 12px; color: var(--ink-soft); }
    .settings-grid input, .settings-grid select {
      padding: 8px; border-radius: 8px; border: 1px solid var(--line); background: #fff;
    }
    .footer-note { font-size: 12px; color: var(--ink-soft); margin-top: 8px; }
  </style>
</head>
<body>
  <div class="header">
    <div class="title">Efficient Tokenizer</div>
    <div class="badge" id="statusBadge">offline</div>
  </div>

  <div class="tabs">
    <div class="tab active" data-tab="report">Report</div>
    <div class="tab" data-tab="replay">Replay</div>
    <div class="tab" data-tab="settings">Settings</div>
  </div>

  <div class="panel active" id="panel-report">
    <div class="card">
      <div class="metric-grid">
        <div class="metric"><h4>Tokens Used</h4><div class="value" id="tokensUsed">-</div></div>
        <div class="metric"><h4>Tokens Saved</h4><div class="value" id="tokensSaved">-</div></div>
        <div class="metric"><h4>Cost Saved Today</h4><div class="value" id="costSavedToday">-</div></div>
        <div class="metric"><h4>Cost Saved Month</h4><div class="value" id="costSavedMonth">-</div></div>
      </div>
    </div>
    <div class="card">
      <div class="row-muted">Stage Breakdown</div>
      <div class="stage-list" id="stageBreakdown"></div>
    </div>
    <button class="btn accent" id="openProxy">Open Proxy Dashboard</button>
    <div class="footer-note">Session totals reset when VS Code restarts.</div>
  </div>

  <div class="panel" id="panel-replay">
    <div class="card" id="replayList">Loading...</div>
  </div>

  <div class="panel" id="panel-settings">
    <div class="card">
      <div class="settings-grid">
        <div>
          <label>Proxy Port</label>
          <input id="portInput" type="number" min="1" max="65535" value="${cfg.port}" />
        </div>
        <div>
          <label>Compression Mode</label>
          <select id="modeInput">
            <option value="lossy" ${cfg.compressionMode === "lossy" ? "selected" : ""}>lossy</option>
            <option value="lossless" ${cfg.compressionMode === "lossless" ? "selected" : ""}>lossless</option>
          </select>
        </div>
        <div>
          <label>Relevance Threshold</label>
          <input id="relevanceInput" type="range" min="0" max="0.5" step="0.01" value="${cfg.relevanceThreshold}" />
        </div>
        <div>
          <label>Dedup Threshold</label>
          <input id="dedupInput" type="range" min="0.7" max="0.99" step="0.01" value="${cfg.dedupThreshold}" />
        </div>
        <div>
          <label>Dry Run</label>
          <select id="dryRunInput">
            <option value="false" ${cfg.dryRun ? "" : "selected"}>false</option>
            <option value="true" ${cfg.dryRun ? "selected" : ""}>true</option>
          </select>
        </div>
      </div>
      <button class="btn accent" id="saveSettings">Save Settings</button>
      <div class="footer-note">Settings update VS Code preferences only.</div>
    </div>
  </div>

  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const tabs = document.querySelectorAll('.tab');
    const panels = {
      report: document.getElementById('panel-report'),
      replay: document.getElementById('panel-replay'),
      settings: document.getElementById('panel-settings')
    };
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        Object.values(panels).forEach(p => p.classList.remove('active'));
        panels[tab.dataset.tab].classList.add('active');
      });
    });

    document.getElementById('saveSettings').addEventListener('click', () => {
      vscode.postMessage({
        type: 'saveSettings',
        payload: {
          port: Number(document.getElementById('portInput').value),
          compressionMode: document.getElementById('modeInput').value,
          relevanceThreshold: Number(document.getElementById('relevanceInput').value),
          dedupThreshold: Number(document.getElementById('dedupInput').value),
          dryRun: document.getElementById('dryRunInput').value === 'true'
        }
      });
    });

    document.getElementById('openProxy').addEventListener('click', () => {
      vscode.postMessage({ type: 'openProxyUrl' });
    });

    window.addEventListener('message', event => {
      const { type, payload } = event.data;
      if (type === 'status') {
        document.getElementById('statusBadge').textContent = payload;
      }
      if (type === 'report') {
        document.getElementById('tokensUsed').textContent = payload.tokensUsed || '-';
        document.getElementById('tokensSaved').textContent = payload.tokensSaved || '-';
        document.getElementById('costSavedToday').textContent = payload.costSavedToday || '-';
        document.getElementById('costSavedMonth').textContent = payload.costSavedMonth || '-';
        const breakdown = document.getElementById('stageBreakdown');
        breakdown.innerHTML = '';
        Object.entries(payload.stageBreakdown || {}).forEach(([stage, value]) => {
          const row = document.createElement('div');
          row.className = 'stage-row';
          row.innerHTML = `<span>${stage}</span><strong>${value}</strong>`;
          breakdown.appendChild(row);
        });
      }
      if (type === 'replay') {
        const list = document.getElementById('replayList');
        list.innerHTML = '';
        if (!payload || payload.length === 0) {
          list.textContent = 'No lossy drops recorded yet.';
          return;
        }
        payload.forEach((item, idx) => {
          const div = document.createElement('div');
          div.className = 'replay-item';
          const header = document.createElement('div');
          header.className = 'replay-header';
          header.innerHTML = `<span>${item.stage}</span><span>conf ${item.confidence}</span>`;
          const body = document.createElement('div');
          body.className = 'replay-body';
          const dropped = (item.removedTexts || []).map(t => `<pre>${t}</pre>`).join('');
          const button = document.createElement('button');
          button.className = 'btn accent';
          button.textContent = 'restore this turn';
          button.addEventListener('click', () => {
            const text = (item.removedTexts || [])[0] || '';
            vscode.postMessage({ type: 'restoreTurn', payload: text });
          });
          body.innerHTML = dropped;
          body.appendChild(button);
          header.addEventListener('click', () => {
            body.style.display = body.style.display === 'block' ? 'none' : 'block';
          });
          div.appendChild(header);
          div.appendChild(body);
          list.appendChild(div);
        });
      }
    });
  </script>
</body>
</html>`;
}

async function applySettings(payload: any): Promise<void> {
  const cfg = vscode.workspace.getConfiguration("efficientTokenizer");
  if (typeof payload?.port === "number") await cfg.update("port", payload.port, vscode.ConfigurationTarget.Global);
  if (typeof payload?.compressionMode === "string") await cfg.update("compressionMode", payload.compressionMode, vscode.ConfigurationTarget.Global);
  if (typeof payload?.relevanceThreshold === "number") await cfg.update("relevanceThreshold", payload.relevanceThreshold, vscode.ConfigurationTarget.Global);
  if (typeof payload?.dedupThreshold === "number") await cfg.update("dedupThreshold", payload.dedupThreshold, vscode.ConfigurationTarget.Global);
  if (typeof payload?.dryRun === "boolean") await cfg.update("dryRun", payload.dryRun, vscode.ConfigurationTarget.Global);
}

async function sendToChatInput(text: string): Promise<void> {
  if (!text) return;
  const candidates: Array<{ id: string; args?: any[] }> = [
    { id: "workbench.action.chat.open", args: [{ query: text }] },
    { id: "workbench.action.chat.open", args: [text] },
    { id: "workbench.action.chat.addToPrompt", args: [text] },
  ];
  for (const cmd of candidates) {
    try {
      await vscode.commands.executeCommand(cmd.id, ...(cmd.args || []));
      return;
    } catch {
      continue;
    }
  }
  await vscode.env.clipboard.writeText(text);
  vscode.window.showInformationMessage("Turn copied to clipboard for manual restore.");
}

// ─── Metrics polling ────────────────────────────────────────────────────────

async function refreshAll(context: vscode.ExtensionContext): Promise<void> {
  const cfg = getConfig();
  const running = await isProxyRunning(cfg.port);
  if (!running) {
    setStatusBarOffline();
    viewProvider?.postMessage("status", "offline");
    return;
  }

  try {
    const metrics = await httpGetJson(`${proxyUrl(cfg.port)}/admin/metrics`);
    const eventsRes = await httpGetJson(`${proxyUrl(cfg.port)}/admin/events`);
    const confidenceRes = await httpGetJson(`${proxyUrl(cfg.port)}/admin/confidence-log`);

    const saved = metrics.total_token_savings || 0;
    const pct = metrics.avg_pct_saved || 0;
    updateStatusBarOnline(cfg.port, saved, pct);
    viewProvider?.postMessage("status", "online");

    const events = Array.isArray(eventsRes?.events) ? eventsRes.events : [];
    const sessionEvents = events.filter((e: any) => (e.ts || 0) >= sessionStartTs);
    const tokensUsed = sessionEvents.reduce((acc: number, e: any) => acc + (e.raw_tokens || 0), 0);
    const tokensSaved = sessionEvents.reduce((acc: number, e: any) => acc + (e.token_savings || 0), 0);
    const stageBreakdown: Record<string, number> = {};
    sessionEvents.forEach((e: any) => {
      const sb = e.savings_by_stage || {};
      Object.keys(sb).forEach((k) => {
        stageBreakdown[k] = (stageBreakdown[k] || 0) + (sb[k] || 0);
      });
    });

    const { costSavedToday, costSavedMonth } = updateCostState(context, events);

    viewProvider?.postMessage("report", {
      tokensUsed: tokensUsed.toLocaleString(),
      tokensSaved: tokensSaved.toLocaleString(),
      costSavedToday: `$${costSavedToday.toFixed(4)}`,
      costSavedMonth: `$${costSavedMonth.toFixed(4)}`,
      stageBreakdown,
    });

    const confidenceLog = Array.isArray(confidenceRes?.confidence_log) ? confidenceRes.confidence_log : [];
    const replayItems = flattenConfidenceLog(confidenceLog);
    viewProvider?.postMessage("replay", replayItems);
  } catch {
    setStatusBarOffline();
    viewProvider?.postMessage("status", "offline");
  }
}

function flattenConfidenceLog(entries: any[]): any[] {
  const items: any[] = [];
  for (const entry of entries) {
    const details = entry.compression_details || [];
    for (const d of details) {
      const removedTexts = d.removed_texts || [];
      if (!removedTexts.length) continue;
      items.push({
        stage: d.stage || "unknown",
        confidence: d.confidence ?? entry.confidence_score,
        removedTexts,
      });
    }
  }
  return items;
}

function updateCostState(context: vscode.ExtensionContext, events: any[]): { costSavedToday: number; costSavedMonth: number } {
  const key = "efficientTokenizer.costState";
  const today = new Date().toISOString().slice(0, 10);
  const month = new Date().toISOString().slice(0, 7);
  const state = context.globalState.get<any>(key) || {
    day: today,
    month: month,
    dayTotal: 0,
    monthTotal: 0,
    lastTs: 0,
  };

  if (state.day !== today) {
    state.day = today;
    state.dayTotal = 0;
  }
  if (state.month !== month) {
    state.month = month;
    state.monthTotal = 0;
  }

  const newEvents = events.filter((e: any) => (e.ts || 0) > (state.lastTs || 0));
  const increment = newEvents.reduce((acc: number, e: any) => acc + (e.cost_usd_saved || 0), 0);
  state.dayTotal += increment;
  state.monthTotal += increment;
  if (newEvents.length) {
    state.lastTs = Math.max(...newEvents.map((e: any) => e.ts || 0));
  }

  context.globalState.update(key, state);
  return { costSavedToday: state.dayTotal, costSavedMonth: state.monthTotal };
}

// ─── Activation ─────────────────────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext): void {
  const cfg = getConfig();
  sessionStartTs = Date.now() / 1000;

  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBarItem.command = "efficientTokenizer.openSidebar";
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  viewProvider = new EfficientTokenizerViewProvider(context);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("efficientTokenizer.dashboardView", viewProvider)
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("efficientTokenizer.startProxy", () => startProxy(context)),
    vscode.commands.registerCommand("efficientTokenizer.stopProxy", () => stopProxy()),
    vscode.commands.registerCommand("efficientTokenizer.openSidebar", () => {
      vscode.commands.executeCommand("workbench.view.extension.efficientTokenizer");
    }),
    vscode.commands.registerCommand("efficientTokenizer.openDashboard", () => {
      const url = `${proxyUrl(cfg.port)}/dashboard`;
      vscode.env.openExternal(vscode.Uri.parse(url));
    }),
    vscode.commands.registerCommand("efficientTokenizer.copyBaseUrl", () => {
      const url = `${proxyUrl(cfg.port)}/v1`;
      vscode.env.clipboard.writeText(url);
      vscode.window.showInformationMessage(`Copied to clipboard: ${url}`);
    })
  );

  isProxyRunning(cfg.port).then((running) => {
    if (running) {
      updateStatusBarOnline(cfg.port, 0, 0);
    } else if (cfg.autoStart) {
      startProxy(context);
    } else {
      setStatusBarOffline();
    }
  });

  metricsInterval = setInterval(() => {
    refreshAll(context);
  }, 5000);
}

export function deactivate(): void {
  if (metricsInterval) clearInterval(metricsInterval);
  if (proxyProcess) {
    proxyProcess.kill();
    proxyProcess = null;
  }
}
