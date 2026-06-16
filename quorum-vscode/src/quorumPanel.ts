// Renders a ConsensusResult into a VS Code WebviewPanel.
//
// All styling uses VS Code theme tokens (var(--vscode-*)) so light and dark
// themes "just work" without a theme listener. CSP is strict: no inline
// scripts, no remote sources, hashes/nonces only for the per-render script
// we ship (currently none — pure HTML/CSS).

import * as vscode from 'vscode';
import type { ConsensusResult, ModelResponse } from './quorumClient';

/**
 * Confidence tier thresholds. Mirrors the quorum-api server-side bucketing
 * so the colour shown in the panel matches what the API reports elsewhere.
 */
const CONFIDENCE_TIERS = {
  high: 0.75,
  medium: 0.5
};

let currentPanel: vscode.WebviewPanel | undefined;

export function showConsensusResult(
  result: ConsensusResult,
  context: vscode.ExtensionContext
): void {
  const column = vscode.window.activeTextEditor?.viewColumn ?? vscode.ViewColumn.Beside;

  if (currentPanel) {
    currentPanel.reveal(column, /* preserveFocus */ true);
    currentPanel.webview.html = renderHtml(currentPanel.webview, result);
    return;
  }

  const panel = vscode.window.createWebviewPanel(
    'quorum.result',
    'Quorum: Consensus',
    { viewColumn: column, preserveFocus: true },
    {
      enableScripts: false,
      retainContextWhenHidden: true,
      localResourceRoots: [context.extensionUri]
    }
  );

  panel.webview.html = renderHtml(panel.webview, result);
  panel.onDidDispose(() => {
    if (currentPanel === panel) {
      currentPanel = undefined;
    }
  }, null, context.subscriptions);

  currentPanel = panel;
}

function renderHtml(webview: vscode.Webview, result: ConsensusResult): string {
  const tier = confidenceTier(result.confidence);
  const confidencePct = Math.round(clamp01(result.confidence) * 100);
  const tierColorVar = {
    high: 'var(--vscode-testing-iconPassed, #3fb950)',
    medium: 'var(--vscode-charts-yellow, #d29922)',
    low: 'var(--vscode-testing-iconFailed, #f85149)'
  }[tier];

  const modelRows = result.models
    .map((m) => renderModelRow(m))
    .join('\n');

  // Strict CSP: no inline/eval/remote. cspSource covers any local resources
  // we might add later (icons, css), nothing else is permitted.
  const csp = [
    `default-src 'none'`,
    `img-src ${webview.cspSource} https: data:`,
    `style-src ${webview.cspSource} 'unsafe-inline'`,
    `font-src ${webview.cspSource}`
  ].join('; ');

  const totalCost = result.totalCostUsd.toFixed(4);

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<title>Quorum Consensus</title>
<style>
  :root {
    --quorum-tier-color: ${tierColorVar};
  }
  body {
    font-family: var(--vscode-font-family);
    font-size: var(--vscode-font-size);
    color: var(--vscode-foreground);
    background: var(--vscode-editor-background);
    padding: 16px 20px;
    line-height: 1.5;
  }
  h1, h2 {
    color: var(--vscode-foreground);
    border-bottom: 1px solid var(--vscode-panel-border);
    padding-bottom: 6px;
    margin-top: 0;
  }
  h1 { font-size: 1.3em; }
  h2 { font-size: 1.05em; margin-top: 1.5em; }

  .answer {
    font-size: 1.15em;
    background: var(--vscode-textBlockQuote-background);
    border-left: 4px solid var(--vscode-textBlockQuote-border, var(--quorum-tier-color));
    padding: 12px 16px;
    margin: 12px 0 24px;
    white-space: pre-wrap;
    border-radius: 2px;
  }

  .confidence {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
  }
  .confidence-label {
    min-width: 120px;
    color: var(--vscode-descriptionForeground);
  }
  .confidence-value {
    font-weight: 600;
    color: var(--quorum-tier-color);
  }
  .confidence-bar {
    flex: 1;
    height: 8px;
    background: var(--vscode-progressBar-background, var(--vscode-editor-inactiveSelectionBackground));
    border-radius: 4px;
    overflow: hidden;
  }
  .confidence-bar-fill {
    height: 100%;
    width: ${confidencePct}%;
    background: var(--quorum-tier-color);
    transition: width 200ms ease;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
    font-size: 0.95em;
  }
  th, td {
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--vscode-panel-border);
    vertical-align: top;
  }
  th {
    color: var(--vscode-descriptionForeground);
    font-weight: 600;
    background: var(--vscode-editorWidget-background);
  }
  tr:hover td {
    background: var(--vscode-list-hoverBackground);
  }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .snippet {
    color: var(--vscode-descriptionForeground);
    max-width: 480px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .status-ok {
    color: var(--vscode-testing-iconPassed, #3fb950);
  }
  .status-err {
    color: var(--vscode-testing-iconFailed, #f85149);
  }

  .total {
    margin-top: 16px;
    color: var(--vscode-descriptionForeground);
    font-size: 0.95em;
  }
  .total strong {
    color: var(--vscode-foreground);
    font-variant-numeric: tabular-nums;
  }

  .empty {
    color: var(--vscode-descriptionForeground);
    font-style: italic;
  }
</style>
</head>
<body>
  <h1>Quorum Consensus</h1>

  <div class="answer">${escapeHtml(result.answer) || '<span class="empty">(no answer returned)</span>'}</div>

  <div class="confidence">
    <span class="confidence-label">Confidence</span>
    <div class="confidence-bar"><div class="confidence-bar-fill"></div></div>
    <span class="confidence-value">${confidencePct}%</span>
  </div>

  <h2>Model breakdown</h2>
  ${result.models.length === 0
    ? `<p class="empty">No model responses returned.</p>`
    : `<table>
      <thead>
        <tr>
          <th>Model</th>
          <th class="num">Latency</th>
          <th class="num">Cost (USD)</th>
          <th>Status</th>
          <th>Response snippet</th>
        </tr>
      </thead>
      <tbody>
        ${modelRows}
      </tbody>
    </table>`
  }

  <p class="total">Total cost: <strong>$${escapeHtml(totalCost)}</strong></p>
</body>
</html>`;
}

function renderModelRow(m: ModelResponse): string {
  const isErr = !!m.error;
  const statusText = m.error
    ? `error: ${escapeHtml(truncate(m.error, 80))}`
    : escapeHtml(m.status ?? 'ok');
  const statusClass = isErr ? 'status-err' : 'status-ok';
  const snippet = escapeHtml(truncate(m.response.replace(/\s+/g, ' ').trim(), 160));

  return `<tr>
    <td>${escapeHtml(m.name)}</td>
    <td class="num">${m.latency_ms} ms</td>
    <td class="num">$${m.cost_usd.toFixed(4)}</td>
    <td class="${statusClass}">${statusText}</td>
    <td class="snippet" title="${escapeHtml(m.response)}">${snippet || '<span class="empty">(empty)</span>'}</td>
  </tr>`;
}

function confidenceTier(c: number): 'high' | 'medium' | 'low' {
  const v = clamp01(c);
  if (v >= CONFIDENCE_TIERS.high) {
    return 'high';
  }
  if (v >= CONFIDENCE_TIERS.medium) {
    return 'medium';
  }
  return 'low';
}

function clamp01(n: number): number {
  if (!Number.isFinite(n)) {
    return 0;
  }
  if (n < 0) {
    return 0;
  }
  if (n > 1) {
    return 1;
  }
  return n;
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n) + '...';
}

/**
 * Minimal HTML escape — we have no inline scripts and we're embedding
 * arbitrary model output, so this is the only thing standing between us and
 * a stored-XSS-equivalent inside the webview.
 */
function escapeHtml(s: string): string {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
