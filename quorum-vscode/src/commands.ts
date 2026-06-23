// User-facing Quorum commands.
//
// Each registerXxx() returns the disposable from vscode.commands.registerCommand
// so extension.ts can push them all onto context.subscriptions in activate().
//
// Conventions:
// - Every command that hits the network wraps the call in withProgress
//   (Notification location) so the user sees "Querying N providers..." and
//   can see Quorum is alive while it waits for consensus.
// - Selection-based commands fail loud (showErrorMessage) when there is no
//   selection or no active editor — silent no-ops cause "did it work?" support.
// - Results are rendered via showConsensusResult, a simple webview panel that
//   renders the answer + per-model breakdown. The webview is intentionally
//   minimal (no scripts, no external assets) so CSP stays tight.

import * as vscode from 'vscode';
import { ConsensusResult, QuorumClient } from './quorumClient';
import { QuorumStatusBar } from './statusBar';

// Module-scoped state for compareImplementations. Persists across the two
// invocations needed to capture selection A then selection B. Cleared after
// the comparison runs (or on failure) so a fresh A/B pair always starts clean.
let pendingCompareA: { code: string; language: string } | undefined;

/**
 * quorum.ask — free-form prompt, no editor context.
 */
export function registerAsk(
  client: QuorumClient,
  output: vscode.OutputChannel,
  statusBar?: QuorumStatusBar
): vscode.Disposable {
  return vscode.commands.registerCommand('quorum.ask', async () => {
    if (!(await ensureKeyOrPrompt())) return;
    const prompt = await vscode.window.showInputBox({
      prompt: 'Ask Quorum...',
      placeHolder: 'e.g. "What is the safest way to store an API key in a VS Code extension?"',
      ignoreFocusOut: true
    });
    if (!prompt) {
      return;
    }
    await runQuery(client, output, prompt, 'Quorum: Ask', statusBar);
  });
}

/**
 * quorum.askAboutSelection — user picks the question, we prepend the selected
 * code as a fenced block tagged with the detected language. Useful for
 * "why does this throw?" / "is this thread-safe?" without leaving the editor.
 */
export function registerAskAboutSelection(
  client: QuorumClient,
  output: vscode.OutputChannel,
  statusBar?: QuorumStatusBar
): vscode.Disposable {
  return vscode.commands.registerCommand('quorum.askAboutSelection', async () => {
    const sel = getSelectionOrError('Quorum: highlight some code first.');
    if (!sel) {
      return;
    }
    const question = await vscode.window.showInputBox({
      prompt: 'What about this code?',
      ignoreFocusOut: true
    });
    if (!question) {
      return;
    }
    const composed = `\`\`\`${sel.language}\n${sel.code}\n\`\`\`\n\n${question}`;
    await runQuery(client, output, composed, 'Quorum: Ask about selection', statusBar);
  });
}

/**
 * quorum.explainSelection — canned "explain this code" prompt with edge-case
 * call-out. Three-to-five sentence cap keeps responses scannable for triage.
 */
export function registerExplainSelection(
  client: QuorumClient,
  output: vscode.OutputChannel,
  statusBar?: QuorumStatusBar
): vscode.Disposable {
  return vscode.commands.registerCommand('quorum.explainSelection', async () => {
    const sel = getSelectionOrError('Quorum: highlight code to explain.');
    if (!sel) {
      return;
    }
    const prompt =
      `Explain the following ${sel.language} code in 3-5 sentences. ` +
      `What does it do? What are the edge cases?\n\n` +
      `\`\`\`${sel.language}\n${sel.code}\n\`\`\``;
    await runQuery(client, output, prompt, 'Quorum: Explain selection', statusBar);
  });
}

/**
 * quorum.reviewSelection — bugs/security/perf review with severity labels.
 * Terse output requested so the panel stays glanceable.
 */
export function registerReviewSelection(
  client: QuorumClient,
  output: vscode.OutputChannel,
  statusBar?: QuorumStatusBar
): vscode.Disposable {
  return vscode.commands.registerCommand('quorum.reviewSelection', async () => {
    const sel = getSelectionOrError('Quorum: highlight code to review.');
    if (!sel) {
      return;
    }
    const prompt =
      `Review the following ${sel.language} code for: bugs, security issues, ` +
      `performance problems, missing edge cases. Be terse. List findings with ` +
      `severity (critical/high/medium/low) and a one-line fix recommendation each.\n\n` +
      `\`\`\`${sel.language}\n${sel.code}\n\`\`\``;
    await runQuery(client, output, prompt, 'Quorum: Review selection', statusBar);
  });
}

/**
 * quorum.compareImplementations — two-step capture. First invocation stores
 * the current selection as A and tells the user to highlight B and re-run.
 * Second invocation captures B and submits the comparison.
 *
 * A "Cancel" button on the toast lets the user abort without leaving stale A
 * state around — otherwise the next invocation would silently treat the new
 * selection as B and compare it against a long-forgotten A.
 */
export function registerCompareImplementations(
  client: QuorumClient,
  output: vscode.OutputChannel,
  statusBar?: QuorumStatusBar
): vscode.Disposable {
  return vscode.commands.registerCommand('quorum.compareImplementations', async () => {
    const sel = getSelectionOrError('Quorum: highlight implementation to compare.');
    if (!sel) {
      return;
    }

    if (!pendingCompareA) {
      pendingCompareA = { code: sel.code, language: sel.language };
      const choice = await vscode.window.showInformationMessage(
        `Quorum: implementation A captured (${sel.code.length} chars). ` +
          `Now highlight implementation B and run "Quorum: Compare two implementations" again.`,
        'Cancel'
      );
      if (choice === 'Cancel') {
        pendingCompareA = undefined;
      }
      return;
    }

    const a = pendingCompareA;
    pendingCompareA = undefined; // consume before async work so a thrown error doesn't strand state
    const b = sel;

    const prompt =
      `Compare implementations A and B for correctness, readability, performance. ` +
      `Which is better and why?\n\n` +
      `A:\n\`\`\`\n${a.code}\n\`\`\`\n\n` +
      `B:\n\`\`\`\n${b.code}\n\`\`\``;
    await runQuery(client, output, prompt, 'Quorum: Compare implementations', statusBar);
  });
}

/**
 * quorum.openSettings — jumps straight to the extension's settings page,
 * filtered by extension id so the user only sees Quorum's keys.
 */
export function registerOpenSettings(): vscode.Disposable {
  return vscode.commands.registerCommand('quorum.openSettings', async () => {
    await vscode.commands.executeCommand(
      'workbench.action.openSettings',
      '@ext:sovereignchain.quorum-vscode'
    );
  });
}

/**
 * quorum.getProLicense — opens the Stripe Checkout for Quorum Pro
 * (£149/mo). Quorum is paid-only: no free tier, no trial queries.
 * Pairs with the warning flow in ensureKeyOrPrompt that fires this
 * command when /v1/consensus is invoked without a license.
 */
export function registerGetProLicense(): vscode.Disposable {
  return vscode.commands.registerCommand('quorum.getProLicense', async () => {
    await vscode.env.openExternal(
      vscode.Uri.parse('https://buy.stripe.com/aFadR9d6E5rf8JGeINdwc0j')
    );
    vscode.window.showInformationMessage(
      'Quorum Pro checkout opened (£149/mo). After payment, your license key arrives by email — paste it into Settings → Quorum → API Key.'
    );
  });
}

/**
 * Helper: when a Quorum command is invoked but quorum.apiKey is empty,
 * pop a friendly toast with three buttons instead of failing with the
 * generic "Invalid API key" error from the server. Returns true if the
 * key is set (caller should proceed), false otherwise (caller should
 * stop).
 */
export async function ensureKeyOrPrompt(): Promise<boolean> {
  const key = vscode.workspace.getConfiguration('quorum').get<string>('apiKey', '');
  if (key && key.trim().length > 0) return true;
  const choice = await vscode.window.showWarningMessage(
    'Quorum requires a paid Pro license (£149/mo). Purchase via Stripe, then paste the key into Settings.',
    'Get Pro License',
    'Open Settings',
    'Dismiss'
  );
  if (choice === 'Get Pro License') {
    await vscode.commands.executeCommand('quorum.getProLicense');
  } else if (choice === 'Open Settings') {
    await vscode.commands.executeCommand('quorum.openSettings');
  }
  return false;
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

interface EditorSelection {
  code: string;
  language: string;
}

/**
 * Pull the active editor's current selection plus the document's language id.
 * Returns undefined and shows the supplied error message when there is no
 * editor or no non-empty selection — callers should just `return` after that.
 */
function getSelectionOrError(errorMessage: string): EditorSelection | undefined {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    void vscode.window.showErrorMessage(errorMessage);
    return undefined;
  }
  const code = editor.document.getText(editor.selection);
  if (code.length === 0) {
    void vscode.window.showErrorMessage(errorMessage);
    return undefined;
  }
  return {
    code,
    language: editor.document.languageId || 'text'
  };
}

/**
 * Shared "run prompt through Quorum with progress + result panel" pipeline
 * so every command has identical UX (progress toast, error toast, webview).
 */
async function runQuery(
  client: QuorumClient,
  output: vscode.OutputChannel,
  prompt: string,
  title: string,
  statusBar?: QuorumStatusBar
): Promise<void> {
  const providerCount = vscode.workspace
    .getConfiguration('quorum')
    .get<string[]>('providers', [])
    .length;
  const progressTitle =
    providerCount > 0
      ? `Querying ${providerCount} providers...`
      : 'Querying Quorum...';

  output.appendLine(`[${new Date().toISOString()}] ${title} — ${prompt.length} chars`);
  statusBar?.setBusy();

  try {
    const result = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: progressTitle,
        cancellable: false
      },
      async () => client.ask(prompt)
    );
    statusBar?.setSuccess(result);
    showConsensusResult(title, prompt, result);
  } catch (err) {
    const msg = (err as Error).message ?? String(err);
    output.appendLine(`  ERROR: ${msg}`);
    statusBar?.setError(err);
    void vscode.window.showErrorMessage(`Quorum failed: ${msg}`);
  }
}

/**
 * Render a consensus result in a webview panel. New panel per query (no
 * reuse) so users can keep multiple results side-by-side. CSP locks the
 * webview down to inline styles only — no scripts, no remote resources.
 */
function showConsensusResult(
  title: string,
  prompt: string,
  result: ConsensusResult
): void {
  const panel = vscode.window.createWebviewPanel(
    'quorum.result',
    title,
    vscode.ViewColumn.Beside,
    {
      enableScripts: false,
      retainContextWhenHidden: true
    }
  );
  panel.webview.html = renderResultHtml(prompt, result);
}

function escapeHtml(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderResultHtml(prompt: string, result: ConsensusResult): string {
  const showCost = vscode.workspace
    .getConfiguration('quorum')
    .get<boolean>('showCostInline', true);

  const confidencePct = Math.round(result.confidence * 100);
  const modelsHtml = result.models
    .map((m) => {
      const errBadge = m.error
        ? `<span class="badge error">error</span>`
        : '';
      const costCell = showCost
        ? `<td class="num">$${m.cost_usd.toFixed(4)}</td>`
        : '';
      return `
        <tr>
          <td><strong>${escapeHtml(m.name)}</strong> ${errBadge}</td>
          <td class="num">${m.weight.toFixed(2)}</td>
          <td class="num">${m.latency_ms} ms</td>
          ${costCell}
        </tr>
        <tr class="response-row">
          <td colspan="${showCost ? 4 : 3}">
            <pre>${escapeHtml(m.error ?? m.response)}</pre>
          </td>
        </tr>`;
    })
    .join('');

  const costHeader = showCost ? '<th>Cost</th>' : '';
  const totalCostLine = showCost
    ? `<p class="muted">Total cost: $${result.totalCostUsd.toFixed(4)}</p>`
    : '';

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline';">
<title>Quorum Result</title>
<style>
  body { font-family: var(--vscode-font-family); padding: 1rem; line-height: 1.5; }
  h1 { font-size: 1.2rem; margin-top: 0; }
  h2 { font-size: 1rem; margin-top: 1.5rem; }
  pre {
    background: var(--vscode-textCodeBlock-background);
    padding: 0.75rem;
    border-radius: 4px;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: var(--vscode-editor-font-family);
    font-size: var(--vscode-editor-font-size);
  }
  .prompt { opacity: 0.85; }
  .answer { border-left: 3px solid var(--vscode-textLink-foreground); padding-left: 0.75rem; }
  .confidence {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    background: var(--vscode-badge-background);
    color: var(--vscode-badge-foreground);
    font-weight: bold;
  }
  table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
  th, td { text-align: left; padding: 0.4rem 0.5rem; border-bottom: 1px solid var(--vscode-panel-border); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .response-row td { padding-top: 0; padding-bottom: 0.75rem; }
  .muted { opacity: 0.7; font-size: 0.9rem; }
  .badge { font-size: 0.75rem; padding: 0.1rem 0.4rem; border-radius: 3px; margin-left: 0.4rem; }
  .badge.error { background: var(--vscode-errorForeground); color: var(--vscode-editor-background); }
</style>
</head>
<body>
  <h1>Consensus answer
    <span class="confidence">${confidencePct}% confidence</span>
  </h1>
  <div class="answer"><pre>${escapeHtml(result.answer)}</pre></div>

  <h2>Prompt</h2>
  <div class="prompt"><pre>${escapeHtml(prompt)}</pre></div>

  <h2>Per-model responses (${result.models.length})</h2>
  ${totalCostLine}
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th class="num">Weight</th>
        <th class="num">Latency</th>
        ${costHeader}
      </tr>
    </thead>
    <tbody>
      ${modelsHtml || '<tr><td colspan="4" class="muted">No per-model data returned.</td></tr>'}
    </tbody>
  </table>
</body>
</html>`;
}
