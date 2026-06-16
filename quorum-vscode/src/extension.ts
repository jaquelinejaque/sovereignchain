// Quorum — Multi-LLM Consensus, VS Code extension entry point.
//
// activate() owns the singletons (QuorumClient, status bar, output channel)
// and registers every contributed command via the helpers in ./commands.ts.
// All disposables go onto context.subscriptions so VS Code tears them down
// cleanly on extension reload / window shutdown.

import * as vscode from 'vscode';
import {
  registerAsk,
  registerAskAboutSelection,
  registerCompareImplementations,
  registerExplainSelection,
  registerOpenSettings,
  registerReviewSelection
} from './commands';
import { QuorumClient } from './quorumClient';
import { QuorumStatusBar } from './statusBar';

const CONFIG_NS = 'quorum';
const SECRET_API_KEY = 'quorum.apiKey';

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  await migrateApiKeyToSecretStorage(context);

  // Shared output channel — every command logs its query timestamp here so
  // there's a paper trail when debugging "did Quorum see my request?".
  const output = vscode.window.createOutputChannel('Quorum');
  context.subscriptions.push(output);

  const client = new QuorumClient(context.secrets);
  const statusBar = new QuorumStatusBar();
  context.subscriptions.push(statusBar);

  // Reflect endpoint changes in the status bar tooltip immediately.
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration(`${CONFIG_NS}.endpoint`)) {
        statusBar.refresh();
      }
    })
  );

  context.subscriptions.push(
    registerAsk(client, output, statusBar),
    registerAskAboutSelection(client, output, statusBar),
    registerExplainSelection(client, output, statusBar),
    registerReviewSelection(client, output, statusBar),
    registerCompareImplementations(client, output, statusBar),
    registerOpenSettings()
  );

  output.appendLine(
    `[${new Date().toISOString()}] Quorum activated ` +
      `(v${context.extension.packageJSON.version ?? 'unknown'}).`
  );
}

export function deactivate(): void {
  // Nothing to dispose — all subscriptions are managed by context.subscriptions.
}

/**
 * One-time migration: if the user has quorum.apiKey set in settings.json
 * (insecure plaintext), move it to VS Code's secret storage and clear the
 * settings value. We only touch the global target — workspace-level
 * overrides are left alone so a shared workspace setting (rare but possible)
 * isn't silently nuked.
 */
async function migrateApiKeyToSecretStorage(context: vscode.ExtensionContext): Promise<void> {
  const cfg = vscode.workspace.getConfiguration(CONFIG_NS);
  const inspect = cfg.inspect<string>('apiKey');
  const plaintext = (inspect?.globalValue ?? inspect?.workspaceValue ?? '') as string;
  if (!plaintext || plaintext.length === 0) {
    return;
  }

  try {
    await context.secrets.store(SECRET_API_KEY, plaintext);
    // Clear from whichever scope it was set in (global preferred).
    if (inspect?.globalValue !== undefined) {
      await cfg.update('apiKey', undefined, vscode.ConfigurationTarget.Global);
    }
    if (inspect?.workspaceValue !== undefined) {
      await cfg.update('apiKey', undefined, vscode.ConfigurationTarget.Workspace);
    }
    void vscode.window.showWarningMessage(
      'Quorum: Moved your API key from settings to VS Code secret storage.'
    );
  } catch (err) {
    // Don't block activation if secret storage is unavailable (e.g. headless).
    console.error('Quorum: failed to migrate apiKey to secret storage', err);
  }
}
