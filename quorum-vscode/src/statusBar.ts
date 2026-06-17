// Status bar item for Quorum.
//
// Shows idle / in-flight / error state for the most recent consensus query.
// Click target is the quorum.ask command. The item lives on the right side of
// the status bar at priority 100 so it sits near other AI tooling.

import * as vscode from 'vscode';
import type { ConsensusResult } from './quorumClient';

const CONFIG_NS = 'quorum';

type State =
  | { kind: 'idle' }
  | { kind: 'busy' }
  | { kind: 'error'; message: string };

export class QuorumStatusBar {
  private readonly item: vscode.StatusBarItem;
  private state: State = { kind: 'idle' };
  private lastResult: ConsensusResult | undefined;

  constructor() {
    this.item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this.item.command = 'quorum.ask';
    this.render();
    this.item.show();
  }

  /** Mark a query as in flight. Call right before QuorumClient.ask(). */
  setBusy(): void {
    this.state = { kind: 'busy' };
    this.render();
  }

  /** Record a successful query result. Call right after QuorumClient.ask() resolves. */
  setSuccess(result: ConsensusResult): void {
    this.lastResult = result;
    this.state = { kind: 'idle' };
    this.render();
  }

  /** Record a failed query. Call right after QuorumClient.ask() rejects. */
  setError(err: unknown): void {
    const message = err instanceof Error ? err.message : String(err);
    this.state = { kind: 'error', message };
    this.render();
  }

  /** Re-render after a configuration change so the tooltip reflects new endpoint. */
  refresh(): void {
    this.render();
  }

  dispose(): void {
    this.item.dispose();
  }

  private render(): void {
    const endpoint = vscode.workspace
      .getConfiguration(CONFIG_NS)
      .get<string>('endpoint', 'https://api.quorum-ai.dev');

    switch (this.state.kind) {
      case 'busy':
        this.item.text = '$(sync~spin) Quorum…';
        this.item.tooltip = this.buildTooltip(endpoint);
        break;
      case 'error':
        this.item.text = '$(error) Quorum';
        this.item.tooltip = `Quorum: last query failed — ${this.state.message}\n\n${this.buildTooltip(endpoint)}`;
        break;
      case 'idle':
      default:
        this.item.text = '$(symbol-class) Quorum';
        this.item.tooltip = this.buildTooltip(endpoint);
        break;
    }
  }

  private buildTooltip(endpoint: string): string {
    const last = this.lastResult;
    if (!last) {
      return `Click to ask Quorum. Endpoint: ${endpoint}.`;
    }
    return (
      `Click to ask Quorum. Endpoint: ${endpoint}. ` +
      `Last query cost: $${last.totalCostUsd.toFixed(6)} | ` +
      `confidence ${(last.confidence * 100).toFixed(0)}%`
    );
  }
}
