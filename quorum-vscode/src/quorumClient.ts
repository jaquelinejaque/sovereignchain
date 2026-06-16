// Typed client for the Quorum consensus API.
//
// Posts to ${endpoint}/v1/consensus with the configured providers, honours an
// AbortSignal-based timeout, maps server errors to friendly messages, and
// retries once against the Cloud Run fallback URL on DNS / connection-refused
// failures (covers the case where the custom domain is mis-resolved).
//
// API key resolution prefers vscode.SecretStorage (so we never leave the key
// in plaintext settings JSON); falls back to the legacy `quorum.apiKey`
// setting only when no secret is stored yet — activation is expected to
// migrate the value into secrets on first run.

import * as vscode from 'vscode';

export interface ModelResponse {
  name: string;
  response: string;
  weight: number;
  latency_ms: number;
  cost_usd: number;
  status?: string;
  error?: string;
}

export interface ConsensusResult {
  answer: string;
  confidence: number; // 0..1
  models: ModelResponse[];
  totalCostUsd: number;
}

export interface AskOpts {
  providers?: string[];
  maxLatencyMs?: number;
}

/** Back-compat alias for the original skeleton name. */
export type AskOptions = AskOpts;

export class QuorumError extends Error {
  public readonly status: number;
  public readonly body: string;

  constructor(message: string, status: number, body: string) {
    super(message);
    this.name = 'QuorumError';
    this.status = status;
    this.body = body;
  }
}

export const FALLBACK_URL = 'https://quorum-api-86770458722.europe-west1.run.app';
const CONFIG_NS = 'quorum';
const SECRET_API_KEY = 'quorum.apiKey';
const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * Test seam: defaults to the runtime global fetch, but tests can swap it via
 * __setFetchForTests. Keeping a module-level reference avoids dragging a
 * fetch parameter through every call site.
 */
type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;
let _fetch: FetchLike = (input, init) => globalThis.fetch(input, init);

/** @internal — used by the smoke test to inject a mock fetch. */
export function __setFetchForTests(fn: FetchLike | null): void {
  _fetch = fn ?? ((input, init) => globalThis.fetch(input, init));
}

export class QuorumClient {
  constructor(private readonly secrets?: vscode.SecretStorage) {}

  /**
   * Run a consensus query across the configured providers.
   *
   * Reads endpoint + apiKey + default providers/timeout from VS Code settings,
   * allowing per-call overrides via `opts`.
   */
  async ask(prompt: string, opts: AskOpts = {}): Promise<ConsensusResult> {
    const cfg = vscode.workspace.getConfiguration(CONFIG_NS);
    const endpoint = cfg.get<string>('endpoint', 'https://api.quorum-ai.dev');
    const apiKey = await this.resolveApiKey(cfg);
    const providers = opts.providers ?? cfg.get<string[]>('providers', []);
    const maxLatencyMs =
      opts.maxLatencyMs ?? cfg.get<number>('maxLatencyMs', DEFAULT_TIMEOUT_MS);

    const path = '/v1/consensus';
    const url = stripTrailingSlash(endpoint) + path;

    const body = JSON.stringify({
      prompt,
      providers,
      max_latency_ms: maxLatencyMs
    });

    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };
    if (apiKey) {
      headers['X-Quorum-API-Key'] = apiKey;
    }

    const res = await this.fetchWithFallback(
      url,
      path,
      { method: 'POST', headers, body },
      maxLatencyMs
    );

    const rawText = await res.text();

    if (!res.ok) {
      throw new QuorumError(
        mapStatusToFriendlyMessage(res.status, rawText),
        res.status,
        rawText
      );
    }

    return parseConsensusPayload(rawText);
  }

  /**
   * Prefer secret storage; fall back to plaintext settings only if secret is
   * absent or empty. Activation does a one-time migration from settings to
   * secrets, so the settings path should only hit on first-run or when the
   * user hasn't migrated yet.
   */
  private async resolveApiKey(cfg: vscode.WorkspaceConfiguration): Promise<string> {
    if (this.secrets) {
      const stored = await this.secrets.get(SECRET_API_KEY);
      if (stored && stored.length > 0) {
        return stored;
      }
    }
    return cfg.get<string>('apiKey', '');
  }

  /**
   * POST with built-in fetch under an AbortSignal timeout. Retries once
   * against FALLBACK_URL on DNS / connection-refused style failures so the
   * extension survives a custom-domain hiccup.
   */
  private async fetchWithFallback(
    url: string,
    path: string,
    init: RequestInit,
    timeoutMs: number
  ): Promise<Response> {
    try {
      return await this.timedFetch(url, init, timeoutMs);
    } catch (err) {
      if (!isDnsOrConnRefused(err)) {
        throw err;
      }
      const fallback = stripTrailingSlash(FALLBACK_URL) + path;
      return await this.timedFetch(fallback, init, timeoutMs);
    }
  }

  private async timedFetch(
    url: string,
    init: RequestInit,
    timeoutMs: number
  ): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await _fetch(url, { ...init, signal: controller.signal });
    } finally {
      clearTimeout(timer);
    }
  }
}

/** Strip trailing slashes so we never produce `//v1/consensus`. */
function stripTrailingSlash(s: string): string {
  return s.replace(/\/+$/, '');
}

/**
 * Heuristic: Node's undici surfaces these as Error.cause.code, but also leaks
 * the code into the message string. We check both to stay robust across Node
 * versions and the VS Code bundled runtime.
 */
function isDnsOrConnRefused(err: unknown): boolean {
  if (!err || typeof err !== 'object') {
    return false;
  }
  const e = err as { message?: string; cause?: { code?: string }; code?: string };
  const code = e.cause?.code ?? e.code ?? '';
  const msg = e.message ?? '';
  return (
    code === 'ENOTFOUND' ||
    code === 'ECONNREFUSED' ||
    code === 'EAI_AGAIN' ||
    /ENOTFOUND|ECONNREFUSED|EAI_AGAIN|getaddrinfo/i.test(msg)
  );
}

function mapStatusToFriendlyMessage(status: number, body: string): string {
  if (status === 401 || status === 403) {
    return 'Invalid API key. Set quorum.apiKey in settings.';
  }
  if (status === 429) {
    return 'Rate limited. Try again in a minute.';
  }
  if (status >= 500) {
    return 'Quorum server error.';
  }
  return `Quorum request failed (HTTP ${status}): ${truncate(body, 200)}`;
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n) + '...';
}

/**
 * Tolerant parser: accepts the documented shape but also normalises a few
 * common variants (snake_case fields, missing totals) so the panel never has
 * to second-guess the wire format.
 */
function parseConsensusPayload(raw: string): ConsensusResult {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    throw new QuorumError(
      `Quorum returned non-JSON response: ${(err as Error).message}`,
      200,
      raw
    );
  }

  const obj = (parsed ?? {}) as Record<string, unknown>;
  const modelsRaw = (obj.models as unknown[]) ?? [];
  const models: ModelResponse[] = modelsRaw.map((m) => {
    const r = (m ?? {}) as Record<string, unknown>;
    return {
      name: String(r.name ?? r.provider ?? 'unknown'),
      response: String(r.response ?? r.text ?? r.answer ?? ''),
      weight: numberOr(r.weight, 0),
      latency_ms: numberOr(r.latency_ms ?? r.latencyMs, 0),
      cost_usd: numberOr(r.cost_usd ?? r.costUsd, 0),
      status: r.status != null ? String(r.status) : undefined,
      error: r.error != null ? String(r.error) : undefined
    };
  });

  const declaredTotal = numberOr(obj.totalCostUsd ?? obj.total_cost_usd, NaN);
  const totalCostUsd = Number.isFinite(declaredTotal)
    ? declaredTotal
    : models.reduce((sum, m) => sum + m.cost_usd, 0);

  return {
    answer: String(obj.answer ?? ''),
    confidence: numberOr(obj.confidence, 0),
    models,
    totalCostUsd
  };
}

function numberOr(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}
