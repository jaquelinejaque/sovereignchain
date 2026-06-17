// Smoke test for QuorumClient using Node's built-in test runner.
//
// We can't import 'vscode' under node (it's only provided by the VS Code
// host), so we pre-populate require.cache with a stub before the client
// module is loaded. Once that's in place, all `vscode.workspace.getConfiguration`
// calls inside the client resolve against our fake settings.
//
// To run after `tsc`:
//   npx tsc -p tsconfig.json --outDir out-test --module commonjs
//   node --test out-test/__test__/quorumClient.test.js
//
// This file is intentionally dependency-free (node:test, node:assert only) so
// the project doesn't gain dev deps just to verify the request shape.

import { test } from 'node:test';
import * as assert from 'node:assert/strict';
import Module from 'node:module';

// ---------------------------------------------------------------------------
// vscode stub — populated into require.cache so the import in quorumClient
// resolves to this fake object rather than failing with MODULE_NOT_FOUND.
// ---------------------------------------------------------------------------

type ConfigMap = Record<string, unknown>;
let fakeConfig: ConfigMap = {
  endpoint: 'https://api.quorum-ai.dev',
  apiKey: 'test-key-abc',
  providers: ['gemini-flash', 'claude-sonnet-4-6'],
  maxLatencyMs: 30000
};

const vscodeStub = {
  workspace: {
    getConfiguration: (_ns: string) => ({
      get: <T,>(key: string, fallback?: T): T => {
        const v = fakeConfig[key];
        return (v === undefined ? fallback : v) as T;
      }
    })
  }
};

// Install the stub before quorumClient.ts is required. Using the same
// internal API VS Code itself relies on for module stubbing in tests.
const stubId = require.resolve('module');
void stubId; // silence unused warning under strict tsconfig
const cache = (Module as unknown as { _cache: Record<string, unknown> })._cache;
const vscodeFakePath = require.resolve.paths('')?.[0] + '/__vscode_stub__';
cache[vscodeFakePath] = {
  id: vscodeFakePath,
  filename: vscodeFakePath,
  loaded: true,
  exports: vscodeStub,
  children: [],
  paths: []
};
const originalResolve = (Module as unknown as {
  _resolveFilename: (request: string, ...rest: unknown[]) => string;
})._resolveFilename;
(Module as unknown as {
  _resolveFilename: (request: string, ...rest: unknown[]) => string;
})._resolveFilename = function (request: string, ...rest: unknown[]): string {
  if (request === 'vscode') {
    return vscodeFakePath;
  }
  return originalResolve.call(this, request, ...rest);
};

// Now safe to import the client (it will pick up the stub).
// eslint-disable-next-line @typescript-eslint/no-var-requires
import {
  QuorumClient,
  QuorumError,
  __setFetchForTests,
  FALLBACK_URL
} from '../quorumClient';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface CapturedRequest {
  url: string;
  init: RequestInit;
}

function makeFetchStub(
  responder: (req: CapturedRequest) => Response | Promise<Response>
): { fetch: (input: string, init?: RequestInit) => Promise<Response>; calls: CapturedRequest[] } {
  const calls: CapturedRequest[] = [];
  const fn = async (input: string, init?: RequestInit): Promise<Response> => {
    const captured = { url: input, init: init ?? {} };
    calls.push(captured);
    return await responder(captured);
  };
  return { fetch: fn, calls };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' }
  });
}

function textResponse(body: string, status: number): Response {
  return new Response(body, { status });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test('ask() posts the documented request shape', async () => {
  fakeConfig = {
    endpoint: 'https://api.quorum-ai.dev/',
    apiKey: 'test-key-abc',
    providers: ['gemini-flash'],
    maxLatencyMs: 12345
  };

  const stub = makeFetchStub(() =>
    jsonResponse({
      answer: 'hello',
      confidence: 0.9,
      models: [],
      totalCostUsd: 0
    })
  );
  __setFetchForTests(stub.fetch);

  try {
    const client = new QuorumClient();
    const result = await client.ask('what is 2+2?', { providers: ['claude-sonnet-4-6'] });

    assert.equal(stub.calls.length, 1, 'fetch called exactly once');
    const call = stub.calls[0];

    // Endpoint joined with /v1/consensus and trailing slash collapsed.
    assert.equal(call.url, 'https://api.quorum-ai.dev/v1/consensus');

    // Method + headers.
    assert.equal(call.init.method, 'POST');
    const headers = call.init.headers as Record<string, string>;
    assert.equal(headers['Content-Type'], 'application/json');
    assert.equal(headers['X-Quorum-API-Key'], 'test-key-abc');

    // AbortSignal attached.
    assert.ok(call.init.signal, 'AbortSignal attached');

    // Body uses snake_case max_latency_ms and per-call providers override.
    const body = JSON.parse(call.init.body as string);
    assert.equal(body.prompt, 'what is 2+2?');
    assert.deepEqual(body.providers, ['claude-sonnet-4-6']);
    assert.equal(body.max_latency_ms, 12345);

    // Parsed response surfaces.
    assert.equal(result.answer, 'hello');
    assert.equal(result.confidence, 0.9);
  } finally {
    __setFetchForTests(null);
  }
});

test('ask() omits X-Quorum-API-Key when apiKey is empty', async () => {
  fakeConfig = {
    endpoint: 'https://api.quorum-ai.dev',
    apiKey: '',
    providers: [],
    maxLatencyMs: 1000
  };

  const stub = makeFetchStub(() =>
    jsonResponse({ answer: 'ok', confidence: 0, models: [], totalCostUsd: 0 })
  );
  __setFetchForTests(stub.fetch);

  try {
    await new QuorumClient().ask('hi');
    const headers = stub.calls[0].init.headers as Record<string, string>;
    assert.equal(headers['X-Quorum-API-Key'], undefined);
  } finally {
    __setFetchForTests(null);
  }
});

test('ask() maps 401 to friendly QuorumError', async () => {
  fakeConfig = { endpoint: 'https://api.quorum-ai.dev', apiKey: 'bad', providers: [], maxLatencyMs: 1000 };
  __setFetchForTests(async () => textResponse('unauthorized', 401));

  try {
    await assert.rejects(
      () => new QuorumClient().ask('hi'),
      (err: unknown) => {
        assert.ok(err instanceof QuorumError, 'is QuorumError');
        assert.equal((err as QuorumError).status, 401);
        assert.match((err as QuorumError).message, /Invalid API key/i);
        return true;
      }
    );
  } finally {
    __setFetchForTests(null);
  }
});

test('ask() maps 429 to rate-limit message', async () => {
  fakeConfig = { endpoint: 'https://api.quorum-ai.dev', apiKey: 'k', providers: [], maxLatencyMs: 1000 };
  __setFetchForTests(async () => textResponse('slow down', 429));

  try {
    await assert.rejects(
      () => new QuorumClient().ask('hi'),
      (err: unknown) => {
        assert.equal((err as QuorumError).status, 429);
        assert.match((err as QuorumError).message, /Rate limited/i);
        return true;
      }
    );
  } finally {
    __setFetchForTests(null);
  }
});

test('ask() maps 5xx to server error message', async () => {
  fakeConfig = { endpoint: 'https://api.quorum-ai.dev', apiKey: 'k', providers: [], maxLatencyMs: 1000 };
  __setFetchForTests(async () => textResponse('boom', 503));

  try {
    await assert.rejects(
      () => new QuorumClient().ask('hi'),
      (err: unknown) => {
        assert.equal((err as QuorumError).status, 503);
        assert.match((err as QuorumError).message, /server error/i);
        return true;
      }
    );
  } finally {
    __setFetchForTests(null);
  }
});

test('ask() retries against FALLBACK_URL on ENOTFOUND', async () => {
  fakeConfig = { endpoint: 'https://api.quorum-ai.dev', apiKey: 'k', providers: [], maxLatencyMs: 1000 };

  const stub = makeFetchStub((req) => {
    if (req.url.startsWith('https://api.quorum-ai.dev')) {
      const err = new Error('getaddrinfo ENOTFOUND api.quorum-ai.dev') as Error & { code?: string };
      err.code = 'ENOTFOUND';
      throw err;
    }
    return jsonResponse({ answer: 'fallback-hit', confidence: 1, models: [], totalCostUsd: 0 });
  });
  __setFetchForTests(stub.fetch);

  try {
    const result = await new QuorumClient().ask('hi');
    assert.equal(stub.calls.length, 2, 'primary + fallback');
    assert.equal(stub.calls[1].url, FALLBACK_URL + '/v1/consensus');
    assert.equal(result.answer, 'fallback-hit');
  } finally {
    __setFetchForTests(null);
  }
});
