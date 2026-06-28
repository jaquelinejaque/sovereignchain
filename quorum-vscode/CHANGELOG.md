# Changelog

All notable changes to the **Quorum** VS Code extension are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-06-21

### Changed (breaking)
- **Paid commercial product.** Quorum no longer ships a free tier with 100
  queries/month. Use requires a paid Pro license — **£15/mo** via
  [Stripe Checkout](https://buy.stripe.com/aFadR9d6E5rf8JGeINdwc0j).
  Enterprise pricing on request. The API gate
  (`/v1/license/validate`) rejects unlicensed traffic.
- **License: Apache-2.0 → FSL-1.1** (Functional Source License). Source
  stays readable for internal review and code inspection; commercial use
  in VS Code is paid. Each release converts to Apache-2.0 two years after
  publication.
- Marketplace `pricing` flag set to **Trial** — install is free, the badge
  signals the extension requires a paid license to function.
- Display name updated to **"Quorum — Multi-LLM Consensus (Pro)"**.

### Added
- New command `Quorum: Get Pro License` — opens the Stripe Checkout
  for Quorum Pro (£15/mo) so a new user can purchase without leaving
  VS Code. Wired into `ensureKeyOrPrompt` so an unlicensed user trying to
  run any Quorum command sees a "Get Pro License / Open Settings" toast.
- README rewritten as a clear commercial product page (target buyer:
  regulated dev teams that need a defensible tamper-evident traceability log).

### Removed
- `quorum.getFreeKey` command. Replaced by `quorum.getProLicense`.
- All copy referencing a free tier, free signup, or 100 free queries.

### Notes
- The 0.1.x line is end-of-life. Users still on 0.1.2 see "license required"
  prompts after upgrading.

## [0.1.2] — 2026-06-18

### Added
- New command `Quorum: Get Free API Key (signup)` — opens
  https://quorum-ai.dev/signup so a new user can grab a free
  100-queries/month key (BYOK; no card) without leaving VS Code.
- Soft-warning toast on `Quorum: Ask` when `quorum.apiKey` is empty,
  with one-click buttons to open signup or jump straight to the
  Quorum settings page. Replaces the cryptic "Invalid API key"
  server error that previously confronted new users.

### Changed
- README rewritten as a 3-step quickstart (get free key → paste in
  settings → register provider keys via /v1/customer/keys). Removed
  the old "download .vsix from Releases" flow in favour of the
  Marketplace install + the free signup link, since the marketplace
  listing is now the primary distribution channel.

## [0.1.1] — 2026-06-17

### Fixed
- Status-bar default endpoint now points to `https://api.quorum-ai.dev`
  (was the bare `quorum-ai.dev` root, which serves the marketing landing
  page and 404s on `/v1/*`). `QuorumClient` already used the correct host;
  only the status-bar tooltip showed the stale URL when the user had not
  overridden `quorum.endpoint` in settings.

### Internal
- Test mocks in `quorumClient.test.ts` realigned to `api.quorum-ai.dev` so
  the captured request URLs match what the client actually emits.
- `.vscodeignore` now excludes `out-test/**` from the published `.vsix`.

## [0.1.0] — 2026-06-16

### Added
- Initial extension skeleton (TypeScript + esbuild bundling).
- Six command palette entries:
  - `quorum.ask` — free-form prompt → consensus answer.
  - `quorum.askAboutSelection` — selection + follow-up question.
  - `quorum.explainSelection` — multi-model explanation of highlighted code.
  - `quorum.reviewSelection` — adversarial bug/security review of a selection.
  - `quorum.compareImplementations` — structured side-by-side of two snippets.
  - `quorum.openSettings` — jump directly to the Quorum settings page.
- Right-click context menu integration (`Explain` + `Review`) when the editor
  has a selection.
- Status bar wiring scaffolding (`$(rocket) Quorum`) — phase-1 placeholder that
  surfaces the active endpoint and click-to-open-settings; the live consensus
  status indicator lands in 0.2.0.
- Secrets migration path documented: settings `quorum.apiKey` will be migrated
  out of `settings.json` and into VS Code `SecretStorage` on first activation in
  0.2.0; 0.1.0 reads from the configuration value as a transitional shim and
  emits a deprecation log line when a plaintext key is detected.
- Settings contributions: `endpoint`, `apiKey`, `providers`, `maxLatencyMs`,
  `showCostInline`.
- Typed `QuorumClient` stub with `ConsensusResult` / `ModelResponse` shapes and
  Cloud Run DNS fallback path (real network call lands in 0.2.0).
- F5 debug launch configuration for the extension host.
- 128×128 marketplace icon (`media/icon.png`) and dark gallery banner.
- Apache-2.0 license.

### Notes
- This release intentionally ships stubs — selecting any command shows an
  informational toast. The phase-2 release wires the real consensus call, the
  live status-bar indicator, the SecretStorage-backed API-key path, and the
  webview result panel.
