# Changelog

All notable changes to the **Quorum** VS Code extension are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
