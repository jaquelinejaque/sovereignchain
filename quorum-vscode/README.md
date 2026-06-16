# Quorum — Multi-LLM Consensus for VS Code

Quorum runs a single prompt across 8+ frontier LLMs (Claude, GPT, Gemini, Llama, DeepSeek, Mistral, Qwen, Phi) directly inside VS Code, scores the answers for semantic agreement, and surfaces exactly where the models disagree — because disagreement between strong models is the most valuable signal you can get before shipping code, taking a security call, or making a regulatory commitment. Backed by the hosted [Quorum](https://quorum-ai.dev) consensus engine; BYOK supported so your provider keys never leave your machine.

## Install

1. Download `quorum-vscode-0.1.0.vsix` from the [Releases page](https://github.com/jaquelinejaque/sovereignchain/releases).
2. In VS Code: open the **Extensions** sidebar, click the `…` menu, choose **Install from VSIX…**, and pick the `.vsix` file.
3. Open **Settings** (`Cmd+,` / `Ctrl+,`), search `quorum`, and set at minimum `quorum.endpoint` and either `quorum.apiKey` or your BYOK provider keys.

Or build from source:

```bash
git clone https://github.com/jaquelinejaque/sovereignchain.git
cd sovereignchain/quorum-vscode
npm install
npm run build
npm run package
code --install-extension quorum-vscode-0.1.0.vsix
```

## Commands

| Command | What it does |
|---|---|
| `Quorum: Ask` | Free-form prompt, returns a consensus answer in a side panel |
| `Quorum: Ask about selection` | Combines your selection with a follow-up question |
| `Quorum: Explain selected code` | Multi-model explanation of the highlighted code |
| `Quorum: Review selected code (bugs/security)` | Adversarial code review across all configured models |
| `Quorum: Compare two implementations` | Pick two snippets, get a structured side-by-side comparison |
| `Quorum: Open settings` | Jumps straight to the Quorum settings page |

The **Explain** and **Review** commands also appear in the editor right-click menu whenever you have a selection.

## Settings

| Setting | Default | Description |
|---|---|---|
| `quorum.endpoint` | `https://quorum-ai.dev` | Quorum server endpoint |
| `quorum.apiKey` | `""` | `X-Quorum-API-Key` header (leave empty to use BYOK mode) |
| `quorum.providers` | `["gemini-flash","llama-3.3-70b","deepseek-v3","claude-sonnet-4-6"]` | Models to query in parallel |
| `quorum.maxLatencyMs` | `30000` | Per-query timeout in milliseconds |
| `quorum.showCostInline` | `true` | Show per-query cost inline in the result panel |

## Sample workflow

1. Highlight a function in any editor (TypeScript, Python, Rust, anything).
2. Right-click → **Quorum: Review selected code (bugs/security)**.
3. Quorum fans the snippet out to all configured models, scores their agreement, and renders a panel showing the consensus verdict, the dissenting opinions, and a per-model latency/cost breakdown.

The same pattern works for explanations (`Explain selected code`), free-form questions about a snippet (`Ask about selection`), and side-by-side comparisons (`Compare two implementations`).

## Screenshots

> Webview screenshots will be added in v0.2.0 once the consensus result panel ships. The phase-1 release uses VS Code notification toasts to confirm command wiring.

## Pricing

Free during beta. **Pro £49/mo** when GA, which covers hosted consensus, all upstream provider calls, and priority routing. **BYOK is fully supported** — point `quorum.endpoint` at your own deployment or leave `quorum.apiKey` empty and supply provider keys directly; in BYOK mode you only pay your upstream providers.

## Known limitations

Being honest about what 0.1.0 is and isn't:

- **Depends on the hosted Quorum engine.** You need either a `quorum-ai.dev` API key or your own self-hosted Quorum instance for the commands to return real answers. Without one, the commands fire but receive no consensus.
- **No streaming yet.** Results arrive as a single payload once the slowest configured model finishes (bounded by `quorum.maxLatencyMs`). Streaming lands in v0.2.0.
- **No inline diagnostics yet.** Findings render in a side panel, not as squiggles in the editor. Inline diagnostics + Code Lens land in v0.3.0.
- **Phase-1 commands are stubs.** This release ships the full command/menu/settings surface and toasts a confirmation; the real consensus calls and webview land in v0.2.0 (tracked in `CHANGELOG.md`).

## Links

- Website: <https://quorum-ai.dev>
- Repository: <https://github.com/jaquelinejaque/sovereignchain>
- Issues / feedback: <https://github.com/jaquelinejaque/sovereignchain/issues>
- License: Apache-2.0 (see `LICENSE`); HSP modules have additional terms (see `LICENSE-HSP` in the parent repo).
