Show HN: Quorum – multi-LLM consensus engine with self-evolution loops

I've been running production decisions through 3-4 LLMs in parallel and scoring agreement before trusting any single answer. Quorum is the cleanup of that workflow into a CLI + FastAPI server.

What it does: fans a prompt out to multiple backends (Claude, Gemini, GPT, local Llama via Ollama, optionally Grok), then returns a consensus score plus the divergence points. Divergence is the actually useful signal — it tells you where the models disagree, which is where you should look harder. Ships with 13 self-evolution loops (prompt rewrites, calibration checks, adversarial re-asks) that fire when agreement is low. Apache 2.0. Stripe billing is optional and off by default. The HSP transport layer is under PCT/US26/11908; the reference implementation is fully open.

Install and run against your own keys:

```bash
pip install quorum-engine
export ANTHROPIC_API_KEY=... GOOGLE_API_KEY=... OPENAI_API_KEY=...
quorum ask --consensus "Is this SQL safe? $(cat query.py)"
```

Python:

```python
from quorum import Quorum
q = Quorum(backends=["claude-opus-4-8", "gemini-2.5-pro", "gpt-5"])
result = q.consensus("Review this diff for race conditions:\n" + diff)
print(result.score, result.divergence_points)  # 0.0-1.0, list[str]
```

Server mode (OpenAI-compatible endpoint):

```bash
quorum serve --port 8080 --evolve-loops 13
curl -s localhost:8080/v1/consensus \
  -H "Content-Type: application/json" \
  -d '{"prompt":"audit this auth flow","min_agreement":0.7}'
```

Honest weakness: the consensus scorer is currently lexical (Jaccard + Sørensen-Dice over tokenized responses). When three models say the same thing in different words, the score underestimates agreement. I have an embeddings-based scorer in a branch but haven't shipped it because the eval set is still small (n=180) and I don't want to overclaim. PRs on `scorer/semantic` welcome.

Solo founder, UK (Sovereign Chain Ltd). Try the CLI, file issues for the dumb stuff, and please push back on the API shape before 0.2.0 freezes it: https://github.com/jaquelinejaque/sovereignchain
