# Contributing to SovereignChain

Thanks for taking the time to contribute. This document covers how to set up
a development environment, run the test suite, and what we expect from a
pull request. Please read it before opening your first PR.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](./CODE_OF_CONDUCT.md).
By participating you agree to uphold it. Report unacceptable behaviour to the
maintainers via the contact in `SECURITY.md`.

---

## Development environment

We use [uv](https://github.com/astral-sh/uv) for fast, reproducible installs.
The `[all]` extra pulls in every optional dependency plus the dev toolchain
(`ruff`, `mypy`, `pytest`, `pytest-cov`).

```bash
# clone
git clone https://github.com/sovereignchain/sovereignchain.git
cd sovereignchain

# create and activate a virtual environment (uv-managed)
uv venv
source .venv/bin/activate

# editable install with every extra (runtime + dev)
uv pip install -e ".[all]"
```

Verify the install:

```bash
python -c "import sovereignchain; print(sovereignchain.__version__)"
ruff --version
mypy --version
pytest --version
```

### Python versions supported

CI runs against **Python 3.10, 3.11, and 3.12** on `ubuntu-latest`. Your local
environment must use one of those three. We do not accept patches that
require 3.13+ features until CI matrix is updated.

---

## Running the test suite

```bash
# full suite with coverage
pytest --cov=src --cov-branch --cov-report=term-missing

# a single file
pytest tests/test_quorum.py -v

# a single test
pytest tests/test_quorum.py::test_threshold_signature -v

# fail fast on first failure
pytest -x
```

Coverage threshold for new code: **90 %**. PRs that drop overall coverage will
be flagged automatically.

### Lint and typecheck locally before pushing

```bash
ruff check .
ruff format --check .
mypy --strict src/
```

The CI workflow (`.github/workflows/ci.yml`) runs exactly these commands. If
it passes locally it will pass on CI.

---

## Pull request checklist

Before requesting review, confirm every box:

- [ ] `ruff check .` passes with no errors or warnings
- [ ] `ruff format --check .` reports no diffs
- [ ] `mypy --strict src/` passes with no errors
- [ ] `pytest` passes on Python 3.10, 3.11, and 3.12 (CI matrix)
- [ ] New behaviour is covered by tests (unit and, where relevant, integration)
- [ ] Coverage for changed files is at least 90 %
- [ ] Public API changes are reflected in docstrings and `README.md` / docs
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
- [ ] The PR description references the issue it closes (`Closes #123`)
- [ ] No secrets, credentials, or personal data in the diff
- [ ] If a security-sensitive area was touched, `SECURITY.md` was reviewed

PRs that fail any of the above will be sent back for revision before review.

---

## HSP-gated modules

Some subsystems are licensed under the **Hybrid Sovereign Protection (HSP)**
licence (see `LICENSE-HSP`) rather than the standard project licence. These
modules implement subject matter covered by **PCT/US26/11908** and require
explicit maintainer sign-off before any change is merged — even cosmetic
ones such as renaming a private symbol or reformatting a comment block.

Currently HSP-gated:

- `src/sovereignchain/quorum/` — threshold signature and consensus core
- `src/sovereignchain/launch/` — bootstrap and launch-attestation pipeline
- `src/sovereignchain/hsp/` — HSP webhook and attestation transport

**Process for HSP-gated changes:**

1. Open a draft PR with the change and prefix the title with `[HSP]`.
2. Tag `@sovereignchain/hsp-maintainers` in the PR description.
3. Wait for a maintainer to assign themselves as reviewer. **Do not merge**
   even if CI is green and other reviewers have approved.
4. The assigned maintainer will run an additional sign-off review covering
   IP-impact and patent-scope. Expect 3–5 business days.
5. Sign-off is recorded as a `Signed-off-by:` trailer in the merge commit by
   the maintainer. No sign-off, no merge.

If you are unsure whether a file is HSP-gated, run:

```bash
grep -l "SPDX-License-Identifier: LicenseRef-HSP-1.0" src/**/*.py
```

Anything that matches is gated. When in doubt, ask in the PR before writing
code — refactors that get rejected at sign-off are painful for everyone.

---

## Reporting bugs and requesting features

- **Security vulnerabilities**: do **not** open a public issue. Follow the
  responsible disclosure process in [`SECURITY.md`](./SECURITY.md).
- **Bugs**: open a GitHub issue with the `bug` template, including the
  minimum reproducer, Python version, and OS.
- **Features**: open a GitHub issue with the `feature` template and wait for
  maintainer triage before starting work. We will not merge unsolicited
  large features without a prior design discussion.

Thanks for contributing.
