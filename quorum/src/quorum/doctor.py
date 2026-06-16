"""Quorum doctor — pre-deploy cross-config conflict detector.

Catches the class of bug where pyproject.toml, .dockerignore, Dockerfile, and
runtime code disagree about reality (readme paths excluded by .dockerignore,
COPY globs that look optional but aren't, Path.home() writes under a non-root
USER, SQLite paths hard-coded outside QUORUM_DATA_DIR, provider env vars
silently missing).

Each check returns a list of Issue(severity, check, message, hint?).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from rich.console import Console
from rich.table import Table
from rich import box


Severity = str  # "error" | "warning" | "info"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    severity: Severity            # error | warning | info
    check: str                    # short check id
    message: str                  # human-readable description
    hint: str | None = None       # optional remediation suggestion


@dataclass
class CheckResult:
    name: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)


# ---------------------------------------------------------------------------
# Provider env-var registry
# ---------------------------------------------------------------------------

# Maps provider display name -> env var that gates it.
PROVIDER_ENV_VARS: dict[str, str] = {
    "Claude":   "ANTHROPIC_API_KEY",
    "OpenAI":   "OPENAI_API_KEY",
    "Gemini":   "GEMINI_API_KEY",
    "Grok":     "XAI_API_KEY",
    "Replicate": "REPLICATE_API_TOKEN",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root(start: Path | None = None) -> Path:
    """Walk upward from start (or CWD) until we find pyproject.toml."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here  # fall back, downstream checks will flag missing files


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _parse_dockerignore(path: Path) -> list[str]:
    """Return list of non-comment, non-empty patterns from .dockerignore."""
    text = _read_text(path)
    if text is None:
        return []
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _dockerignore_excludes(patterns: list[str], target: str) -> bool:
    """Return True if `target` is excluded (not re-included with !) by patterns."""
    excluded = False
    for pat in patterns:
        negated = pat.startswith("!")
        body = pat[1:] if negated else pat
        if body == target or body == f"./{target}" or body == "*" or body == "**":
            excluded = not negated
        elif body.endswith("/*") and target == body[:-2]:
            excluded = not negated
    return excluded


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_pyproject_readme(root: Path) -> CheckResult:
    """Verify readme declared in pyproject.toml actually exists on disk."""
    result = CheckResult(name="pyproject ↔ filesystem (readme)")
    pyproject = root / "pyproject.toml"
    text = _read_text(pyproject)
    if text is None:
        result.issues.append(Issue("error", "pyproject.missing",
                                   f"pyproject.toml not found at {pyproject}"))
        return result
    try:
        data = tomllib.loads(text)
    except Exception as exc:  # pragma: no cover
        result.issues.append(Issue("error", "pyproject.parse",
                                   f"pyproject.toml is not valid TOML: {exc}"))
        return result

    readme = data.get("project", {}).get("readme")
    if not readme:
        result.issues.append(Issue("info", "pyproject.readme.absent",
                                   "No `readme` declared in [project]; skipping."))
        return result

    # readme may be a string or a table {file = "..."} or {text = "..."}
    readme_file: str | None = None
    if isinstance(readme, str):
        readme_file = readme
    elif isinstance(readme, dict) and "file" in readme:
        readme_file = readme["file"]

    if readme_file is None:
        result.issues.append(Issue("info", "pyproject.readme.inline",
                                   "readme is inline text — no file to verify."))
        return result

    if not (root / readme_file).exists():
        result.issues.append(Issue(
            "error", "pyproject.readme.missing",
            f"pyproject.toml declares readme = '{readme_file}' but file is missing.",
            hint=f"Create {readme_file} or update [project].readme.",
        ))
    return result


def check_dockerignore_readme(root: Path) -> CheckResult:
    """Ensure the readme isn't silently excluded from the build context."""
    result = CheckResult(name="pyproject ↔ .dockerignore (readme)")
    dockerignore = root / ".dockerignore"
    pyproject = root / "pyproject.toml"
    if not dockerignore.exists():
        result.issues.append(Issue("info", "dockerignore.absent",
                                   ".dockerignore not present; nothing to check."))
        return result

    text = _read_text(pyproject)
    if text is None:
        return result
    try:
        data = tomllib.loads(text)
    except Exception:
        return result
    readme = data.get("project", {}).get("readme")
    readme_file = readme if isinstance(readme, str) else (
        readme.get("file") if isinstance(readme, dict) else None
    )
    if not readme_file:
        return result

    patterns = _parse_dockerignore(dockerignore)
    if _dockerignore_excludes(patterns, readme_file):
        result.issues.append(Issue(
            "error", "dockerignore.readme.excluded",
            f".dockerignore excludes '{readme_file}', but pyproject.toml needs it at build time.",
            hint=f"Add `!{readme_file}` to .dockerignore or remove the excluding pattern.",
        ))
    return result


_COPY_RE = re.compile(r"^\s*COPY\s+(?:--[^\s]+\s+)*(?P<src>\S+)\s+(?P<dst>\S+)\s*$")
_AMBIGUOUS_GLOB = re.compile(r"\[[^\]]+\]")  # e.g. README[.]md or pyproject[.]toml


def check_dockerfile_copy_globs(root: Path) -> CheckResult:
    """Catch `COPY foo[.]bar dst/` — looks optional but BuildKit treats it literally."""
    result = CheckResult(name="Dockerfile ↔ COPY glob safety")
    dockerfile = root / "Dockerfile"
    text = _read_text(dockerfile)
    if text is None:
        result.issues.append(Issue("info", "dockerfile.absent",
                                   "Dockerfile not present; skipping."))
        return result

    for idx, raw in enumerate(text.splitlines(), start=1):
        m = _COPY_RE.match(raw)
        if not m:
            continue
        src = m.group("src")
        if _AMBIGUOUS_GLOB.search(src):
            result.issues.append(Issue(
                "error", "dockerfile.copy.ambiguous_glob",
                f"Line {idx}: `COPY {src} ...` uses `[…]` glob — BuildKit does NOT treat this as optional.",
                hint="Use two separate COPY lines, or copy the file unconditionally.",
            ))
    return result


_USER_RE = re.compile(r"^\s*USER\s+(?P<user>\S+)\s*$", re.MULTILINE)
_PATH_HOME_RE = re.compile(r"Path\.home\(\)")


def check_dockerfile_user_vs_home_writes(root: Path) -> CheckResult:
    """If the container runs as non-root, Path.home() writes will explode."""
    result = CheckResult(name="Dockerfile USER ↔ Path.home() writes")
    dockerfile = root / "Dockerfile"
    text = _read_text(dockerfile)
    if text is None:
        return result

    users = _USER_RE.findall(text)
    if not users:
        return result
    final_user = users[-1].strip()
    if final_user in ("root", "0"):
        return result

    src_dir = root / "src"
    if not src_dir.exists():
        return result

    hits: list[tuple[Path, int]] = []
    for py in src_dir.rglob("*.py"):
        body = _read_text(py)
        if body is None:
            continue
        for lineno, line in enumerate(body.splitlines(), start=1):
            if _PATH_HOME_RE.search(line):
                hits.append((py, lineno))

    for py, lineno in hits:
        rel = py.relative_to(root)
        result.issues.append(Issue(
            "warning", "runtime.home_write_under_nonroot",
            f"{rel}:{lineno} uses Path.home() — fails when container USER={final_user}.",
            hint="Read QUORUM_DATA_DIR (fallback to Path.home()) so the container can override.",
        ))
    return result


_SQLITE_RE = re.compile(r"""['"]([^'"]*\.db|[^'"]*\.sqlite3?)['"]""")
_QUORUM_DATA_DIR_RE = re.compile(r"""QUORUM_DATA_DIR""")


def check_sqlite_paths(root: Path) -> CheckResult:
    """Flag SQLite file paths that don't go through QUORUM_DATA_DIR."""
    result = CheckResult(name="Storage paths use QUORUM_DATA_DIR")
    src_dir = root / "src"
    if not src_dir.exists():
        return result

    for py in src_dir.rglob("*.py"):
        body = _read_text(py)
        if body is None:
            continue
        if not _SQLITE_RE.search(body):
            continue
        # File mentions a sqlite path — does it also reference the override env var?
        if _QUORUM_DATA_DIR_RE.search(body):
            continue
        # Find each match's line for a precise pointer.
        for lineno, line in enumerate(body.splitlines(), start=1):
            m = _SQLITE_RE.search(line)
            if not m:
                continue
            rel = py.relative_to(root)
            result.issues.append(Issue(
                "warning", "storage.sqlite.hardcoded",
                f"{rel}:{lineno} references '{m.group(1)}' without QUORUM_DATA_DIR.",
                hint='Use os.getenv("QUORUM_DATA_DIR", default_dir) before joining the filename.',
            ))
    return result


def check_provider_env_vars(_root: Path) -> CheckResult:
    """Surface which provider env vars are set vs missing in the current shell."""
    result = CheckResult(name="Provider API keys (current env)")
    for provider, var in PROVIDER_ENV_VARS.items():
        if os.environ.get(var):
            result.issues.append(Issue(
                "info", f"provider.{provider.lower()}.present",
                f"{var} is set — provider {provider} will be loaded.",
            ))
        else:
            result.issues.append(Issue(
                "warning", f"provider.{provider.lower()}.missing",
                f"{var} missing — provider {provider} won't be loaded.",
                hint=f"export {var}=... in your shell or deploy secret store.",
            ))
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

ALL_CHECKS: list[Callable[[Path], CheckResult]] = [
    check_pyproject_readme,
    check_dockerignore_readme,
    check_dockerfile_copy_globs,
    check_dockerfile_user_vs_home_writes,
    check_sqlite_paths,
    check_provider_env_vars,
]


def run_all_checks(root: Path | None = None) -> list[CheckResult]:
    base = _project_root(root)
    return [check(base) for check in ALL_CHECKS]


# ---------------------------------------------------------------------------
# Rich rendering
# ---------------------------------------------------------------------------

_SEVERITY_GLYPH = {
    "error":   ("[bold red]✗[/]",     "red"),
    "warning": ("[bold yellow]⚠[/]",  "yellow"),
    "info":    ("[bold cyan]i[/]",    "cyan"),
}


def render_results(results: list[CheckResult], console: Console | None = None) -> int:
    """Print results with rich. Returns process exit code (0 ok, 1 errors)."""
    console = console or Console()
    errors = warnings = passes = 0

    for result in results:
        if not result.issues:
            console.print(f"[bold green]✓[/] {result.name}")
            passes += 1
            continue

        # Show header with the worst severity glyph.
        worst = "info"
        for issue in result.issues:
            if issue.severity == "error":
                worst = "error"
                break
            if issue.severity == "warning" and worst != "error":
                worst = "warning"

        glyph, color = _SEVERITY_GLYPH[worst]
        console.print(f"{glyph} [bold {color}]{result.name}[/]")

        tbl = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        tbl.add_column(width=2)
        tbl.add_column(overflow="fold")

        for issue in result.issues:
            g, _ = _SEVERITY_GLYPH[issue.severity]
            line = issue.message
            if issue.hint:
                line += f"\n  [dim]hint:[/] {issue.hint}"
            tbl.add_row(g, line)
            if issue.severity == "error":
                errors += 1
            elif issue.severity == "warning":
                warnings += 1
        console.print(tbl)

    summary = (
        f"[bold]Summary:[/] "
        f"[green]{passes} passed[/] · "
        f"[yellow]{warnings} warnings[/] · "
        f"[red]{errors} errors[/]"
    )
    console.print(summary)
    return 1 if errors else 0


def run_doctor(root: Path | None = None) -> int:
    """Entry point used by `quorum doctor` CLI command."""
    console = Console()
    console.print("[bold]Quorum doctor[/] — pre-deploy cross-config audit\n")
    results = run_all_checks(root)
    return render_results(results, console)


if __name__ == "__main__":
    import sys
    sys.exit(run_doctor())
