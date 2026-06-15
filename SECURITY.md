# Security Policy

The SovereignChain maintainers take security seriously. This document
describes how to report a vulnerability, what to expect from us in response,
and how we handle disclosure timelines.

---

## Supported versions

| Version | Supported |
|---------|-----------|
| `main`  | yes       |
| latest tagged release | yes |
| older releases        | no  |

Security fixes are applied to `main` and backported only to the latest tagged
release. Users on older versions should upgrade.

---

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Send a report to:

- **Email**: `security@sovereignchain.io`
- **PGP**: fingerprint `7F4E 9C2A 5B8D 1A03 9F62  4C1E 8D77 2B5A E4F0 99C3`
  (key published at `https://sovereignchain.io/.well-known/security.asc`)
- **GitHub Security Advisories**: use the private advisory form at
  <https://github.com/sovereignchain/sovereignchain/security/advisories/new>

Encrypt anything sensitive (PoCs, traces, credentials inadvertently exposed)
with the PGP key. Reports sent in plaintext are still triaged — we will not
penalise you for not using PGP — but we strongly prefer encrypted reports.

### What to include

A good report contains:

1. A clear description of the vulnerability and its impact.
2. The affected version(s), commit SHA, or release tag.
3. Step-by-step reproduction instructions. A minimal PoC is ideal.
4. Your assessment of severity (CVSS 3.1 vector if you can).
5. Whether the issue is already public or known to other parties.
6. The name and contact details you would like us to use for credit
   (or "anonymous" if you prefer no credit).

---

## Our response process

| Stage | Target time |
|-------|-------------|
| Acknowledgement of report | within **48 hours** |
| Initial triage and severity assignment | within **5 business days** |
| Fix developed and tested in private branch | within **30 days** for high/critical |
| Coordinated public disclosure | within **90 days** of initial report |

We follow a **90-day disclosure window**. If a fix is not ready by day 90 we
will still publish a coordinated advisory (with mitigations and workarounds)
unless the reporter and maintainers jointly agree to a short extension based
on the complexity of the fix. We will not unilaterally extend the window.

If we cannot reproduce the issue, we will say so explicitly and ask for more
information before closing.

### Credit

With your permission we will credit you in the published advisory, the
release notes, and (where applicable) the CVE record. We do not currently
operate a paid bug bounty programme but reporters of valid vulnerabilities
are listed in `SECURITY-ACKNOWLEDGEMENTS.md`.

---

## Safe-harbour

We will not pursue civil or criminal action against good-faith security
researchers who:

- Make a sincere effort to avoid privacy violations, service degradation,
  and destruction of data.
- Only access the minimum data necessary to demonstrate the vulnerability.
- Give us reasonable time to remediate before public disclosure (the 90-day
  window above).
- Do not exploit the vulnerability beyond what is required to demonstrate
  it, and do not pivot to other systems.

This safe-harbour does not extend to social engineering of staff, physical
attacks, or denial-of-service testing against production infrastructure.

---

## HSP webhook security (special note)

The Hybrid Sovereign Protection (HSP) webhook transport — implemented in
`src/sovereignchain/hsp/` and covered by the licence terms in `LICENSE-HSP`
— carries attestations between sovereign nodes and is a particularly
sensitive surface. Subject matter in this module is covered by international
patent application **PCT/US26/11908**.

If your report touches the HSP webhook surface (signature verification,
replay-window enforcement, nonce handling, attestation chain validation,
quorum-threshold computation, or the launch-attestation pipeline in
`src/sovereignchain/launch/`), please:

1. Mark the report subject line with the prefix `[HSP]`.
2. Encrypt with PGP — plaintext reports on HSP issues will be acknowledged
   but discussion of details will be moved to an encrypted channel before
   any technical content is exchanged.
3. Expect a longer triage window (up to **10 business days**) because HSP
   reports are reviewed jointly by the security team and by the patent
   counsel responsible for PCT/US26/11908. Acknowledgement still happens
   within 48 hours.
4. Do not include PoC traffic against any live HSP endpoint in the report.
   Run PoCs only against a local node or a test fixture from
   `tests/hsp/fixtures/`. Live-endpoint exploitation, even for research,
   falls outside the safe-harbour above and may also implicate the patent
   claims.

HSP vulnerabilities follow the same 90-day disclosure window but the
coordinated advisory will additionally reference the relevant PCT/US26/11908
claim number(s) so downstream implementers can assess their exposure.

---

## Out of scope

The following are explicitly **not** considered vulnerabilities for the
purposes of this policy:

- Findings from automated scanners without a working proof of concept.
- Self-XSS or issues requiring an already-compromised local machine.
- Missing security headers on documentation sites (`docs.sovereignchain.io`)
  unless they lead to a concrete exploit.
- Rate-limit or volumetric denial-of-service against public endpoints.
- Vulnerabilities in third-party dependencies that have not yet been
  triaged by their upstream maintainers — please report those upstream
  first and CC us.

---

## Questions

For non-vulnerability security questions (hardening guidance, deployment
review, threat-model discussion) use `security@sovereignchain.io` with the
subject prefix `[QUESTION]`. These are answered on a best-effort basis and
do not have an SLA.
