# Quorum Landing Page

Single-file static landing page for **Quorum** — the Switzerland of LLMs.

- `index.html` — self-contained, no build step, no JS framework, < 25 kB.
- Dark mode by default with a light-mode toggle (persists via `localStorage`).
- Mobile-first responsive layout.

## Local preview

```bash
cd /tmp/sovereignchain/quorum/landing
python3 -m http.server 8080
# open http://localhost:8080
```

That's it — no bundler, no `node_modules`.

---

## Deploy

Three options, ordered by my preference for this kind of static one-pager.

### 1. Cloudflare Pages (recommended)

Zero build, free tier, edge-cached globally, custom domain in 30 seconds.

**Via Wrangler (one shot):**

```bash
npm install -g wrangler
cd /tmp/sovereignchain/quorum/landing
wrangler pages deploy . --project-name=quorum --branch=main
```

First run will prompt a browser login to your Cloudflare account, then create the project and return a `*.pages.dev` URL.

**Via dashboard:**

1. Push this folder to a GitHub repo (e.g. `jaquelinejaque/quorum-landing`).
2. Cloudflare dashboard → Workers & Pages → Create → Pages → Connect to Git.
3. Pick the repo. Build command: *(leave empty)*. Output directory: `/` (root).
4. Save and Deploy. Cloudflare returns `https://quorum.pages.dev`.
5. Add custom domain (e.g. `quorum.sovereignchain.co.uk`) in *Custom domains*. DNS is autoconfigured if your domain is on Cloudflare.

### 2. Vercel

Good DX, instant previews per branch, also free for this size.

**Via Vercel CLI:**

```bash
npm install -g vercel
cd /tmp/sovereignchain/quorum/landing
vercel deploy --prod
```

Answer the prompts (scope, project name `quorum`, no framework, root directory `./`). Vercel returns a `*.vercel.app` URL.

**Via dashboard:**

1. Push the folder to GitHub.
2. vercel.com → *Add New → Project* → import the repo.
3. Framework preset: *Other*. Root directory: `./`. Build command: *empty*. Output dir: *empty*.
4. Deploy. Add a custom domain under *Project → Settings → Domains*.

A minimal `vercel.json` is **not required** but if you want explicit headers:

```json
{
  "headers": [
    { "source": "/(.*)", "headers": [{ "key": "Cache-Control", "value": "public, max-age=3600" }] }
  ]
}
```

### 3. GitHub Pages

Slowest TTFB of the three, but zero infra outside GitHub.

```bash
# from /tmp/sovereignchain/quorum/landing
git init
git add index.html README.md
git commit -m "init quorum landing"
git branch -M main
git remote add origin git@github.com:jaquelinejaque/quorum-landing.git
git push -u origin main
```

Then on GitHub:

1. Repo → *Settings → Pages*.
2. Source: *Deploy from a branch*.
3. Branch: `main`, folder: `/ (root)`. Save.
4. Wait ~1 min. URL is `https://jaquelinejaque.github.io/quorum-landing/`.
5. Add a custom domain via the *Custom domain* field (and a `CNAME` file in the repo root). Add a `CNAME` DNS record from your domain to `jaquelinejaque.github.io`.

---

## Custom domain checklist (all three providers)

- [ ] Add domain in provider dashboard.
- [ ] Point DNS: either a `CNAME` to the provider hostname, or use the provider's nameservers (Cloudflare).
- [ ] Verify HTTPS certificate is issued (usually automatic, ~1–5 min).
- [ ] Hit the apex (`quorum.sovereignchain.co.uk`) and `/` — both should 200.
- [ ] Run Lighthouse (Performance + Accessibility ≥ 95 expected — it's a static page).

## Edits

It's one file. Open `index.html`, change copy, redeploy. There is no CMS and that is the feature.
