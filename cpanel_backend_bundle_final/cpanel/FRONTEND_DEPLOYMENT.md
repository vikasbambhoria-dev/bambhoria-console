# Frontend (Node.js SPA) Deployment to cPanel / GoDaddy

This guide shows how to host your built frontend at the domain root (https://bambhoriaquantum.in/) while the Flask backend continues to run at https://bambhoriaquantum.in/app.

If your app is a React/Vue/Svelte/Vite SPA, you do NOT run Node.js on the server. You only upload the built static files.

---

## 1) Configure API base URL at build time

Set your frontend to call the backend under `/app`.

- Vite: use env var `VITE_API_BASE=https://bambhoriaquantum.in/app`
- Create React App: `REACT_APP_API_BASE=https://bambhoriaquantum.in/app`
- Next.js (static export): use `NEXT_PUBLIC_API_BASE=https://bambhoriaquantum.in/app`

Then update your code where you create the API client:

```js
const API_BASE = import.meta?.env?.VITE_API_BASE || process.env.NEXT_PUBLIC_API_BASE || process.env.REACT_APP_API_BASE || '';
// Example: fetch(`${API_BASE}/api/status`)
```

SPA Router notes:
- If using React Router or Vue Router in history mode, keep basename `/`.
- Do NOT set a basename of `/app` for the frontend; `/app` is reserved for the Python backend.

---

## 2) Build locally

Run your project build locally. Examples:

- Vite: `npm ci && npm run build` → outputs to `dist/`
- CRA: `npm ci && npm run build` → outputs to `build/`
- Next.js static export: `npm ci && npm run build && npm run export` → outputs to `out/`

After the build, verify `index.html` opens locally (basic sanity check).

Optional: Use `scripts/zip_frontend.ps1` in this repo to zip the built folder for upload.

---

## 3) Upload to cPanel

- Open cPanel → File Manager → `public_html/`
- Backup existing files if needed.
- Upload the contents of your built folder (e.g., `dist/`, `build/`, or `out/`) directly into `public_html/`.
  - You should see: `public_html/index.html`, `public_html/assets/...`, etc.
- Do NOT upload `node_modules/` or your source files.

---

## 4) Add .htaccess for SPA, caching, and gzip

Create or edit `public_html/.htaccess` and include the recommended rules. A sample is provided in this repo at `cpanel/frontend.htaccess.sample`.

Key points:
- Rewrite all non-file paths to `/index.html` so SPA routes work.
- Exclude `/app/` from SPA rewrites so requests go to the Flask backend.
- Add far-future caching for hashed assets.
- Enable gzip (mod_deflate) if available.

---

## 5) Verify end-to-end

- https://bambhoriaquantum.in/ should serve your frontend.
- Frontend API calls should go to `https://bambhoriaquantum.in/app/...` and succeed.
- Backend health: https://bambhoriaquantum.in/app/health
- Backend docs: https://bambhoriaquantum.in/app/docs

If you changed your domain or subpath, update `VITE_API_BASE` (or equivalent) and rebuild.

---

## 6) Optional: CDN and security

- Cloudflare/Static CDN: point DNS and enable caching for `/assets/` paths.
- Security headers for static files can be added in `.htaccess`. The backend already has robust headers.

---

## 7) Troubleshooting

- White page on refresh for SPA routes: you likely missed the SPA rewrite rules in `.htaccess`.
- API 404/401: confirm `API_BASE` points to `/app`, and backend is healthy and authenticated.
- Mixed content: ensure your site uses HTTPS everywhere.
- Cache not updating: bump asset names (build hashes) or purge CDN/browser cache.

---

## 8) Rollback

Keep a zipped copy of the previous `public_html/` contents. To rollback, replace files with the previous version.

---

## 9) CI: Download backend cPanel ZIP (ready-made)

If you prefer not to build the backend bundle locally, this repo publishes a cPanel-ready backend ZIP via GitHub Actions:

- Workflow: `.github/workflows/package_backend.yml`
- Docs: `cpanel/CI_CD.md`

How to use:
1. Push to the `main` branch (or run the workflow manually from GitHub → Actions → "Package Backend (cPanel bundle)").
2. Open the workflow run → download the artifact named `cpanel_backend_bundle`.
3. Upload `cpanel/cpanel_backend_bundle.zip` to your cPanel Python App root and deploy per your backend guide (`DEPLOYMENT_GUIDE.md`).
