# CRAG Study Agent

**[Open the live app → crag-study-agent.vercel.app](https://crag-study-agent.vercel.app)**

A study assistant for typed and handwritten PDFs, images, and text notes. Upload documents, ask questions, and inspect page-linked source passages. The app combines Gemini OCR, local semantic and keyword retrieval, and Groq answer generation with optional independent source checking.

## Try it online

1. Open the [live app](https://crag-study-agent.vercel.app) and select **Open your study room**.
2. Upload a PDF, TXT, JPG, PNG, or WEBP file. For handwritten pages, choose **Handwritten / mixed pages: OCR every page**.
3. Wait for processing, then select **Study this document**.
4. Ask a question or choose a study mode. Open the cited source passages to review the evidence.

No signup is required. Each browser gets a private guest library. Keep that browser's site data to retain access; another browser or device starts a separate library.

## Public demo limits

Each private guest session has **5 questions across all study modes** and **2 total uploads**. The backend reserves usage atomically in Supabase before expensive work starts. Accepted requests count even if later processing fails; invalid inputs rejected before reservation do not count.

Refreshing the page, starting a conversation, deleting a file, and restarting the backend do not reset these allowances. Existing documents and answers remain readable. The allowance follows the guest identity stored in the browser; it is not a per-person limit and a new guest identity receives a new allowance. Existing users start with a fresh allowance when this feature is introduced.

## Hosting

| Component | Deployment |
|---|---|
| Frontend | [Vercel](https://crag-study-agent.vercel.app) |
| Backend | Google Cloud Compute Engine, running FastAPI in Docker behind Caddy HTTPS |
| Database, authentication and original files | Supabase |
| OCR | Gemini API |
| Answer generation | Groq API |
| Embeddings and reranking | Local models on the backend VM |

The frontend is connected to this GitHub repository through Vercel; pushes to `main` deploy the frontend automatically. Backend changes still require a separate Docker deployment on Google Cloud. The backend uses Google Cloud trial credits; continued availability depends on hosting resources and provider quotas.

## Features

- Ask questions, create revision notes, summaries, exam answers, important questions, and MCQs.
- Extract scanned and handwritten pages with per-page progress, saved OCR results, and retryable incomplete pages.
- Retrieve evidence using BGE embeddings, BM25 keyword search, and a cross-encoder reranker.
- Show citations, omitted details, answer-check status, and processing timings.
- Show actual provider cooldowns separately from normal answer processing, including an approximate retry countdown.
- Maintain private guest libraries using Supabase anonymous authentication. There is no login/signup screen; documents and conversations are scoped to the guest identity.

## Stack and layout

| Directory | Purpose |
|---|---|
| `frontend/` | React, TypeScript, Vite and Tailwind UI |
| `backend/app/` | FastAPI API, LangGraph answer pipeline, OCR and retrieval services |
| `backend/tests/` | Offline regression tests |
| `supabase/migrations/` | Database, vector search, ownership and storage setup |

## Requirements

- Python 3.11 and Node.js 22.12+.
- Hosted Supabase with **Allow anonymous sign-ins** enabled and saved.
- Gemini and Groq API keys with access to the configured models.
- Space and memory for local embedding/reranking weights. First startup may download models and take several minutes.

The configured OCR default is `gemini-3.5-flash-lite`; answer generation defaults to `openai/gpt-oss-120b` on Groq. These are independent settings in `backend/.env.example`, not promises of model availability or free quota for every account.

## Supabase setup

For a new project, apply the migrations in order:

1. `supabase/migrations/0001_init.sql`
2. `supabase/migrations/20260914191832_reliable_handwritten_pipeline.sql`
3. `supabase/migrations/20261101130516_guest_usage_limits.sql`

For an existing project, review and apply only migrations not already applied. The second migration preserves existing rows, creates private original-file storage, and restricts data access to the authenticated backend. Legacy incompatible documents may require re-uploading. Unowned legacy rows are not assigned automatically to new guests.

Enable anonymous sign-ins under Supabase Authentication. The browser uses Supabase for identity; document and chat requests go through the backend's token and ownership checks. **Never put service-role or model API keys in frontend variables.**

## Run locally

Run these PowerShell commands from the project root. Copy example environment files only on first setup; do not overwrite an existing configured `.env`.

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `backend/.env`:

```dotenv
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_backend_service_role_key
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
ENVIRONMENT=development
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Start the backend from `backend/`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

`main:app` needs a colon. `app.main:app` also works. For code development, append `--reload`, but avoid editing/reloading while documents are processing. From the project root, `powershell -ExecutionPolicy Bypass -File .\start-backend.ps1` is an alternative; pass `-Port 8001` if port 8000 is unavailable and update the frontend API URL accordingly.

In a second terminal, from the project root:

```powershell
cd frontend
npm ci
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `frontend/.env`:

```dotenv
VITE_API_BASE_URL=http://localhost:8000
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your_public_anon_key
```

Run `npm run dev` and open `http://localhost:5173`. Restart Vite after changing its environment variables. The backend loads `backend/.env` regardless of the working directory.

On macOS/Linux, create the environment with `python3.11 -m venv .venv`, use `.venv/bin/python`, and copy files with `cp`. If the project moves between drives, recreate its virtual environment rather than relying on copied executable launchers; reinstall frontend dependencies with `npm ci`.

## Documents, privacy and answer checks

Keep browser site data to retain access to the guest library. Clearing it or using another browser profile creates a separate identity; it does not delete the original stored documents. Use the app's delete action to remove documents.

For handwritten or mixed PDFs with a printed text layer, select **Handwritten / mixed pages: OCR every page**. Check extraction progress before asking questions. A partial document answers only from readable pages and warns about missing or unclear material. **Resume pending / unclear pages** retries incomplete pages while reusing successful transcriptions. Current defaults allow 30 MB uploads and up to 150 pages.

Gemini OCR batches up to two adjacent scans per request. Native text and saved OCR bypass vision calls. Rate-limit waits respect the provider's cooldown; long waits leave pages pending for a later retry. Raising retry counts does not increase free quotas. OCR may misread equations, diagrams, or handwriting, so compare important answers with the original pages.

**Adaptive (faster)** can use one generation call plus local citation checks for specific questions with strong evidence. Such answers are labelled as not independently checked. **Full source check** adds model-based checking. Broad questions retain independent checking and bounded repair; supported items can be retained with an explicit partial-answer warning. Neither mode guarantees factual correctness.

The live quota notice appears only during a confirmed provider cooldown. Its countdown estimates the next retry, not the answer completion time. Polling errors do not discard the answer. Requests can still fail if an infrastructure timeout expires.

## Tests

Backend, from `backend/`:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Frontend, from `frontend/`:

```powershell
npm test
npm run lint
npm run build
```

## Docker backend and Vercel frontend

The live app uses this deployment layout. For another deployment:

1. Configure backend secrets, `ENVIRONMENT=production`, and `CORS_ORIGINS` with the exact HTTPS frontend origin.
2. On a Docker-capable host, run `docker compose -f docker-compose.production.yml up -d --build`. This binds port 8000 to host localhost for an HTTPS reverse proxy. Preserve the `model_cache` volume.
3. Run **one backend worker/replica**, without reload or overlapping rolling deployments. OCR restart recovery, quota coordination and request progress use a single-process design; multiple replicas require a shared queue/progress store.
4. In Vercel, set the project root to `frontend`. The existing `vercel.json` builds with `npm run build`, publishes `dist`, and handles SPA routes. Set the three public `VITE_*` variables above, using the deployed HTTPS backend URL, then rebuild.
5. The browser calls the backend directly. Configure proxy timeouts for provider waits. Check `/ready`, direct `/workspace` navigation, two isolated guest libraries, typed/handwritten uploads, citations, history, deletion and quota status on the deployed domains.

`/health` checks process liveness. `/ready` waits for local models and the startup database check; it does not guarantee ongoing database access or provider quota. Interrupted queued/processing uploads resume from saved work at startup. Failed/partial documents remain manually retryable.

## Deployment verification

The current deployment passed these live checks:

- Docker build and backend startup, including model loading and the database readiness check.
- Public HTTPS frontend and backend access.
- Browser guest authentication and frontend-to-backend requests.
- A TXT document upload, storage, indexing, and a cited answer.
- A separate browser guest seeing an empty library.
- Test-document deletion and no browser runtime errors during the tested flow.

The live smoke test used typed text; it does not establish handwritten OCR quality or performance for every document. Adaptive answers may be marked as not independently checked. Processing time and availability depend on document size, model calls, and provider quotas.
