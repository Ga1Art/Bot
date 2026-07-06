# Server First Launch With Docker And Termius

This guide assumes a Linux VPS with Docker and Docker Compose plugin installed.

## How The Bot Works Now

Lead Radar has four runtime parts:

- `db`: PostgreSQL database in Docker.
- `api`: FastAPI app. It also starts the scheduler.
- `bot`: Telegram polling bot.
- `collector`: one-shot Docker service for manual collection checks.

The API scheduler runs automatically:

- every hour at minute `15`: runs collectors.
- every hour at minute `25`: syncs Google Sheets if configured.
- once per day at configured time: runs optional AI lead analysis if configured.
- every day at `09:00` and `16:00`: sends Telegram digest.

Current active order collectors:

- `b2b_center`: public B2B-Center keyword search.
- `bidzaar`: public Bidzaar buy requests API.
- `fabrikant`: public Fabrikant procedure search.
- `rostender`: public Rostender aggregator keyword search.
- `synapse`: public Synapse aggregator keyword search.
- `eis`: `zakupki.gov.ru`; may fail on some networks because of TLS.

Event calendar collectors are disabled by default and should stay context-only.

Telegram bot commands:

- The bot also shows a persistent button menu for the same daily actions.
- `/start`: shows bot status and command list.
- `/new`: latest new leads.
- `/queue`: current working queue.
- `/hot`: priority A/B hot leads.
- `/deadlines`: nearest deadlines for hot leads.
- `/summary`: compact hot lead summary.
- `/take <lead_id>`: move a lead to `in_work`.
- `/mine`: show leads currently assigned to the current Telegram user.
- `/export`: send queue CSV.
- `/hotexport`: send hot leads CSV.
- `/syncsheets`: sync queue to Google Sheets.
- `/synchot`: sync hot leads to Google Sheets.

Inline buttons on lead cards:

- `В работу` and `Подходит`: positive feedback for future scoring.
- `Не профиль`, `Далеко`, `Бюджет`, `Дедлайн`, `Дубль`, `Другое`: rejection reasons used by feedback learning.
- `AI-анализ`: runs optional AI analysis for this lead when AI is configured.

Lead statuses:

- `new`: imported and not processed.
- `in_work`: manager took it.
- `rejected` / `done` if changed from inline buttons or API.
- `context`: not a real order, or old event-calendar item.

Deadline rule:

- Commercial sources are shown only when the application deadline is at least tomorrow in Moscow time.
- Leads with deadlines today, expired deadlines, missing commercial-source deadlines, or closed statuses are moved to `context` by the `expire` command.

Geography rule:

- Hot leads are not limited to Moscow anymore.
- Moscow and Moscow region get the highest geography score.
- Other European Russia regions are included with lower score bonuses as distance/logistics complexity increases.

Feedback and AI rule:

- Feedback learning works without external services after migrations are applied.
- AI scoring is disabled by default and does not affect the bot unless `ENABLE_AI_SCORING=true` and `GEMINI_API_KEY` are set.
- AI only adds a bounded score adjustment; the base rule-based score remains visible.

## 1. Connect To The Server In Termius

1. Open Termius.
2. Add a host with server IP, SSH user, and SSH key/password.
3. Connect to the host.
4. Create a project directory:

```bash
mkdir -p /opt/lead-radar
cd /opt/lead-radar
```

## 2. Install Docker On The Server

Ubuntu/Debian quick path:

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
newgrp docker
docker --version
docker compose version
```

If `newgrp docker` does not apply cleanly, disconnect and reconnect in Termius.

## 3. Upload Project Files

Option A: Git:

```bash
cd /opt
git clone <YOUR_REPO_URL> lead-radar
cd /opt/lead-radar
```

Option B: Termius SFTP:

1. Open SFTP panel in Termius.
2. Upload the project folder contents into `/opt/lead-radar`.
3. Do not upload local `.venv`, `.pytest_cache`, or old logs.

## 4. Prepare `.env`

Create `.env` from example:

```bash
cp .env.example .env
nano .env
```

For Docker server launch, use these important values:

```env
APP_ENV=prod
APP_HOST=0.0.0.0
APP_PORT=8010
DATABASE_URL=postgresql://lead_radar:lead_radar@db:5432/lead_radar

TELEGRAM_BOT_TOKEN=<token from BotFather>
TELEGRAM_CHAT_ID=<target chat id>

ENABLE_CROCUS_COLLECTOR=false
ENABLE_EXPONET_CITY_COLLECTORS=false
ENABLE_EXPOCENTR_COLLECTOR=false
ENABLE_B2B_CENTER_COLLECTOR=true
ENABLE_BIDZAAR_COLLECTOR=true
ENABLE_FABRIKANT_COLLECTOR=true
ENABLE_ROSTENDER_COLLECTOR=true
ENABLE_SYNAPSE_COLLECTOR=true
ENABLE_EIS_COLLECTOR=false

TENDER_SEARCH_MAX_PAGES=1
BIDZAAR_SEARCH_MAX_PAGES=1
ROSTENDER_SEARCH_MAX_PAGES=1
SYNAPSE_SEARCH_MAX_PAGES=1
ROSTENDER_SEARCH_KEYWORDS=выставочный стенд,изготовление выставочного стенда,монтаж выставочного стенда,застройка стенда,оформление стенда,выставочное оборудование

ENABLE_FEEDBACK_LEARNING=true
FEEDBACK_LEARNING_WEIGHT=8
FEEDBACK_LEARNING_CAP=24

ENABLE_AI_SCORING=false
AI_PROVIDER=gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
AI_SCORE_WEIGHT=15
AI_MIN_BASE_SCORE=45
AI_DAILY_ANALYSIS_LIMIT=20
AI_ANALYSIS_HOUR=23
AI_ANALYSIS_MINUTE=35
```

Recommended first server launch: set `ENABLE_EIS_COLLECTOR=false`. After B2B-Center, Bidzaar, Fabrikant, Rostender, and Synapse are stable, enable EIS and check whether the server network can access `zakupki.gov.ru`.

If Google Sheets is needed:

```env
GOOGLE_SHEETS_CREDENTIALS_FILE=secrets/google-service-account.json
GOOGLE_SHEETS_SPREADSHEET_ID=<spreadsheet id>
GOOGLE_SHEETS_RANGE=Queue!A1
GOOGLE_SHEETS_HOT_RANGE=Hot!A1
```

Then upload the service account JSON to:

```bash
mkdir -p secrets
nano secrets/google-service-account.json
chmod 600 secrets/google-service-account.json
```

## 5. Build And Start Database

```bash
docker compose build
docker compose up -d db
docker compose ps
```

Wait until `lead_radar_db` is healthy.

## 6. Apply Migrations

```bash
docker compose run --rm migrate
```

This creates the database tables.

## 7. Run First Collection Manually

```bash
docker compose run --rm collector
```

Expected result: `b2b_center`, `bidzaar`, `fabrikant`, `rostender`, and `synapse` should return `success`.

If EIS is enabled and fails with TLS, disable it in `.env`:

```env
ENABLE_EIS_COLLECTOR=false
```

Then rerun:

```bash
docker compose run --rm collector
```

## 8. Start API And Bot

```bash
docker compose up -d api bot
docker compose ps
```

Check API:

```bash
curl http://127.0.0.1:8010/health
```

Expected:

```json
{"status":"ok"}
```

Check logs:

```bash
docker compose logs -f api
docker compose logs -f bot
```

In Telegram, send:

```text
/start
/summary
/queue
/hot
```

## 9. Useful Daily Commands

Show containers:

```bash
docker compose ps
```

Restart after `.env` changes:

```bash
docker compose up -d --force-recreate api bot
```

Run collectors manually:

```bash
docker compose run --rm collector
```

Move expired or closed open leads to `context`:

```bash
docker compose run --rm expire
```

Apply new migrations after code update:

```bash
docker compose run --rm migrate
```

Recalculate scores after feedback/scoring changes:

```bash
docker compose --profile tools run --rm rescore
```

Run optional AI analysis manually:

```bash
docker compose --profile tools run --rm ai_analyzer
```

View logs:

```bash
docker compose logs -f api
docker compose logs -f bot
docker compose logs -f db
```

After updating collector logic:

```bash
docker compose build
docker compose run --rm migrate
docker compose run --rm expire
docker compose run --rm collector
docker compose up -d --force-recreate api bot
```

Stop services:

```bash
docker compose down
```

Stop and delete database volume only when you intentionally want to erase data:

```bash
docker compose down -v
```

## 10. First Launch Checklist

- Docker installed.
- Project uploaded to `/opt/lead-radar`.
- `.env` filled on server.
- `DATABASE_URL` uses `db:5432`, not `localhost:55432`.
- Telegram token and chat id filled.
- Google JSON uploaded only if Sheets sync is needed.
- `docker compose build` completed.
- `docker compose run --rm migrate` completed.
- `docker compose run --rm collector` returns leads or at least successful source statuses.
- `docker compose up -d api bot` completed.
- `/start` works in Telegram.
