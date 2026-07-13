# Lead Radar

Starter scaffold for a lead-monitoring service focused on tenders and exhibition/event opportunities.

## Stack

- FastAPI
- SQLAlchemy
- PostgreSQL
- APScheduler
- python-telegram-bot

## First launch

1. Copy `.env.example` to `.env`
2. Fill the manual values listed in the "Manual configuration" section below
3. Create and activate a virtual environment
4. Install dependencies:

```powershell
.\run.ps1 -Action setup
```

5. Start PostgreSQL:

```powershell
.\run.ps1 -Action db
```

The bundled PostgreSQL container is exposed on `localhost:55432` to avoid conflicts with local PostgreSQL services already installed on the machine.

6. Apply database migrations:

```powershell
.\run.ps1 -Action migrate
```

7. Check readiness:

```powershell
.\run.ps1 -Action doctor
```

8. Run collectors and score the first queue:

```powershell
.\run.ps1 -Action collect
.\run.ps1 -Action rescore
.\run.ps1 -Action previewscore
```

If you want the working queue to contain only real orders/tenders and hide previously collected exhibition calendar items:

```powershell
.\run.ps1 -Action ordermode
```

9. Run the API:

```powershell
.\run.ps1 -Action api
```

10. Run the Telegram bot in a separate terminal after Telegram values are filled:

```powershell
.\run.ps1 -Action bot
```

## Manual configuration

Values that must be filled by hand in `.env`:

- `TELEGRAM_BOT_TOKEN`: token from BotFather
- `TELEGRAM_CHAT_ID`: allowed chat id for protected admin commands. Multiple ids can be separated with commas; use `/chatid` in Telegram to see the current chat id.
- `TELEGRAM_NOTIFICATION_CHAT_ID`: target group chat id for the morning digest. If empty, the first id from `TELEGRAM_CHAT_ID` is used.
- `GOOGLE_SHEETS_CREDENTIALS_FILE`: local path to a Google service account JSON file, only if Sheets sync is needed
- `GOOGLE_SHEETS_SPREADSHEET_ID`: spreadsheet id from the Google Sheets URL, only if Sheets sync is needed
- `GOOGLE_SHEETS_RANGE`: target range, for example `Queue!A1`
- `GOOGLE_SHEETS_HOT_RANGE`: target range for hot prospects, for example `Hot!A1`

Google Sheets first setup:

1. Create a Google Cloud service account and download its JSON key.
2. Put the JSON path into `.env` as `GOOGLE_SHEETS_CREDENTIALS_FILE`.
3. Create or prepare a spreadsheet:

```powershell
.\run.ps1 -Action setupsheets -ShareWith your.email@gmail.com
```

The command prints the values to put into `.env`, including `GOOGLE_SHEETS_SPREADSHEET_ID`.
If you already created the spreadsheet manually, put its id into `.env` first and run the same command; it will make sure the `Queue` and `Hot` tabs exist.

Optional values to review before production use:

- `APP_HOST`: keep `127.0.0.1` for local-only access; use `0.0.0.0` only when the API must be reachable from other machines
- `APP_PORT`: default is `8010`
- `ENABLE_FEEDBACK_LEARNING`: enables score nudges from manager feedback; works without external AI
- `ENABLE_AI_SCORING`: enables LLM scoring and explanations; keep `false` until `GEMINI_API_KEY` is filled
- `AI_PROVIDER`: default is `gemini`
- `GEMINI_API_KEY`: Google AI Studio API key for optional AI lead analysis
- `GEMINI_MODEL`: default is `gemini-2.5-flash`
- `AI_DAILY_ANALYSIS_LIMIT`: max active leads analyzed by AI per scheduled run
- `AI_ANALYSIS_HOUR` / `AI_ANALYSIS_MINUTE`: daily AI analysis schedule in Moscow time
- `ENABLE_SCHEDULED_MORNING_COLLECTION`: runs one automatic morning collection and digest when `true`; daytime collection stays manual via `/collectnow`
- `ENABLE_INSTANT_TELEGRAM_NOTIFICATIONS`: sends every new A/B lead immediately when `true`; default `false` to avoid Telegram spam
- `MANUAL_COLLECT_TIMEOUT_SECONDS`: max time the Telegram `/collectnow` command waits for a manual order-source collection result
- `RECENT_COLLECTION_WINDOW_MINUTES`: grouping window used by the Telegram `Новые` button around the latest collection run that saved leads
- `PRIORITY_REGIONS`: default is `Москва,Московская область`
- `ENABLE_EXPOCENTR_COLLECTOR`: disabled by default until the source is validated in your environment
- `ENABLE_B2B_CENTER_COLLECTOR`, `ENABLE_BIDZAAR_COLLECTOR`, `ENABLE_FABRIKANT_COLLECTOR`, `ENABLE_ROSTENDER_COLLECTOR`, `ENABLE_SYNAPSE_COLLECTOR`: enabled by default as real order/tender sources
- `ENABLE_EIS_COLLECTOR`: optional public procurement source; disable it if `zakupki.gov.ru` fails on your network
- `ENABLE_CROCUS_COLLECTOR` and `ENABLE_EXPONET_CITY_COLLECTORS`: disabled by default because they are event calendars, not customer orders

## Routine commands

Run all collectors in one pass:

```powershell
.\run.ps1 -Action collect
```

Recalculate scores after changing `.env` scoring settings:

```powershell
.\run.ps1 -Action rescore
```

Run optional AI analysis for active leads:

```powershell
.\run.ps1 -Action analyzeai
```

Sync the current queue to Google Sheets:

```powershell
.\run.ps1 -Action syncsheets
```

Sync only hot prospects to Google Sheets:

```powershell
.\run.ps1 -Action synchot
```

## PowerShell helper

Use the root script [run.ps1](/abs/path/c:/Users/galee/New%20good%20project/run.ps1:1) to avoid long commands:

```powershell
.\run.ps1 -Action setup
.\run.ps1 -Action db
.\run.ps1 -Action migrate
.\run.ps1 -Action api
.\run.ps1 -Action bot
.\run.ps1 -Action collect
.\run.ps1 -Action rescore
.\run.ps1 -Action previewscore
.\run.ps1 -Action analyzeai
.\run.ps1 -Action syncsheets
.\run.ps1 -Action synchot
.\run.ps1 -Action setupsheets
.\run.ps1 -Action ordermode
.\run.ps1 -Action doctor
.\run.ps1 -Action all
```

## Scoring tuning

Lead scoring can now be tuned directly through `.env` without code changes.

Main knobs:

- `SCORING_WHITELIST_KEYWORDS`: words and phrases that increase lead priority
- `SCORING_BLACKLIST_KEYWORDS`: words and phrases that lower priority
- `SCORING_PRIORITY_A_THRESHOLD`: minimum score for priority `A`
- `SCORING_PRIORITY_B_THRESHOLD`: minimum score for priority `B`
- `SCORING_WHITELIST_WEIGHT`: score added for each whitelist match
- `SCORING_WHITELIST_CAP`: maximum whitelist bonus
- `SCORING_BLACKLIST_WEIGHT`: penalty for each blacklist match
- `SCORING_BLACKLIST_CAP`: maximum blacklist penalty
- Geography is distance-weighted: Moscow and Moscow region get the strongest score bonus; other European Russia regions are included with gradually lower bonuses.

Typical workflow:

1. Edit `.env`
2. Recalculate scores:

```powershell
.\run.ps1 -Action rescore
```

3. Preview explanations for recent leads:

```powershell
.\run.ps1 -Action previewscore
```

If there are too many priority `A` leads, first lower `SCORING_WHITELIST_WEIGHT` or raise `SCORING_PRIORITY_A_THRESHOLD`.

## Feedback And AI Layer

The base scoring rules remain the source of truth. Manager feedback and AI only add bounded adjustments on top of the base score.

- Feedback learning is enabled by `ENABLE_FEEDBACK_LEARNING=true` and does not require external services.
- Telegram lead buttons `Подходит`, `Точно профиль`, `Хороший бюджет`, `Срочно`, `Не профиль`, `Далеко`, `Бюджет`, `Дедлайн`, `Дубль`, `Другое` are saved as training signals.
- Lead cards include `Почему?`, which explains rule-based score factors, component scores, feedback learning, duplicate status, and AI reasoning when available.
- Lead quality is stored as components: `fit_score`, `business_score`, `urgency_score`, `logistics_score`, plus `quality_reason`. The old `relevance_score` remains the main sorting score.
- Cross-source duplicates are conservatively marked as `context` with `is_duplicate=true` and `duplicate_reason`, instead of being shown as separate queue items.
- `rescore` recalculates existing leads using current scoring settings and accumulated feedback.
- Optional AI scoring is enabled only when `ENABLE_AI_SCORING=true` and `GEMINI_API_KEY` are set.
- AI analysis stores `ai_score`, explanation, tags, risks, and a recommended action, but it is capped by `AI_SCORE_WEIGHT` so it cannot fully override rule-based scoring.
- AI runs in scheduled/manual batch mode, not during collector ingest, so source collection stays fast even if Gemini is slow or unavailable.
- `COMPANY_PROFILE_TEXT` can tune the AI profile without code changes.
- The scheduler runs optional AI analysis once per day at `AI_ANALYSIS_HOUR:AI_ANALYSIS_MINUTE`; if AI is disabled, the job exits immediately.

## First build targets

- Alembic migrations are included
- Event calendar collectors are available but disabled by default.
- Active order collectors target B2B-Center, Bidzaar, Fabrikant, Rostender, Synapse, and optionally EIS on `zakupki.gov.ru`
- New `A/B` leads are not sent instantly by default; use the morning digest or `/collectnow`
- Scheduler runs one morning collection and digest at `DIGEST_HOUR_MORNING:00`
- Scheduler moves expired unprocessed `new` leads out of the queue every hour at minute `5`; `in_work` leads are kept.
- Scheduler syncs Google Sheets every hour at minute `25` when configured
- Scheduler runs optional AI analysis once per day when configured
- `ENABLE_EIS_COLLECTOR=false` is recommended on networks where `zakupki.gov.ru` fails at TLS level
- API queue endpoint: `GET /leads/queue`
- API hot-only filter: `GET /leads?hot_only=true`
- API hot prospects endpoint: `GET /leads/hot`
- API hot export: `GET /leads/export/hot`
- API hot stats: `GET /stats/hot`
  includes `by_priority`, `by_status`, `by_region`, `by_source`, `top_customers`, `top_events`, `top_deadlines`
- API CSV export: `GET /leads/export`
- API Google Sheets sync: `POST /leads/sync-sheets`
- API hot prospects Google Sheets sync: `POST /leads/sync-sheets/hot`
- Bot command `/new` shows the latest unprocessed leads
- Bot command `/queue` shows the current working queue
- The Telegram `Очередь` button shows 10 leads at a time and offers `Показать еще` until the queue ends
- Bot command `/hot` shows only hot `A/B` leads from priority regions
- Bot command `/deadlines` shows ближайшие дедлайны по горячим лидам
- Bot command `/summary` shows a compact hot leads summary with statuses, top sources, and nearest deadlines
- Bot command `/take <lead_id>` moves a lead to `in_work` and records who took it
- Bot command `/mine` shows the leads currently in work for the current Telegram user
- Bot command `/export` sends a CSV file with the current queue
- Bot command `/hotexport` sends a CSV file with hot prospects
- Bot command `/syncsheets` pushes the current queue to Google Sheets
- Bot command `/synchot` pushes hot prospects to Google Sheets
- Bot shows a persistent Telegram button menu for the same daily actions: new leads, queue, hot leads, my leads, deadlines, summary, CSV export, and Google Sheets sync.
- Bot lead cards include feedback buttons and optional `AI-анализ`.
