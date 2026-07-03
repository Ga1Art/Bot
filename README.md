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
- `TELEGRAM_CHAT_ID`: target chat id for notifications and digests
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
- `PRIORITY_REGIONS`: default is `Москва,Московская область`
- `ENABLE_EXPOCENTR_COLLECTOR`: disabled by default until the source is validated in your environment
- `ENABLE_B2B_CENTER_COLLECTOR`, `ENABLE_FABRIKANT_COLLECTOR`, `ENABLE_ROSTENDER_COLLECTOR`, `ENABLE_SYNAPSE_COLLECTOR`: enabled by default as real order/tender sources
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

## First build targets

- Alembic migrations are included
- Event calendar collectors are available but disabled by default.
- Active order collectors target B2B-Center, Fabrikant, Rostender, Synapse, and optionally EIS on `zakupki.gov.ru`
- New `A/B` leads are sent to Telegram once
- Scheduler runs collectors every hour at minute `15`
- Scheduler syncs Google Sheets every hour at minute `25` when configured
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
