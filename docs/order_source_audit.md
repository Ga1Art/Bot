# Order Source Audit

Last checked: 2026-06-26.

This document tracks real order sources for exhibition stands, temporary constructions, branded zones, POSM, signage, and exhibition exposition buildout. Event calendars are context only and should not drive the working queue.

## Automated Now

| Source | Status | What We Parse | Notes |
| --- | --- | --- | --- |
| B2B-Center | enabled | Public keyword search results from `https://www.b2b-center.ru/market/` | Public keyword search works without login. Extra filters, combined queries, favorites, and templates require registration. |
| Fabrikant | enabled | Public procedure search results from `https://www.fabrikant.ru/procedure/search` | Public HTML search works without login and includes title, organizer, dates, and sometimes budget. |
| Rostender | enabled | Public keyword search results from `https://rostender.info/extsearch` | Public aggregator HTML works without login. Current parser keeps only profile orders with actionable deadlines. |
| Synapse | enabled | Public keyword search results from `https://synapsenet.ru/search` | Aggregator over multiple platforms, including B2B, Roseltorg, Fabrikant, OTC, and public procurement. Current parser keeps only profile orders with actionable deadlines. |
| EIS / zakupki.gov.ru | enabled but network-blocked here | 44-FZ and 223-FZ public search | Collector is implemented, but this local Windows/network environment currently fails TLS before HTML is returned. Try from Docker, another network, VPN off/on, or a server. |

## Excluded Sources

| Source | Decision | Reason |
| --- | --- | --- |
| zakupki.mos.ru | excluded | Manual review showed very few relevant orders. Existing matches are only partially suitable and not attractive enough commercially. Do not spend integration time here unless the business situation changes. |

## Needs Account, API, Export, Or Deeper Integration

| Source | Priority | Action Needed | Why Not Plain Parsing Yet |
| --- | --- | --- | --- |
| Kontur.Zakupki | high as paid aggregator | Buy/trial account and configure alerts/API/export. | Better as paid aggregator integration than HTML parsing. Can cover many platforms at once. |
| SBIS Torgi | high as paid aggregator | Buy/trial account and configure alerts/API/export. | Better as paid aggregator integration than HTML parsing. |
| Zakupki360 | medium as paid aggregator | Buy/trial account and configure alerts/API/export. | Public page is mostly a cabinet entry point; better as paid/export integration than HTML parsing. |
| Tenderplan | medium as paid aggregator | Buy/trial account and configure alerts/API/export. | Better as paid aggregator integration than HTML parsing. |
| Bicotender | medium as paid aggregator | Check account/API/export options. | Public search page returns a broad current tender feed instead of reliably applying keyword relevance; too noisy for unattended parsing. |
| TenderGuru | medium as paid aggregator/API | Check API/export terms or paid search access. | Public HTML search form does not return parseable keyword result cards without deeper account/API integration. |
| Roseltorg | high | Register, check public search endpoint and API/export availability. | There are corporate and commercial procurements, but stable automation needs endpoint inspection. |
| RTS-Tender / B2B-RTS | high | Register supplier account, create saved searches, check notification/export/API options. | Likely relevant commercial and regulated procedures, but robust access usually needs account/session. |
| Sberbank-AST | medium-high | Register, check public search and export. | Large procurement platform; pages often require JS/session and may be poor for unauthenticated scraping. |
| Tender.Pro | medium | Register or provide public search URL examples that return target orders. | Commercial tenders may fit, but source needs endpoint analysis. |
| Bidzaar | medium | Register company account and check if search results/export are available. | Commercial procurement source, likely requires account. |

## Latest Public Parsing Check

Checked on 2026-07-03:

| Source | Result |
| --- | --- |
| Rostender | Works through public HTML. Search parameter must be `keywords`, not `keyword`. Best regular mode is one page with a source-specific keyword shortlist; deeper paging is possible but can be slow. |
| Synapse | Works through public HTML and adds aggregator coverage over B2B, Roseltorg, Fabrikant, OTC, and public procurement. One page is the recommended regular mode. |
| Bicotender | Public page is accessible, but keyword filtering is unreliable/noisy in unauthenticated HTML. Keep for account/API/export follow-up. |
| TenderGuru | Public page exposes search forms and API links, but unauthenticated HTML did not return usable tender cards for the target keywords. Keep for API/export follow-up. |
| Kontur.Zakupki | Returns access restriction for public scraping. Treat as paid aggregator/API source. |
| SBIS/Saby Torgi | Public page is mostly dynamic; no reliable static result cards in HTML. Treat as account/API/export source. |
| Zakupki360/Tenderplan | Public pages are mostly landing/cabinet entry points in this environment. Treat as account/API/export sources. |
| Roseltorg/RTS/Sberbank-AST/ETP GPB | Direct public parsing is unstable in this environment because of TLS/session/JS issues. Prefer account/export/API or aggregator coverage through Synapse/Rostender. |

## Account/API/Export Setup Plan

### Paid Aggregators: Kontur.Zakupki, SBIS Torgi, Zakupki360, Tenderplan

Best integration path: API, export, or email alerts.

What to do manually:

1. Start trial or paid account.
2. Create saved search profiles with stand/exhibition keywords and priority regions.
3. Check available delivery methods: API, Excel/CSV export, email alerts, webhook.
4. Ask support for API docs, rate limits, export format, and whether search profiles have stable IDs.

What to add to `.env` later:

```env
ENABLE_KONTUR_ZAKUPKI_COLLECTOR=true
KONTUR_ZAKUPKI_API_TOKEN=
KONTUR_ZAKUPKI_SEARCH_ID=
KONTUR_ZAKUPKI_EXPORT_URL=

ENABLE_SBIS_TORGI_COLLECTOR=true
SBIS_TORGI_API_TOKEN=
SBIS_TORGI_SEARCH_ID=
SBIS_TORGI_EMAIL_INBOX=

ENABLE_ZAKUPKI360_COLLECTOR=true
ZAKUPKI360_API_TOKEN=
ZAKUPKI360_EXPORT_URL=
ZAKUPKI360_EMAIL_INBOX=

ENABLE_TENDERPLAN_COLLECTOR=true
TENDERPLAN_API_TOKEN=
TENDERPLAN_SEARCH_ID=
TENDERPLAN_EXPORT_URL=
```

Bot integration: yes. This is likely the fastest way to expand coverage. Aggregator leads can use the same queue, scoring, Google Sheets sync, and Telegram buttons as current sources.

### Roseltorg

Best integration path: account-based integration by section.

What to do manually:

1. Register as supplier.
2. Check commercial procurement, corporate procurement, 223-FZ, and Roseltorg.Business sections.
3. Create saved searches by keywords.
4. Check export, notifications, or documented integration/API.
5. If only web search is available, capture the search request URL and response format from DevTools.

What to add to `.env` later:

```env
ENABLE_ROSELTORG_COLLECTOR=true
ROSELTORG_SECTION=com,rb,corp
ROSELTORG_EXPORT_URL=
ROSELTORG_API_URL=
ROSELTORG_API_TOKEN=
```

Bot integration: yes. Preserve the section name because participation rules differ between commercial, 223-FZ, and corporate procedures.

### RTS-Tender / B2B-RTS

Best integration path: registered supplier account plus saved searches/export/API if available.

What to do manually:

1. Register or accredit the company as a supplier.
2. Check sections for commercial procurement and regulated procurement.
3. Create saved searches with the core keywords.
4. Check whether results can be exported to Excel/CSV or delivered by email.
5. If no export exists, capture the authenticated search request from DevTools.

What to add to `.env` later:

```env
ENABLE_RTS_TENDER_COLLECTOR=true
RTS_TENDER_EXPORT_URL=
RTS_TENDER_API_URL=
RTS_TENDER_API_TOKEN=
RTS_TENDER_COOKIE=
```

Bot integration: yes. If participation requires paid access or accreditation, add an `access_note` in `raw_payload`.

### Sberbank-AST

Best integration path: registered supplier account or public search endpoint if stable.

What to do manually:

1. Register or accredit as supplier.
2. Check 44-FZ, 223-FZ, commercial procurement, and small purchases sections.
3. Run saved searches by keywords.
4. Check export and notification features.
5. If no export/API exists, capture authenticated search requests from DevTools.

What to add to `.env` later:

```env
ENABLE_SBER_AST_COLLECTOR=true
SBER_AST_EXPORT_URL=
SBER_AST_API_URL=
SBER_AST_API_TOKEN=
SBER_AST_COOKIE=
```

Bot integration: yes. Mark `requires_accreditation=true` when participation is not immediate.

### Tender.Pro

Best integration path: public search if stable; otherwise account export or email alerts.

What to do manually:

1. Register an account.
2. Search target keywords and verify whether relevant orders exist.
3. Check if Tender.Pro can send email alerts or export search results.
4. If emails are available, prefer email ingestion over fragile browser scraping.

What to add to `.env` later:

```env
ENABLE_TENDER_PRO_COLLECTOR=true
TENDER_PRO_SEARCH_URL=
TENDER_PRO_EXPORT_URL=
TENDER_PRO_EMAIL_INBOX=
```

Bot integration: yes. If the best path is email alerts, parsed email leads can still enter the same queue.

### Bidzaar

Best integration path: supplier account and official notifications/export/API.

What to do manually:

1. Register the company as supplier.
2. Search for event, stand, POSM, branding, and exposition keywords.
3. Check supplier alerts, saved filters, or export.
4. Ask support whether API access or webhook notifications are available.

What to add to `.env` later:

```env
ENABLE_BIDZAAR_COLLECTOR=true
BIDZAAR_API_URL=
BIDZAAR_API_TOKEN=
BIDZAAR_EXPORT_URL=
```

Bot integration: yes. This may be especially useful for commercial procurement.

## Bot Integration Model

All account/API/export sources should feed the same `LeadCreate` structure as current collectors:

- `source_type=tender`
- `source_name`
- `external_id`
- `title`
- `description`
- `url`
- `published_at`
- `deadline_at`
- `region`
- `budget_max`
- `customer_name`
- `keywords_matched`
- `raw_payload`

Recommended bot features after these integrations:

1. Source filters: `/hot b2b_center`, `/hot fabrikant`, `/hot aggregators`.
2. Access hints: show `нужна аккредитация`, `платное участие`, `требуется ЭП`, or `можно откликнуться сразу`.
3. Manager workflow: buttons `В работу`, `Не подходит`, `Запросить документы`, `Нужна оценка`.
4. Daily digest by source: commercial sources first, aggregators second, public procurement third.
5. Duplicate merging: if aggregator and original platform report the same order, keep one lead and store source aliases in `raw_payload`.

## Preferred Integration Order

1. One paid aggregator with API/export because it can cover many platforms quickly.
2. Roseltorg commercial/corporate sections.
3. RTS-Tender / B2B-RTS.
4. Sberbank-AST.
5. Bidzaar and Tender.Pro after we confirm relevant order volume.

## Manual Watchlist

These may contain smaller business requests, but parsing can be blocked by account rules, anti-bot protection, chat-only workflows, or marketplace terms. Keep them as a manager checklist until we decide on an account-based integration.

| Source | What To Search |
| --- | --- |
| Avito Services | `выставочный стенд`, `застройка стенда`, `фотозона`, `брендированная стойка`, `POSM` |
| Profi.ru | `изготовление стендов`, `оформление мероприятий`, `фотозона`, `выставочный стенд` |
| YouDo | `стенд`, `фотозона`, `оформление мероприятия`, `монтаж стенда` |
| Telegram channels/chats for tenders and event contractors | Search manually first, then add bot/API parsing only for channels where reposting/parsing is allowed. |

## Current Keyword Strategy

The commercial collectors reuse `EIS_SEARCH_KEYWORDS` for now:

- `выставочный стенд`
- `изготовление выставочного стенда`
- `монтаж выставочного стенда`
- `застройка стенда`
- `оформление стенда`
- `оформление выставочной экспозиции`
- `брендирование стенда`
- `экспозиционный стенд`
- `выставочное оборудование`
- `регистрационная стойка`
- `фотозона`

The parser also applies a profile filter so broad words do not flood the queue with unrelated real estate, medical, or equipment purchases.

## Recommended Next Steps

1. Run the current open-source collectors daily: B2B-Center, Fabrikant, Rostender, Synapse, and EIS where TLS works.
2. Choose one paid aggregator trial and prioritize API/export over HTML scraping.
3. Check Roseltorg and RTS-Tender next because they are more likely to contain commercial/corporate orders than `zakupki.mos.ru`.
4. Keep event calendars as context only, not as queue leads.
