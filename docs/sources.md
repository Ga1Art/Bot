# Source Backlog

Lead Radar separates two data classes:

- `order sources`: tenders, commercial procurement, marketplace requests, and RFQ-style posts where a customer is looking for a contractor.
- `event context sources`: exhibition calendars and venue schedules. These are useful context, but they are not orders.

See `docs/order_source_audit.md` for the detailed implementation status and manual watchlist.

## Enabled By Default

- `b2b_center` - public keyword search on B2B-Center commercial procedures.
- `fabrikant` - public keyword search on Fabrikant commercial and regulated procedures.
- `rostender` - public keyword search on Rostender aggregator pages.
- `synapse` - public keyword search on Synapse tender aggregator pages.
- `eis` - public procurement search on `zakupki.gov.ru`; currently can fail at TLS level in this local Windows/network environment.

## Disabled By Default

- `crocus_expo` - event calendar only.
- `exponet_*` - event calendars only.
- `expocentr` - event calendar only.

These sources can still be useful for understanding upcoming demand, but they should not drive the working queue by default.

## Excluded

- `zakupki.mos.ru` - excluded after manual review because there are very few relevant orders, and found matches are only partially suitable and not attractive enough commercially.

## Target Order Platforms

High-priority sources to connect next:

- `Контур.Закупки`, `СБИС Торги`, `Закупки360`, `Тендерплан` - aggregators that may be better as paid/API integrations than as HTML scrapers.
- `Росэлторг` - regulated, corporate, and commercial procurement.
- `РТС-тендер` / `B2B-RTS` - regulated and commercial procurement.
- `Сбербанк-АСТ` - regulated and commercial procurement.
- `Tender.Pro` - commercial tenders.
- `Bidzaar` - commercial procurement.

Recommended account/API priority:

1. One paid aggregator with alerts/API/export.
2. Roseltorg commercial/corporate sections.
3. RTS-Tender / B2B-RTS.
4. Sberbank-AST.
5. Bidzaar and Tender.Pro after we confirm relevant order volume.

Marketplace-style sources to evaluate separately:

- `Авито Услуги`
- `Профи`
- `YouDo`
- Telegram channels/chats for tenders and event contractors

These are closer to small business requests, but anti-bot limits and account rules matter more there.

## Search Keywords

Core keywords:

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
- `POSM`

## Rollout Plan

1. Keep B2B-Center, Fabrikant, Rostender, and Synapse running daily.
2. Use EIS only where `zakupki.gov.ru` works without TLS failures.
3. Add one paid aggregator via API/export/email alerts.
4. Add commercial ETP connectors where open search is stable or an API/account is available.
5. Keep event calendars as context only, not as queue items.
