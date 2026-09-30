# Troubleshooting

## Writes return 202 but no data appears

Query the same window with `?consistency=strong`. The default is eventual and
can lag a few seconds behind a write.

## Device clocks drift

Lumen rejects points more than 24 hours in the past. Send
`X-Lumen-Offset-Ms` to correct for a known device offset rather than rewriting
timestamps on the device.

## I get 401 after rotating a key

Only the most recent key is valid; rotation is immediate and does not overlap.
Update the device fleet before rotating in production.

## A batch returned 200 but some points are missing

Check the `rejected` array in the `/v2/ingest-report` response body. Each entry
carries the `id` and a reason. Common causes: a future timestamp, a malformed
`ts`, or an `id` longer than 64 characters.

## Queries time out on long windows

Narrow the time range or add a `group by` — Lumen is optimised for narrow,
highly selective reads. The 13-month default retention is not intended to be
scanned in one query.
