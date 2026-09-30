# Write API

Send one JSON object per measurement. Batches are newline-delimited JSON, capped
at 1000 records.

```bash
curl https://ingest.lumen.dev/v2/write \
  -H "Authorization: Bearer $LUMEN_KEY" \
  -H "Content-Type: application/x-ndjson" \
  --data-binary @metrics.ndjson
```

A `200` means the batch was durably queued, not that every point parsed.
Per-record rejections arrive in the `/v2/ingest-report` response body and in the
project log.

## What is the maximum batch size?

1000 records per request. Larger batches are rejected with `413` rather than
truncated, so you never lose points silently.

## Which timestamp does a point use?

The `ts` field, in epoch milliseconds. If it is more than 24 hours in the past
or more than 1 hour in the future, Lumen rejects the record. Send
`X-Lumen-Offset-Ms` to declare a known device clock offset.

## What happens if a write fails?

Retry with exponential backoff. Writes are idempotent per `id` field: resending
a record with the same `id` overwrites rather than duplicating.
