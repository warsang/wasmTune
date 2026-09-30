# Lumen

Lumen is a time-series store for constrained edge devices. Devices push
measurements to an ingest endpoint; Lumen keeps them queryable for 13 months.

- **Ingest**: newline-delimited JSON over HTTPS. Batches capped at 1000 records.
- **Query**: SQL-ish read API with a 13-month default retention window.
- **Storage**: columnar, time-partitioned by day, compressed with zstd.

Lumen has no agents, no collectors and no sidecar. A device is a key and an
HTTPS endpoint.
