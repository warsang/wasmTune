# Quickstart

Install the SDK and point it at a project. There is no server to deploy.

```bash
npm i @lumen/sdk
lumen init --project my-project
lumen push --config lumen.config.ts
```

`lumen init` writes a project manifest, generates a write key scoped to that
project, and creates the ingest endpoint. Keys are shown exactly once and stored
in `~/.config/lumen/credentials.json`.

## How long does setup take?

About a minute. There is nothing to run and nothing to keep alive — the ingest
endpoint is managed.

## Where are the keys stored?

In `~/.config/lumen/credentials.json`, created on first run. Delete the file to
revoke local access; the key itself stays valid until you rotate it in the
dashboard.
