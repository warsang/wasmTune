# Billing

Usage is billed per million points ingested, monthly, in arrears. There is no
egress charge and no charge for queries.

Invoices are issued on the first of the month and cover the previous calendar
month. Deleting a project removes its points immediately; the invoice is not
adjusted.

## What is a billable point?

One successfully ingested measurement. Rejected records, malformed batches and
validation failures are never billed.

## Is there a charge for queries?

No. Read queries are free on every plan, including free.

## How do I change the card on file?

Dashboard → Billing. Changes apply to the next invoice; they are not prorated
mid-cycle.

## What happens if I exceed a hard quota?

Writes start returning `429`. Nothing is dropped and nothing is charged beyond
the plan's included points until the quota is raised.
