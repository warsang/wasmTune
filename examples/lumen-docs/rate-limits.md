# Rate limits

Limits are per project, per minute. Burst is `limit × 1.5`. Exceeding either
returns `429` with a `Retry-After` header in seconds.

| Plan | Writes / minute | Retention |
|---|---|---|
| Free | 1,000 | 30 days |
| Team | 100,000 | 13 months |
| Enterprise | negotiated | negotiated |

A `429` is always safe to retry. Use the `Retry-After` value rather than your
own backoff so a recovering project is not slowed down by guesswork.

## Which header tells me my limit?

`X-Lumen-RateLimit-Remaining`, decremented on every write. It resets on a fixed
window, not a sliding one, so read it as "how much is left this window".

## Are read queries rate limited?

Yes, separately. Reads are limited at 10× the write rate because they are
cheaper for us to serve.
