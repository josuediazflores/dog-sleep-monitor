# Feeding PupLog from the Pi

How sleep sessions detected on a Raspberry Pi reach the PupLog iOS app.

## Why the Pi does not write to CloudKit

The obvious design is for the Pi to write `Event` records straight into the
CloudKit zone the app already syncs. It does not work.

CloudKit Web Services offers two ways to authenticate. A **server-to-server
key** is the only one an unattended machine can hold, and it can only reach the
**public** database. Reaching a private database requires an API token plus a
web auth token obtained by a person signing in with an Apple ID, which a
headless Pi cannot do. Apple's developer relations state this directly: a
server-to-server key authenticates "to make API calls to a public database", and
private-database access needs an API token.

PupLog keeps everything in `PupLogZone` in the owner's **private** database,
shared zone-wide via `CKShare`. So the Pi is locked out by design, and putting
dog data in the public database to work around that would make it readable by
any authenticated user of the app and would break the sharing model entirely.

## The architecture

The app stays the only CloudKit writer.

```
  Pi                                  iPhone
  ┌──────────────────┐                ┌─────────────────────────┐
  │ watch  ──> CSV   │                │  puller                 │
  │ serve  ──> JSON  │ ──HTTP GET──>  │    ↓                    │
  └──────────────────┘   over the     │  PupStore.addEvent      │
     read-only          tailnet       │    ↓                    │
                                      │  CloudSync ──> CloudKit │
                                      └─────────────────────────┘
                                                        ↓
                                            other parents' phones
```

This preserves everything already built: the `CKShare` participant model,
the documented conflict rules, `CKSyncEngine` retry and offline queues, and the
fact that no new credential has to exist anywhere in CloudKit.

## Endpoints

Base URL is the Pi's tailnet address, e.g. `http://100.x.y.z:8787`.

| method | path | auth | returns |
| --- | --- | --- | --- |
| GET | `/health` | none | `{"ok":true}` and nothing else |
| GET | `/v1/state` | bearer | current state, for the live sleep banner |
| GET | `/v1/events?since=&min_minutes=&merge_minutes=` | bearer | completed sleep sessions |
| GET | `/v1/frame.jpg` | bearer | the current camera still, `image/jpeg` |

Anything other than GET returns 405. There is no write path at all.

### `/v1/frame.jpg`

The frame `watch` last sampled, written atomically to a fixed path so a reader
never catches a half-written jpeg. It is the clean frame -- the green ROI
rectangle drawn on `snapshots/` is a debugging aid and does not appear here.

Served `no-store`. That is not tuning: browser clients cache-bust with a
counter starting at zero, so a remount repeats the same URL and a cached
response would pin the image to whatever was current the first time.

`503` in two cases, both deliberate: no frame has been produced yet, or the
newest frame is older than `frame_max_age_s` (default 3600). Serving last
night's picture behind a fresh-looking panel is worse than serving nothing.

`X-Frame-Timestamp` and `X-Frame-Age-S` are on the response for `curl` and for
service-to-service callers. **A browser cannot read them** -- headers of an
`<img>` load are not exposed to JavaScript -- so a UI must take freshness from
`/v1/state` instead, and gate whether it sets `src` at all on what that says.

An `<img>` also cannot send an `Authorization` header. A browser-facing
consumer therefore needs a same-origin proxy that holds the bearer token
server-side; do not weaken the auth on this route to work around it. It is the
most sensitive thing this program produces.

`/health` is deliberately unauthenticated and deliberately empty: it proves the
process is alive and reveals nothing about a dog, a home, or a version number
worth pivoting on.

### `/v1/events`

```json
{
  "events": [
    { "id": "c-1786572214000", "kind": "sleep",
      "ts":    "2026-08-12T22:03:34Z",
      "start": "2026-08-12T22:03:34Z",
      "end":   "2026-08-12T22:06:09Z",
      "source": "camera", "partial": false, "stirs": 1, "duration_s": 155 }
  ],
  "count": 1,
  "server_time": "2026-08-12T22:55:27Z",
  "state": { "state": "awake", "since": "...", "stale": false }
}
```

Each event decodes directly into `PupEvent`. A synthesized `Codable` decoder
ignores unknown keys, so `source`, `partial`, `stirs`, and `duration_s` cost the
app nothing while remaining available to a richer client.

```swift
struct Feed: Decodable { let events: [PupEvent]; let serverTime: Date }

var request = URLRequest(url: base.appending(path: "v1/events"))
request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
let (data, _) = try await URLSession.shared.data(for: request)
let decoder = JSONDecoder()
decoder.dateDecodingStrategy = .iso8601
decoder.keyDecodingStrategy = .convertFromSnakeCase
let feed = try decoder.decode(Feed.self, from: data)
```

Dates are UTC with a `Z` suffix, which `.iso8601` handles without a custom
formatter.

### Idempotency, which is the important part

`id` is `c-<start-instant-in-ms>`, derived from the data rather than assigned.
Pulling the same session twice produces the same id both times.

That matters because `PupStore.applyCloudChanges` already merges events with
`Dictionary(userEvents.map { ($0.id, $0) }, uniquingKeysWith:)`, and
`SYNC-PLAN.md` records the rule as "events are immutable and append-only; sync =
set union by id". So a re-pull is a no-op, a partial pull self-heals, and the
puller needs no cursor bookkeeping or dedupe logic. Use `since` only to keep
responses small.

The `c-` prefix also distinguishes camera-detected events from `u-` user events
and `s-` seed data, which gives the UI a way to attribute them and gives
"reset my entries" a way to leave them alone (or not).

### Two flags worth honoring

- **`partial: true`** means the session touched a stretch where the monitor was
  not running, so the end time is a lower bound rather than an observation. Show
  it differently or drop it, but do not present it as measured.
- **`state.stale: true`** means no fresh sample has arrived. The server keeps
  answering after `watch` dies, so without checking this the app would happily
  report "asleep for 6 hours" when really nothing has been watching for 6 hours.

## Semantics decided on the server

Two knobs, in `config.json`, because they are judgments about dogs rather than
about transport:

| setting | default | effect |
| --- | --- | --- |
| `api_min_session_minutes` | 10 | shorter stretches of stillness are not reported as sleep |
| `api_merge_stirs_minutes` | 10 | an awake span shorter than this merges the sleeps on either side into one, counted in `stirs` |
| `frame_max_age_s` | 3600 | past this `/v1/frame.jpg` returns 503 rather than serving a stale picture as current |
| `api_log_tail_bytes` | 131072 | `/v1/state` parses only the tail of the log; `/v1/events` still reads it whole |

### Liveness: `stale` says the reading is old, `monitor` says why

`/v1/state` carries two extra objects. They exist because the server reads CSVs
off disk and has no idea whether anything is producing them, and because the
feed-down path writes **no log row at all** -- so "the camera is unplugged" and
"nothing has been watching" produce an identical-looking staleness.

```json
"monitor": { "alive": true, "pid": 7415, "beat_age_s": 3,
             "feed_ok": false, "feed_down_for_s": 1840, "uptime_s": 5121 },
"frame":   { "available": false, "ts": "2026-08-13T10:12:44", "age_s": 14987 }
```

Read them together:

| `monitor.alive` | `monitor.feed_ok` | means | say |
| --- | --- | --- | --- |
| true | true | everything working | live |
| true | false | monitor up, **camera** down | camera offline |
| false | — | **monitor** dead or wedged | not being watched |

`alive` comes from a heartbeat file rewritten on every loop iteration including
the feed-down branch, with three sample intervals of slack. A client that shows
a sleep banner without checking it will cheerfully report "asleep for 6 hours"
when nothing has been running for 6 hours.

### Presence: `state` says "unknown", `presence` says why

```json
"presence": { "value": "occupied", "reliable": false, "ref_corr": 0.53,
              "references": 5, "trust_floor": 0.6,
              "source": "model", "p_dog": 0.9731, "model_error": null,
              "shadow": false }
```

`shadow: true` means a model is loaded and `p_dog` is real, but the
references still produced `value` (`source` will say `"reference"`). It is a
rehearsal; render `p_dog` as information, not as the verdict.

`state: "unknown"` is one word for two situations that need different
sentences. When `presence.reliable` is `false`, every empty-pen reference has
stopped matching the scene (camera moved, crate rearranged, a lighting
condition with no capture) and both `away` and `asleep` are deliberately
disabled until `reference` is re-run on an empty pen -- that is an actionable
outage, not a transient. When `reliable` is `true` or `null`, "unknown" is
just the machine between verdicts. `ref_corr` is the frame's correlation with
the closest reference; render it against `trust_floor` if you want to show how
far off the scene is. `reliable: null` means not measured (no sample yet this
run, or a heartbeat from an older monitor).

**`source` says which layer produced `value`**, and it decides which of the
other fields mean anything:

| `source` | what is answering | the field that matters |
| --- | --- | --- |
| `"model"` | the learned dog/no-dog classifier | `p_dog` |
| `"reference"` | diffing against stored empty-pen frames | `reliable`, `ref_corr` |
| `"none"` | nothing; presence is not measured | neither |
| `null` | a heartbeat written before this field existed | neither |

Under `"model"`, `reliable` is always `true` and `ref_corr` is along for the
ride: the classifier never reads the references, so their staleness says
nothing about whether it can see a dog, and a client that greys out presence on
`ref_corr` alone will hide a layer that is working fine. Under `"none"` there
is no presence layer at all and an empty pen reads as asleep, which is worth
saying out loud rather than rendering as a transient "unknown".

`p_dog` is the classifier's raw probability for the last sample, before the
deadband turned it into a verdict. `null` when no model is running, or when
that sample's forward pass failed. Values near 0.5 are the model declining to
vote, which is why `value` can sit still while `p_dog` moves.

`model_error` is `null` in the normal case. A string means a model is
configured but not running -- a missing file, an ONNX op the Pi's OpenCV
cannot import, a forward pass that threw -- and presence has quietly fallen
back to reference-diff, so `source` will read `"reference"` or `"none"`
alongside it. Treat it exactly like `reliable: false`: an actionable outage
with a human fix, not a transient. The monitor keeps watching either way.

### Session: `elapsed_s` resets on every stir, `session` does not

```json
"session": { "kind": "sleep", "start": "2026-08-21T03:11:02Z",
             "elapsed_s": 4380, "stirs": 2, "in_stir": false }
```

Dogs are polyphasic: a reposition mid-sleep flips `state` to `awake` for a
minute and resets the top-level `since`/`elapsed_s`, so an hour of sleep with
one stir reads "asleep 3m". `session` is the merged in-progress sleep block --
the same merge `/v1/events` applies retrospectively (awake/unknown gaps under
`api_merge_stirs_minutes` fold in as stirs), reported live. `null` whenever no
sleep block is open: she is properly awake, away, or the block was closed by a
data gap. `in_stir: true` means she is moving right now but the movement has
not yet outlasted the merge window -- render "stirring", not "woke up". The
block only becomes a `/v1/events` entry once it closes; an open block is never
exported there, so the two views never disagree about a finished sleep.

So a night reads as one 7h04m sleep with 5 stirs, not six separate sleep events.
A data gap always breaks a merge, since sleep across an unobserved stretch is an
assumption, not a measurement.

## Security

**Transport: put it on the tailnet, not the LAN.** WireGuard gives encryption
and device identity for free, works from anywhere without port forwarding, and
means the endpoint is not exposed to other devices on the home network.

```bash
# on the Pi
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale ip -4          # the address the app will use
```

Install Tailscale on the iPhone from the App Store and sign in with the same
account. Then bind the server to the tailnet address:

```bash
MONITOR_API_TOKEN=... python3 monitor.py serve --bind 100.x.y.z
```

**Auth: a bearer token**, compared with `hmac.compare_digest` so a wrong token
cannot be recovered byte by byte from response timing. Generate one and keep it
out of the repo:

```bash
echo "MONITOR_API_TOKEN=$(openssl rand -hex 32)" >> .env
```

Store it in the iOS **Keychain**, not `UserDefaults` and not source. The server
refuses to start without a token of at least 24 characters.

**What this does not protect against**, stated plainly:

- Plain HTTP on a bare LAN sends the token in the clear to anything sniffing that
  network. On a tailnet WireGuard covers it. For real TLS, `tailscale serve
  --https` issues a certificate.
- The token is a single shared secret with no rotation or per-device revocation.
  Rotating it means editing `.env`, restarting, and updating the phone.
- Anyone holding the token learns when the dog is unattended. That is the actual
  sensitivity of this endpoint and the reason it is not served openly.

## Still to build, on the app side

1. A puller: `URLSession` fetch on `scenePhase == .active` and on a timer, into
   the existing store. The app has no `URLSession` anywhere today, so this is
   genuinely new code.
2. Keychain storage for the base URL and token, plus a settings row to enter them.
3. Attribution in the UI: whether a camera-detected sleep looks different from a
   hand-logged one, and whether it is editable or deletable.
4. Whether the live `/v1/state` drives the existing `ActiveSession` banner or
   only shows up after a session completes.

Items 3 and 4 are product decisions, not technical ones.
