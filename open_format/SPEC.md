# The Fireweed Memory Protocol (FMP) — normalization spec v1

**Status:** the open half of the open-core boundary. This document, `reference_reader.py`,
`conformance.py` and the trap corpus are permanently open; the engine that DECIDES what enters a
substrate is not. The split is exact: **everything that describes is here, everything that
adjudicates is private.**

**Why it exists:** weights obsolesce in a decade. A record that can only be read by the vendor that
wrote it is not a record, it is a rental. An FMP snapshot must stay readable with nothing but a JSON
parser and this document — which is why the reference reader imports only the standard library, and
why that constraint is a conformance check rather than a preference.

---

## 1. Container

A snapshot is one UTF-8 JSON object. Top level:

| field | type | meaning |
|---|---|---|
| `snapshot_version` | int | format version. **2** is current. A reader MUST refuse a version it does not know. |
| `fireweed_version` | string | the writer's engine version. Informational; never load-bearing. |
| `nodes` | array | the claims. §2 |
| `entities` | array | the things claims are about. §3 |
| `relations` | array | typed edges. §4 |
| `sessions_seen`, `total_sessions`, `seen_domains`, `session_timestamp`, `session_anchor`, `ingested_session_ids` | — | writer bookkeeping. A reader MAY ignore all of it. |

**Refusing is mandatory, not optional.** A reader that silently reads an unknown version is how a
format rots: it will misinterpret fields that changed meaning and report the result as fact.

## 2. Nodes (claims)

Required: `node_id` (unique), `claim`. Load-bearing optional fields:

| field | meaning |
|---|---|
| `normalized_claim` | canonical form used for matching. Derived; never authoritative over `claim`. |
| `node_type` | `fact` \| `event` \| `state` \| `preference` \| `constraint` \| `inference` \| `reflection` \| `summary` |
| `status.memory_state` | `active` \| `disputed` \| `superseded` \| `quarantined` \| `frozen` |
| `entities[].entity_id` | references §3 |
| `domains` | topical tags, unordered |
| `provenance` | §5 |

**The memory-state contract.**

- `active` and `disputed` are both READABLE. A `disputed` node is held in standing contradiction,
  not retired: both sides of an unresolved conflict stay answerable, because silently picking a
  winner is a decision the format must not make on the reader's behalf.
- `superseded` nodes are **retained in the file, never deleted.** A reader can always reconstruct
  what was once believed and when it stopped being believed. Belief revision is part of the record.
- Erasure is the one exception: erased content is genuinely gone, replaced by a tombstone. That is a
  deliberate asymmetry — supersession is history, erasure is a promise.

## 3. Entities

`entity_id` (unique), `canonical_name`, `entity_type`
(`person` \| `place` \| `organization` \| `object` \| `concept` \| `event`), `aliases`.

> **Known defect, stated rather than hidden:** writers at time of writing emit `person` for nearly
> every entity, including organizations and objects. `entity_type` is therefore NOT yet trustworthy
> and a reader must not build behaviour on it. Tracked in `docs/DESIGN_read_gate.md` §5, where it is
> the blocker for object typing.

## 4. Relations

`relation_id`, `relation_type`, `source_id`, `target_id`.

**The list is heterogeneous, and the endpoint domain depends on the type.** This is the one place a
naive reader will get it wrong — the first run of the conformance suite reported four "dangling"
relations that were simply entity edges:

| relation_type | `source_id` / `target_id` refer to |
|---|---|
| `supersedes`, `contradicts`, `supports`, `derived_from`, `causes`, `motivates`, `before` | **node** ids |
| `co_occurs` | **entity** ids |

A reader MUST resolve endpoints in the domain the type declares.

## 5. Provenance and receipts

| field | meaning |
|---|---|
| `source_turn_id` | the turn or document-derived id the claim came from |
| `source_span` | the **verbatim** evidence the claim was admitted on |
| `confidence` | the proposer's confidence. Never a gate; informational only. |
| `grounding_class` | `grounded_verbatim` (subject named in the cited span) or `grounded_resolved` (subject resolved from outside it) |
| `doc_hash`, `byte_start`, `byte_end` | the receipt coordinate, when the claim is document-bound |

**A receipt exists only when `doc_hash`, `byte_start` and `byte_end` are ALL present.** A partial
coordinate is not a weak receipt — it is no receipt. Reporting one would be exactly the fabrication
the format exists to prevent, and the conformance suite checks it.

**Verification** is two steps, and both must pass:

1. `"sha256:" + sha256(document)` equals `doc_hash`
2. `document[byte_start:byte_end]` decodes to `source_span`

Change any byte of the source and one of the two fails. That is the whole provenance guarantee, and
it is checkable by anyone holding the document — no engine, no network, no trust.

## 6. Conformance

    python open_format/conformance.py <snapshot.json>

20 checks. It does not trust the reference reader: it states properties of the FORMAT and runs them
against whatever reader it is handed, so passing with your own implementation means yours conforms.
Three of the checks assert the reader REFUSES malformed input, and two assert receipts FAIL on a
tampered source — a suite that can only pass is not evidence.

## 7. Stability

Within a major version: fields may be ADDED; existing field meanings never change. A reader must
ignore fields it does not recognise. Removing or repurposing a field requires a version bump, which
existing readers are required to refuse.
