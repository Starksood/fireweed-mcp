# fireweed-mcp

<!-- mcp-name: io.github.Starksood/fireweed-mcp -->

Agent memory where every fact carries a receipt.

```
remember(claim    = "Priya joined Acme in 2019 under duress.",
         evidence = "Priya Raman joined Acme in 2019 as a logistics analyst.")

REFUSED (asserts_more_than_evidence) — the claim adds something the evidence does not say.
  claim   : Priya joined Acme in 2019 under duress.
  evidence: Priya Raman joined Acme in 2019 as a logistics analyst.
```

```
recall("Priya's salary")

ABSTAINED (unknown_predicate) — no claims ground "salary"; 1 claim about Priya Raman exists
This is a refusal, not an empty result.
```

```
forget("Priya")

ERASED Priya Raman — certificate issued
  signature            : hmac-sha256:f4d0768ef3b0fec624afec12f25bfd91…
  nodes in closure     : 1
  every probe abstains : True
  bystanders surviving : 1
```

That last one is the artifact behind *"delete me from your agent's memory — and prove it."*

## Install

```bash
uvx fireweed-mcp          # try it
pip install fireweed-mcp  # keep it
```

```bash
claude mcp add fireweed -- uvx fireweed-mcp
```

No dependencies. No API keys. No model — nothing in this server calls an LLM.

## What it does

| tool | |
|---|---|
| `remember` | admits a claim **only if the evidence you cite supports it**. Refusals are typed and say what to fix. |
| `recall` | grounded claims **with the byte range they came from**; abstains and names the term it could not ground |
| `verify_receipts` | re-hash every source, re-slice every range — **tamper-evident** |
| `forget` | erasure with exact closure and a **signed certificate**; bystanders survive |
| `export_memory` | the whole substrate as a portable open-format blob |

## Why the refusals are the point

Most memory servers store what the model says and return what's nearest. This one **adjudicates**.

The rule is *the model proposes, deterministic code decides.* Across an RPC boundary that stops
being a slogan: **your agent is the proposer**, and it cannot talk its way past the gate, because
the gate is not a prompt. Pass a claim and the text you're quoting; pure functions check that the
evidence names the subject, preserves the relation, invents no numbers, and asserts nothing the
span doesn't say. What survives is stored with a byte range into the source.

Then anyone can check it afterwards — including someone who trusts neither your agent nor this
server. That is the whole product.

## What it does NOT do

Stated up front, because this project's last headline number turned out to be measuring nothing
(see [the retraction](https://github.com/Starksood/Fireweed_Fabric/blob/main/RETRACTION.md), which
ships with a script that proves it):

- **It does not extract memories from free text.** You supply the claim and the evidence.
  Automatic extraction needs a perceiver model; this server deliberately has none.
- **It does not make an LLM truthful.** It governs what enters the *record* and what can be proven
  about it. Your model can still say whatever it likes in its own prose.
- **Recall is weak, and measured on both axes.** On questions whose answer is genuinely absent, the
  gate abstains on only **38%** — it checks that the question's *topic* is grounded, not that the
  asked-for *value* exists. On a paired corpus of questions the stored facts *do* answer, it returns
  the right answer **24%** of the time against **75%** for plain retrieval-and-read. Both numbers
  come from a calibrated instrument that prints its own controls before measuring; the method and
  the failures behind it are written up in the private repo's evaluation notes.
  **The write path, receipts and erasure are unaffected and are the parts to rely on.**

## Your data

`~/.fireweed/mcp/` (`FIREWEED_MCP_STORE` to change). The substrate is an open format — see
[`open_format/SPEC.md`](open_format/SPEC.md) — and `open_format/reference_reader.py` reads it with
the standard library alone. Your memory outlives this server, this engine, and any model. A test
asserts that round trip.

Optional: `pip install "fireweed-mcp[semantic]"` enables paraphrase matching in `recall`. Without
it the gate refuses *more* — the safe direction — and `memory_stats` tells you which mode you're in.

## License

**FSL-1.1-ALv2** — source-available. Free for everything except building a competing product;
converts to **Apache 2.0 on 2028-01-01**. Full text in [`LICENSE.md`](LICENSE.md).

Want to use Fireweed in a commercial product or competing service? → **sanyamsood2@gmail.com**
