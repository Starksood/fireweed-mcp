"""Constants and thresholds validated across v15 experiments.

These values were tuned empirically against the Maya benchmark and validated
across 40+ experiments. Do not change these without re-validating against
the full benchmark suite.

Ported from v15: memory_loop.py, novel_domain_boost.py, exp041, exp040
"""

# Reinforcement formula weights
# r = 0.3 * local_frequency + 0.7 * cross_session_recurrence
LOCAL_FREQUENCY_WEIGHT = 0.3
CROSS_SESSION_WEIGHT = 0.7

# Novel-domain boost constants
NOVEL_DOMAIN_FIRST_MENTION_BOOST = 0.20
NOVEL_DOMAIN_CROSS_SESSION_FLOOR = 0.72  # 0.7 * 0.72 = 0.504 → crosses HOT threshold
NOVEL_DOMAIN_FIRST_MENTION_MAX_R = 0.62

# HOT/CORE layer thresholds
COLD_THRESHOLD = 0.40   # r < 0.40
WARM_THRESHOLD = 0.50   # 0.40 ≤ r < 0.50
HOT_THRESHOLD = 0.50    # 0.50 ≤ r < 0.87 (synthesis eligibility)
CORE_THRESHOLD = 0.87   # r ≥ 0.87

# Preflight gates (exp041)
PREFLIGHT_MIN_HOT_CORE_NODES = 6
PREFLIGHT_MIN_HOT_CORE_DOMAINS = 5
PREFLIGHT_MAX_FALSE_MERGE_RATE = 0.20

# Milestone B gates (exp040)
WIN_MIN_SEMANTIC_PRECISION = 0.50
WIN_MIN_SEMANTIC_TARGETS_HIT = 3
WIN_MAX_PERSONALITY_INFERS = 0
WIN_MAX_PARSER_FAILURE_RATE = 0.15

# Domain safety (exp026)
STRICT_DOMAIN_MERGE = True
FALLBACK_FLOOR = 0.55

# Entity-keyed canonical merge (two-clock canonicality). A tiny perceiver emits
# surface paraphrases of one fact ("a nurse at Riverside" / "a nurse at the Riverside
# clinic"); without merging them, a recurring fact never accumulates reinforcement.
# The resolver merges two claims that (a) resolve to the SAME entity set — referent
# identity is the entity linker's job, so "Paris" vs "Paris, Texas" (distinct
# entities) never merge — (b) share polarity, and (c) have content-token Jaccard at
# or above this floor, which separates true paraphrases (~0.6) from different
# predications of the same entities ("nurse" vs "doctor", ~0.5). See resolver.py.
CANON_MERGE_MIN_JACCARD = 0.55

# Validation lifecycle (Phase 5)
# A node transitions from "provisional" to "validated" after appearing in this many
# distinct sessions. 2 is the minimum for spacing-effect confidence.
VALIDATION_SESSION_THRESHOLD = 2

# Synthesis planner (Phase 5) — Jaccard-based pair filtering
# Pairs below SYNTHESIS_JACCARD_MIN are unrelated noise; above MAX are near-duplicates.
# These bounds come from ARCHITECTURE.md Stage 7 (text Jaccard variant).
SYNTHESIS_JACCARD_MIN = 0.05
SYNTHESIS_JACCARD_MAX = 0.65

# Synthesis runner (Phase 6)
# Inferences with LLM-reported confidence below this threshold are discarded.
# 0.60 is conservative — better to produce fewer, higher-quality inferences.
SYNTHESIS_MIN_CONFIDENCE = 0.60

# Inference node reinforcement floor.
# Synthesis-derived nodes start HOT immediately (same as novel-domain boost).
# 0.72 * 0.7 = 0.504 → crosses HOT threshold on first creation.
SYNTHESIS_INFER_CSR_FLOOR = 0.72

# Retrieval scoring weights — E2 Iteration 1 (kiro-task-036), restored E2 close
# Formula: score = W_JACCARD * jaccard + W_ENTITY * entity_precision + W_REINFORCE * reinforcement
# entity_precision = min(1.0, |expand_possessive(query_entities) ∩ node_entities| / |query_entities|)
# Iteration 1 rationale: raising W_JACCARD fixes motivation/multi-hop ranking; precision entity
# overlap removes multi-entity penalty that caused memory_pollution_02 regression at low W_REINFORCE.
# Re-validated against E1 eval suite: 16/21 Maya, 0 regressions vs 14-case baseline.
# E2 close: W_REINFORCE restored to approved §16.2 value (0.05); B.0 simulation confirmed
# r=0.05 is empirically identical to r=0.10 across all 21 Maya cases. W_JACCARD raised to
# 0.55 to maintain weight sum = 1.0 (0.55 + 0.40 + 0.05 = 1.00).
RETRIEVAL_W_JACCARD = 0.55
RETRIEVAL_W_ENTITY = 0.40
RETRIEVAL_W_REINFORCE = 0.05

# Jaccard abstention gate — E2 Iteration 2 (kiro-task-037)
# If the maximum Jaccard score across all candidates is below this threshold, abstain.
# Prevents entity-matched but off-topic nodes from producing spurious results.
# 0.12 is chosen between abstention_01 max_j (0.111) and paraphrase_recall_01 max_j (0.154).
# Re-validated: 19/21 Maya, 22/24 overall, 0 regressions.
RETRIEVAL_MIN_JACCARD_ABSTAIN = 0.12
# Most of the query's terms must be present for a claim to count as evidence. Paired with
# RETRIEVAL_MIN_JACCARD_ABSTAIN as an OR: jaccard catches broad topical overlap, coverage catches
# the short-query/long-claim case jaccard structurally cannot. 0.6 keeps a partial match out --
# "Maya salary" against a claim naming only Maya scores 0.5 and still abstains.
RETRIEVAL_MIN_COVERAGE_ABSTAIN = 0.6

# Claim extractor (X2) — writer-side LLM tenant constants
EXTRACTION_MAX_INPUT_CHARS = 8000
EXTRACTION_MAX_CLAIMS_PER_CALL = 20
EXTRACTION_MIN_CONFIDENCE = 0.3

# Adapter (X3) — project memory adapter
ADAPTER_GIT_LOG_DEFAULT_MAX_COMMITS = 1000  # safety cap on git log walks
ADAPTER_FILE_MAX_BYTES = 1_000_000  # 1MB; larger files are skipped with a clear error

# Temporal contradiction detection (X5)
POLARITY_OPPOSITION_THRESHOLD = 0.95       # Polarities must be very opposite (likes ↔ hates)
SCOPE_MATCH_THRESHOLD = 0.70               # Scopes must substantially match (same object)
CONTRADICTION_MAX_TIME_GAP_DAYS = 30       # Contradictions must be within 30 days
MODIFY_CONTRADICTION_CONFIDENCE_THRESHOLD = 0.65  # Min confidence to route to MODIFY

# Constitutional calibration (X3 Calibration Phase 2)
ABSTENTION_SIMILARITY_THRESHOLD = 0.50     # Minimum Jaccard similarity to answer (was 0.12, raising to 0.50)
ABSTENTION_EVIDENCE_QUALITY_WEIGHT = 0.60  # Weight for evidence quality in abstention scoring

# Percept buffer (two-clock perception→consolidation bridge)
# The buffer decouples Clock 1 (perception, LLM-bound, never blocks) from Clock 2
# (consolidation, Fireweed write-path, batched). See percept_buffer.py.
#
# Empirical knob behaviour — bench/run_coherence_staleness_sweep.py, gemma-3-1b
# stream, refines the original hypothesis:
#   * Final-state coherence is BATCH-INVARIANT. Duplicate claims collapse to the
#     same nodes at any batch size because the deterministic resolver dedups by
#     content at write time, not at the batch boundary. (Same 12 nodes from 25
#     accepted percepts whether batch_size=1 or 16.)
#   * Staleness grows ~linearly with batch_size and is CAPPED by max_idle_seconds
#     (the liveness floor). max_idle is the real freshness guarantee.
#   * Non-obvious coupling: 1 drain == 1 turn == 1 decay cycle, so SMALLER batches
#     mean MORE turns mean MORE decay between reinforcements — the graph forgets
#     faster. batch_size is effectively a metabolism-rate dial (the "nurse" claim
#     settled at r=0.28 with batch_size=1 vs r=0.70 with batch_size=16). Prefer a
#     larger batch to preserve reinforcement; use max_idle_seconds to bound
#     staleness; only drop batch_size when freshness must beat retention.
PERCEPT_BUFFER_CAPACITY = 256          # Max staged percepts before lowest-salience eviction
PERCEPT_BUFFER_BATCH_SIZE = 32         # Percepts drained per consolidation cycle (= 1 turn)
PERCEPT_BUFFER_MAX_IDLE_SECONDS = 2.0  # Drain even a partial buffer after this idle gap (liveness)

# Perceiver (Clock 1) — tiny-model robustness. A small perceptual model often omits
# the self-reported "confidence" field even when it extracts a good claim/evidence
# pair. Confidence is a self-report, not content, so a *missing* value defaults to
# this neutral midpoint rather than discarding the percept; a *malformed* value
# (wrong type) is still rejected as a model-confusion signal. See perceiver.py.
PERCEIVER_DEFAULT_CONFIDENCE = 0.5

# Claim-grounding guard (Clock 1 faithfulness). A creative tiny model can cite a real
# verbatim evidence span yet elaborate the CLAIM beyond it ("Riverside clinic" ->
# "...is a public health facility"). The verbatim-evidence check does not catch that,
# so we additionally require this fraction of a claim's content tokens (len>2) to
# appear in the source text; below it the percept is rejected as an over-claim. 0.6
# keeps faithful rephrasings ("works as a nurse" from "is a nurse", ~0.67) while
# rejecting invented detail ("public health facility", ~0.5). See perceiver.py.
PERCEIVER_MIN_CLAIM_GROUNDING = 0.6

# Significance — the meaning axis (Stage 3, W1). The original three-axis model is
# T(recency)/R(recurrence)/M(significance); R lives in Reinforcement, T in the decay
# side-table, and M here. The perceiver PROPOSES a rationale ("why this matters") and a
# cause ("the source clause this is caused-by"); this code DECIDES admission by grounding,
# so a fabricated motive is rejected (a fabricated motive is worse than a fact-list).
# Significance never mutates r or the Node schema — it is a side-table like decay. See
# significance.py.
RATIONALE_MIN_GROUNDING = 0.6          # frac of rationale content tokens that must occur in the source
CAUSE_MIN_CONTAINMENT = 0.8           # frac of cause-clause content tokens that must occur in the source
SIGNIFICANCE_RATIONALE_WEIGHT = 0.5   # alpha — how much a grounded rationale lifts significance over r
SIGNIFICANCE_CAUSAL_WEIGHT = 0.3      # beta  — how much each grounded causal link lifts significance
SIGNIFICANCE_CAUSAL_SATURATION = 3    # causal links past this add nothing (degree is normalized by it)
SIGNIFICANCE_RETRIEVAL_WEIGHT = 0.10  # +0.10*significance prior in retrieval scoring (rosetta_stone §16.2)

# Typed field edges (Stage 3, W2). A grounded cause clause (W1) is promoted to a typed
# `causes` edge only if it matches an existing active node's claim by at least this content
# overlap — otherwise it stays a dangling annotation (an unlinked truth beats an invented edge).
CAUSE_EDGE_MIN_MATCH = 0.5
MOTIVATES_EDGE_MIN_MATCH = 0.5   # same gate for promoting a grounded rationale to a `motivates` edge

# Opportunity-scored consolidation ops (Stage 3, W4). FREEZE protects meaningful-but-not-yet
# permanent memories from decay: a node with grounded significance prior >= this, below CORE,
# is made decay-immune. Significance (M) shields against forgetting (T). Budget bounds the
# (cheap, idempotent) op per turn — the same score->rank->spend shape REFLECT/COMPRESS will use.
FREEZE_MIN_SIGNIFICANCE = 0.4
FREEZE_BUDGET_PER_TURN = 8

# REFLECT (W4): the self observes a PATTERN across a cluster of its own facts (evidence->pattern,
# NOT personality-overreach — the is_personality_claim guard filters that). A reflection is
# admitted only if it covers >=2 cluster facts, its content tokens are >= REFLECT_MIN_GROUNDING
# present across the cluster, it is not a restatement of any single fact, and it is novel vs
# existing reflections. Reflections start tentative (low r) and fade unless re-observed. Budget
# keeps the (LLM) op bounded per turn.
REFLECT_MIN_CLUSTER = 3
# Lower than W1's rationale floor (0.6): a rationale tracks ONE fact closely, but a reflection
# GENERALIZES over many and so legitimately introduces connective words. 0.4 (stemmed) admits a
# genuinely grounded pattern (a live gemma-3-4b run grounded a real food-prep pattern at 0.45)
# while still rejecting over-abstract narration (0.08-0.33).
REFLECT_MIN_GROUNDING = 0.4
REFLECT_CSR_FLOOR = 0.5          # -> r ~= 0.35 (tentative; below WARM, like v15 reflections)
REFLECT_BUDGET_PER_TURN = 1
# Lexical novelty ceiling for a new reflection vs existing ones. Was 0.80; the 212-commit ops run
# produced three reflections of the same observation, the closest pair scoring 0.714 — under the old
# ceiling. 0.65 rejects those. Semantic duplicates (one pair sat at 0.44 lexically) need the encoder
# check in consolidation_ops._too_similar_semantically; no lexical threshold can reach them.
REFLECT_NOVELTY_MAX = 0.65

# COMPRESS (W4): fold a cluster of COLD, forgettable, MEANINGLESS facts into one grounded
# summary so the graph metabolizes clutter instead of drowning in it. Anti-destruction guards
# (ported from v15 memory_loop.py:1577-1647) are non-negotiable: min group of 3; only COLD
# nodes (r < COMPRESS_MAX_R); never a frozen/disputed/inference/reflection node, and never one
# carrying grounded significance (those are protected by W1/W3/W4-FREEZE). Sources are marked
# superseded-by the summary (non-destructive; provenance preserved), not deleted.
COMPRESS_MAX_R = 0.40            # only fold genuinely COLD nodes (below WARM)
COMPRESS_MIN_GROUP = 3
COMPRESS_MIN_GROUNDING = 0.5     # a summary condenses (not generalizes), so it must stay close to source
COMPRESS_CSR_FLOOR = 0.6        # -> summary r ~= 0.42 (WARM-ish; survives, won't instantly re-compress)
COMPRESS_BUDGET_PER_TURN = 1

# Recursive self-modeling (Stage 3, W5). A later observation RESOLVES an open prediction only if
# it shares the prediction's subject and at least this fraction of the observation's content
# tokens — keeping resolution precise (few false matches) at the cost of recall.
PREDICTION_RESOLVE_MIN_OVERLAP = 0.4

# Three-axis forgetting (Stage 4, pillar 4). Decay made T/R/M explicit: T (immutable) = CORE
# (r>=CORE_THRESHOLD) + FROZEN nodes never decay; R = the linear reinforcement decay; M (slow
# drift) = grounded significance dampens the per-turn decay rate, so a meaningful memory fades
# more slowly than a bare one of equal r. Damping x significance_prior (<=~0.8) scales the
# slowdown; 0.6 makes a maximally-significant sub-freeze node decay ~half as fast.
SIGNIFICANCE_DECAY_DAMPING = 0.6

# Decay — "forgetting as metabolism" (Clock 2 consolidation step).
# Ported from v15 memory_loop.py MemoryFabric.decay()/_compute_decayed_r().
# Decay runs once per consolidation turn (1 drain = 1 turn). Reinforcement (r)
# bleeds linearly from its value at last access while a node is idle; identity
# (CORE) nodes are immune. See decay.py.
DECAY_BASE_PER_TURN = 0.012            # r lost per idle turn before polarity modulation
DECAY_SURVIVAL_FLOOR = 0.25            # Nodes younger than FLOOR_TURNS cannot fall below this
DECAY_SURVIVAL_FLOOR_TURNS = 5         # Grace period (turns) for newborn nodes
DECAY_HOT_STABILIZATION_TURNS = 3      # Turns a freshly-HOT node decays slower (post-tetanic window)
DECAY_HOT_STABILIZATION_FACTOR = 0.80  # Decay multiplier during the HOT stabilization window
# Polarity modulation — v16 has no scalar valence, so categorical Predicate.polarity
# maps to the v15 valence behaviour: emotionally-charged claims (positive OR negative)
# persist; neutral facts fade fastest. Lower multiplier = slower decay.
DECAY_POLARITY_MODIFIER_POSITIVE = 0.80
DECAY_POLARITY_MODIFIER_NEGATIVE = 0.60
DECAY_POLARITY_MODIFIER_NEUTRAL = 1.00
# Confidence decay for inference nodes (guesses fade unless re-confirmed).
INFER_CONFIDENCE_DECAY_TURNS = 10      # Idle grace before inference confidence starts decaying
INFER_CONFIDENCE_DECAY_RATE = 0.10     # Confidence lost per decay interval
INFER_CONFIDENCE_DECAY_INTERVAL = 5    # Turns between confidence decay steps

# ── Read Gate (docs/DESIGN_read_gate.md) ──────────────────────────────────────
# Predicate grounding: minimum cosine between the question's demand head and any single token of
# the substrate's own vocabulary for the head to count as a paraphrase of a grounded predicate.
# MEASURED, not tuned to taste: the answerable/abstain classes separate at 0.465 -> 0.551 on the
# demo + Maya corpora (design doc §2b), so 0.50 sits in the gap. Token-to-token — comparing against
# whole CLAIMS interleaves the classes at every threshold.
READ_GATE_MIN_PREDICATE_SIM = 0.50
# Document-frequency cap above which a token is genuinely non-discriminative and is dropped from
# the grounding test. Tokens BELOW the cap that are absent from the corpus are UNGROUNDED, never
# dropped — that sign is the review's Correction 1.
READ_GATE_DF_CAP = 0.5

# Predicate grounding in DOCUMENT mode (docs/FINDING_predicate_fabrication.md): minimum cosine
# between an unsupported claim token and any evidence token. MEASURED, not chosen: fabrications land
# at 0.240-0.359 and legitimate rewording ("monthly" for "per month") at 0.819. 0.55 sits in the gap.
PREDICATE_MIN_SUPPORT = 0.55
