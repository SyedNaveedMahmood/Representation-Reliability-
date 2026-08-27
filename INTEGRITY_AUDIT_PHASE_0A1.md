# Integrity Audit — Phase 0A.1

Scope: cache/probe data path of Phase 0A before any E01 work.
Method: adversarial regression tests written BEFORE fixes where required;
prior E00 probe results invalidated and re-run on a schema-v2 clean pipeline.

## Verdict table

| # | Issue | Confirmed? | Scientific impact | Fix | Regression test |
|---|---|---|---|---|---|
| 1 | Multi-shard X/metadata row ordering (`runners/probe.py`) | **Yes** — reproducer failed: *"row 0 (sample s001) is paired with another sample's vector"* | **Critical.** With ≥2 shards every probe/control design matrix paired activations with other samples' labels; E00 AUROC ≈0.53 was an artifact of label randomization. Invalidates `E00_bf9efb94222b`, `E00_18a016f37eb9`, `E00_50519d0b487d` probe metrics (runs preserved as evidence). TF-IDF text baseline unaffected (never touched activations) — reported separately. | Formal contract in `ActivationCacheReader.load_rows(requested)`: output row *i* ↔ requested metadata row *i* regardless of shard interleaving (positional write-back); `ShardedMatrixLoader` delegates to it with insertion-order filtering; no caller-side sorting | `tests/test_probe_alignment.py::test_multi_shard_rows_match_their_metadata` (3 shards, identity-encoded vectors, asserted per row) — failed pre-fix, passes post-fix |
| 2 | Shard-boundary/resume identity (`extraction/activations.py`) | **Yes** — old loop could close a physical shard while `batch_prompts` still held unprocessed units (`batch_size=8` crossed `units_per_shard=9`), so marker contents ≠ assumed arithmetic; resume used `prefix × units_per_shard`, i.e. counts, not facts | Interrupted runs would duplicate/skip exactly one work unit per boundary → silent sample/row corruption on resume | Batches no longer cross declared boundaries silently: at each boundary all pending forward-pass rows are flushed into the closing shard; `_complete.json` stores truthful `(unit_start, unit_end_exclusive)`; resume reads chained markers (validating `n_rows == Δunits·rows_per_unit`), prunes to the contiguous prefix | `tests/test_resume_identity.py`: cases A (bs=8<9), B (bs=16>9), C (partial final shard), D (deleted middle shard) — resumed cache must equal fresh extraction in IDs, selectors, sites, layers, vectors, cardinality |
| 3 | Random-label control protocol (`runners/e00.py`) | **Yes** — C selected against REAL validation labels while only train labels were shuffled | The null was contaminated by real-signal hyperparameter selection, weakening Gate-1 comparisons | Clean null: independent deterministic permutations of train AND validation labels (`probes.linear.randomized_control_labels`; both seeds recorded per control row); evaluation stays on real untouched discovery-test labels | `tests/test_controls_and_holdout.py::test_randomized_control_labels_independent_and_deterministic` |
| 4 | Confirmation-label access (`runners/e00.py`, `data/splits.py`) | **Yes** — discovery built a full-df label map incl. holdout rows; `validate_splits()` computed confirmation label sets; `samples.parquet` carried holdout labels | Spirit-of-holdout violation: labels existed in process memory/artifacts during discovery even though unused for fitting | `data.splits.build_discovery_label_map` refuses confirmation ids loudly (`ConfirmationSplitAccessError`); `validate_splits` marks confirmation `labels_observed=False`; discovery `samples.parquet` nulls holdout `target_label/truth_label`; runner records only a confirmation-ID digest + split hash | `tests/test_controls_and_holdout.py::test_discovery_label_map_refuses_confirmation_ids`, `::test_validate_splits_does_not_observe_confirmation_labels`; enforced end-to-end inside corrected runs |
| 5 | Stale-cache identity (`extraction/cache.py`, `activations.py`) | **Yes** — cache dir keyed only by config hash; generator/prompt/revision changes silently compatible; manifest revisions recorded "unspecified" | A cache from different prompts/revisions could masquerade as current data | Cache schema v2 namespace (`<exp>_v2_<hash>`): strict `cache_identity.json` (schema version, ordered dataset content hash, model/tokenizer resolved commits, sites/layers/selectors, dtypes, tokenization settings); reuse REFUSED on mismatch or missing identity (`LegacyCacheError`; nothing deleted) | `tests/test_resume_identity.py::test_cache_identity_mismatch_refuses_reuse`, `::test_legacy_cache_without_identity_never_adopted`, `::test_identity_covers_dataset_content_and_geometry` |
| 6 | Generation-config plumbing (`adapters/hf.py`) | **Yes** — adapter looked for `generation` under runtime config but it lives under model config; values merely coincided with defaults | Behavioral evaluation would run on implicit defaults regardless of YAML; decoder-only padding side unspecified | Adapter stores `model_cfg.generation`; `default_generation_kwargs()` reads it; tokenizer `padding_side="right"` explicit at load | `tests/test_generation_plumbing.py` (kwargs track `max_new_tokens`/`do_sample`) + padding-policy test |

## Additional repairs

- **Probe-path integrity**: probe loading previously checked only marker row
  counts, bypassing SHA verification. All probe-path loads now go through the
  reader's verified shard cache (sha256 once per process; validated tensors
  memoized). Test: `tests/test_probe_alignment.py::test_probe_path_verifies_shard_integrity`.
- **Misleading "round-trip" diagnostic renamed/strengthened**:
  `verify_alignment()` → `verify_reload_consistency()` (repeat-read check), plus
  a genuine source-vector→disk→reader multi-shard round-trip test (bit-exact
  fp32): `tests/test_cache_roundtrip.py::test_true_source_vector_roundtrip_multi_shard`.
- **Pre-v2 markers without unit ranges are non-interpretable**:
  `tests/test_cache_roundtrip.py::test_pre_v2_marker_without_unit_range_not_interpretable`.
- **Design-matrix identity assertions**: duplicates / missing / unexpected /
  length-mismatch all raise, with an expected-sample mapping supplied by the
  runner: `tests/test_probe_alignment.py::test_exact_identity_validation_against_expected_samples`.
- **site promoted to first-class dimension**: design identity is
  `site × layer × token_selector`; `site` propagates into probe metrics,
  controls, coefficient filenames (`{site}_{selector}_layer{NN}.npz`), figure
  names, and summaries: `tests/test_probe_alignment.py::test_two_sites_are_never_mixed`.
- **Resolved HF revisions**: after load, actual commit hashes are read from
  cached snapshot metadata (hub fallback attempted; explicit note if
  unavailable) and recorded in the run manifest AND cache identity. Unpinned
  `main` is never treated as pinned.
  Test: `tests/test_hf_extraction_contract.py::test_resolved_revisions_are_recorded`.

## Disproved suspicions

None — each of the six suspected issues was confirmed either by a minimal
reproducer (#1), by adversarial construction of the failure geometry (#2),
or by code-level inspection under the audit checklist (#3–#6).

## Consequence for prior results

The multi-shard ordering bug means every previously reported probe AUROC
(~0.5–0.55 incl. "best layers") paired activations with wrong labels. Those
numbers must be treated as invalid measurements, not negative evidence about
Qwen3-0.6B representations. The corrected, identity-validated runs
(`E00_b137cfabe7d3`, `E00_5162f18f1901`) show strong decodability (see
SUMMARY_SO_FAR.md §"Corrected E00"). TF-IDF surface-baseline results from the
old runs remain valid text-only measurements (no activation involvement); at
n=2000 they were ~chance (0.454 / 0.516), consistent with the anti-leakage
construction.

