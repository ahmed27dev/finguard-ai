"""
test_config.py - config.yaml must describe the code that actually runs.

The first version of this file listed a different NLI model, a chunk size in
tokens when the code counts characters, and four thresholds that no longer
existed. Nothing read it, so nothing caught the drift, and the report would
have quoted values the system never used.

These tests make the file load-bearing without refactoring every module to
read from it: change a constant and forget the file, and the suite fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CONFIG = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))


def test_model_names_match():
    from retrieval.indexer import MODEL_NAME as BI
    from retrieval.reranker import MODEL_NAME as CE
    from flagging.generator import MODEL_NAME as GEN
    from flagging.faithfulness_check import MODEL_NAME as NLI

    m = CONFIG["models"]
    assert m["bi_encoder"] == BI
    assert m["cross_encoder"] == CE
    assert m["generator"] == GEN
    assert m["nli_model"] == NLI


def test_chunking_matches():
    from retrieval.chunker import (
        TARGET_CHARS, MAX_CHARS, OVERLAP_CHARS, MERGE_MAX_CHARS,
        STANDALONE_MIN_CHARS,
    )

    c = CONFIG["chunking"]
    assert c["target_chars"] == TARGET_CHARS
    assert c["max_chars"] == MAX_CHARS
    assert c["overlap_chars"] == OVERLAP_CHARS
    assert c["merge_below_chars"] == MERGE_MAX_CHARS
    assert c["standalone_min_chars"] == STANDALONE_MIN_CHARS


def test_retrieval_matches():
    from retrieval.retrieve import CANDIDATES, TOP_K, MIN_SUPPORT_SCORE

    r = CONFIG["retrieval"]
    assert r["bi_encoder_top_k"] == CANDIDATES
    assert r["cross_encoder_top_k"] == TOP_K
    assert r["min_support_score"] == MIN_SUPPORT_SCORE


def test_parsing_matches():
    from retrieval.parser import SCALE_LOOKBACK

    assert CONFIG["parsing"]["scale_lookback_blocks"] == SCALE_LOOKBACK


def test_anomaly_thresholds_match():
    from flagging.anomaly_detector import (
        DIVERGENCE_MIN_GAP, SEGMENT_MIN_GAP, SUSTAINED_MIN_GAP,
        MAGNITUDE_THRESHOLD, MAGNITUDE_ALIGNED_PP, REVENUE_THRESHOLD,
    )

    a = CONFIG["anomaly_detection"]
    assert a["divergence_min_gap_pp"] == DIVERGENCE_MIN_GAP
    assert a["segment_min_gap_pp"] == SEGMENT_MIN_GAP
    assert a["sustained_min_gap_pp"] == SUSTAINED_MIN_GAP
    assert a["magnitude_threshold_pct"] == MAGNITUDE_THRESHOLD
    assert a["magnitude_aligned_pp"] == MAGNITUDE_ALIGNED_PP
    assert a["revenue_threshold_pct"] == REVENUE_THRESHOLD


def test_canonical_metric_names_match():
    """Names in the config must be the ones metric_extractor can produce."""
    from flagging.metric_extractor import CANONICAL_PATTERNS

    produced = {name for name, _ in CANONICAL_PATTERNS}
    listed = set(CONFIG["anomaly_detection"]["canonical_metrics"])
    assert listed == produced, (
        f"only in config: {listed - produced}; "
        f"only in code: {produced - listed}")


def test_faithfulness_matches():
    from flagging.faithfulness_check import ENTAILMENT_MIN

    assert CONFIG["faithfulness"]["entailment_min"] == ENTAILMENT_MIN