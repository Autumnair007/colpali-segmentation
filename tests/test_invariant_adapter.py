import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import pandas as pd
import pytest
import torch
import torch.nn.functional as F


def test_query_group_split_keeps_raw_answer_groups_together():
    from experiments.invariant_splits import build_query_groups, flatten_query_ids, split_query_groups

    selected = pd.DataFrame(
        {
            "query_id": list(range(18)),
            "query": [f"q{i}" for i in range(18)],
        }
    )
    all_queries = pd.DataFrame(
        {
            "query_id": list(range(18)),
            "language": ["english", "french", "spanish"] * 6,
            "raw_answers": [[f"answer-{i // 3}"] for i in range(18)],
        }
    )

    groups = build_query_groups(selected, all_queries, seed=13)
    splits = split_query_groups(groups)

    seen = {}
    for split, split_groups in splits.items():
        for group in split_groups:
            for query_id in group["query_ids"]:
                seen[query_id] = split
    for answer_idx in range(6):
        split_names = {seen[query_id] for query_id in range(answer_idx * 3, answer_idx * 3 + 3)}
        assert len(split_names) == 1
    assert sorted(flatten_query_ids(groups)) == list(range(18))


def test_variant_split_holds_out_main_variant():
    from experiments.invariant_splits import HELD_OUT_MAIN_VARIANT, split_variants

    components = ["PD", "MB", "GN", "JC", "LR", "CS"]
    variants = []
    for mask in range(1, 2 ** len(components)):
        variants.append("_".join(component for idx, component in enumerate(components) if mask & (1 << idx)))

    splits = split_variants(variants, seed=13)

    assert HELD_OUT_MAIN_VARIANT in splits["test"]
    assert HELD_OUT_MAIN_VARIANT not in splits["train"]
    assert HELD_OUT_MAIN_VARIANT not in splits["val"]
    assert not (set(splits["train"]) & set(splits["val"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["val"]) & set(splits["test"]))


def test_residual_adapter_starts_as_identity_for_normalized_embeddings():
    from robust.invariant_learning import ResidualEmbeddingAdapter

    adapter = ResidualEmbeddingAdapter(dim=4, hidden_dim=8)
    embeddings = F.normalize(torch.randn(5, 4), p=2, dim=-1)
    output = adapter(embeddings)

    assert output.shape == embeddings.shape
    assert torch.allclose(output, embeddings, atol=1e-6)
    assert torch.allclose(output.norm(dim=-1), torch.ones(5), atol=1e-6)


def test_invariant_losses_are_finite_with_multi_positive_qrels():
    from robust.invariant_learning import ResidualEmbeddingAdapter, invariant_adapter_loss, late_interaction_scores

    adapter = ResidualEmbeddingAdapter(dim=4, hidden_dim=8)
    queries = [F.normalize(torch.randn(3, 4), p=2, dim=-1), F.normalize(torch.randn(2, 4), p=2, dim=-1)]
    clean_docs = [F.normalize(torch.randn(4, 4), p=2, dim=-1) for _ in range(3)]
    degraded_docs = [F.normalize(torch.randn(4, 4), p=2, dim=-1) for _ in range(3)]
    adapted_docs = [adapter(doc) for doc in degraded_docs]

    clean_scores = late_interaction_scores(queries, clean_docs, device="cpu", doc_batch_size=2)
    adapted_scores = late_interaction_scores(queries, adapted_docs, device="cpu", doc_batch_size=2)
    losses = invariant_adapter_loss(
        adapted_scores=adapted_scores,
        clean_teacher_scores=clean_scores,
        relevant_pages=[{0, 2}, {1}],
        original_embeddings=degraded_docs,
        adapted_embeddings=adapted_docs,
    )

    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.score_distill)
    assert torch.isfinite(losses.qrels_rank)
    assert torch.isfinite(losses.identity)


def test_embedding_cache_accepts_arbitrary_variant_and_rejects_mismatch(tmp_path):
    from experiments.invariant_embeddings import embedding_cache_path, load_embedding_cache, save_embedding_cache

    path = embedding_cache_path(tmp_path, "doc", "degraded", "GN_JC")
    page_paths = [tmp_path / "page_001.png"]
    save_embedding_cache(path, "doc", "degraded", "GN_JC", page_paths, [torch.randn(2, 4)])

    payload = load_embedding_cache(path, "doc", "degraded", "GN_JC", page_paths=page_paths, expected_count=1)
    assert len(payload["embeddings"]) == 1
    with pytest.raises(ValueError, match="variant"):
        load_embedding_cache(path, "doc", "degraded", "GN", page_paths=page_paths)
