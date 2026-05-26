import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))


def test_save_results_uses_singleview_tag(tmp_path):
    from experiments.run_local_hr_benchmark import save_results

    path = save_results(
        tmp_path,
        {
            "mode": "degraded",
            "variant": "PD_MB_GN_JC_LR_CS",
            "method": "singleview",
            "metrics": {"ndcg@5": 0.1},
        },
    )

    assert path.name.endswith("_degraded_singleview_PD_MB_GN_JC_LR_CS.json")


def test_compute_metrics_uses_page_indices_as_relevance():
    import torch

    from experiments.run_local_hr_benchmark import compute_metrics

    scores = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.2, 0.1]])
    metrics = compute_metrics(scores, query_ids=[10, 11], relevant_pages={10: {1}, 11: {0}})

    assert metrics["ndcg@5"] == 1.0
    assert metrics["recall@5"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["n_queries"] == 2
    assert metrics["n_docs"] == 3
