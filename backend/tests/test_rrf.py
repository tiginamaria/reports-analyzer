from app.retrieval.search import rrf_merge


def test_rrf_prefers_items_in_both_lists():
    merged = rrf_merge([["a", "b", "c"], ["c", "d", "a"]])
    ids = [i for i, _ in merged]
    assert set(ids[:2]) == {"a", "c"}
    assert ids[-1] in ("b", "d")
