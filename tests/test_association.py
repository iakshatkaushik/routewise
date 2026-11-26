import random

import pytest

from llm_router.association import TransactionDB, apriori, generate_rules, naive_frequent_itemsets

TEXTBOOK = [  # classic market-basket example
    {"bread", "milk"},
    {"bread", "diaper", "beer", "eggs"},
    {"milk", "diaper", "beer", "cola"},
    {"bread", "milk", "diaper", "beer"},
    {"bread", "milk", "diaper", "cola"},
]


def test_textbook_supports_and_rule():
    db = TransactionDB(TEXTBOOK)
    freq = apriori(db, min_support=0.4, max_len=3)
    assert freq[("bread",)] == pytest.approx(0.8)
    assert freq[("beer", "diaper")] == pytest.approx(0.6)
    assert freq[("beer", "bread", "diaper")] == pytest.approx(0.4)  # exactly at min_support: kept
    assert ("beer", "bread", "diaper") not in apriori(db, min_support=0.5, max_len=3)
    rules = {(r.antecedent, r.consequent): r for r in generate_rules(freq)}
    r = rules[(("beer",), ("diaper",))]
    assert r.confidence == pytest.approx(1.0)
    assert r.lift == pytest.approx(1.0 / 0.8)


@pytest.mark.parametrize("seed", range(5))
def test_apriori_matches_naive_and_counts_fewer(seed):
    rng = random.Random(seed)
    items = [f"i{k}" for k in range(12)]
    tx = [set(rng.sample(items, rng.randint(1, 6))) for _ in range(300)]
    db = TransactionDB(tx)
    a = apriori(db, 0.05, 4)
    ca = db.candidates_counted
    n = naive_frequent_itemsets(db, 0.05, 4)
    assert a.keys() == n.keys()
    for k in a:
        assert a[k] == pytest.approx(n[k])
    assert ca < db.candidates_counted
