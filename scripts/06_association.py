"""Step 6 - association rules: which prompt properties go together with cheap-model failure?

Each prompt becomes a "transaction" of items such as `length=long`, `has_code`,
`cluster=7` and `outcome=FAIL`. Apriori (written from scratch in llm_router.association)
finds frequent itemsets; rules of the form {properties} -> {outcome=FAIL} are ranked by lift.
The naive algorithm is run on the same data to show how many candidates Apriori saves.
"""
import time

import numpy as np

from _common import save_json
from llm_router import association as A
from llm_router import config, data
from llm_router.features import STAT_NAMES, prompt_stats


def transactions(df, clusters):
    S = prompt_stats(df.prompt)
    col = {n: S[:, i] for i, n in enumerate(STAT_NAMES)}
    chars = np.expm1(col["log_chars"])
    length = np.select([chars < 80, chars < 300, chars < 1200], ["short", "medium", "long"], "very_long")
    out = []
    for i in range(len(df)):
        t = {f"length={length[i]}", f"source={df.source.values[i]}", f"cluster={clusters[i]}",
             "outcome=FAIL" if df.label.values[i] else "outcome=OK"}
        if col["code_markers"][i] >= 2 or col["code_words"][i] >= 1: t.add("has_code")
        if col["math_markers"][i] >= 1: t.add("has_math")
        if col["frac_non_ascii"][i] > 0.2: t.add("non_english")
        if col["n_question_marks"][i] >= 1: t.add("is_question")
        if col["starts_imperative"][i]: t.add("imperative")
        if col["creative_words"][i] >= 1: t.add("creative")
        if col["constraint_words"][i] >= 3: t.add("many_constraints")
        if col["reasoning_words"][i] >= 1: t.add("reasoning")
        if col["has_context_block"][i]: t.add("has_context_block")
        if col["mcq_options"][i] >= 2: t.add("multiple_choice")
        out.append(t)
    return out


if __name__ == "__main__":
    train = data.load("train")
    clusters = np.load(config.DATA_PROCESSED / "cluster_train.npy")
    db = A.TransactionDB(transactions(train, clusters))
    print(f"{db.n} transactions, {len(db.items)} distinct items")

    min_support, max_len = 0.003, 3
    t = time.perf_counter()
    freq = A.apriori(db, min_support, max_len)
    t_ap, cand_ap = time.perf_counter() - t, db.candidates_counted
    t = time.perf_counter()
    freq_naive = A.naive_frequent_itemsets(db, min_support, max_len)
    t_nv, cand_nv = time.perf_counter() - t, db.candidates_counted
    assert set(freq) == set(freq_naive), "Apriori and naive algorithm disagree"
    print(f"Apriori: {len(freq)} frequent itemsets, {cand_ap} candidates, {t_ap:.1f}s | "
          f"naive: {cand_nv} candidates, {t_nv:.1f}s")

    fail_rules = A.generate_rules(freq, min_confidence=0.2, consequent_filter=lambda c: c == ("outcome=FAIL",))
    ok_rules = A.generate_rules(freq, min_confidence=0.95, consequent_filter=lambda c: c == ("outcome=OK",))
    for r in fail_rules[:12]:
        print(f"  {r.antecedent} -> FAIL  conf={r.confidence:.2f} lift={r.lift:.2f} sup={r.support:.4f}")
    save_json("06_association.json", {
        "n_transactions": db.n, "n_items": len(db.items), "min_support": min_support, "max_len": max_len,
        "base_fail_rate": float(train.label.mean()),
        "apriori": {"frequent_itemsets": len(freq), "candidates_counted": cand_ap, "seconds": round(t_ap, 2)},
        "naive": {"frequent_itemsets": len(freq_naive), "candidates_counted": cand_nv, "seconds": round(t_nv, 2)},
        "rules_to_fail": [r.as_dict() for r in fail_rules[:25]],
        "rules_to_ok": [r.as_dict() for r in ok_rules[:15]],
    })
