"""Association rule mining, written from scratch (syllabus unit: Association Rules Learning).

Two algorithms find the same frequent itemsets:
- `naive_frequent_itemsets`: tests every combination of items up to `max_len`.
- `apriori`: grows itemsets level by level and prunes any candidate with an infrequent
  subset (the Apriori property), so it counts far fewer candidates.

Support counting is vectorised: each item is a boolean column, and the support of an
itemset is the mean of the AND of its columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class Rule:
    antecedent: tuple
    consequent: tuple
    support: float
    confidence: float
    lift: float

    def as_dict(self):
        return {"if": list(self.antecedent), "then": list(self.consequent),
                "support": round(self.support, 4), "confidence": round(self.confidence, 4),
                "lift": round(self.lift, 3)}


class TransactionDB:
    """Boolean matrix view of a list of transactions (each a set of item names)."""

    def __init__(self, transactions):
        self.items = sorted({i for t in transactions for i in t})
        self.index = {item: j for j, item in enumerate(self.items)}
        self.matrix = np.zeros((len(transactions), len(self.items)), dtype=bool)
        for r, t in enumerate(transactions):
            for item in t:
                self.matrix[r, self.index[item]] = True
        self.n = len(transactions)
        self.candidates_counted = 0

    def support(self, itemset) -> float:
        self.candidates_counted += 1
        cols = [self.index[i] for i in itemset]
        return float(self.matrix[:, cols].all(axis=1).mean()) if cols else 1.0


def naive_frequent_itemsets(db: TransactionDB, min_support: float, max_len: int = 3) -> dict:
    db.candidates_counted = 0
    out = {}
    for k in range(1, max_len + 1):
        for itemset in combinations(db.items, k):
            s = db.support(itemset)
            if s >= min_support:
                out[itemset] = s
    return out


def apriori(db: TransactionDB, min_support: float, max_len: int = 3) -> dict:
    db.candidates_counted = 0
    level = {}
    for item in db.items:
        s = db.support((item,))
        if s >= min_support:
            level[(item,)] = s
    out = dict(level)
    k = 2
    while level and k <= max_len:
        prev = sorted(level)
        prev_set = set(prev)
        candidates = set()
        # join step: merge two (k-1)-itemsets that share their first k-2 items
        for a, b in combinations(prev, 2):
            if a[:-1] == b[:-1]:
                cand = tuple(sorted(set(a) | set(b)))
                # prune step: every (k-1)-subset must itself be frequent
                if all(sub in prev_set for sub in combinations(cand, k - 1)):
                    candidates.add(cand)
        level = {}
        for cand in sorted(candidates):
            s = db.support(cand)
            if s >= min_support:
                level[cand] = s
        out.update(level)
        k += 1
    return out


def generate_rules(itemsets: dict, min_confidence: float = 0.0, consequent_filter=None) -> list[Rule]:
    rules = []
    for itemset, sup in itemsets.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for ante in combinations(itemset, r):
                cons = tuple(i for i in itemset if i not in ante)
                if consequent_filter and not consequent_filter(cons):
                    continue
                if ante not in itemsets or cons not in itemsets:
                    continue
                conf = sup / itemsets[ante]
                if conf >= min_confidence:
                    rules.append(Rule(ante, cons, sup, conf, conf / itemsets[cons]))
    return sorted(rules, key=lambda r: (-r.lift, -r.support))
