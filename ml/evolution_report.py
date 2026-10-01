"""Read-only views of a living population (ml/evolution.py) for the
population page (frontend/web/population.html): summary, causes of death,
timeline, what evolved across generations, the graveyard and family trees.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ml import evolution

CAUSES = ("fatal_miss", "repeated_misses", "exhaustion")
_NUMERIC_GENES = ("boldness", "temperament", "news_sensitivity", "window")


def current(key: str) -> evolution.EvolutionaryForecaster | None:
    """The population the app is running for `key`, else the saved one."""
    ev = evolution._registry.get(key) or evolution.load_state(key)
    if ev is None or getattr(ev, "version", 1) != evolution.STATE_VERSION or ev.newborn:
        return None
    return ev


def _downsample(rows: list[dict], max_points: int) -> list[dict]:
    if len(rows) <= max_points:
        return rows
    idx = np.linspace(0, len(rows) - 1, max_points).round().astype(int)
    return [rows[i] for i in idx]


def report(ev: evolution.EvolutionaryForecaster, max_points: int = 400, max_generation_buckets: int = 40) -> dict:
    t = ev.t_last
    diag = ev.diagnostics(t)
    graveyard = ev.graveyard

    by_cause = Counter(d["cause"] for d in graveyard)
    causes = [{
        "cause": c,
        "deaths": by_cause.get(c, 0),
        "share_pct": round(by_cause.get(c, 0) / len(graveyard) * 100, 1) if graveyard else 0.0,
        "avg_age": round(float(np.mean([d["age"] for d in graveyard if d["cause"] == c])), 1) if by_cause.get(c) else None,
    } for c in CAUSES]

    # Timeline: one point per snapshot (every 24 candles), deaths in between.
    hist = ev.history
    timeline = []
    prev_deaths = 0
    for h in hist:
        timeline.append({
            "time": h["time"], "price": h.get("price"), "mood": h["mood"], "alive": h["alive"],
            "deaths": h["deaths"] - prev_deaths, "avg_generation": h.get("avg_generation"),
            "shares": h.get("shares", {}), **{g: h.get(g) for g in _NUMERIC_GENES},
        })
        prev_deaths = h["deaths"]
    # Downsampling keeps deaths honest: sum them into the kept points.
    if len(timeline) > max_points:
        keep = np.linspace(0, len(timeline) - 1, max_points).round().astype(int)
        merged, start = [], 0
        for k in keep:
            point = dict(timeline[k])
            point["deaths"] = sum(p["deaths"] for p in timeline[start:k + 1])
            merged.append(point)
            start = k + 1
        timeline = merged

    # Generations: every organism that ever lived (dead + alive), bucketed.
    people = [{"generation": d["generation"], "age": d["age"], "dead": True, "cause": d["cause"],
               "inputs": d["inputs"], **{g: d[g] for g in _NUMERIC_GENES}} for d in graveyard]
    people += [{"generation": o.generation, "age": o.age(t), "dead": False, "cause": None,
                "inputs": list(o.groups), **{g: o.genes()[g] for g in _NUMERIC_GENES}} for o in ev.alive]
    max_gen = max((p["generation"] for p in people), default=0)
    size = max(1, int(np.ceil((max_gen + 1) / max_generation_buckets)))
    generations = []
    for lo in range(0, max_gen + 1, size):
        group = [p for p in people if lo <= p["generation"] < lo + size]
        if not group:
            continue
        dead = [p for p in group if p["dead"]]
        inputs = Counter(g for p in group for g in p["inputs"])
        generations.append({
            "from": lo, "to": lo + size - 1, "organisms": len(group), "alive": len(group) - len(dead),
            "avg_lifespan": round(float(np.mean([p["age"] for p in dead])), 1) if dead else None,
            "causes": {c: sum(p["cause"] == c for p in dead) for c in CAUSES},
            "input_shares": {g: round(n / len(group), 3) for g, n in inputs.most_common()},
            **{g: round(float(np.mean([p[g] for p in group])), 3) for g in _NUMERIC_GENES},
        })

    return {
        "summary": {
            "lived_from": diag["lived_from"], "lived_to": ev.index[t].isoformat(), "candles_lived": diag["candles_lived"],
            "alive": diag["population"], "births": diag["births"], "deaths": diag["deaths"],
            "max_generation": diag["max_generation"], "emotion": diag["emotion"], "mood": diag["mood"],
            "avg_lifespan_of_dead": round(float(np.mean([d["age"] for d in graveyard])), 1) if graveyard else None,
            "oldest_alive": max((o.age(t) for o in ev.alive), default=0),
            "market_sentiment": diag["market_sentiment"],
        },
        "track_record": diag["track_record"],
        "causes": causes,
        "gene_survival": diag["gene_survival"],
        "genes": list(ev.group_names),
        "timeline": timeline,
        "generations": generations,
        "alive": sorted(_alive_rows(ev), key=lambda o: o["energy"], reverse=True),
    }


def _alive_rows(ev: evolution.EvolutionaryForecaster) -> list[dict]:
    diag_leaders = {o["id"] for o in ev.diagnostics(ev.t_last)["leaders"]}
    t = ev.t_last
    rows = []
    for o in ev.alive:
        feel = ev._feeling(o, t)
        rows.append({"id": o.id, "generation": o.generation, "parents": list(o.parents),
                     "born": ev.index[o.born_at].isoformat(), "age": o.age(t), "energy": round(o.energy, 1),
                     "mood": round(feel, 3), "emotion": evolution.emotion_name(feel), "wins": o.wins,
                     "losses": o.losses, "combos": o.combos, "voting": o.id in diag_leaders, **o.genes()})
    return rows


def deaths_page(ev: evolution.EvolutionaryForecaster, page: int = 1, page_size: int = 50, cause: str | None = None,
                generation: int | None = None, organism_id: int | None = None, order: str = "desc") -> dict:
    rows = ev.graveyard
    if cause:
        rows = [d for d in rows if d["cause"] == cause]
    if generation is not None:
        rows = [d for d in rows if d["generation"] == generation]
    if organism_id is not None:
        rows = [d for d in rows if d["id"] == organism_id or organism_id in d["parents"]]
    if order == "desc":
        rows = rows[::-1]
    total = len(rows)
    page_size = max(1, min(page_size, 200))
    pages = max(1, -(-total // page_size))
    page = max(1, min(page, pages))
    return {"total": total, "page": page, "pages": pages, "page_size": page_size,
            "rows": rows[(page - 1) * page_size: page * page_size]}


def organism(ev: evolution.EvolutionaryForecaster, organism_id: int, depth: int = 6) -> dict | None:
    """One organism (dead or alive), its ancestors up to `depth` generations
    back, and its children."""
    by_id = {d["id"]: {**d, "dead": True} for d in ev.graveyard}
    by_id.update({o["id"]: {**o, "dead": False} for o in _alive_rows(ev)})
    me = by_id.get(organism_id)
    if me is None:
        return None
    ancestors, frontier = [], list(me.get("parents", []))
    for level in range(1, depth + 1):
        nxt = []
        for pid in frontier:
            rec = by_id.get(pid)
            ancestors.append({"level": level, "id": pid, "known": rec is not None,
                              **({k: rec[k] for k in ("generation", "dead", "age", "inputs", "cause") if k in rec} if rec else {})})
            if rec:
                nxt += rec.get("parents", [])
        frontier = list(dict.fromkeys(nxt))
        if not frontier:
            break
    children = [{"id": r["id"], "generation": r["generation"], "dead": r["dead"], "age": r["age"]}
                for r in by_id.values() if organism_id in r.get("parents", [])]
    return {"organism": me, "ancestors": ancestors, "children": children}
