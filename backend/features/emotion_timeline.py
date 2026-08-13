def summarize_timeline(points: list[dict]) -> dict:
    if not points:
        return {"available": False, "series": [], "dominant": None}
    keys = set()
    for p in points:
        keys.update((p.get("emotions") or {}).keys())
    series = {k: [] for k in sorted(keys)}
    for p in points:
        t = p.get("t", 0)
        emo = p.get("emotions") or {}
        for k in series:
            series[k].append({"t": t, "v": float(emo.get(k, 0))})
    avgs = {k: (sum(x["v"] for x in v) / len(v) if v else 0) for k, v in series.items()}
    dominant = max(avgs, key=avgs.get) if avgs else None
    return {"available": True, "series": series, "averages": avgs, "dominant": dominant}
