"""Forecasting core: earners/players decomposition with damped growth.

All fits are closed-form or small grid searches (standard library only), and every
step is deterministic so two runs on one case write identical bytes.
"""
import math
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    c = clamp(pct)
    return math.log(c / (100.0 - c))


def sigmoid(x):
    if x > 40:
        return 99.99
    if x < -40:
        return 0.01
    return 100.0 / (1.0 + math.exp(-x))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_at(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _wls(xs, ys, ws):
    """Weighted least squares line y = a + b x; returns (a, b, resid_std)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return (ys[-1] if ys else 0.0), 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    a = my - b * mx
    var = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    return a, b, math.sqrt(max(var, 0.0))


def project_players(ts, ps, h_days):
    """Project players_tracked forward with a saturating inflow model.

    P(t) = C + A * (1 - exp(-t / tau)); grid-search tau, solve C, A linearly on the
    observed plateaus. Returns (P_at_cut_extrapolated, P_target, rel_uncertainty).
    """
    t_cut = ts[-1]
    # Use distinct plateaus to avoid over-weighting flat stale stretches.
    pts = []
    for t, p in zip(ts, ps):
        if not pts or p != pts[-1][1]:
            pts.append((t, p))
    if len(pts) < 2:
        p = ps[-1]
        # Launch-night prior: a small, stale pool is about to refresh upward as
        # new players track the game. Anticipate multiplicative inflow over the
        # horizon (decelerating with pool size) instead of assuming it stays put.
        launch = max(0.0, 1.0 - p / 6000.0)
        g = 0.35 + 3.2 * launch
        p_tgt = p * (1.0 + g * math.sqrt(max(h_days, 0.0)))
        rel = 0.5 + 0.35 * launch
        return p, p_tgt, min(rel, 0.85)
    best = None
    for tau in [0.5, 0.8, 1.2, 1.8, 2.6, 3.6, 5.0, 7.0, 10.0, 14.0]:
        xs = [1.0 - math.exp(-t / tau) for t, _ in pts]
        ys = [p for _, p in pts]
        ws = [1.0] * len(pts)
        a, b, res = _wls(xs, ys, ws)  # P = a + b*(1-exp(-t/tau))
        if b < 0:
            continue
        err = res / max(ys[-1], 1.0)
        if best is None or err < best[0]:
            best = (err, tau, a, b)
    if best is None:
        p = ps[-1]
        launch = max(0.0, 1.0 - p / 6000.0)
        g = 0.30 + 3.0 * launch
        return p, p * (1.0 + g * math.sqrt(max(h_days, 0.0))), min(0.5 + 0.35 * launch, 0.85)
    _, tau, a, b = best
    p_cut = a + b * (1.0 - math.exp(-t_cut / tau))
    p_tgt = a + b * (1.0 - math.exp(-(t_cut + h_days) / tau))
    p_cut = max(p_cut, ps[-1])
    p_tgt = max(p_tgt, p_cut)
    rel = min(0.6, 0.12 + 0.10 * h_days / max(tau, 0.5))
    return p_cut, p_tgt, rel


TAU_DAMP = 1.6  # days; earner inflow decelerates over the horizon


def _states(ts, ser):
    """Collapse consecutive identical earner counts into refresh states.

    Exophase holds a page constant between refreshes, so repeated identical
    reads carry no new signal; fitting them would bias the slope toward zero.
    Keep the first read of each plateau (its timestamp) and its earner count.
    """
    pts = []
    for t, e in zip(ts, ser):
        if e is None:
            continue
        if not pts or abs(e - pts[-1][1]) > 1e-9:
            pts.append((t, e))
    return pts


def _trophy_slope(ts, ser):
    """Damped-ready log-linear slope of earner count; returns (slope, resid, e_cut, n_pts)."""
    allpts = [(t, e) for t, e in zip(ts, ser) if e is not None]
    if not allpts:
        return 0.0, 0.5, 0.0, 0
    e_cut = allpts[-1][1]
    pts = _states(ts, ser)
    if len(pts) < 3 or e_cut <= 0:
        return 0.0, 0.6, e_cut, len(allpts)
    # recent ~70% of the refresh states, weight later states more
    k = max(3, int(len(pts) * 0.7))
    win = pts[-k:]
    xs = [t for t, _ in win]
    ys = [math.log(e + 1.0) for _, e in win]
    ws = [1.0 + 0.5 * j / max(len(win) - 1, 1) for j in range(len(win))]
    a, b, res = _wls(xs, ys, ws)
    return b, res, e_cut, len(allpts)


def forecast_case(case):
    trophies = case["trophies"]
    hist = case["history"]
    h_days = case["horizon_minutes"] / 1440.0
    n = len(trophies)
    t0 = parse_at(hist[0]["at"])
    ts = [(parse_at(r["at"]) - t0) / 86400.0 for r in hist]
    ps = [float(r["players_tracked"]) for r in hist]
    es = []
    for i in range(n):
        ser = []
        for r in hist:
            rate = r["rates"][i]
            ser.append(None if rate is None else rate / 100.0 * float(r["players_tracked"]))
        es.append(ser)
    p_cut, p_tgt, p_rel = project_players(ts, ps, h_days)

    # per-trophy slopes, then tier pools for sparse trophies
    info = []
    tier_slopes = {}
    for i, tr in enumerate(trophies):
        b, res, e_cut, npts = _trophy_slope(ts, es[i])
        tier = tr.get("tier", "bronze")
        info.append({"b": b, "res": res, "e": e_cut, "n": npts, "tier": tier})
        if e_cut >= 25 and npts >= 4:
            tier_slopes.setdefault(tier, []).append(b)
    tier_pool = {t: sum(v) / len(v) for t, v in tier_slopes.items()}
    all_b = [d["b"] for d in info if d["e"] >= 25 and d["n"] >= 4]
    global_pool = sum(all_b) / len(all_b) if all_b else 0.15

    rows = []
    for i, tr in enumerate(trophies):
        d = info[i]
        e_cut = d["e"]
        frac = e_cut / max(ps[-1], 1.0)
        # intro trophy: essentially everyone, stays pinned near 100
        if frac > 0.985:
            q = [round4(clamp(99.9 + 0.02 * z)) for z in Z]
            rows.append({"id": tr["id"], "q": q})
            continue
        # choose slope: own if signal, else shrink to tier/global pool
        if e_cut >= 25 and d["n"] >= 4:
            b = d["b"]
            fit_res = d["res"]
        else:
            pool = tier_pool.get(d["tier"], global_pool)
            w = min(1.0, e_cut / 25.0)
            b = w * d["b"] + (1 - w) * pool
            fit_res = max(d["res"], 0.45)
        b = max(-0.5, min(b, 2.5))
        # damped extrapolation of ln(e+1)
        integ = TAU_DAMP * (1.0 - math.exp(-h_days / TAU_DAMP))
        ln_e_tgt = math.log(e_cut + 1.0) + b * integ
        e_tgt = math.exp(ln_e_tgt) - 1.0
        e_tgt = max(e_cut, min(e_tgt, p_tgt))
        rate_tgt = clamp(100.0 * e_tgt / max(p_tgt, 1.0))
        med = logit(rate_tgt)
        # spread: fit uncertainty + horizon + denominator + rounding, in quadrature
        lev = integ / max(ts[-1] - ts[0], 0.25)
        sig_fit = fit_res * (0.6 + 0.8 * min(lev, 2.0))
        sig_hor = 0.42 * math.sqrt(h_days)
        sig_den = p_rel * 1.2
        sig_round = 0.006 / max(rate_tgt, 0.02)
        sigma = math.sqrt(sig_fit ** 2 + sig_hor ** 2 + sig_den ** 2 + min(sig_round, 1.2) ** 2)
        sigma = max(0.35, min(sigma, 3.2))
        q = [round4(sigmoid(med + z * sigma)) for z in Z]
        rows.append({"id": tr["id"], "q": q})
    return {"trophies": rows}
