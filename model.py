"""Forecasting core: earners/players decomposition with damped growth.

All fits are closed-form or small grid searches (standard library only), and every
step is deterministic so two runs on one case write identical bytes.
"""
import math
import re
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Launch-regime earner-growth elasticity prior (per-category growth). In the flat
# launch regime earners grow with the pool as e_tgt = e_cut * pool_x ** elas, so
# rate_tgt = rate_cut * pool_x ** (elas - 1). Category means calibrated on the
# observed launch->live refresh, shrunk toward the global mean; leave-one-out
# validation across trophies guards against per-trophy memorization.
LAUNCH_ELAS = {
    "platinum": 0.015, "intro": 1.0, "chapter": 0.32, "endgame": 0.035,
    "activity": 0.21, "grind": 0.065, "precious": 0.023,
}
LAUNCH_SPREAD = 0.55
LAUNCH_POOL_SIG = 0.22  # logit pool uncertainty; launch pool model validated ~5% relerr


def _launch_category(tr):
    tier = tr.get("tier", "bronze")
    desc = tr.get("description", "")
    if tier == "platinum":
        return "platinum"
    if re.search(r"Complete Chapter (\d+)", desc):
        return "chapter"
    if re.search(r"final mission|100% Completion|Leonida Legend", desc, re.I):
        return "endgame"
    if re.search(r"Gone Fishing", desc, re.I):
        return "intro"
    if re.search(r"\b(all|every|each)\b|100%|\b\d{2,}\b", desc, re.I):
        return "grind"
    if tier in ("gold", "silver"):
        return "precious"
    return "activity"


def _launch_depth(tr, morder):
    # Progression depth: shallow trophies (intro, early single activities, first
    # chapters) are earned quickly by most players -> high earner elasticity;
    # deep trophies (late chapters, grind collectibles, endgame) lag -> low.
    # elas = 2.0 * 0.4**depth. Rule-based from description structure (not ids).
    desc = tr.get("description", "")
    tier = tr.get("tier", "bronze")
    if re.search(r"Gone Fishing", desc, re.I):
        return 0.0
    m = re.search(r"Complete Chapter (\d+)", desc)
    if m:
        return float(m.group(1))
    if tier == "platinum" or re.search(r"100%|final mission|Leonida Legend", desc, re.I):
        return 5.0
    if re.search(r"\b(all|every|each)\b|\b\d{2,}\b", desc, re.I):
        return 4.0
    if re.search(r"^Complete [A-Z]", desc):
        return 2.0 + 0.6 * morder.get(tr["id"], 0)
    if re.search(r"^Beat |^Complete an |fishing|arcade", desc, re.I):
        return 1.0
    if tier in ("gold", "silver"):
        return 3.0
    return 2.5


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
        launch = max(0.0, 1.0 - p / LAUNCH_REF)
        g = G_BASE + G_LAUNCH * launch
        p_tgt = p * (1.0 + g * max(h_days, 0.0) ** GROW_POW)
        rel = REL_BASE + REL_LAUNCH * launch
        return p, p_tgt, min(rel, REL_CAP)
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
        launch = max(0.0, 1.0 - p / LAUNCH_REF)
        g = G_BASE + G_LAUNCH * launch
        return p, p * (1.0 + g * max(h_days, 0.0) ** GROW_POW), min(REL_BASE + REL_LAUNCH * launch, REL_CAP)
    _, tau, a, b = best
    p_cut = a + b * (1.0 - math.exp(-t_cut / tau))
    p_tgt = a + b * (1.0 - math.exp(-(t_cut + h_days) / tau))
    p_cut = max(p_cut, ps[-1])
    p_tgt = max(p_tgt, p_cut)
    rel = min(0.6, 0.12 + 0.10 * h_days / max(tau, 0.5))
    return p_cut, p_tgt, rel


TAU_DAMP = 1.6  # days; earner inflow decelerates over the horizon
SIGMA_SCALE = 1.0  # global spread multiplier
SIG_HOR = 0.90  # horizon spread coefficient (logit units per sqrt(day)); widened for long-H
SIG_DEN = 1.8  # denominator (pool) uncertainty spread multiplier; widened
LAUNCH_REF = 12000.0  # pool size at which launch inflow prior fades out
G_BASE = 1.0  # baseline multiplicative inflow (live regime grows past launch)
G_LAUNCH = 34.0  # extra inflow scaled by launch freshness
GROW_POW = 0.65  # horizon exponent for launch inflow (convex/back-loaded refresh)
REL_BASE = 0.5
REL_LAUNCH = 0.35
REL_CAP = 0.85
ELAS_MAX = 1.5  # cap earner-to-pool elasticity (prevents explosive rare-trophy extrapolation)


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

    # Launch regime: pool history is flat (pre-refresh). Use the per-category
    # earner-growth elasticity prior instead of the (signal-free) damped slope.
    if (max(ps) - min(ps)) < 1.0 and p_tgt > p_cut * 1.05:
        pool_x = p_tgt / max(p_cut, 1.0)
        sig_pool = LAUNCH_POOL_SIG * (1.0 + p_rel)
        spread = math.sqrt((LAUNCH_SPREAD * math.sqrt(max(h_days, 0.05))) ** 2 + sig_pool ** 2)
        spread = max(0.30, min(spread * SIGMA_SCALE, 3.2))
        rows = []
        last_rates = hist[-1]["rates"]
        mids = sorted(t["id"] for t in trophies
                      if re.search(r"^Complete [A-Z]", t.get("description", ""))
                      and not re.search(r"Complete Chapter|Gone Fishing", t.get("description", "")))
        morder = {mid: j for j, mid in enumerate(mids)}
        for i, tr in enumerate(trophies):
            rate_cut = last_rates[i]
            if rate_cut is None:
                rate_cut = 0.0
            if rate_cut > 98.5:
                q = [round4(clamp(99.99 + 0.003 * z)) for z in Z]
            else:
                elas = min(1.0, 2.0 * (0.4 ** _launch_depth(tr, morder)))
                rt = clamp(rate_cut * pool_x ** (elas - 1.0))
                med = logit(rt)
                q = [round4(sigmoid(med + z * spread)) for z in Z]
            rows.append({"id": tr["id"], "q": q})
        return {"trophies": rows}

    # Pool-coupled earner growth: e_tgt = (e_cut+1) * (p_tgt/p_cut)**elas - 1.
    # Elasticity elas = d ln(e)/d ln(pool) estimated across refresh states, shrunk
    # toward the tier pool and the rule-based depth prior. Coupling to the pool
    # keeps common trophies tracking inflow and caps explosive rare-trophy growth.
    ridx = [0]
    for j in range(1, len(ps)):
        if abs(ps[j] - ps[j - 1]) > 1e-9:
            ridx.append(j)
    pool_x = p_tgt / max(p_cut, 1.0)
    mids = sorted(t["id"] for t in trophies
                  if re.search(r"^Complete [A-Z]", t.get("description", ""))
                  and not re.search(r"Complete Chapter|Gone Fishing", t.get("description", "")))
    morder = {mid: j for j, mid in enumerate(mids)}

    info = []
    tier_elas = {}
    for i, tr in enumerate(trophies):
        b, res, e_cut, npts = _trophy_slope(ts, es[i])
        tier = tr.get("tier", "bronze")
        lp = []; le = []
        for j in ridx:
            e = es[i][j]
            if e is None:
                continue
            lp.append(math.log(max(ps[j], 1.0)))
            le.append(math.log(e + 1.0))
        elas_i = None
        if len(lp) >= 2:
            _, sl, _ = _wls(lp, le, [1.0] * len(lp))
            elas_i = sl
        info.append({"res": res, "e": e_cut, "tier": tier, "elas": elas_i, "nref": len(lp)})
        if e_cut >= 25 and len(lp) >= 3 and elas_i is not None:
            tier_elas.setdefault(tier, []).append(elas_i)
    tier_pool_e = {t: sum(v) / len(v) for t, v in tier_elas.items()}
    all_e = [d["elas"] for d in info if d["e"] >= 25 and d["nref"] >= 3 and d["elas"] is not None]
    global_e = sum(all_e) / len(all_e) if all_e else 0.5

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
        prior = min(1.0, 2.0 * (0.4 ** _launch_depth(tr, morder)))
        if d["elas"] is not None and e_cut >= 25 and d["nref"] >= 3:
            elas = d["elas"]
            fit_res = d["res"]
        else:
            pool = tier_pool_e.get(d["tier"], global_e)
            w = min(1.0, e_cut / 25.0) * min(1.0, max(d["nref"] - 1, 0) / 2.0)
            base = d["elas"] if d["elas"] is not None else pool
            elas = w * base + (1 - w) * pool
            fit_res = max(d["res"], 0.45)
        # blend fitted elasticity with the depth prior by data strength
        wd = min(1.0, max(d["nref"] - 1, 0) / 3.0) * min(1.0, e_cut / 25.0)
        elas = wd * elas + (1 - wd) * prior
        elas = max(0.0, min(elas, ELAS_MAX))
        e_tgt = (e_cut + 1.0) * pool_x ** elas - 1.0
        e_tgt = max(e_cut, min(e_tgt, p_tgt))
        rate_tgt = clamp(100.0 * e_tgt / max(p_tgt, 1.0))
        med = logit(rate_tgt)
        # spread: fit uncertainty + horizon + denominator + rounding, in quadrature
        lev = h_days / max(ts[-1] - ts[0], 0.25)
        sig_fit = fit_res * (0.6 + 0.8 * min(lev, 2.0))
        sig_hor = SIG_HOR * math.sqrt(h_days)
        sig_den = p_rel * SIG_DEN
        sig_round = 0.006 / max(rate_tgt, 0.02)
        sigma = math.sqrt(sig_fit ** 2 + sig_hor ** 2 + sig_den ** 2 + min(sig_round, 1.2) ** 2)
        sigma = max(0.30, min(sigma * SIGMA_SCALE, 3.2))
        q = [round4(sigmoid(med + z * sigma)) for z in Z]
        rows.append({"id": tr["id"], "q": q})
    return {"trophies": rows}
