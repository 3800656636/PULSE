"""Independent implementation of PULSE (Zenodo 23123295), Appendix C.

The simulator owns global time and all-player collisions. Policies see only their
own state, local pair position and feedback. Zero-based arms; inclusive schedules.
"""
import math
import numpy as np
from numba import njit

ORDINARY, PROBE, VERIFY = 0, 1, 2
METHODS = {"PULSE": 0, "ACE": 1, "UCB(2.0)": 2, "RD-UCB": 3, "D-MC": 4}


@njit(cache=True)
def uniform(states, j):
    """Private xorshift64* stream; reward generation uses a separate stream."""
    x = states[j]
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    states[j] = x
    return float((x * np.uint64(2685821657736338717)) >> np.uint64(11)) / 9007199254740992.0


@njit(cache=True)
def choose(mask, count, u, wanted=True):
    rank = min(int(u * count), count - 1)
    for k in range(len(mask)):
        if mask[k] == wanted:
            if rank == 0:
                return k
            rank -= 1
    raise ValueError("Empty or inconsistent sampling set")


@njit(cache=True)
def push(queue, pos, sums, j, k, value):
    p = pos[j, k]
    sums[j, k] += value - queue[j, k, p]
    queue[j, k, p] = value
    pos[j, k] = (p + 1) % queue.shape[2]


@njit(cache=True)
def clear_queue(queue, pos, sums, j, k):
    queue[j, k, :] = 0
    pos[j, k] = 0
    sums[j, k] = 0


@njit(cache=True)
def detector_step(occupied, suspicion, pending, arm, label, collision):
    """Appendix C detector. Return (new pending target, removed arm or -1)."""
    removed = -1
    if label == PROBE:
        if collision:
            suspicion[arm] = False
        elif not suspicion[arm]:
            suspicion[arm] = True
        else:
            pending = arm
    elif label == VERIFY:
        if pending != arm:
            raise ValueError("Verification target must equal pending target")
        pending = -1
        if collision:
            suspicion[arm] = False
        else:
            occupied[arm] = False
            suspicion[:] = False
            removed = arm
    return pending, removed


@njit(cache=True)
def update_reward(counts, means, upper, lower, j, k, reward, collision, label, confidence_log):
    # Appendix C explicitly excludes VERIFY, even when collision free.
    if not collision and label != VERIFY:
        counts[j, k] += 1
        means[j, k] += (reward - means[j, k]) / counts[j, k]
        radius = math.sqrt(confidence_log / counts[j, k])
        upper[j, k] = means[j, k] + radius
        lower[j, k] = means[j, k] - radius


def make_schedule(horizon=2_000_000):
    """Paper schedule; nondefault horizons proportionally scale both events."""
    arrival = max(2, int(round(horizon * 0.04)))
    departure = max(arrival + 1, int(round(horizon * 0.05)))
    starts = np.array([1] * 4 + [arrival] * 4 + [1] * 2, dtype=np.int64)
    ends = np.array([departure] * 4 + [horizon] * 6, dtype=np.int64)
    return starts, ends, departure


def validate(horizon, means, starts, ends, m):
    if horizon < 100 or len(means) < 2 * m:
        raise ValueError("Require T >= 100 and K >= 2m")
    if len(starts) != len(ends) or np.any(starts < 1) or np.any(ends > horizon) or np.any(starts > ends):
        raise ValueError("Invalid inclusive activity intervals")
    for t in set(starts.tolist() + ends.tolist()):
        if np.sum((starts <= t) & (ends >= t)) > m:
            raise ValueError("Activity schedule exceeds m")
    if np.any(np.diff(means) >= 0):
        raise ValueError("Means must be strictly decreasing")


@njit(cache=True)
def simulate_adaptive(method, horizon, mu, starts, ends, m, noise, states,
                      epsilon, confidence, lp, lq, checkpoints, departure):
    """PULSE/ACE. Pair selections precede feedback; each join sets its own parity.

    Curves: time, pseudo regret, collided player rounds, active player rounds.
    Events: round, player, kind, arm, false release flag. Kinds 1 add, 2 pending,
    3 verification collision, 4 removal, 5 seat, 6 return to exploration.
    """
    M, K = len(starts), len(mu)
    occupied = np.zeros((M, K), dtype=np.bool_)
    suspicion = np.zeros((M, K), dtype=np.bool_)
    pending = np.full(M, -1, dtype=np.int64)
    home = np.full(M, -1, dtype=np.int64)
    counts = np.zeros((M, K), dtype=np.int64)
    means = np.zeros((M, K))
    upper, lower = np.full((M, K), np.inf), np.full((M, K), -np.inf)
    pq = np.zeros((M, K, lp), dtype=np.int8)
    pp, ps = np.zeros((M, K), dtype=np.int64), np.zeros((M, K), dtype=np.int64)
    qq = np.zeros((M, K, lq), dtype=np.int8)
    qp, qs = np.zeros((M, K), dtype=np.int64), np.zeros((M, K), dtype=np.int64)
    first, second = np.zeros(M, dtype=np.int64), np.zeros(M, dtype=np.int64)
    first_collision, first_reward = np.zeros(M, dtype=np.bool_), np.zeros(M)
    label = np.zeros(M, dtype=np.int64)
    actions, arm_counts = np.full(M, -1, dtype=np.int64), np.zeros(K, dtype=np.int64)
    eligible = np.zeros(K, dtype=np.bool_)
    curves = np.zeros((len(checkpoints), 4))
    events = np.zeros((100_000, 5), dtype=np.int64)
    ne, cp = 0, 1
    regret, before, collided, active_total = 0.0, 0.0, 0, 0
    probes, verifications, false_releases, seats = 0, 0, 0, 0
    threshold_p, threshold_q = math.ceil(0.85 * lp), math.ceil(0.142 * lq)
    confidence_log = confidence * math.log(horizon)
    optimal = np.zeros(M + 1)
    for j in range(1, M + 1):
        optimal[j] = optimal[j - 1] + mu[j - 1]
    for t in range(1, horizon + 1):
        active = 0
        arm_counts[:] = 0
        # Only simulator scheduling uses absolute time; player pair parity is local.
        for j in range(M):
            actions[j] = -1
            if starts[j] <= t <= ends[j]:
                active += 1
                if (t - starts[j]) % 2 == 0:
                    nocc = 0
                    for k in range(K):
                        nocc += int(occupied[j, k])
                    correction = home[j] < 0 and nocc >= m
                    # Fixed private draws per pair reduce needless coupling drift.
                    u0, u1 = uniform(states, j), uniform(states, j)
                    u2, u3 = uniform(states, j), uniform(states, j)
                    label[j] = ORDINARY
                    if correction:
                        first[j] = choose(occupied[j], nocc, u0)
                        second[j] = choose(occupied[j], nocc, u1)
                        if method == 0:
                            if pending[j] >= 0 and second[j] == pending[j]:
                                label[j] = VERIFY
                            elif pending[j] < 0:
                                label[j] = PROBE
                    else:
                        if home[j] >= 0:
                            first[j] = home[j]
                        else:
                            first[j] = choose(occupied[j], K - nocc, u0, False)
                        second[j] = first[j]
                        if u2 < epsilon:
                            if method == 0 and pending[j] >= 0:
                                second[j], label[j] = pending[j], VERIFY
                            else:
                                nelig = 0
                                for k in range(K):
                                    eligible[k] = occupied[j, k] and (method == 1 or home[j] < 0 or upper[j, k] > lower[j, home[j]])
                                    nelig += int(eligible[k])
                                if nelig > 0:
                                    second[j] = choose(eligible, nelig, u3)
                                    if method == 0:
                                        label[j] = PROBE
                    actions[j] = first[j]
                else:
                    actions[j] = second[j]
                arm_counts[actions[j]] += 1
        active_total += active
        earned_mean = 0.0
        for j in range(M):
            k = actions[j]
            if k >= 0:
                if arm_counts[k] > 1:
                    collided += 1
                else:
                    earned_mean += mu[k]
        regret += optimal[active] - earned_mean
        # Snapshot formal homes before any player's feedback update this round.
        homes_before = home.copy()
        for j in range(M):
            k = actions[j]
            if k < 0:
                continue
            collision = arm_counts[k] > 1
            reward = 0.0 if collision else mu[k] + noise[t - 1, j]
            if (t - starts[j]) % 2 == 0:
                first_collision[j], first_reward[j] = collision, reward
                continue
            was_exploiting = home[j] >= 0
            nocc_before = 0
            for a in range(K):
                nocc_before += int(occupied[j, a])
            was_correction = not was_exploiting and nocc_before >= m
            removed = -1
            if method == 0:
                old_pending = pending[j]
                pending[j], removed = detector_step(occupied[j], suspicion[j], pending[j], k, label[j], collision)
                if label[j] == PROBE:
                    probes += 1
                elif label[j] == VERIFY:
                    verifications += 1
                kind = 0
                if removed >= 0:
                    kind = 4
                elif old_pending < 0 and pending[j] >= 0:
                    kind = 2
                elif label[j] == VERIFY and collision:
                    kind = 3
                if kind:
                    flag = 0
                    if removed >= 0:
                        for other in range(M):
                            if other != j and starts[other] <= t <= ends[other] and homes_before[other] == removed:
                                flag = 1
                        false_releases += flag
                    events[ne] = np.array([t, j, kind, k, flag])
                    ne += 1
            else:
                # ACE's long release queue consumes both listed-arm observations.
                if occupied[j, first[j]]:
                    push(qq, qp, qs, j, first[j], int(not first_collision[j]))
                if occupied[j, k]:
                    push(qq, qp, qs, j, k, int(not collision))
                for a in range(K):
                    if occupied[j, a] and qs[j, a] >= threshold_q:
                        removed = a
                        occupied[j, a] = False
                        clear_queue(qq, qp, qs, j, a)
                        flag = 0
                        for other in range(M):
                            if other != j and starts[other] <= t <= ends[other] and homes_before[other] == a:
                                flag = 1
                        false_releases += flag
                        events[ne] = np.array([t, j, 4, a, flag])
                        ne += 1
                        break
            # Identical retained estimation profile for PULSE and ACE.
            update_reward(counts, means, upper, lower, j, first[j], first_reward[j], first_collision[j], ORDINARY, confidence_log)
            update_reward(counts, means, upper, lower, j, k, reward, collision, label[j], confidence_log)
            if removed >= 0:
                clear_queue(pq, pp, ps, j, removed)
                if was_exploiting and (method == 0 or upper[j, removed] > lower[j, home[j]]):
                    old_home = home[j]
                    home[j] = -1
                    events[ne] = np.array([t, j, 6, old_home, 0])
                    ne += 1
            # Occupied-arm discovery applies only to ordinary repeated exploration.
            if not was_exploiting and not was_correction and first[j] == k and label[j] == ORDINARY and not occupied[j, k]:
                push(pq, pp, ps, j, k, int(first_collision[j] and collision))
                if ps[j, k] >= threshold_p:
                    occupied[j, k] = True
                    clear_queue(pq, pp, ps, j, k)
                    suspicion[j, :] = False
                    if pending[j] >= 0 and not occupied[j, pending[j]]:
                        pending[j] = -1
                    events[ne] = np.array([t, j, 1, k, 0])
                    ne += 1
            # Seating requires this pair's two collision-free ordinary samples.
            # A player returning from exploitation cannot seat using that old pair.
            if not was_exploiting and home[j] < 0 and pending[j] < 0 and not was_correction and first[j] == k and label[j] == ORDINARY and not first_collision[j] and not collision and not occupied[j, k]:
                nocc = 0
                dominated = True
                for a in range(K):
                    nocc += int(occupied[j, a])
                    if a != k and not occupied[j, a] and lower[j, k] < upper[j, a]:
                        dominated = False
                if nocc < m and dominated:
                    home[j] = k
                    seats += 1
                    events[ne] = np.array([t, j, 5, k, 0])
                    ne += 1
            if ne >= len(events) - 10:
                raise RuntimeError("Event buffer exceeded")
        if t == departure:
            before = regret
        if cp < len(checkpoints) and t == checkpoints[cp]:
            curves[cp] = np.array([float(t), regret, float(collided), float(active_total)])
            cp += 1
    metrics = np.array([regret, regret - before, float(collided), float(active_total),
                        float(false_releases), float(probes), float(verifications), float(seats)])
    return metrics, curves, events[:ne], home, counts


@njit(cache=True)
def simulate_baseline(method, horizon, mu, starts, ends, noise, states, checkpoints, departure):
    """Selfish UCB, randomized UCB, and a documented local-epoch D-MC variant.

    Collided zero rewards enter all baseline estimates. D-MC uses local T/5
    epochs with half-epoch uniform learning and collision-based population
    estimation. This adaptation is not an exact author-code implementation.
    """
    M, K = len(starts), len(mu)
    counts, empirical = np.zeros((M, K), dtype=np.int64), np.zeros((M, K))
    actions, arm_counts = np.full(M, -1, dtype=np.int64), np.zeros(K, dtype=np.int64)
    home = np.full(M, -1, dtype=np.int64)
    n_hat, epoch_collisions = np.ones(M, dtype=np.int64), np.zeros(M, dtype=np.int64)
    top = np.zeros((M, K), dtype=np.int64)
    epoch_len = max(2, horizon // 5)
    learn_len = epoch_len // 2
    optimal = np.zeros(M + 1)
    for j in range(1, M + 1):
        optimal[j] = optimal[j - 1] + mu[j - 1]
    regret, before, collided, active_total, cp = 0.0, 0.0, 0, 0, 1
    curves = np.zeros((len(checkpoints), 4))
    for t in range(1, horizon + 1):
        arm_counts[:] = 0
        active = 0
        for j in range(M):
            actions[j] = -1
            if not starts[j] <= t <= ends[j]:
                continue
            active += 1
            local = t - starts[j] + 1
            if method == 4:
                e = (local - 1) % epoch_len
                if e == 0:
                    counts[j, :] = 0
                    empirical[j, :] = 0
                    epoch_collisions[j] = 0
                    home[j] = -1
                if e < learn_len:
                    k = min(int(uniform(states, j) * K), K - 1)
                else:
                    if e == learn_len:
                        p = min(0.999999, epoch_collisions[j] / learn_len)
                        estimate = int(round(1.0 + math.log(1.0 - p) / math.log(1.0 - 1.0 / K)))
                        n_hat[j] = min(K, max(1, estimate))
                        order = np.argsort(empirical[j])
                        for a in range(n_hat[j]):
                            top[j, a] = order[K - 1 - a]
                    if home[j] >= 0:
                        k = home[j]
                    else:
                        rank = min(int(uniform(states, j) * n_hat[j]), n_hat[j] - 1)
                        k = top[j, rank]
            else:
                best, k = -np.inf, 0
                # Random initialization order avoids a shared deterministic clock.
                unseen = 0
                for a in range(K):
                    unseen += int(counts[j, a] == 0)
                if unseen:
                    rank = min(int(uniform(states, j) * unseen), unseen - 1)
                    for a in range(K):
                        if counts[j, a] == 0:
                            if rank == 0:
                                k = a
                                break
                            rank -= 1
                else:
                    for a in range(K):
                        score = empirical[j, a] + math.sqrt(2.0 * math.log(horizon) / counts[j, a])
                        if method == 3:
                            u1 = max(uniform(states, j), 1e-16)
                            u2 = uniform(states, j)
                            score += math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2) / local
                        if score > best:
                            best, k = score, a
            actions[j] = k
            arm_counts[k] += 1
        active_total += active
        earned = 0.0
        for j in range(M):
            k = actions[j]
            if k < 0:
                continue
            collision = arm_counts[k] > 1
            if collision:
                collided += 1
                reward = 0.0
            else:
                earned += mu[k]
                reward = mu[k] + noise[t - 1, j]
            if method != 4 or (t - starts[j]) % epoch_len < learn_len:
                counts[j, k] += 1
                empirical[j, k] += (reward - empirical[j, k]) / counts[j, k]
            if method == 4:
                if (t - starts[j]) % epoch_len < learn_len:
                    epoch_collisions[j] += int(collision)
                elif home[j] < 0 and not collision:
                    home[j] = k
        regret += optimal[active] - earned
        if t == departure:
            before = regret
        if cp < len(checkpoints) and t == checkpoints[cp]:
            curves[cp] = np.array([float(t), regret, float(collided), float(active_total)])
            cp += 1
    metrics = np.array([regret, regret - before, float(collided), float(active_total), np.nan, 0., 0., 0.])
    return metrics, curves
