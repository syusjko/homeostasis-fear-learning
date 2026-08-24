import torch
import numpy as np
import csv
import os
from scipy import stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device = {device}")

torch.manual_seed(0)

# ============================================================
# 1. 환경 및 상수
# ============================================================
D_OBS = 8
D_H = 16
D_SIG = 16
K_RES = 3
J_MAX = 8

ENERGY_DECAY = 0.01
FOOD_GAIN = 0.25
THREAT_DAMAGE_PREDATOR = 0.7
THREAT_DAMAGE_POISON = 0.1
POISON_ENERGY_COST = 0.2
SIGNAL_COST = 0.005
T_SAFE = 80
CURIOSITY_DECAY = 0.012
CURIOSITY_GAIN = 0.20
P_STRIKE = 0.4
SIGNAL_DECAY = 0.05

THREAT_PRESENT_PROB = 0.3
THREAT_DETECT_PROB = 0.2
PREDATOR_PROB = 0.5

D_IN = D_OBS + K_RES + D_SIG


def make_world_constants(seed):
    g = torch.Generator(device=device).manual_seed(seed)
    mu_food = torch.randn(D_OBS, device=device, generator=g)
    mu_threat_predator = torch.randn(D_OBS, device=device, generator=g)
    mu_threat_poison = torch.randn(D_OBS, device=device, generator=g)
    mu_threat_safe = torch.randn(D_OBS, device=device, generator=g)
    W_in = torch.randn(D_H, D_IN, device=device, generator=g) * 0.5
    return dict(mu_food=mu_food,
                mu_threat_predator=mu_threat_predator,
                mu_threat_poison=mu_threat_poison,
                mu_threat_safe=mu_threat_safe,
                W_in=W_in)


def make_env(progress):
    p_both = 0.05 * progress
    p_safe = 0.08 * progress
    p_food = 0.25
    p_threat = 0.25
    p_neutral = max(0.0, 1.0 - p_food - p_threat - p_both - p_safe)
    return dict(
        p_food=p_food, p_threat=p_threat, p_both=p_both, p_safe=p_safe,
        p_neutral=p_neutral, flee_cost=0.01 + 0.02 * progress,
        threat_damage_predator=THREAT_DAMAGE_PREDATOR,
        threat_damage_poison=THREAT_DAMAGE_POISON,
        poison_energy_cost=POISON_ENERGY_COST,
        energy_decay=ENERGY_DECAY,
        food_gain=FOOD_GAIN, T_SAFE=T_SAFE, p_strike=P_STRIKE,
        curiosity_decay=CURIOSITY_DECAY, curiosity_gain=CURIOSITY_GAIN,
        signal_cost=SIGNAL_COST,
        threat_present_prob=THREAT_PRESENT_PROB,
        threat_detect_prob=THREAT_DETECT_PROB,
        predator_prob=PREDATOR_PROB
    )


def sample_stimuli_and_threat(n, t, env, W):
    threat_present = torch.rand(1, device=device).item() < env['threat_present_prob']
    threat_type = 0 if torch.rand(1, device=device).item() < env['predator_prob'] else 1

    r = torch.rand(n, device=device)
    cat_food = r < env['p_food']
    detector = torch.rand(n, device=device) < env['threat_detect_prob']

    cat_perceived = torch.zeros(n, dtype=torch.long, device=device)
    cat_perceived[cat_food] = 1
    if threat_present:
        cat_perceived[detector] = 2 + threat_type

    x = torch.randn(n, D_OBS, device=device) * 0.25
    x[cat_perceived == 1] += W['mu_food']
    x[cat_perceived == 2] += W['mu_threat_predator']
    x[cat_perceived == 3] += W['mu_threat_poison']

    return x, cat_perceived, (threat_present, threat_type)


# ============================================================
# 2. 유전자형
# ============================================================
GENE_NAMES = [
    'lam', 'eta_r', 'gamma_lam', 'theta_e', 'theta_i', 'sigma',
    'alpha_p', 'alpha_w', 'alpha_s', 'delta0', 'beta', 'merge_dist',
    'hunger_w', 'sig_out_scale', 'sig_in_scale', 'reciprocity_weight',
    'social_lr', 'social_threshold'
]

GENE_RANGE = {
    'lam': (0.6, 0.95), 'eta_r': (0.001, 0.01), 'gamma_lam': (0.5, 0.9),
    'theta_e': (0.05, 0.25), 'theta_i': (0.05, 0.4), 'sigma': (0.5, 2.0),
    'alpha_p': (0.05, 0.3), 'alpha_w': (0.05, 0.3), 'alpha_s': (0.01, 0.3),
    'delta0': (0.005, 0.03), 'beta': (0.5, 3.0), 'merge_dist': (0.3, 1.2),
    'hunger_w': (0.0, 3.0),
    'sig_out_scale': (0.0, 2.0),
    'sig_in_scale': (0.0, 2.0),
    'reciprocity_weight': (0.0, 2.0),
    'social_lr': (0.001, 0.2),
    'social_threshold': (0.0, 1.0),
}


def init_genotype(n):
    g = {}
    for name in GENE_NAMES:
        lo, hi = GENE_RANGE[name]
        g[name] = torch.empty(n, device=device).uniform_(lo, hi)
    g['sigma_mut'] = torch.full((n,), 0.25, device=device)
    return g


def mutate_genotype(g, parent_idx):
    n = parent_idx.shape[0]
    sm = g['sigma_mut'][parent_idx]
    sm_child = torch.clamp(sm * torch.exp(0.3 * torch.randn(n, device=device)),
                            min=0.05, max=1.0)
    child = {'sigma_mut': sm_child}
    for name in GENE_NAMES:
        val = g[name][parent_idx]
        noise = torch.randn(n, device=device)
        new_val = val * torch.exp(sm_child * noise)
        lo, hi = GENE_RANGE[name]
        # 느슨한 클램핑 (벽에 달라붙는 것 방지)
        new_val = torch.clamp(new_val, min=lo * 0.2, max=hi * 3.0)
        child[name] = new_val
    return child


# ============================================================
# 3. 개체 상태 및 신경망 연산
# ============================================================
def init_agents(n):
    return dict(
        h=torch.zeros(n, D_H, device=device),
        W=torch.zeros(n, D_H, D_H, device=device),
        v=torch.ones(n, K_RES, device=device),
        proto=torch.zeros(n, J_MAX, D_H, device=device),
        proto_active=torch.zeros(n, J_MAX, dtype=torch.bool, device=device),
        w_ji=torch.zeros(n, J_MAX, K_RES, device=device),
        s_ji=torch.zeros(n, J_MAX, K_RES, device=device),
        trace=torch.zeros(n, J_MAX, device=device),
        alive=torch.ones(n, dtype=torch.bool, device=device),
        survival_t=torch.zeros(n, dtype=torch.long, device=device),
        social_signal_prev=torch.zeros(n, D_SIG, device=device),
        social_w=torch.zeros(n, D_SIG, device=device),
        gen_sim=None,
        signal_credit=torch.zeros(n, n, device=device),
    )


def compute_genetic_similarity(g, n):
    keys = ['sigma', 'theta_i', 'hunger_w', 'sig_out_scale', 'sig_in_scale',
            'reciprocity_weight', 'social_lr', 'social_threshold']
    vectors = []
    for k in keys:
        vec = g[k]
        norm = (vec - vec.mean()) / (vec.std() + 1e-8)
        vectors.append(norm)
    mat = torch.stack(vectors, dim=1)
    sim = mat @ mat.T
    return sim


def compute_hidden(x, v, social_signal, h_prev, W_prev, W):
    input_vec = torch.cat([x, v, social_signal], dim=1)
    rec = torch.bmm(W_prev, h_prev.unsqueeze(-1)).squeeze(-1)
    return torch.tanh(input_vec @ W['W_in'].T + rec)


def compute_activation(h_t, proto, proto_active, sigma):
    dist2 = ((proto - h_t.unsqueeze(1)) ** 2).sum(-1)
    c = torch.exp(-dist2 / (2 * sigma.unsqueeze(1) ** 2 + 1e-8))
    return c * proto_active.float()


# ============================================================
# 4. 핵심 스텝 함수
# ============================================================
def step(state, g, t, env, W, force_stimulus=None, force_no_damage=False,
         learn=True, explore=True):
    n = state['h'].shape[0]

    if force_stimulus is None:
        x, cat_perceived, threat_info = sample_stimuli_and_threat(n, t, env, W)
    else:
        x, cat_perceived, threat_info = force_stimulus

    threat_present, threat_type = threat_info

    h_prev = state['h']
    W_prev = state['W']
    v_prev = state['v']
    social_prev = state['social_signal_prev']

    h_t = compute_hidden(x, v_prev, social_prev * g['sig_in_scale'].unsqueeze(1),
                          h_prev, W_prev, W)

    c = compute_activation(h_t, state['proto'], state['proto_active'], g['sigma'])
    score = torch.einsum('nj,njk->nk', c, state['w_ji'] - state['s_ji'])
    max_score, _ = score.max(dim=1)

    energy = v_prev[:, 0]
    social_influence = social_prev[:, 0]
    threshold = g['theta_i'] * (1.0 + g['hunger_w'] * (1.0 - energy)) * (1.0 - 0.5 * torch.tanh(social_influence))

    social_score = (social_prev * state['social_w']).sum(dim=1)

    avoid_det = (max_score > threshold) | (social_score > g['social_threshold']) | (cat_perceived == 2)

    if explore:
        explore_mask = torch.rand(n, device=device) < 0.05
        avoid_random = torch.rand(n, device=device) < 0.5
        avoid = torch.where(explore_mask, avoid_random, avoid_det)
    else:
        avoid = avoid_det

    is_food = (cat_perceived == 1)
    approach = ~avoid

    energy_delta = -env['energy_decay'] * torch.ones(n, device=device)
    energy_delta += torch.where(is_food & approach,
                                 torch.full((n,), env['food_gain'], device=device),
                                 torch.zeros(n, device=device))
    energy_delta -= torch.where(avoid,
                                 torch.full((n,), env['flee_cost'], device=device),
                                 torch.zeros(n, device=device))

    signal_sent = (cat_perceived >= 2).float()
    energy_delta -= torch.where(signal_sent.bool(), torch.full((n,), env['signal_cost'], device=device),
                                 torch.zeros(n, device=device))

    damage = torch.zeros(n, device=device)
    if threat_present:
        if threat_type == 0:
            strike_roll = torch.rand(n, device=device) < env['p_strike']
            damage = torch.where(approach & strike_roll,
                                  torch.full((n,), env['threat_damage_predator'], device=device),
                                  torch.zeros(n, device=device))
        else:
            energy_delta -= torch.where(approach,
                                         torch.full((n,), env['poison_energy_cost'], device=device),
                                         torch.zeros(n, device=device))
            damage = torch.where(approach,
                                  torch.full((n,), env['threat_damage_poison'], device=device),
                                  torch.zeros(n, device=device))

    if force_no_damage:
        damage = torch.zeros(n, device=device)

    novelty = torch.clamp(1.0 - max_score.detach(), min=0.0, max=1.0)
    curiosity_delta = -env.get('curiosity_decay', 0.0) * torch.ones(n, device=device)
    curiosity_delta += torch.where(approach, env.get('curiosity_gain', 0.0) * novelty,
                                    torch.zeros(n, device=device))

    v_new = v_prev.clone()
    v_new[:, 0] = torch.clamp(v_prev[:, 0] + energy_delta, 0.0, 1.0)
    v_new[:, 1] = torch.clamp(v_prev[:, 1] - damage, 0.0, 1.0)
    v_new[:, 2] = torch.clamp(v_prev[:, 2] + curiosity_delta, 0.0, 1.0)
    e_t = v_new - v_prev

    signal_out = h_t * g['sig_out_scale'].unsqueeze(1)
    signal_out = signal_out * signal_sent.unsqueeze(1)

    social_w = state['social_w'].clone()
    if learn:
        reward = (1.0 - damage)
        lr = g['social_lr'].unsqueeze(1)
        if avoid.any():
            social_w = social_w + lr * reward.unsqueeze(1) * social_prev * avoid.float().unsqueeze(1)
        if approach.any():
            social_w = social_w - lr * damage.unsqueeze(1) * social_prev * approach.float().unsqueeze(1)
        social_w = torch.clamp(social_w, -5.0, 5.0)

    lam = g['lam'].view(n, 1, 1)
    eta_r = g['eta_r'].view(n, 1, 1)
    W_t = lam * W_prev + eta_r * torch.einsum('ni,nj->nij', h_t, h_prev)
    W_t = torch.clamp(W_t, -5.0, 5.0)

    gl = g['gamma_lam'].unsqueeze(1)
    trace_new = gl * state['trace'] + (1.0 - gl) * c

    drop = (-e_t) > g['theta_e'].unsqueeze(1)
    any_drop = drop.any(dim=1)

    proto = state['proto'].clone()
    proto_active = state['proto_active'].clone()
    w_ji = state['w_ji'].clone()
    s_ji = state['s_ji'].clone()

    if learn and any_drop.any():
        idx = any_drop.nonzero(as_tuple=True)[0]
        c_sub = c[idx]
        max_c, argmax_j = c_sub.max(dim=1)
        merge_thresh = torch.exp(-1.0 / (2.0 * g['merge_dist'][idx] ** 2 + 1e-8))
        do_merge = max_c > merge_thresh

        usage = w_ji[idx].max(dim=-1).values
        usage = torch.where(proto_active[idx], usage, torch.full_like(usage, -1.0))
        replace_j = usage.argmin(dim=1)
        target_j = torch.where(do_merge, argmax_j, replace_j)

        alpha_p = g['alpha_p'][idx]
        alpha_w = g['alpha_w'][idx]
        h_cur = h_t[idx]
        trace_cur = trace_new[idx]
        e_cur = e_t[idx]
        theta_e_cur = g['theta_e'][idx]
        m = idx.shape[0]
        m_ar = torch.arange(m, device=device)

        new_created = ~do_merge
        base_trace = trace_cur[m_ar, target_j]
        tgt_trace = torch.where(new_created, torch.ones_like(base_trace), base_trace)

        old_pos = proto[idx, target_j]
        pulled = old_pos + alpha_p.unsqueeze(-1) * tgt_trace.unsqueeze(-1) * (h_cur - old_pos)
        proto[idx, target_j] = torch.where(new_created.unsqueeze(-1), h_cur, pulled)
        proto_active[idx, target_j] = True

        excess = torch.clamp(-e_cur - theta_e_cur.unsqueeze(-1), min=0.0)
        dw = alpha_w.unsqueeze(-1) * tgt_trace.unsqueeze(-1) * excess
        w_ji[idx, target_j] += dw

    if learn:
        predicted_threat_j = c.unsqueeze(-1) * (w_ji - s_ji)
        positive_pred = predicted_threat_j.clamp(min=0.0)
        avoided_but_safe = avoid.unsqueeze(1) & ~drop
        integrity_mask = torch.zeros(K_RES, device=device)
        integrity_mask[1] = 1.0
        avoided_but_safe_integrity = avoided_but_safe * integrity_mask.unsqueeze(0)
        ds_wrong = positive_pred * avoided_but_safe_integrity.unsqueeze(1).float()
        ds_wrong = ds_wrong * g['alpha_s'].view(n, 1, 1)

        safe_margin = torch.clamp(g['theta_e'].unsqueeze(1) - (-e_t), min=0.0)
        safe_margin_integrity = safe_margin * integrity_mask.unsqueeze(0)
        ds_safe = torch.einsum('nj,nk->njk', trace_new, safe_margin_integrity) * (0.1 * g['alpha_s']).view(n, 1, 1)
        s_ji = s_ji + ds_wrong + ds_safe

        if (e_t[:, 1] < -0.3).any():
            s_ji[:, :, 1] = s_ji[:, :, 1] * 0.5

        if not force_no_damage:
            drop_mag = torch.clamp(-e_t, min=0.0)
            delta_i = g['delta0'].unsqueeze(1) * torch.exp(-g['beta'].unsqueeze(1) * drop_mag)
            w_ji = w_ji * (1.0 - delta_i).view(n, 1, K_RES)

        weak = (w_ji.max(dim=-1).values < 1e-3) & (trace_new < 1e-3)
        proto_active = proto_active & (~weak)

    if n > 1 and state['gen_sim'] is not None:
        mask = 1.0 - torch.eye(n, device=device)
        sim_weighted = state['gen_sim'] * mask
        credit_weighted = sim_weighted * (1.0 + g['reciprocity_weight'].unsqueeze(1) * state['signal_credit'])
        social_signal_received = torch.einsum('ij,jk->ik', credit_weighted, signal_out)
        row_sum = credit_weighted.sum(dim=1, keepdim=True) + 1e-8
        social_signal_received = social_signal_received / row_sum

        new_credit = signal_sent.unsqueeze(1) * sim_weighted
        state['signal_credit'] = state['signal_credit'] * (1.0 - SIGNAL_DECAY) + new_credit
    else:
        social_signal_received = torch.zeros_like(signal_out)

    new_alive = state['alive'] & ~(v_new <= 0).any(dim=1)
    survival_t = state['survival_t'] + state['alive'].long()

    state.update(
        h=h_t, W=W_t, v=v_new, proto=proto, proto_active=proto_active,
        w_ji=w_ji, s_ji=s_ji, trace=trace_new, alive=new_alive,
        survival_t=survival_t,
        social_signal_prev=social_signal_received,
        social_w=social_w
    )
    return state, dict(cat=cat_perceived, avoid=avoid, e_t=e_t, score=score,
                        signal_out=signal_out, threat_present=threat_present,
                        threat_type=threat_type, social_score=social_score)


def run_social(n, g, t_max, env, W):
    state = init_agents(n)
    state['gen_sim'] = compute_genetic_similarity(g, n)
    for t in range(t_max):
        if not state['alive'].any():
            break
        state, info = step(state, g, t, env, W)
    return state


# ============================================================
# 5. 진화 루프
# ============================================================
def evolve(pop_size, generations, t_max, W, stage_gen=40, transition_gen=60):
    g = init_genotype(pop_size)
    best_idx_overall = None
    best_T_overall = 0.0

    for gen in range(generations):
        if gen < stage_gen:
            progress = 0.0
        else:
            raw = min(1.0, (gen - stage_gen) / transition_gen)
            progress = raw ** 1.5
        env = make_env(progress)

        state = run_social(pop_size, g, t_max, env, W)
        surv = state['survival_t'].float()

        max_T, max_idx = torch.max(surv, 0)
        if max_T > best_T_overall:
            best_T_overall = max_T
            best_idx_overall = max_idx.item()

        n_parents = max(1, pop_size // 5)
        _, top_idx = torch.topk(surv, n_parents)

        offspring_per = pop_size // n_parents
        child_idx = torch.repeat_interleave(top_idx, offspring_per)
        if child_idx.shape[0] < pop_size:
            extra = torch.randint(n_parents, (pop_size - child_idx.shape[0],), device=device)
            child_idx = torch.cat([child_idx, top_idx[extra]])
        g = mutate_genotype(g, child_idx)

    if best_idx_overall is None:
        best_idx_overall = 0
    return g, best_idx_overall


def _extract_individual_state(state_full, idx, pop_size):
    ind = {}
    for key, val in state_full.items():
        if isinstance(val, torch.Tensor) and val.shape[0] == pop_size:
            ind[key] = val[idx:idx + 1].clone()
        else:
            ind[key] = val
    ind['gen_sim'] = None
    ind['signal_credit'] = torch.zeros(1, 1, device=device)
    ind['social_signal_prev'] = torch.zeros(1, D_SIG, device=device)
    return ind


def _repeat_state(state, n_trials):
    """개체 상태(batch=1)를 n_trials개로 복제해 배치 차원을 만든다."""
    out = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            if v.dim() == 1:
                out[k] = v.repeat(n_trials)
            else:
                out[k] = v.repeat(n_trials, *([1] * (v.dim() - 1)))
        else:
            out[k] = v
    out['gen_sim'] = None
    return out


def _repeat_genotype(g1, n_trials):
    return {k: (v.repeat(n_trials) if v.dim() == 1 else v.repeat(n_trials, *([1] * (v.dim() - 1))))
            for k, v in g1.items()}


# ============================================================
# 6. 벡터화된 교차개체 신호 테스트 (핵심 성능 개선)
#    - n_trials를 배치 차원으로 처리하여 파이썬 반복문 제거
#    - discrimination을 부호(direction)와 크기(specificity)로 분리
# ============================================================
def _get_sender_signal_batched(sender_state, sender_g, W, threat_type, n_trials):
    env = make_env(1.0)
    if threat_type == 0:
        mu = W['mu_threat_predator']
        cat_val = 2
    else:
        mu = W['mu_threat_poison']
        cat_val = 3
    threat_x = mu.unsqueeze(0).repeat(n_trials, 1) + torch.randn(n_trials, D_OBS, device=device) * 0.05
    cat_perceived = torch.full((n_trials,), cat_val, dtype=torch.long, device=device)

    s_batch = _repeat_state(sender_state, n_trials)
    g_batch = _repeat_genotype(sender_g, n_trials)

    _, info = step(s_batch, g_batch, 0, env, W,
                    force_stimulus=(threat_x, cat_perceived, (True, threat_type)),
                    learn=False, explore=False)
    return info['signal_out'].detach()  # [n_trials, D_SIG]


def _paired_receiver_test_batched(receiver_state, receiver_g, W, signals, n_trials,
                                   noise_low=0.05, noise_high=0.45):
    env = make_env(1.0)
    noise_std = torch.empty(n_trials, 1, device=device).uniform_(noise_low, noise_high)
    neutral_x = torch.randn(n_trials, D_OBS, device=device) * noise_std
    neutral_cat = torch.zeros(n_trials, dtype=torch.long, device=device)
    threat_info = (False, 0)

    idx = torch.arange(n_trials, device=device) % signals.shape[0]
    sig = signals[idx]

    g_batch = _repeat_genotype(receiver_g, n_trials)

    r_base = _repeat_state(receiver_state, n_trials)
    r_base['social_signal_prev'] = torch.zeros(n_trials, D_SIG, device=device)
    _, info_base = step(r_base, g_batch, 0, env, W,
                         force_stimulus=(neutral_x, neutral_cat, threat_info),
                         learn=False, explore=False)
    base_avoid = info_base['avoid'].float()

    r_treat = _repeat_state(receiver_state, n_trials)
    r_treat['social_signal_prev'] = sig
    _, info_treat = step(r_treat, g_batch, 0, env, W,
                          force_stimulus=(neutral_x, neutral_cat, threat_info),
                          learn=False, explore=False)
    treat_avoid = info_treat['avoid'].float()

    deltas = (treat_avoid - base_avoid).cpu().numpy()
    return deltas, base_avoid.mean().item(), treat_avoid.mean().item()


def cross_agent_signal_test_paired(individuals, W, n_trials=100):
    pair_results = []
    n_ind = len(individuals)

    for s in range(n_ind):
        sender_state = individuals[s]['state']
        sender_g = individuals[s]['g']
        # 발신자별 신호는 재사용 (같은 발신자에 대해 매 수신자마다 다시 계산할 필요 없음)
        pred_signal = _get_sender_signal_batched(sender_state, sender_g, W, 0, n_trials)
        pois_signal = _get_sender_signal_batched(sender_state, sender_g, W, 1, n_trials)

        for r in range(n_ind):
            if s == r:
                continue
            receiver_state = individuals[r]['state']
            receiver_g = individuals[r]['g']

            pred_deltas, pred_base, pred_treat = _paired_receiver_test_batched(
                receiver_state, receiver_g, W, pred_signal, n_trials)
            pois_deltas, pois_base, pois_treat = _paired_receiver_test_batched(
                receiver_state, receiver_g, W, pois_signal, n_trials)

            # ttest: 분산이 0인 퇴화 사례(모든 델타가 동일)에서 나오는 경고를 피하기 위해 방어적으로 처리
            if np.std(pred_deltas) > 1e-8:
                t_pred, p_pred = stats.ttest_1samp(pred_deltas, 0.0)
            else:
                t_pred, p_pred = (0.0, 1.0) if abs(pred_deltas.mean()) < 1e-8 else (np.inf, 0.0)
            if np.std(pois_deltas) > 1e-8:
                t_pois, p_pois = stats.ttest_1samp(pois_deltas, 0.0)
            else:
                t_pois, p_pois = (0.0, 1.0) if abs(pois_deltas.mean()) < 1e-8 else (np.inf, 0.0)

            pair_results.append(dict(
                seed=individuals[s].get('seed', -1),
                sender=individuals[s]['label'], receiver=individuals[r]['label'],
                pred_baseline=pred_base, pred_treat=pred_treat, pred_delta=pred_deltas.mean(),
                pois_baseline=pois_base, pois_treat=pois_treat, pois_delta=pois_deltas.mean(),
                pred_p=p_pred, pois_p=p_pois,
            ))

    return pair_results


def aggregate_receiver_scores(pair_results, individuals,
                               nonresponse_thresh=0.1, specificity_thresh=0.2):
    """
    수신자별로 집계. 분류를 방향(direction)과 무관하게 먼저 '반응 크기'로 나누고,
    그다음 방향으로 세분화한다 (기존 버전의 방향성 버그 수정).
    """
    by_receiver = {ind['label']: [] for ind in individuals}
    for r in pair_results:
        by_receiver[r['receiver']].append(r)

    receiver_scores = []
    for ind in individuals:
        rows = by_receiver[ind['label']]
        if not rows:
            continue
        pred_delta = np.mean([r['pred_delta'] for r in rows])
        pois_delta = np.mean([r['pois_delta'] for r in rows])

        # 방향(있는 그대로의 부호) - "포식자 쪽으로 치우쳤는가"
        direction_score = pred_delta - pois_delta
        # 특이성(부호 무관) - "둘 중 하나에 선택적으로 반응하는가"
        specificity = abs(pred_delta - pois_delta)
        # 경보 강도(둘 다에 반응하는 정도)
        alarm_score = (pred_delta + pois_delta) / 2.0

        if alarm_score < nonresponse_thresh and specificity < specificity_thresh:
            group = "nonresponder"
        elif specificity > specificity_thresh:
            group = "predator_specialist" if direction_score > 0 else "poison_specialist"
        else:
            group = "generalist"

        receiver_scores.append(dict(
            seed=ind.get('seed', -1), label=ind['label'],
            pred_delta=pred_delta, pois_delta=pois_delta,
            direction_score=direction_score, specificity=specificity, alarm_score=alarm_score,
            group=group, survival_t=ind.get('survival_t', float('nan')),
            **{gene: ind['g'][gene].item() for gene in
               ['sigma', 'theta_i', 'theta_e', 'hunger_w', 'sig_out_scale',
                'sig_in_scale', 'reciprocity_weight', 'social_lr', 'social_threshold']}
        ))
    return receiver_scores


# ============================================================
# 7. 실제 다개체 환경에서의 자연발생적 신호 효과
# ============================================================
def run_social_instrumented(n, g, t_max, env, W):
    state = init_agents(n)
    state['gen_sim'] = compute_genetic_similarity(g, n)

    log_true, log_false = [], []
    prev_signal_sent = torch.zeros(n, dtype=torch.bool, device=device)

    for t in range(t_max):
        if not state['alive'].any():
            break
        state, info = step(state, g, t, env, W)

        cat = info['cat']
        detector_mask = cat >= 2
        non_detector_mask = ~detector_mask
        avoid = info['avoid']

        if info['threat_present'] and non_detector_mask.any():
            avoid_rate = avoid[non_detector_mask].float().mean().item()
            if prev_signal_sent.any():
                log_true.append(avoid_rate)
            else:
                log_false.append(avoid_rate)

        prev_signal_sent = detector_mask.clone()

    return state, log_true, log_false


def in_situ_group_test(g, W, n_reps=3, t_max=300):
    all_true, all_false = [], []
    env = make_env(1.0)
    pop_size = next(iter(g.values())).shape[0]

    for _ in range(n_reps):
        _, lt, lf = run_social_instrumented(pop_size, g, t_max, env, W)
        all_true.extend(lt)
        all_false.extend(lf)

    all_true = np.array(all_true)
    all_false = np.array(all_false)

    result = dict(mean_after_signal=float(all_true.mean()) if len(all_true) else float('nan'),
                   mean_no_signal=float(all_false.mean()) if len(all_false) else float('nan'),
                   n_after_signal=len(all_true), n_no_signal=len(all_false))
    if len(all_true) >= 3 and len(all_false) >= 3:
        t_val, p_val = stats.ttest_ind(all_true, all_false, equal_var=False)
        result['t_stat'] = t_val
        result['p_value'] = p_val
    return result


# ============================================================
# 8. 다중비교 보정 (Benjamini-Hochberg FDR)
# ============================================================
def benjamini_hochberg(p_values, alpha=0.05):
    """반환: 각 항목이 FDR 기준을 통과하는지 여부(bool 배열), 보정된 p값"""
    p_values = np.array(p_values)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresh
    # 가장 큰 순위에서부터 처음으로 통과하는 지점까지 모두 유의로 처리
    if passed.any():
        max_pass_rank = np.max(np.where(passed)[0])
        significant = np.zeros(n, dtype=bool)
        significant[order[:max_pass_rank + 1]] = True
    else:
        significant = np.zeros(n, dtype=bool)
    # 보정된 p값(대략적, BH 방식)
    adj_p = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adj_p_full = np.empty(n)
    adj_p_full[order] = np.clip(adj_p, 0, 1)
    return significant, adj_p_full


# ============================================================
# 9. 다중 시드 실행 (규모 확대 + 통계 수정)
# ============================================================
def run_full_experiment(n_seeds=20, pop_size=256, generations=80, t_max=200,
                         top_k=20, cross_trials=150, in_situ_reps=3,
                         out_dir="/mnt/user-data/outputs"):
    os.makedirs(out_dir, exist_ok=True)

    all_pair_results = []
    all_receiver_scores = []
    seed_summaries = []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        W = make_world_constants(seed * 100 + 1)

        g, best_idx = evolve(pop_size, generations, t_max, W)

        env_final = make_env(1.0)
        state_last = run_social(pop_size, g, t_max, env_final, W)
        surv_last = state_last['survival_t'].float()
        _, top_indices = torch.topk(surv_last, min(top_k, pop_size))

        individuals = []
        for i, idx in enumerate(top_indices):
            g1 = {k: v[idx:idx + 1].clone() for k, v in g.items()}
            ind_state = _extract_individual_state(state_last, idx.item(), pop_size)
            individuals.append(dict(
                g=g1, state=ind_state, label=f"s{seed}_r{i}",
                seed=seed, survival_t=surv_last[idx].item()
            ))

        pair_results = cross_agent_signal_test_paired(individuals, W, n_trials=cross_trials)
        receiver_scores = aggregate_receiver_scores(pair_results, individuals)

        in_situ = in_situ_group_test(g, W, n_reps=in_situ_reps, t_max=max(t_max, 300))

        counts = {}
        for grp in ["predator_specialist", "poison_specialist", "generalist", "nonresponder"]:
            counts[grp] = sum(1 for r in receiver_scores if r['group'] == grp)

        print(f"시드 {seed:2d}: 포식자특화={counts['predator_specialist']:2d}  독초특화={counts['poison_specialist']:2d}  "
              f"범용경보={counts['generalist']:2d}  무반응={counts['nonresponder']:2d}  (top_k={top_k})  "
              f"| 다개체환경 신호효과 delta={in_situ['mean_after_signal']-in_situ['mean_no_signal']:+.3f} "
              f"(p={in_situ.get('p_value', float('nan')):.4f})")

        all_pair_results.extend(pair_results)
        all_receiver_scores.extend(receiver_scores)
        seed_summaries.append(dict(seed=seed, **counts, **in_situ))

    # ---------- 전체 요약 ----------
    total = len(all_receiver_scores)
    groups = {grp: [r for r in all_receiver_scores if r['group'] == grp]
              for grp in ["predator_specialist", "poison_specialist", "generalist", "nonresponder"]}

    print(f"\n=== 전체 {n_seeds}시드 {total}개체 요약 ===")
    for name, grp in groups.items():
        print(f"{name:22s}: {len(grp)}/{total} ({100*len(grp)/total:.1f}%)")
    print(f"(참고: '구별형' = 포식자특화 + 독초특화 = {len(groups['predator_specialist'])+len(groups['poison_specialist'])}/{total} "
          f"({100*(len(groups['predator_specialist'])+len(groups['poison_specialist']))/total:.1f}%))")

    # ---------- 그룹별 생존시간 비교 ----------
    print(f"\n=== 그룹별 생존시간(survival_t) 비교 ===")
    group_survs = {}
    for name, grp in groups.items():
        vals = np.array([r['survival_t'] for r in grp])
        group_survs[name] = vals
        if len(vals) > 0:
            print(f"  {name:22s}: n={len(vals):3d}  평균={vals.mean():.1f}  표준편차={vals.std():.1f}")
        else:
            print(f"  {name:22s}: n=0")

    valid_groups = [v for v in group_survs.values() if len(v) >= 2]
    if len(valid_groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*valid_groups)
        print(f"  일원분산분석(ANOVA): F={f_stat:.2f} p={p_anova:.4f}")

    # ---------- 유전자형과 특이성(specificity)/방향(direction)의 상관 + 다중비교 보정 ----------
    print(f"\n=== 유전자형 변수와 특이성(specificity)의 상관 (부호 무관, n={total}, BH-FDR 보정) ===")
    genes_to_test = ['sigma', 'theta_i', 'theta_e', 'hunger_w', 'sig_out_scale',
                      'sig_in_scale', 'reciprocity_weight', 'social_lr', 'social_threshold']
    spec_vals = np.array([r['specificity'] for r in all_receiver_scores])
    dir_vals = np.array([r['direction_score'] for r in all_receiver_scores])

    corr_rows = []
    p_list = []
    for gene in genes_to_test:
        vals = np.array([r[gene] for r in all_receiver_scores])
        if np.std(vals) < 1e-8:
            continue
        r_spec, p_spec = stats.pearsonr(vals, spec_vals)
        r_dir, p_dir = stats.pearsonr(vals, dir_vals)
        corr_rows.append((gene, r_spec, p_spec, r_dir, p_dir))
        p_list.append(p_spec)

    sig_mask, adj_p = benjamini_hochberg(p_list) if p_list else (np.array([]), np.array([]))
    corr_rows_sorted = sorted(zip(corr_rows, sig_mask, adj_p), key=lambda x: x[2])

    for (gene, r_spec, p_spec, r_dir, p_dir), is_sig, ap in corr_rows_sorted:
        flag = " *(FDR통과)" if is_sig else ""
        print(f"  {gene:20s}: 특이성 r={r_spec:+.3f} p={p_spec:.4f} (보정p={ap:.4f}){flag}   |   방향 r={r_dir:+.3f} p={p_dir:.4f}")

    # ---------- CSV 저장 ----------
    if all_pair_results:
        pair_csv = os.path.join(out_dir, "cross_agent_pair_results.csv")
        with open(pair_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_pair_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_pair_results)
        print(f"\n저장됨: {pair_csv}")

    if all_receiver_scores:
        receiver_csv = os.path.join(out_dir, "receiver_classification_genotype.csv")
        with open(receiver_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_receiver_scores[0].keys()))
            writer.writeheader()
            writer.writerows(all_receiver_scores)
        print(f"저장됨: {receiver_csv}")

    if seed_summaries:
        seed_csv = os.path.join(out_dir, "seed_level_summary.csv")
        with open(seed_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(seed_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(seed_summaries)
        print(f"저장됨: {seed_csv}")

    if corr_rows:
        corr_csv = os.path.join(out_dir, "genotype_discrimination_correlations.csv")
        with open(corr_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["gene", "pearson_r_specificity", "p_specificity", "adj_p_BH",
                              "fdr_significant", "pearson_r_direction", "p_direction"])
            for (gene, r_spec, p_spec, r_dir, p_dir), is_sig, ap in corr_rows_sorted:
                writer.writerow([gene, r_spec, p_spec, ap, is_sig, r_dir, p_dir])
        print(f"저장됨: {corr_csv}")

    return dict(pair_results=all_pair_results, receiver_scores=all_receiver_scores,
                seed_summaries=seed_summaries, correlations=corr_rows_sorted)


# ============================================================
# 10. 실행
# ============================================================
if __name__ == "__main__":
    # Colab T4 기준: 교차신호 테스트가 벡터화되어 top_k^2에 비례하던 병목이
    # 크게 줄었습니다. 아래는 규모를 키운 기본값입니다 - 먼저 이대로 돌려보고,
    # 시간이 남으면 n_seeds/pop_size/top_k를 더 키우세요.
    results = run_full_experiment(
        n_seeds=20, pop_size=256, generations=80, t_max=200,
        top_k=20, cross_trials=150, in_situ_reps=3,
    )
