"""
항상성 기반 위협 학습 시뮬레이션 - 최종 통합본
=================================================
이 대화에서 합의된 설계의 전체 구현:

  뼈대
  - 은닉 상태: h_t = tanh(W_in x_t + W_v v_{t-1} + W_t h_{t-1})   [내수용감각 포함]
  - fast weight(개체 생애 내 학습): W_t = lam W_{t-1} + eta (h_t 외적 h_{t-1})
  - 항상성 자원 v = [energy, integrity, curiosity], 놀라움 e_t = v_t - v_{t-1}
  - 프로토타입 기반 위협 기억: eligibility trace + 흥분성/억제성 연관(w_ji, s_ji)
    -> 소거(extinction)와 재발(reinstatement)을 구조적으로 지원
  - 행동: 명시적 계획 없이, 프로토타입 활성화 점수가 임계값을 넘으면 즉각 회피 반사

  진화 (slow weights 역할)
  - 유전자형은 스칼라 하이퍼파라미터만 (학습 규칙의 모수). W_0, 프로토타입 초기위치는
    유전되지 않음 (볼드윈 효과 방지).
  - 로그공간 승법 변이 + 자기적응 변이폭(step size 자체도 진화) + 느슨한 클램핑
  - 순위 기반 선택 (생존시간 T)

  이번 라운드에 추가된 것
  - 확률적 위협(p_strike<1): 접근해도 항상 다치지 않음 -> 불확실성
  - 호기심 자원: 낯선 자극에 접근하면 보상 -> 무조건 회피에 대한 반대 압력
  - 내수용감각: v_{t-1}이 h_t 계산에 직접 입력됨 -> "배고픈 상태" 자체를 표상 가능
  - 다중 시드 반복 실행 (단일 실행의 노이즈 문제 해결)
  - 일반화 테스트를 여러 고정 v 조건에서 따로 실행 (내부상태가 결과를 뒤덮는지 확인)

주의: 이 코드는 "의식/의지가 창발했다"를 증명하지 않습니다. 확인하는 것은 두려움의
행동적 서명(반사, 일반화, 소거/재발, 위협기억의 비대칭 보존)이 이 최소 규칙 집합에서
창발적으로 나타나는가 하는, 훨씬 좁고 검증 가능한 질문입니다.
"""

import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device = {device}")

# ============================================================
# 1. 환경
# ============================================================
D_OBS = 8
D_H = 16
K_RES = 3          # [energy, integrity, curiosity]
J_MAX = 8
NOISE_STD = 0.25

ENERGY_DECAY = 0.01
FOOD_GAIN = 0.25
THREAT_DAMAGE = 0.5
T_SAFE = 80
CURIOSITY_DECAY = 0.012
CURIOSITY_GAIN = 0.20
P_STRIKE = 0.4


def make_world_constants(seed):
    """자극 클러스터 중심과 감각/내수용 사영을 시드별로 고정."""
    g = torch.Generator(device=device).manual_seed(seed)
    mu_food = torch.randn(D_OBS, device=device, generator=g)
    mu_threat = torch.randn(D_OBS, device=device, generator=g)
    mu_threat_safe = torch.randn(D_OBS, device=device, generator=g)
    W_in = torch.randn(D_H, D_OBS, device=device, generator=g) * 0.5
    W_v = torch.randn(D_H, K_RES, device=device, generator=g) * 0.8
    return dict(mu_food=mu_food, mu_threat=mu_threat, mu_threat_safe=mu_threat_safe,
                W_in=W_in, W_v=W_v)


def make_env(progress):
    p_both = 0.05 * progress
    p_safe = 0.08 * progress
    p_food = 0.25
    p_threat = 0.25
    p_neutral = max(0.0, 1.0 - p_food - p_threat - p_both - p_safe)
    return dict(p_food=p_food, p_threat=p_threat, p_both=p_both, p_safe=p_safe,
                p_neutral=p_neutral, flee_cost=0.01 + 0.02 * progress,
                threat_damage=THREAT_DAMAGE, energy_decay=ENERGY_DECAY,
                food_gain=FOOD_GAIN, T_SAFE=T_SAFE, p_strike=P_STRIKE,
                curiosity_decay=CURIOSITY_DECAY, curiosity_gain=CURIOSITY_GAIN)


def sample_stimuli(n, t, env, W):
    r = torch.rand(n, device=device)
    cat = torch.zeros(n, dtype=torch.long, device=device)
    t1 = env['p_food']
    t2 = t1 + env['p_threat']
    t3 = t2 + env['p_both']
    t4 = t3 + env['p_safe']
    cat[(r < t1)] = 1
    cat[(r >= t1) & (r < t2)] = 2
    cat[(r >= t2) & (r < t3)] = 3
    cat[(r >= t3) & (r < t4)] = 4
    x = torch.randn(n, D_OBS, device=device) * NOISE_STD
    x[cat == 1] += W['mu_food']
    x[cat == 2] += W['mu_threat']
    x[cat == 3] += 0.5 * W['mu_food'] + 0.5 * W['mu_threat']
    x[cat == 4] += W['mu_threat_safe']
    if t >= env['T_SAFE']:
        cat[cat == 4] = 0
    return x, cat


# ============================================================
# 2. 유전자형
# ============================================================
GENE_NAMES = ['lam', 'eta_r', 'gamma_lam', 'theta_e', 'theta_i', 'sigma',
              'alpha_p', 'alpha_w', 'alpha_s', 'delta0', 'beta', 'merge_dist', 'hunger_w']

GENE_RANGE = {
    'lam': (0.6, 0.95), 'eta_r': (0.001, 0.01), 'gamma_lam': (0.5, 0.9),
    'theta_e': (0.05, 0.25), 'theta_i': (0.01, 0.4), 'sigma': (0.5, 2.0),
    'alpha_p': (0.05, 0.3), 'alpha_w': (0.05, 0.3), 'alpha_s': (0.01, 0.15),
    'delta0': (0.005, 0.03), 'beta': (0.5, 3.0), 'merge_dist': (0.3, 1.2),
    'hunger_w': (0.0, 3.0),
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
    sm_child = torch.clamp(sm * torch.exp(0.3 * torch.randn(n, device=device)), min=0.05, max=1.0)
    child = {'sigma_mut': sm_child}
    for name in GENE_NAMES:
        val = g[name][parent_idx]
        noise = torch.randn(n, device=device)
        new_val = val * torch.exp(sm_child * noise)
        lo, hi = GENE_RANGE[name]
        # 느슨한 클램핑: 벽에 달라붙는 것을 방지 (원래 범위의 0.2~3배까지 허용)
        new_val = torch.clamp(new_val, min=lo * 0.2, max=hi * 3.0)
        child[name] = new_val
    return child


# ============================================================
# 3. 개체 상태 / 한 스텝
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
    )


def compute_hidden(x, v, h_prev, W_prev, W):
    rec = torch.bmm(W_prev, h_prev.unsqueeze(-1)).squeeze(-1)
    return torch.tanh(x @ W['W_in'].T + v @ W['W_v'].T + rec)


def compute_activation(h_t, proto, proto_active, sigma):
    dist2 = ((proto - h_t.unsqueeze(1)) ** 2).sum(-1)
    c = torch.exp(-dist2 / (2 * sigma.unsqueeze(1) ** 2 + 1e-8))
    return c * proto_active.float()


def step(state, g, t, env, W, force_stimulus=None, force_no_damage=False, learn=True, explore=True):
    n = state['h'].shape[0]
    x, cat = sample_stimuli(n, t, env, W) if force_stimulus is None else force_stimulus

    h_prev, W_prev, v_prev = state['h'], state['W'], state['v']
    h_t = compute_hidden(x, v_prev, h_prev, W_prev, W)

    c = compute_activation(h_t, state['proto'], state['proto_active'], g['sigma'])
    score = torch.einsum('nj,njk->nk', c, state['w_ji'] - state['s_ji'])
    max_score, _ = score.max(dim=1)

    energy = v_prev[:, 0]
    threshold = g['theta_i'] * (1.0 + g['hunger_w'] * (1.0 - energy))
    avoid_det = max_score > threshold

    if explore:
        explore_mask = torch.rand(n, device=device) < 0.05
        avoid_random = torch.rand(n, device=device) < 0.5
        avoid = torch.where(explore_mask, avoid_random, avoid_det)
    else:
        avoid = avoid_det

    is_food = (cat == 1) | (cat == 3)
    is_threat = (cat == 2) | (cat == 3) | ((cat == 4) & (t < env['T_SAFE']))
    approach = ~avoid

    energy_delta = -env['energy_decay'] * torch.ones(n, device=device)
    energy_delta += torch.where(is_food & approach, torch.full((n,), env['food_gain'], device=device),
                                 torch.zeros(n, device=device))
    energy_delta -= torch.where(avoid, torch.full((n,), env['flee_cost'], device=device),
                                 torch.zeros(n, device=device))

    strike_roll = torch.rand(n, device=device) < env.get('p_strike', 1.0)
    damage = torch.where(is_threat & approach & strike_roll,
                          torch.full((n,), env['threat_damage'], device=device),
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

    lam = g['lam'].view(n, 1, 1)
    eta_r = g['eta_r'].view(n, 1, 1)
    W_t = lam * W_prev + eta_r * torch.einsum('ni,nj->nij', h_t, h_prev)
    W_t = torch.clamp(W_t, -5.0, 5.0)

    gl = g['gamma_lam'].unsqueeze(1)
    trace_new = gl * state['trace'] + (1 - gl) * c

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
        safe_margin = torch.clamp(g['theta_e'].unsqueeze(1) - (-e_t), min=0.0)
        ds = torch.einsum('nj,nk->njk', trace_new, safe_margin) * g['alpha_s'].view(n, 1, 1)
        s_ji = s_ji + ds
        s_ji = s_ji * (1.0 - 0.02)

        drop_mag = torch.clamp(-e_t, min=0.0)
        delta_i = g['delta0'].unsqueeze(1) * torch.exp(-g['beta'].unsqueeze(1) * drop_mag)
        w_ji = w_ji * (1.0 - delta_i).view(n, 1, K_RES)

        weak = (w_ji.max(dim=-1).values < 1e-3) & (trace_new < 1e-3)
        proto_active = proto_active & (~weak)

    new_alive = state['alive'] & ~(v_new <= 0).any(dim=1)
    survival_t = state['survival_t'] + state['alive'].long()

    state.update(h=h_t, W=W_t, v=v_new, proto=proto, proto_active=proto_active,
                  w_ji=w_ji, s_ji=s_ji, trace=trace_new, alive=new_alive, survival_t=survival_t)
    return state, dict(cat=cat, avoid=avoid, e_t=e_t, score=score)


def run_life(n, g, t_max, env, W, force_schedule=None, learn=True, explore=True):
    state = init_agents(n)
    for t in range(t_max):
        if not state['alive'].any() and force_schedule is None:
            break
        fs, fnd = (force_schedule[t] if force_schedule is not None and t < len(force_schedule) else (None, False))
        state, info = step(state, g, t, env, W, force_stimulus=fs, force_no_damage=fnd, learn=learn, explore=explore)
    return state


# ============================================================
# 4. 진화 루프
# ============================================================
def evolve(pop_size, generations, t_max, W, stage_gen=30, transition_gen=70, verbose=True):
    g = init_genotype(pop_size)
    history = []
    for gen in range(generations):
        progress = 0.0 if gen < stage_gen else min(1.0, (gen - stage_gen) / transition_gen)
        env = make_env(progress)
        state = run_life(pop_size, g, t_max, env, W)
        surv = state['survival_t'].float()
        history.append(surv.mean().item())

        n_parents = max(1, pop_size // 5)
        _, top_idx = torch.topk(surv, n_parents)
        offspring_per = pop_size // n_parents
        child_idx = torch.repeat_interleave(top_idx, offspring_per)
        if child_idx.shape[0] < pop_size:
            extra = torch.randint(n_parents, (pop_size - child_idx.shape[0],), device=device)
            child_idx = torch.cat([child_idx, top_idx[extra]])
        g = mutate_genotype(g, child_idx)

        if verbose and ((gen + 1) % 20 == 0 or gen == generations - 1):
            print(f"  Gen {gen+1:3d} | prog={progress:.2f} | avg T={surv.mean():6.1f} | "
                  f"theta_i={g['theta_i'][top_idx[0]]:.4f} sigma={g['sigma'][top_idx[0]]:.3f} "
                  f"hunger_w={g['hunger_w'][top_idx[0]]:.2f}")
    return g, top_idx[0].item(), history


def select_one(g, idx):
    return {k: v[idx:idx + 1].clone() for k, v in g.items()}


# ============================================================
# 5. 행동 서명 진단 스위트
# ============================================================
def pretrain_one(g1, env, W, t_max=300):
    state = init_agents(1)
    for t in range(t_max):
        if not state['alive'][0]:
            state = init_agents(1)
        state, _ = step(state, g1, t, env, W, learn=True)
    return state


def batch_avoid(g1, x, v_fixed, state, W, n_trials):
    """고정된 h_prev/W_prev/prototype 상태에서, 주어진 (x, v) 배치에 대해 회피 여부 계산."""
    h_batch = state['h'].repeat(n_trials, 1)
    W_batch = state['W'].repeat(n_trials, 1, 1)
    proto_batch = state['proto'].repeat(n_trials, 1, 1)
    proto_active_batch = state['proto_active'].repeat(n_trials, 1)
    w_ji_batch = state['w_ji'].repeat(n_trials, 1, 1)
    s_ji_batch = state['s_ji'].repeat(n_trials, 1, 1)
    sigma_batch = g1['sigma'].repeat(n_trials)

    h_t = compute_hidden(x, v_fixed, h_batch, W_batch, W)
    c = compute_activation(h_t, proto_batch, proto_active_batch, sigma_batch)
    score = torch.einsum('nj,njk->nk', c, w_ji_batch - s_ji_batch)
    max_score, _ = score.max(dim=1)
    threshold = g1['theta_i'].item()
    return (max_score > threshold).float()


def test_category_avoidance(g1, state, env, W, n_trials=500):
    x, cat = sample_stimuli(n_trials, 0, env, W)
    v_fixed = state['v'].repeat(n_trials, 1)
    avoid = batch_avoid(g1, x, v_fixed, state, W, n_trials)
    print("\n[① 카테고리별 회피율] (개체 자신의 현재 내부상태 v 사용)")
    names = {0: '중립', 1: '먹이', 2: '위협', 3: '먹이+위협', 4: 'safe_after'}
    for k, name in names.items():
        mask = cat == k
        if mask.sum() > 0:
            print(f"  {name:12s} (n={mask.sum().item():4d}): 회피율 = {avoid[mask].mean().item():.3f}")


def test_generalization_multi_v(g1, state, W, n_alphas=11, n_trials=200):
    print("\n[② 일반화: 서로 다른 내부상태(v) 조건에서의 위협-보간 회피율]")
    v_conditions = {
        '건강함 [1.0,1.0,1.0]': torch.tensor([[1.0, 1.0, 1.0]], device=device),
        '지침 [0.3,0.3,0.9]':   torch.tensor([[0.3, 0.3, 0.9]], device=device),
        '개체 실제상태':          state['v'],
    }
    alphas = torch.linspace(0, 1, n_alphas)
    for label, v_cond in v_conditions.items():
        print(f"  -- {label} --")
        v_fixed = v_cond.repeat(n_trials, 1)
        for a in alphas:
            x = torch.randn(n_trials, D_OBS, device=device) * 0.25 + a * W['mu_threat']
            avoid = batch_avoid(g1, x, v_fixed, state, W, n_trials)
            rate = avoid.mean().item()
            bar = '#' * int(rate * 30)
            print(f"    alpha={a:.1f}  회피율={rate:.3f} {bar}")


def test_extinction_reinstatement(g1, env, W, n_exposures=60, n_probe=100):
    print("\n[③ 소거와 재발]")
    state = init_agents(1)
    threat_x = (W['mu_threat'] + torch.randn(D_OBS, device=device) * 0.1).unsqueeze(0)

    for t in range(5):
        # 형성 단계는 p_strike=1로 통제 (확실히 위협 연관이 생기도록)
        env_strong = dict(env, p_strike=1.0)
        state, _ = step(state, g1, t, env_strong, W,
                         force_stimulus=(threat_x, torch.tensor([2], device=device)),
                         force_no_damage=False, learn=True)

    def probe(state, n_trials=n_probe):
        x = threat_x.repeat(n_trials, 1) + torch.randn(n_trials, D_OBS, device=device) * 0.05
        v_fixed = state['v'].repeat(n_trials, 1)
        return batch_avoid(g1, x, v_fixed, state, W, n_trials).mean().item()

    r0 = probe(state)
    print(f"  형성 직후 회피율: {r0:.3f}")

    for t in range(5, 5 + n_exposures):
        state, _ = step(state, g1, t, env, W,
                         force_stimulus=(threat_x, torch.tensor([2], device=device)),
                         force_no_damage=True, learn=True)
    r1 = probe(state)
    print(f"  무해 반복노출 {n_exposures}회 후 회피율: {r1:.3f}  (감소하면 소거)")

    other_x = torch.randn(1, D_OBS, device=device) + torch.randn(D_OBS, device=device) * 0.1
    for t in range(5 + n_exposures, 5 + n_exposures + 3):
        env_strong = dict(env, p_strike=1.0)
        state, _ = step(state, g1, t, env_strong, W,
                         force_stimulus=(other_x, torch.tensor([2], device=device)),
                         force_no_damage=False, learn=True)
    r2 = probe(state)
    print(f"  무관한 스트레스 사건 이후 회피율: {r2:.3f}  (다시 증가하면 재발)")
    return r0, r1, r2


def test_asymmetric_memory(g1, env, W, idle_steps=200):
    print("\n[④ 위협 기억의 보존]")
    state = init_agents(1)
    threat_x = (W['mu_threat'] + torch.randn(D_OBS, device=device) * 0.1).unsqueeze(0)
    env_strong = dict(env, p_strike=1.0)
    for t in range(5):
        state, _ = step(state, g1, t, env_strong, W,
                         force_stimulus=(threat_x, torch.tensor([2], device=device)),
                         force_no_damage=False, learn=True)
    w0 = state['w_ji'].max().item()
    for t in range(5, 5 + idle_steps):
        neutral_x = (torch.randn(D_OBS, device=device) * 0.25).unsqueeze(0)
        state, _ = step(state, g1, t, env, W,
                         force_stimulus=(neutral_x, torch.tensor([0], device=device)),
                         force_no_damage=True, learn=True)
    w1 = state['w_ji'].max().item()
    print(f"  위협 연관 강도: 형성 직후 {w0:.4f} -> {idle_steps}스텝 유휴 후 {w1:.4f}")


# ============================================================
# 6. 다중 시드 실행
# ============================================================
def run_full_experiment(n_seeds=5, pop_size=256, generations=80, t_max=200):
    """여러 시드로 반복 실행해서 결과의 안정성을 확인."""
    all_results = []
    for seed in range(n_seeds):
        print(f"\n{'='*60}\n시드 {seed}\n{'='*60}")
        torch.manual_seed(seed)
        W = make_world_constants(seed * 100 + 1)

        g, best_idx, hist = evolve(pop_size, generations, t_max, W, verbose=True)
        g1 = select_one(g, best_idx)
        env_final = make_env(1.0)

        state = pretrain_one(g1, env_final, W, t_max=300)
        test_category_avoidance(g1, state, env_final, W)
        r0, r1, r2 = test_extinction_reinstatement(g1, env_final, W)
        test_asymmetric_memory(g1, env_final, W)

        all_results.append(dict(seed=seed, final_avg_T=hist[-1],
                                 extinction_success=(r0 - r1),
                                 reinstatement_success=(r2 - r1)))

    print(f"\n\n{'='*60}\n전체 {n_seeds}개 시드 요약\n{'='*60}")
    ext = [r['extinction_success'] for r in all_results]
    rei = [r['reinstatement_success'] for r in all_results]
    print(f"소거 정도 (형성직후 - 소거후 회피율): 평균={np.mean(ext):.3f}, 표준편차={np.std(ext):.3f}")
    print(f"재발 정도 (재발후 - 소거후 회피율):   평균={np.mean(rei):.3f}, 표준편차={np.std(rei):.3f}")
    print("\n(둘 다 평균이 뚜렷이 양수이고 표준편차가 크지 않다면, 소거/재발이 이 규칙에서")
    print(" 시드에 안정적으로 창발한다고 볼 수 있는 최소한의 근거가 됩니다.)")
    return all_results


if __name__ == "__main__":
    # Colab T4 기준: pop_size를 256~1024까지 올려도 GPU 부담이 크지 않습니다.
    # 처음 돌려볼 때는 아래 작은 설정으로 먼저 확인하고, 문제없으면 키우세요.
    results = run_full_experiment(n_seeds=3, pop_size=128, generations=60, t_max=200)

    # 마지막 시드의 개체로 상세 일반화(다중 v 조건) 테스트까지 보고 싶다면:
    # W = make_world_constants(1)
    # g, best_idx, _ = evolve(128, 60, 200, W)
    # g1 = select_one(g, best_idx)
    # state = pretrain_one(g1, make_env(1.0), W)
    # test_generalization_multi_v(g1, state, W)
