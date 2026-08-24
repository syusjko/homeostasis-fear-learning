"""
[미검증] V11: ES 훈련-평가 목적함수 불일치 해결 시도
========================================================
V9/V10에서 반복 확인된 현상: argmax(진짜 결정론적 정책)가 상태와 무관하게
100% EAT만 선택하여 생존율이 0으로 붕괴. 반면 샘플링 기반 평가에서는
합리적인 성능을 보임 (예: 샘플링 2500 vs argmax 700).

원인 진단: ES 훈련 롤아웃이 소프트맥스 샘플링(온도=1.0)으로만 진행되어,
"상태를 무시하고 EAT 로짓을 근소하게 1위로 유지하는" 결함 정책도
샘플링 노이즈 덕분에 훈련 중 벌점을 받지 않았음. 여기에 FOOD_LIMIT=300
(그리드의 75%)이라 "가만히 있어도 되는" 환경이 이 결함을 가려줌.

이 파일의 수정: (1) FOOD_LIMIT 300->60, (2) 훈련 롤아웃에 온도 어닐링
(세대가 진행될수록 1.0->0.3으로 낮춰 샘플링이 argmax에 가까워지게 함),
(3) 스케일이 무의미했던 엔트로피 보너스 제거.

주의: 이 스크립트는 대화 종료 시점까지 실제로 실행/검증되지 않았습니다.
다음 실행자가 결과를 확인하고 README의 "다음 단계"에 업데이트해야 합니다.
"""

import torch
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device = {device}")

# ============================================================
# 1. 하이퍼파라미터
# ============================================================
GRID_SIZE = 20
INPUT_DIM = 34
HIDDEN_DIM = 64
WILL_HIDDEN_DIM = 32
ACTION_DIM = 8
SIGNAL_DIM = 6

POP_SIZE = 2048
N_AGENTS = 16
GENERATIONS = 80
T_MAX = 150

ENERGY_DECAY = 0.03
ENERGY_DECAY_SHELTER = 0.01
FOOD_ENERGY_GAIN = 0.2
FOOD_SPAWN_PROB = 0.10
FOOD_LIMIT = 60  # [V11] 300 -> 60. "안 움직여도 되는" 환경 제거.

FRUIT_ENERGY_GAIN = 0.5
FRUIT_SPAWN_PROB = 0.005
FRUIT_LIMIT = 20

BASE_X = 0
BASE_Y = 0
STORE_ENERGY_COST = 0.1
BASE_ENERGY_CAP = 5.0
BASE_REPRODUCE_THRESHOLD = 2.0
BASE_REPRODUCE_PROB = 0.02

THREAT_DAMAGE = 0.15
THREAT_SPAWN_PROB = 0.01
THREAT_DAMAGE_AVOID = 0.05
THREAT_MOVE_PROB = 0.2
THREAT_DECAY_PROB = 0.05

PREDATOR_DAMAGE = 0.30
PREDATOR_ATTACK_PROB = 0.3
PREDATOR_MOVE_PROB = 0.5
PREDATOR_MAX_STEP = 3
PREDATOR_FEAR_DECAY = 0.05
PREDATOR_FEAR_SIGNAL_SCALE = 0.3
PREDATOR_FEAR_GROUP_SCALE = 1.5
PREDATOR_BASE_DAMAGE_REDUCTION = 0.7
PREDATOR_FLEE_THRESHOLD = 0.75
PREDATOR_FLEE_SLOPE = 6.0
PREDATOR_RESPAWN_INTERVAL = 30

SOCIAL_BOND_DECAY = 0.005
PROXIMITY_BOND_GAIN = 0.02
PROXIMITY_RADIUS = 3

MOVE_COST = 0.002
AVOID_COST = 0.0015

REWARD_VISIT_WEIGHT = 5.0
FRUIT_VISIT_WEIGHT = 10.0
STORE_WEIGHT = 5.0
REPRODUCE_WEIGHT = 30.0
GROUP_DEFENSE_WEIGHT = 15.0

SIGMA = 0.3
LR = 0.05

# [V11] 훈련 롤아웃 소프트맥스 온도 어닐링 범위
TEMP_START = 1.0
TEMP_END = 0.3

A_REST = 0
A_UP = 1
A_DOWN = 2
A_LEFT = 3
A_RIGHT = 4
A_EAT = 5
A_AVOID = 6
A_STORE = 7


def init_theta_vec(use_will=True, seed=0):
    torch.manual_seed(seed)
    W1 = torch.randn(INPUT_DIM, HIDDEN_DIM) * 0.1
    W1[0, 0:8] += 0.2
    W1[1, 8:16] += 0.2
    b1 = torch.zeros(HIDDEN_DIM)
    W2 = torch.randn(HIDDEN_DIM, ACTION_DIM) * 0.1
    b2 = torch.zeros(ACTION_DIM)
    W3 = torch.randn(HIDDEN_DIM, SIGNAL_DIM) * 0.1
    W3[0:8, 0] += 0.2
    W3[8:16, 1] += 0.2
    b3 = torch.zeros(SIGNAL_DIM)
    b2[A_EAT] += 0.1
    b2[A_AVOID] += 0.1
    base_vec = torch.cat([W1.flatten(), b1, W2.flatten(), b2, W3.flatten(), b3])
    if use_will:
        W1_will = torch.randn(WILL_HIDDEN_DIM, HIDDEN_DIM) * 0.1
        b1_will = torch.zeros(WILL_HIDDEN_DIM)
        W2_will = torch.randn(ACTION_DIM, WILL_HIDDEN_DIM) * 0.1
        b2_will = torch.zeros(ACTION_DIM)
        W3_will = torch.randn(SIGNAL_DIM, WILL_HIDDEN_DIM) * 0.1
        b3_will = torch.zeros(SIGNAL_DIM)
        will_vec = torch.cat([W1_will.flatten(), b1_will, W2_will.flatten(), b2_will, W3_will.flatten(), b3_will])
        return torch.cat([base_vec, will_vec])
    else:
        dummy_size = (WILL_HIDDEN_DIM * HIDDEN_DIM) + WILL_HIDDEN_DIM + \
                     (ACTION_DIM * WILL_HIDDEN_DIM) + ACTION_DIM + \
                     (SIGNAL_DIM * WILL_HIDDEN_DIM) + SIGNAL_DIM
        return torch.cat([base_vec, torch.zeros(dummy_size)])


def unflatten_params(candidate_vecs, batch_size, use_will=True):
    W1_size = INPUT_DIM * HIDDEN_DIM; b1_size = HIDDEN_DIM
    W2_size = HIDDEN_DIM * ACTION_DIM; b2_size = ACTION_DIM
    W3_size = HIDDEN_DIM * SIGNAL_DIM; b3_size = SIGNAL_DIM
    W1 = candidate_vecs[:, :W1_size].reshape(batch_size, INPUT_DIM, HIDDEN_DIM)
    idx = W1_size
    b1 = candidate_vecs[:, idx:idx+b1_size]; idx += b1_size
    W2 = candidate_vecs[:, idx:idx+W2_size].reshape(batch_size, HIDDEN_DIM, ACTION_DIM); idx += W2_size
    b2 = candidate_vecs[:, idx:idx+b2_size]; idx += b2_size
    W3 = candidate_vecs[:, idx:idx+W3_size].reshape(batch_size, HIDDEN_DIM, SIGNAL_DIM); idx += W3_size
    b3 = candidate_vecs[:, idx:idx+b3_size]
    if not use_will:
        return W1, b1, W2, b2, W3, b3, None, None, None, None, None, None
    idx += b3_size
    W1w_size = WILL_HIDDEN_DIM * HIDDEN_DIM; b1w_size = WILL_HIDDEN_DIM
    W2w_size = ACTION_DIM * WILL_HIDDEN_DIM; b2w_size = ACTION_DIM
    W3w_size = SIGNAL_DIM * WILL_HIDDEN_DIM; b3w_size = SIGNAL_DIM
    W1w = candidate_vecs[:, idx:idx+W1w_size].reshape(batch_size, WILL_HIDDEN_DIM, HIDDEN_DIM); idx += W1w_size
    b1w = candidate_vecs[:, idx:idx+b1w_size]; idx += b1w_size
    W2w = candidate_vecs[:, idx:idx+W2w_size].reshape(batch_size, ACTION_DIM, WILL_HIDDEN_DIM); idx += W2w_size
    b2w = candidate_vecs[:, idx:idx+b2w_size]; idx += b2w_size
    W3w = candidate_vecs[:, idx:idx+W3w_size].reshape(batch_size, SIGNAL_DIM, WILL_HIDDEN_DIM); idx += W3w_size
    b3w = candidate_vecs[:, idx:idx+b3w_size]
    return W1, b1, W2, b2, W3, b3, W1w, b1w, W2w, b2w, W3w, b3w


def policy_forward(W1, b1, W2, b2, W3, b3, obs, will_params=None):
    z1 = torch.bmm(obs.unsqueeze(1), W1).squeeze(1) + b1
    h_base = torch.relu(z1)
    base_action_logits = torch.bmm(h_base.unsqueeze(1), W2).squeeze(1) + b2
    base_signal_logits = torch.bmm(h_base.unsqueeze(1), W3).squeeze(1) + b3
    if will_params is not None and will_params[0] is not None:
        W1w, b1w, W2w, b2w, W3w, b3w = will_params
        will_z1 = torch.bmm(h_base.unsqueeze(1), W1w.transpose(1, 2)).squeeze(1) + b1w
        will_h = torch.relu(will_z1)
        will_action_logits = torch.bmm(will_h.unsqueeze(1), W2w.transpose(1, 2)).squeeze(1) + b2w
        will_signal_logits = torch.bmm(will_h.unsqueeze(1), W3w.transpose(1, 2)).squeeze(1) + b3w
        return base_action_logits + will_action_logits, base_signal_logits + will_signal_logits, \
               (h_base, will_h, will_action_logits, will_signal_logits)
    return base_action_logits, base_signal_logits, h_base


def init_population(n_worlds, n_agents):
    return {
        'energy': torch.ones(n_worlds, n_agents, device=device) * 0.8,
        'integrity': torch.ones(n_worlds, n_agents, device=device) * 1.0,
        'social_bond': torch.ones(n_worlds, n_agents, n_agents, device=device) * 0.6,
        'pos_x': torch.randint(0, GRID_SIZE, (n_worlds, n_agents), device=device),
        'pos_y': torch.randint(0, GRID_SIZE, (n_worlds, n_agents), device=device),
        'alive': torch.ones(n_worlds, n_agents, dtype=torch.bool, device=device),
        'survival_t': torch.zeros(n_worlds, n_agents, dtype=torch.long, device=device),
        'fruit_visits': torch.zeros(n_worlds, dtype=torch.long, device=device),
        'store_events': torch.zeros(n_worlds, dtype=torch.long, device=device),
        'reproduce_events': torch.zeros(n_worlds, dtype=torch.long, device=device),
        'group_defense_events': torch.zeros(n_worlds, dtype=torch.long, device=device),
        'signals': torch.zeros(n_worlds, n_agents, SIGNAL_DIM, device=device),
        'threat_encounters': torch.zeros(n_worlds, n_agents, dtype=torch.long, device=device),
        'predator_encounters': torch.zeros(n_worlds, n_agents, dtype=torch.long, device=device),
    }


def spawn_food_batch(food_grids, n_worlds):
    total_cells = GRID_SIZE * GRID_SIZE
    n_spawns = torch.distributions.Binomial(total_cells, FOOD_SPAWN_PROB).sample((n_worlds,)).to(device)
    current_food = food_grids.reshape(n_worlds, -1).sum(dim=1)
    max_allowed = torch.clamp(FOOD_LIMIT - current_food, min=0)
    n_spawns = torch.min(n_spawns, max_allowed)
    total_spawns = int(n_spawns.sum().item())
    if total_spawns > 0:
        batch_indices = torch.repeat_interleave(torch.arange(n_worlds, device=device), n_spawns.long())
        flat_indices = torch.randint(0, total_cells, (total_spawns,), device=device)
        food_flat = food_grids.reshape(n_worlds, -1)
        food_flat[batch_indices, flat_indices] = 1.0
        food_grids = food_flat.reshape(n_worlds, GRID_SIZE, GRID_SIZE)
    return food_grids


def spawn_fruit_batch(fruit_grids, n_worlds):
    total_cells = GRID_SIZE * GRID_SIZE
    n_spawns = torch.distributions.Binomial(total_cells, FRUIT_SPAWN_PROB).sample((n_worlds,)).to(device)
    current_fruit = fruit_grids.reshape(n_worlds, -1).sum(dim=1)
    max_allowed = torch.clamp(FRUIT_LIMIT - current_fruit, min=0)
    n_spawns = torch.min(n_spawns, max_allowed)
    total_spawns = int(n_spawns.sum().item())
    if total_spawns > 0:
        batch_indices = torch.repeat_interleave(torch.arange(n_worlds, device=device), n_spawns.long())
        flat_indices = torch.randint(0, total_cells, (total_spawns,), device=device)
        fruit_flat = fruit_grids.reshape(n_worlds, -1)
        fruit_flat[batch_indices, flat_indices] = 1.0
        fruit_grids = fruit_flat.reshape(n_worlds, GRID_SIZE, GRID_SIZE)
    return fruit_grids


def spawn_threats_batch(threat_grids, n_worlds):
    total_cells = GRID_SIZE * GRID_SIZE
    n_spawns = torch.distributions.Binomial(total_cells, THREAT_SPAWN_PROB).sample((n_worlds,)).to(device)
    total_spawns = int(n_spawns.sum().item())
    if total_spawns > 0:
        batch_indices = torch.repeat_interleave(torch.arange(n_worlds, device=device), n_spawns.long())
        flat_indices = torch.randint(0, total_cells, (total_spawns,), device=device)
        threat_flat = threat_grids.reshape(n_worlds, -1)
        threat_flat[batch_indices, flat_indices] = 1.0
        threat_grids = threat_flat.reshape(n_worlds, GRID_SIZE, GRID_SIZE)
    return threat_grids


def move_threats_batch(threat_grids, n_worlds):
    flat = threat_grids.reshape(n_worlds, -1)
    batch_indices, flat_indices = torch.nonzero(flat > 0.5, as_tuple=True)
    if batch_indices.numel() == 0:
        return threat_grids
    survive_mask = torch.rand(batch_indices.shape[0], device=device) > THREAT_DECAY_PROB
    b_idx_surv = batch_indices[survive_mask]; f_idx_surv = flat_indices[survive_mask]
    if b_idx_surv.numel() == 0:
        return torch.zeros_like(threat_grids)
    move_mask = torch.rand(b_idx_surv.shape[0], device=device) < THREAT_MOVE_PROB
    x = f_idx_surv // GRID_SIZE; y = f_idx_surv % GRID_SIZE
    dx = torch.randint(-1, 2, (b_idx_surv.shape[0],), device=device)
    dy = torch.randint(-1, 2, (b_idx_surv.shape[0],), device=device)
    dx = torch.where(move_mask, dx, torch.zeros_like(dx))
    dy = torch.where(move_mask, dy, torch.zeros_like(dy))
    new_x = torch.clamp(x + dx, 0, GRID_SIZE - 1); new_y = torch.clamp(y + dy, 0, GRID_SIZE - 1)
    new_flat_indices = new_x * GRID_SIZE + new_y
    new_flat = torch.zeros_like(flat)
    new_flat[b_idx_surv, new_flat_indices] = 1.0
    return new_flat.reshape(n_worlds, GRID_SIZE, GRID_SIZE)


def get_nearest_direction_from_positions(pos_x, pos_y, grids):
    n_worlds, n_agents = pos_x.shape
    flat_grids = grids.reshape(n_worlds, -1)
    has_object = flat_grids > 0.5
    coords_x = torch.arange(GRID_SIZE, device=device).repeat(GRID_SIZE)
    coords_y = torch.arange(GRID_SIZE, device=device).repeat_interleave(GRID_SIZE)
    dx = coords_x.unsqueeze(0).unsqueeze(0) - pos_x.unsqueeze(-1).float()
    dy = coords_y.unsqueeze(0).unsqueeze(0) - pos_y.unsqueeze(-1).float()
    dist = torch.abs(dx) + torch.abs(dy)
    dist = torch.where(has_object.unsqueeze(1).expand(-1, n_agents, -1), dist, torch.tensor(float('inf'), device=device))
    min_dist, min_idx = torch.min(dist, dim=2)
    nearest_dx = torch.sign(torch.gather(dx, 2, min_idx.unsqueeze(-1)).squeeze(-1))
    nearest_dy = torch.sign(torch.gather(dy, 2, min_idx.unsqueeze(-1)).squeeze(-1))
    has_any = has_object.any(dim=1).unsqueeze(1).expand(-1, n_agents).float()
    return nearest_dx * has_any, nearest_dy * has_any


def compute_neighbor_info(pos_x, pos_y, alive, radius=PROXIMITY_RADIUS):
    n_worlds, n_agents = pos_x.shape
    dx = pos_x.unsqueeze(2) - pos_x.unsqueeze(1); dy = pos_y.unsqueeze(2) - pos_y.unsqueeze(1)
    dist = torch.abs(dx) + torch.abs(dy)
    eye = torch.eye(n_agents, device=device).unsqueeze(0).expand(n_worlds, -1, -1).bool()
    mask = (~eye) & (dist > 0) & (dist <= radius) & alive.unsqueeze(1) & alive.unsqueeze(2)
    return mask.sum(dim=2).float(), mask


def get_nearest_agent_direction(pos_x, pos_y, alive):
    n_worlds, n_agents = pos_x.shape
    dx = pos_x.unsqueeze(2) - pos_x.unsqueeze(1); dy = pos_y.unsqueeze(2) - pos_y.unsqueeze(1)
    dist = torch.abs(dx) + torch.abs(dy)
    eye = torch.eye(n_agents, device=device).unsqueeze(0).expand(n_worlds, -1, -1).bool()
    dist = torch.where(eye, torch.tensor(float('inf'), device=device), dist)
    alive_mask = alive.unsqueeze(1).expand(-1, n_agents, -1).float()
    dist = torch.where(alive_mask > 0.5, dist, torch.tensor(float('inf'), device=device))
    min_dist, min_idx = torch.min(dist, dim=2)
    nearest_dx = torch.sign(torch.gather(dx, 2, min_idx.unsqueeze(-1)).squeeze(-1)) * (min_dist < float('inf')).float()
    nearest_dy = torch.sign(torch.gather(dy, 2, min_idx.unsqueeze(-1)).squeeze(-1)) * (min_dist < float('inf')).float()
    return nearest_dx, nearest_dy


def get_direction_to(pos_x, pos_y, target_x, target_y):
    return torch.sign(target_x - pos_x.float()), torch.sign(target_y - pos_y.float())


def get_nearest_agent_direction_from_predator(predator_x, predator_y, agent_x, agent_y, alive):
    dx = agent_x - predator_x.unsqueeze(1); dy = agent_y - predator_y.unsqueeze(1)
    dist = torch.abs(dx) + torch.abs(dy)
    dist = torch.where(alive > 0.5, dist, torch.tensor(float('inf'), device=device))
    min_dist, min_idx = torch.min(dist, dim=1)
    nearest_dx = torch.sign(torch.gather(dx, 1, min_idx.unsqueeze(-1)).squeeze(-1)) * (min_dist < float('inf')).float()
    nearest_dy = torch.sign(torch.gather(dy, 1, min_idx.unsqueeze(-1)).squeeze(-1)) * (min_dist < float('inf')).float()
    return nearest_dx, nearest_dy


def step_population(state, W1, b1, W2, b2, W3, b3, food_grids, fruit_grids, threat_grids,
                     base_energy, predator_pos, predator_fear, will_params=None,
                     deterministic_action=False, temperature=1.0):
    n_worlds, n_agents = state['energy'].shape
    alive = state['alive'].clone()
    pos_x = state['pos_x'].long(); pos_y = state['pos_y'].long()
    in_base = (pos_x == BASE_X) & (pos_y == BASE_Y)
    energy_decay = torch.where(in_base, ENERGY_DECAY_SHELTER, ENERGY_DECAY)
    state['energy'] -= energy_decay
    state['social_bond'] = state['social_bond'] - SOCIAL_BOND_DECAY

    dx = pos_x.unsqueeze(2) - pos_x.unsqueeze(1); dy = pos_y.unsqueeze(2) - pos_y.unsqueeze(1)
    dist = torch.abs(dx) + torch.abs(dy)
    eye = torch.eye(n_agents, device=device).unsqueeze(0).expand(n_worlds, -1, -1).bool()
    proximity = (~eye) & (dist > 0) & (dist <= PROXIMITY_RADIUS) & alive.unsqueeze(1) & alive.unsqueeze(2)
    state['social_bond'] += proximity.float() * PROXIMITY_BOND_GAIN
    state['social_bond'] = torch.clamp(state['social_bond'], min=0.0, max=1.0)

    world_indices = torch.arange(n_worlds, device=device).unsqueeze(1).expand(-1, n_agents)
    local_food = food_grids[world_indices, pos_x, pos_y]
    local_fruit = fruit_grids[world_indices, pos_x, pos_y]
    local_threat = threat_grids[world_indices, pos_x, pos_y]

    food_dx, food_dy = get_nearest_direction_from_positions(pos_x, pos_y, food_grids)
    fruit_dx, fruit_dy = get_nearest_direction_from_positions(pos_x, pos_y, fruit_grids)
    threat_dx, threat_dy = get_nearest_direction_from_positions(pos_x, pos_y, threat_grids)
    neighbor_count, neighbor_mask = compute_neighbor_info(pos_x, pos_y, alive)
    nearest_agent_dx, nearest_agent_dy = get_nearest_agent_direction(pos_x, pos_y, alive)
    base_dx, base_dy = get_direction_to(pos_x, pos_y, BASE_X, BASE_Y)
    base_energy_expanded = base_energy.unsqueeze(1).expand(-1, n_agents)
    in_base_float = in_base.float()

    predator_x = predator_pos[0].long(); predator_y = predator_pos[1].long()
    predator_dx, predator_dy = get_direction_to(
        pos_x, pos_y, predator_x.unsqueeze(1).expand(-1, n_agents), predator_y.unsqueeze(1).expand(-1, n_agents))
    predator_fear_expanded = predator_fear.unsqueeze(1).expand(-1, n_agents)

    neighbor_signals = torch.bmm(neighbor_mask.float(), state['signals'])
    neighbor_count_safe = torch.clamp(neighbor_count, min=1.0).unsqueeze(-1)
    neighbor_signal_mean = neighbor_signals / neighbor_count_safe

    predator_signal = torch.zeros(n_worlds, n_agents, SIGNAL_DIM, device=device)
    predator_signal[:, :, 0] = 0.9; predator_signal[:, :, 1] = 0.1
    predator_signal[:, :, 2] = 0.9; predator_signal[:, :, 3] = 0.1
    predator_signal[:, :, 4] = 0.9; predator_signal[:, :, 5] = 0.1

    obs = torch.stack([
        state['energy'], state['integrity'], state['social_bond'].mean(dim=2),
        local_food, local_threat, food_dx, food_dy, threat_dx, threat_dy,
        fruit_dx, fruit_dy, neighbor_count, nearest_agent_dx, nearest_agent_dy,
        base_dx, base_dy, base_energy_expanded, in_base_float, predator_dx, predator_dy,
        local_fruit, predator_signal[:, :, 0], predator_signal[:, :, 1], predator_signal[:, :, 2],
        predator_signal[:, :, 3], predator_signal[:, :, 4], predator_signal[:, :, 5],
        neighbor_signal_mean[:, :, 0], neighbor_signal_mean[:, :, 1], neighbor_signal_mean[:, :, 2],
        neighbor_signal_mean[:, :, 3], neighbor_signal_mean[:, :, 4], neighbor_signal_mean[:, :, 5],
        predator_fear_expanded,
    ], dim=2)
    obs_flat = obs.reshape(n_worlds * n_agents, -1)

    if will_params is not None and will_params[0] is not None:
        action_logits_flat, signal_logits_flat, _ = policy_forward(W1, b1, W2, b2, W3, b3, obs_flat, will_params)
    else:
        action_logits_flat, signal_logits_flat, _ = policy_forward(W1, b1, W2, b2, W3, b3, obs_flat)

    action_logits = action_logits_flat.reshape(n_worlds, n_agents, ACTION_DIM)
    signal_logits = signal_logits_flat.reshape(n_worlds, n_agents, SIGNAL_DIM)
    probs = torch.softmax(action_logits / temperature, dim=2)  # [V11] 온도 스케일링
    if deterministic_action:
        actions = torch.argmax(probs, dim=2)
    else:
        actions = torch.multinomial(probs.reshape(-1, ACTION_DIM), 1).squeeze(1).reshape(n_worlds, n_agents)

    signals = torch.sigmoid(signal_logits)
    state['signals'] = signals

    dx_move = torch.zeros(n_worlds, n_agents, device=device); dy_move = torch.zeros(n_worlds, n_agents, device=device)
    dx_move[actions == A_UP] = -1; dx_move[actions == A_DOWN] = 1
    dy_move[actions == A_LEFT] = -1; dy_move[actions == A_RIGHT] = 1
    state['pos_x'] = torch.clamp(state['pos_x'] + dx_move.long(), 0, GRID_SIZE - 1)
    state['pos_y'] = torch.clamp(state['pos_y'] + dy_move.long(), 0, GRID_SIZE - 1)

    move_mask = (actions >= A_UP) & (actions <= A_RIGHT)
    state['energy'] -= move_mask.float() * MOVE_COST

    eat_mask = (actions == A_EAT) & (local_food > 0.5)
    state['energy'] += eat_mask.float() * FOOD_ENERGY_GAIN
    eat_indices = torch.nonzero(eat_mask, as_tuple=False)
    if eat_indices.numel() > 0:
        w_idx, a_idx = eat_indices[:, 0], eat_indices[:, 1]
        food_grids[w_idx, state['pos_x'][w_idx, a_idx].long(), state['pos_y'][w_idx, a_idx].long()] = 0.0

    fruit_eat_mask = (actions == A_EAT) & (local_fruit > 0.5)
    state['energy'] += fruit_eat_mask.float() * FRUIT_ENERGY_GAIN
    fruit_indices = torch.nonzero(fruit_eat_mask, as_tuple=False)
    if fruit_indices.numel() > 0:
        w_idx, a_idx = fruit_indices[:, 0], fruit_indices[:, 1]
        fruit_grids[w_idx, state['pos_x'][w_idx, a_idx].long(), state['pos_y'][w_idx, a_idx].long()] = 0.0
        state['fruit_visits'] += 1

    store_mask = (actions == A_STORE) & in_base & (state['energy'] > STORE_ENERGY_COST + 0.1)
    state['energy'] -= store_mask.float() * STORE_ENERGY_COST
    base_energy += store_mask.float().sum(dim=1) * STORE_ENERGY_COST
    base_energy = torch.clamp(base_energy, max=BASE_ENERGY_CAP)
    state['store_events'] += store_mask.any(dim=1).long()

    avoid_mask = (actions == A_AVOID)
    state['energy'] -= avoid_mask.float() * AVOID_COST

    threat_present = local_threat > 0.5
    state['threat_encounters'] += (threat_present & alive).long()
    damage = torch.where(threat_present, THREAT_DAMAGE, torch.tensor(0.0, device=device))
    damage = torch.where(avoid_mask & threat_present, THREAT_DAMAGE_AVOID, damage)
    state['integrity'] = torch.clamp(state['integrity'] - damage, min=0.0)

    in_predator_cell = (pos_x == predator_x.unsqueeze(1)) & (pos_y == predator_y.unsqueeze(1))
    state['predator_encounters'] += (in_predator_cell & alive).long()
    dist_to_predator = torch.abs(pos_x - predator_x.unsqueeze(1)) + torch.abs(pos_y - predator_y.unsqueeze(1))
    nearby_count = ((dist_to_predator <= PROXIMITY_RADIUS) & alive).sum(dim=1).float()
    signal_sum = signals.mean(dim=1).sum(dim=1)
    fear_gain = nearby_count * PREDATOR_FEAR_GROUP_SCALE + signal_sum * PREDATOR_FEAR_SIGNAL_SCALE
    predator_fear = torch.clamp(predator_fear + fear_gain - PREDATOR_FEAR_DECAY, min=0.0, max=1.0)
    flee_prob = torch.sigmoid(PREDATOR_FLEE_SLOPE * (predator_fear - PREDATOR_FLEE_THRESHOLD))
    flee_mask = torch.rand(n_worlds, device=device) < flee_prob

    attack_probs = torch.rand(n_worlds, n_agents, device=device) < PREDATOR_ATTACK_PROB
    attack_mask = in_predator_cell & attack_probs & alive & ~flee_mask.unsqueeze(1)
    damage_predator = torch.where(attack_mask, PREDATOR_DAMAGE, torch.tensor(0.0, device=device))
    damage_predator = torch.where(in_base & attack_mask, damage_predator * (1 - PREDATOR_BASE_DAMAGE_REDUCTION), damage_predator)
    state['integrity'] = torch.clamp(state['integrity'] - damage_predator, min=0.0)
    state['group_defense_events'] += (flee_mask & (nearby_count >= 2)).long()

    base_agents_count = (in_base & alive).sum(dim=1)
    reproduce_mask = (base_energy >= BASE_REPRODUCE_THRESHOLD) & (base_agents_count >= 2) & \
                      (torch.rand(n_worlds, device=device) < BASE_REPRODUCE_PROB)
    state['reproduce_events'] += reproduce_mask.long()
    if reproduce_mask.any():
        state['energy'] += reproduce_mask.unsqueeze(1).float() * 0.3
        state['integrity'] += reproduce_mask.unsqueeze(1).float() * 0.2
        base_energy -= reproduce_mask.float() * 1.0

    state['energy'] = torch.clamp(state['energy'], min=0.0, max=1.0)
    state['integrity'] = torch.clamp(state['integrity'], min=0.0, max=1.0)
    state['social_bond'] = torch.clamp(state['social_bond'], min=0.0, max=1.0)
    dead = (state['energy'] <= 0) | (state['integrity'] <= 0)
    state['alive'] = alive & ~dead
    state['survival_t'] += state['alive'].long()

    nearest_dx, nearest_dy = get_nearest_agent_direction_from_predator(predator_x, predator_y, pos_x, pos_y, alive)
    flee_step = torch.randint(1, PREDATOR_MAX_STEP + 1, (n_worlds,), device=device).float()
    chase_step = torch.randint(1, PREDATOR_MAX_STEP + 1, (n_worlds,), device=device).float()
    chase_move = ~flee_mask & (torch.rand(n_worlds, device=device) < PREDATOR_MOVE_PROB)
    dx_total = torch.where(flee_mask, -nearest_dx * flee_step,
                            torch.where(chase_move, nearest_dx * chase_step, torch.zeros_like(nearest_dx)))
    dy_total = torch.where(flee_mask, -nearest_dy * flee_step,
                            torch.where(chase_move, nearest_dy * chase_step, torch.zeros_like(nearest_dy)))
    predator_x = torch.clamp(predator_x + dx_total.long(), 0, GRID_SIZE - 1)
    predator_y = torch.clamp(predator_y + dy_total.long(), 0, GRID_SIZE - 1)

    return state, food_grids, fruit_grids, threat_grids, base_energy, (predator_x, predator_y), predator_fear, actions


def _expand(params, use_will):
    W1, b1, W2, b2, W3, b3 = params[0], params[1], params[2], params[3], params[4], params[5]
    will_params = params[6:] if use_will else None
    W1 = W1.repeat_interleave(N_AGENTS, dim=0); b1 = b1.repeat_interleave(N_AGENTS, dim=0)
    W2 = W2.repeat_interleave(N_AGENTS, dim=0); b2 = b2.repeat_interleave(N_AGENTS, dim=0)
    W3 = W3.repeat_interleave(N_AGENTS, dim=0); b3 = b3.repeat_interleave(N_AGENTS, dim=0)
    if will_params is not None and will_params[0] is not None:
        wp = tuple(p.repeat_interleave(N_AGENTS, dim=0) for p in will_params)
    else:
        wp = None
    return W1, b1, W2, b2, W3, b3, wp


def _maybe_respawn_predator(state, predator_x, predator_y, t, n_worlds):
    if t % PREDATOR_RESPAWN_INTERVAL == 0 and t > 0:
        dist_all = torch.abs(state['pos_x'] - predator_x.unsqueeze(1)) + torch.abs(state['pos_y'] - predator_y.unsqueeze(1))
        nearest_idx = torch.argmin(dist_all, dim=1)
        ar = torch.arange(n_worlds, device=device)
        predator_x = torch.clamp(state['pos_x'][ar, nearest_idx].long() + torch.randint(-2, 3, (n_worlds,), device=device), 0, GRID_SIZE-1)
        predator_y = torch.clamp(state['pos_y'][ar, nearest_idx].long() + torch.randint(-2, 3, (n_worlds,), device=device), 0, GRID_SIZE-1)
    return predator_x.long(), predator_y.long()


def evaluate_candidates(candidate_vecs, batch_size, use_will, temperature=1.0):
    params = unflatten_params(candidate_vecs, batch_size, use_will)
    W1, b1, W2, b2, W3, b3, wp = _expand(params, use_will)
    n_worlds = batch_size
    state = init_population(n_worlds, N_AGENTS)
    food_grids = torch.zeros(n_worlds, GRID_SIZE, GRID_SIZE, device=device)
    fruit_grids = torch.zeros(n_worlds, GRID_SIZE, GRID_SIZE, device=device)
    threat_grids = torch.zeros(n_worlds, GRID_SIZE, GRID_SIZE, device=device)
    base_energy = torch.zeros(n_worlds, device=device)
    predator_x = torch.randint(2, GRID_SIZE-2, (n_worlds,), device=device)
    predator_y = torch.randint(2, GRID_SIZE-2, (n_worlds,), device=device)
    predator_fear = torch.zeros(n_worlds, device=device)

    for t in range(T_MAX):
        food_grids = spawn_food_batch(food_grids, n_worlds)
        fruit_grids = spawn_fruit_batch(fruit_grids, n_worlds)
        threat_grids = spawn_threats_batch(threat_grids, n_worlds)
        threat_grids = move_threats_batch(threat_grids, n_worlds)
        predator_x, predator_y = _maybe_respawn_predator(state, predator_x, predator_y, t, n_worlds)
        state, food_grids, fruit_grids, threat_grids, base_energy, (predator_x, predator_y), predator_fear, _ = step_population(
            state, W1, b1, W2, b2, W3, b3, food_grids, fruit_grids, threat_grids, base_energy,
            (predator_x, predator_y), predator_fear, wp, temperature=temperature)

    fitness = (state['survival_t'].sum(dim=1).float()
               + state['store_events'].float() * STORE_WEIGHT
               + state['fruit_visits'].float() * FRUIT_VISIT_WEIGHT
               + state['reproduce_events'].float() * REPRODUCE_WEIGHT
               + state['group_defense_events'].float() * GROUP_DEFENSE_WEIGHT
               + state['social_bond'].mean(dim=(1, 2)) * 2.0
               + base_energy * 5.0)
    return fitness


def es_update(theta_vec, batch_size, sigma, lr, use_will, temperature=1.0):
    param_dim = theta_vec.shape[0]
    noise = torch.randn(batch_size, param_dim, device=device) * sigma
    candidate_vecs = theta_vec.unsqueeze(0) + noise
    fitness = evaluate_candidates(candidate_vecs, batch_size, use_will, temperature=temperature)
    advantages = (fitness - fitness.mean()) / (fitness.std() + 1e-8)
    grad = (advantages.unsqueeze(1) * noise).mean(dim=0)
    return theta_vec + lr * grad, fitness


def _run_eval_rollout(theta_vec, batch_size, use_will, deterministic_action, temperature=1.0):
    candidate_vecs = theta_vec.unsqueeze(0).repeat(batch_size, 1)
    params = unflatten_params(candidate_vecs, batch_size, use_will)
    W1, b1, W2, b2, W3, b3, wp = _expand(params, use_will)
    state = init_population(batch_size, N_AGENTS)
    food_grids = torch.zeros(batch_size, GRID_SIZE, GRID_SIZE, device=device)
    fruit_grids = torch.zeros(batch_size, GRID_SIZE, GRID_SIZE, device=device)
    threat_grids = torch.zeros(batch_size, GRID_SIZE, GRID_SIZE, device=device)
    base_energy = torch.zeros(batch_size, device=device)
    predator_x = torch.randint(2, GRID_SIZE-2, (batch_size,), device=device)
    predator_y = torch.randint(2, GRID_SIZE-2, (batch_size,), device=device)
    predator_fear = torch.zeros(batch_size, device=device)
    for t in range(T_MAX):
        food_grids = spawn_food_batch(food_grids, batch_size)
        fruit_grids = spawn_fruit_batch(fruit_grids, batch_size)
        threat_grids = spawn_threats_batch(threat_grids, batch_size)
        threat_grids = move_threats_batch(threat_grids, batch_size)
        predator_x, predator_y = _maybe_respawn_predator(state, predator_x, predator_y, t, batch_size)
        state, food_grids, fruit_grids, threat_grids, base_energy, (predator_x, predator_y), predator_fear, _ = step_population(
            state, W1, b1, W2, b2, W3, b3, food_grids, fruit_grids, threat_grids, base_energy,
            (predator_x, predator_y), predator_fear, wp, deterministic_action=deterministic_action, temperature=temperature)
    fitness = (state['survival_t'].sum(dim=1).float()
               + state['store_events'].float() * STORE_WEIGHT
               + state['fruit_visits'].float() * FRUIT_VISIT_WEIGHT
               + state['reproduce_events'].float() * REPRODUCE_WEIGHT
               + state['group_defense_events'].float() * GROUP_DEFENSE_WEIGHT
               + state['social_bond'].mean(dim=(1, 2)) * 2.0
               + base_energy * 5.0)
    return fitness.mean().item()


def evaluate_deterministic(theta_vec, n_reps=1, use_will=True):
    return np.mean([_run_eval_rollout(theta_vec, 32, use_will, False, 1.0) for _ in range(n_reps)])


def evaluate_argmax(theta_vec, n_reps=1, use_will=True):
    return np.mean([_run_eval_rollout(theta_vec, 32, use_will, True, 1.0) for _ in range(n_reps)])


def diagnose_action_collapse(theta_vec, use_will=True):
    action_names = ['REST', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'EAT', 'AVOID', 'STORE']
    for mode_name, det in [('샘플링', False), ('argmax(결정론)', True)]:
        candidate_vecs = theta_vec.unsqueeze(0).repeat(32, 1)
        params = unflatten_params(candidate_vecs, 32, use_will)
        W1, b1, W2, b2, W3, b3, wp = _expand(params, use_will)
        state = init_population(32, N_AGENTS)
        food_grids = torch.zeros(32, GRID_SIZE, GRID_SIZE, device=device)
        fruit_grids = torch.zeros(32, GRID_SIZE, GRID_SIZE, device=device)
        threat_grids = torch.zeros(32, GRID_SIZE, GRID_SIZE, device=device)
        base_energy = torch.zeros(32, device=device)
        predator_x = torch.randint(2, GRID_SIZE-2, (32,), device=device)
        predator_y = torch.randint(2, GRID_SIZE-2, (32,), device=device)
        predator_fear = torch.zeros(32, device=device)
        action_counts = torch.zeros(ACTION_DIM, dtype=torch.long)
        survival_trace = []
        for t in range(T_MAX):
            food_grids = spawn_food_batch(food_grids, 32)
            fruit_grids = spawn_fruit_batch(fruit_grids, 32)
            threat_grids = spawn_threats_batch(threat_grids, 32)
            threat_grids = move_threats_batch(threat_grids, 32)
            state, food_grids, fruit_grids, threat_grids, base_energy, (predator_x, predator_y), predator_fear, actions = step_population(
                state, W1, b1, W2, b2, W3, b3, food_grids, fruit_grids, threat_grids, base_energy,
                (predator_x, predator_y), predator_fear, wp, deterministic_action=det)
            alive_actions = actions[state['alive']]
            for a in range(ACTION_DIM):
                action_counts[a] += (alive_actions == a).sum().item()
            survival_trace.append(state['alive'].float().mean().item())
        total = action_counts.sum().item()
        print(f"\n--- {mode_name} 행동 분포 (use_will={use_will}) ---")
        for a, name in enumerate(action_names):
            pct = 100 * action_counts[a].item() / max(total, 1)
            print(f"  {name:6s}: {pct:5.1f}% {'#' * int(pct/2)}")
        print(f"  생존율: 시작={survival_trace[0]:.2f} -> 중반={survival_trace[len(survival_trace)//2]:.2f} -> 종료={survival_trace[-1]:.2f}")


def run_experiment(use_will=True, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"\n{'='*60}\n실험 시작: {'의지 있음' if use_will else '의지 없음'} (V11) | seed={seed}\n{'='*60}")
    theta_vec = init_theta_vec(use_will=use_will, seed=seed).to(device)

    start_time = time.time()
    mean_history = []
    for gen in range(GENERATIONS):
        progress = gen / max(GENERATIONS - 1, 1)
        temperature = TEMP_START + (TEMP_END - TEMP_START) * progress
        theta_vec, fitness = es_update(theta_vec, POP_SIZE, SIGMA, LR, use_will, temperature=temperature)
        mean_history.append(fitness.mean().item())
        if gen % 20 == 0 or gen == GENERATIONS - 1:
            det_perf = evaluate_deterministic(theta_vec, 1, use_will)
            print(f"[{gen+1:3d}/{GENERATIONS}] temp={temperature:.2f} 평균={fitness.mean():.1f} "
                  f"평가(샘플링)={det_perf:.1f} 경과={time.time()-start_time:.1f}s")

    print(f"\n학습 완료: {time.time()-start_time:.1f}초")
    print(f"최종 평가(샘플링): {evaluate_deterministic(theta_vec, 2, use_will):.1f}")
    print(f"최종 평가(argmax): {evaluate_argmax(theta_vec, 2, use_will):.1f}")
    diagnose_action_collapse(theta_vec, use_will=use_will)
    return theta_vec, mean_history


if __name__ == "__main__":
    N_SEEDS = 3
    will_results, base_results = [], []
    for seed in range(N_SEEDS):
        _, mw = run_experiment(use_will=True, seed=seed)
        _, mb = run_experiment(use_will=False, seed=seed)
        will_results.append(np.mean(mw[-10:]))
        base_results.append(np.mean(mb[-10:]))
    print("\n=== 최종 비교 ===")
    for i in range(N_SEEDS):
        print(f"seed {i}: 의지있음={will_results[i]:.1f} 의지없음={base_results[i]:.1f}")
    print(f"의지있음 평균={np.mean(will_results):.1f}±{np.std(will_results):.1f}")
    print(f"의지없음 평균={np.mean(base_results):.1f}±{np.std(base_results):.1f}")
