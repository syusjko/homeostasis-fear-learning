# Homeostasis-based Threat/Fear Learning: Exploration of Minimal Architecture
# 항상성 기반 위협/공포 학습: 최소 아키텍처 탐구

> "If attention dominated the world with a single elegant linear algebra formula, can consciousness or will also be designed with such minimal rules?" 
> 
> "어텐션 메커니즘이 간결한 선형대수 공식 하나로 세상을 지배했다면, 의식이나 의지도 그런 최소한의 규칙으로 설계할 수 있을까?"라는 철학적 질문에서 출발해, **"공포 학습의 행동적 서명이 극도로 단순한 규칙에서 창발하는가?"**라는 실증적 질문으로 구체화된 연구 프로젝트입니다.

---

## 📑 Table of Contents (목차)
1. [Abstract (요약)](#1-abstract-요약)
2. [Research Objectives (연구 목표)](#2-research-objectives-연구-목표)
3. [Key Design Principles (핵심 설계 원칙)](#3-key-design-principles-핵심-설계-원칙)
4. [Conceptual Architecture & Behavioral Signatures (개념적 아키텍처 및 행동 서명)](#4-conceptual-architecture--behavioral-signatures-개념적-아키텍처-및-행동-서명)
5. [Repository Structure (저장소 구조)](#5-repository-structure-저장소-구조)
6. [Current Status & Verification (현재 상태 및 검증 내역)](#6-current-status--verification-현재-상태-및-검증-내역)
7. [Execution Guide (실행 방법)](#7-execution-guide-실행-방법)

---

## 1. Abstract (요약)

**English:** This project simulates an artificial life environment to verify if complex behavioral signatures of animal fear learning can emerge from extremely simplified rules. Rather than implementing explicit planning or high-level cognitive models, the agents rely entirely on homeostatic resources, simple prediction errors, and prototype-based reflexes.

**Korean:** 본 프로젝트는 인공 생명체 시뮬레이션을 통해 동물의 공포 학습에서 나타나는 복잡한 행동 서명(Behavioral Signatures)이 극도로 단순화된 규칙으로부터 창발(Emergence)할 수 있는지 검증합니다. 명시적인 계획이나 고차원적 인지 모델 없이, 오직 항상성(Homeostasis) 기반의 자원 관리, 단순 예측 오차, 그리고 프로토타입 기반의 반사 행동만을 사용합니다.

---

## 2. Research Objectives (연구 목표)

### What this project does NOT answer (본 연구가 다루지 않는 것)
- **"Does this system have consciousness/will?"** 
  - We concluded that this is behaviorally unverifiable (refer to the Zombie Argument in `docs/DESIGN_JOURNEY.md`).
  - **"이 시스템에 자의식이나 의지가 있는가?"**라는 질문은 행동적 관찰만으로는 증명이 불가능하다는 결론을 내렸습니다.

### What this project attempts to answer (본 연구가 답하려는 것)
Can the following four behavioral signatures of animal fear emerge from a minimal rule set?
다음 4가지 동물 공포의 행동적 서명이 최소 규칙 집합에서 창발하는가?
1. **Fast Reflex (빠른 반사)**
2. **Generalization (일반화)**: Avoidance behavior triggered by similar, non-exact threat stimuli. (정확히 동일한 자극이 아니어도 유사한 위협 범주에 대한 회피 전이)
3. **Extinction & Relapse (소거 및 재발)**: The disappearance of fear response and its unexpected return. (공포 반응의 소멸과 갑작스러운 재발)
4. **Asymmetric Retention of Threat Memory (위협 기억의 비대칭적 보존)**

---

## 3. Key Design Principles (핵심 설계 원칙)

These principles were refined through multiple iterations of critical review. (여러 번의 비판적 리뷰를 거쳐 확립된 원칙입니다.)

1. **Separation of Timescales (온라인 학습과 진화의 분리)**
   - Local Hebbian updates manage within-lifetime adaptation (fast weights), while only scalar hyperparameters evolve across generations (slow weights).
   - 생애 내의 적응은 국소적 Hebbian 갱신(빠른 가중치)이 담당하고, 세대 간 진화는 스칼라 하이퍼파라미터(느린 가중치)만이 변이와 도태를 거칩니다.

2. **Prevention of the Baldwin Effect (볼드윈 효과 방지)**
   - Traits acquired via lifetime learning (e.g., weights, prototype positions) are never inherited. All agents start from a blank state. Inheriting acquired traits shifts selection pressure toward "fast convergence" rather than true adaptation, stagnating evolution.
   - 생애 동안 학습으로 변하는 값들은 유전자형에 포함하지 않습니다. 이를 어기면 선택압이 실제 생존 능력이 아닌 '빠른 수렴 속도'에 맞춰져 진화가 정체됩니다.

3. **Excitatory vs. Inhibitory Separation (위협 기억의 흥분성/억제성 분리)**
   - Threat memory is implemented via eligibility traces with separated excitatory and inhibitory associations. This allows extinction to occur without destroying the original memory, enabling relapse phenomena.
   - 억제성 연관을 따로 두어야, 공포가 소거된 후에도 무관한 스트레스 등에 의해 다시 공포가 '재발'하는 실제 동물의 특성을 재현할 수 있습니다.

4. **Necessity of Uncertainty & Counter-Incentives (불확실성과 반대 유인의 필수성)**
   - True discrimination emerges only when threats are probabilistic and accompanied by rewards (e.g., food). If threats are deterministic, simple unconditional avoidance becomes the optimal strategy, rendering generalization/extinction experiments meaningless.
   - 위협이 100% 확률로 피해를 준다면 무조건적인 회피가 최적의 전략이 되어버립니다. 진정한 식별 능력은 불확실성과 보상(반대 유인)이 공존할 때 창발합니다.

5. **Direct Interoceptive Grounding (내수용감각의 직접 표상 연결)**
   - Homeostatic states (e.g., hunger) must be directly fed into the hidden representation to allow the agent to associate internal needs with specific external stimuli.
   - '배고픔'과 같은 항상성 상태가 은닉 상태 계산에 직접 입력되지 않으면, 내부 상태와 외부 자극을 결합하여 학습할 원천적인 방법이 없습니다.

---

## 4. Conceptual Architecture & Behavioral Signatures (개념적 아키텍처 및 행동 서명)

Below are conceptual representations generated to validate our hypotheses regarding extinction/relapse and the separation of evolutionary timescales. 
(아래는 소거/재발 현상 및 진화적 시간 척도의 분리에 대한 가설을 시각화한 개념도입니다.)

### Extinction and Relapse (공포의 소거와 재발)
![Extinction and Relapse](docs/assets/extinction_relapse.png)
*Figure 1: Behavioral signature of threat memory. The fear response is acquired, extinguished, but can spontaneously recover (relapse) due to the underlying separation of excitatory and inhibitory weights.*

### Evolution vs. Learning (진화와 생애 온라인 학습의 분리)
![Evolution vs Learning](docs/assets/evolution_learning.png)
*Figure 2: Separation of timescales. Genotypic parameters (Slow weights) evolve over generations, providing the framework within which phenotypic adaptation (Fast weights) occurs rapidly within a single lifetime.*

---

## 5. Repository Structure (저장소 구조)

```text
📦 homeostasis-fear-learning
 ┣ 📂 threat_fear_sim/    # Single-agent evolution & core threat learning verification (가장 정제된 단일 개체 시뮬레이터)
 ┣ 📂 social_signaling/   # Multi-agent environment for alarm signals (다개체 환경 포식자/독초 구별 시뮬레이터)
 ┣ 📂 will_network_es/    # Neural network experiments for a "Will Gate" using ES (진화 전략 기반 의지 네트워크 초안)
 ┗ 📂 docs/               
   ┣ 📜 DESIGN_JOURNEY.md # Full logical progression, abandoned ideas, and rationale (설계 원칙 및 논증 과정)
   ┣ 📜 LESSONS.md        # Traps encountered and reusable methodology checklists (시행착오 및 교훈 체크리스트)
   ┗ 📂 assets/           # Generated graphs and visual resources (시각 자료)
```

---

## 6. Current Status & Verification (현재 상태 및 검증 내역)

| Experiment (실험) | Status (상태) | Notes (비고) |
|:---:|---|---|
| **`threat_fear_sim`** | **Verified (확인됨)** | Extinction and relapse successfully reproduced in small-scale runs. High variance across seeds requires multi-seed repetition.<br>*(소규모 실행에서 소거/재발 재현 확인. 시드 간 변동성이 커 다중 시드 반복 테스트 필요)* |
| **`social_signaling`** | **Verified (확인됨)** | Signal transmission is robust. Genetic correlation with discrimination ability was mostly identified as statistical illusion (See `docs/LESSONS.md` #6, #7).<br>*(신호 전달 효과는 견고하나, 구별 능력과의 유전자 상관은 통계적 착시로 판명)* |
| **`will_network_es`** | **Unresolved (미해결 버그)** | Deterministic `argmax` policy collapses into a single action regardless of state. Temperature annealing (`v11`) pending verification.<br>*(상태와 무관하게 하나의 행동으로 붕괴하는 현상 발생. 온도 어닐링 기법 미검증 상태)* |

---

## 7. Execution Guide (실행 방법)

It is recommended to run these simulations in environments with GPU acceleration (e.g., Google Colab with T4 or higher).
(Colab 등 GPU가 지원되는 환경에서 실행하는 것을 권장합니다.)

**For Threat/Fear Simulation:**
```bash
python threat_fear_sim/sim.py
```

**For Social Signaling Simulation:**
```bash
python social_signaling/sim.py
```

> [!WARNING]
> Before running `will_network_es`, please read `#8` and `#9` in `docs/LESSONS.md`.
> (`will_network_es`를 실행하기 전에 반드시 `docs/LESSONS.md`의 8, 9번 항목을 확인하십시오.)
