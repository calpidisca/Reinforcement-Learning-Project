# Superpermutation RL Research Codebase

이 프로젝트는 강화학습(RL) 알고리즘을 사용하여 superpermutation 문제를 해결하는 연구 코드베이스입니다.

## 목표

n = 3, 4, 5, 6에 대해 여러 RL 알고리즘과 두 가지 MDP formulation을 비교합니다:
- **Symbol-based**: 한 번에 하나의 심볼을 추가
- **Word-based**: 한 번에 하나의 순열을 추가 (cost 기반 상태)

## 설치

### 가상환경 설정 (권장)

Python 가상환경(venv)을 사용하는 것을 권장합니다.

#### Linux / macOS

```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

#### Windows

**PowerShell:**
```powershell
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
.\venv\Scripts\Activate.ps1

# 패키지 설치
pip install -r requirements.txt
```

**CMD:**
```cmd
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate.bat

# 패키지 설치
pip install -r requirements.txt
```


#### 가상환경 비활성화
```bash
deactivate
```

## 프로젝트 구조

```
project_root/
  envs/
    __init__.py
    utils.py              # 유틸리티 함수 (permutation, overlap, coverage, canonicalization)
    symbol_env.py         # Symbol-based 환경
    word_env_cost.py      # Word-based 환경
  algorithms/
    __init__.py
    random_policy.py     # Random baseline
    greedy_policy.py      # Greedy baseline
  utils/
    __init__.py
    common.py            # 공통 유틸리티 함수 (run name, metadata, seed 설정 등)
    types.py             # 타입 정의 (EnvType, AlgoName)
  config.py              # 실험 설정 (중앙 관리)
  callbacks.py            # SB3 콜백 (에피소드 메트릭 로깅)
  train_one.py           # 단일 RL 실행
  sweep.py               # 전체 실험 스윕
  metrics.py             # 메트릭 집계 및 top-3 superpermutation 추출
  requirements.txt
  README.md
  generate_ci_report.py
  logs/                   # 실행 시 생성되는 로그 디렉토리
  metrics/                   # 실행 시 생성되는 메트릭 디렉토리
  report_ci/                   # 실행 시 생성되는 차트 디렉토리
```

## 사용 방법

### 단일 실행 예시

```python
from train_one import train_one_run

train_one_run(
    env_type="symbol",
    n=4,
    algo_name="ppo",
    hyperparams={"learning_rate": 3e-4, "gamma": 0.99},
    seed=42,
    total_timesteps=200_000,
    log_root="logs",
)
```

### 전체 스윕 실행

```bash
python sweep.py
```

이 명령은 다음 조합에 대해 실험을 실행합니다:
- n ∈ {3, 4, 5, 6}
- env_type ∈ {"symbol", "word_cost"}
- 알고리즘: PPO, MaskablePPO, A2C, DQN, Double+Dueling DQN, Random, Greedy
- 여러 random seeds (0, 1, 2, 42, 100, 123, 999)
- 하이퍼파라미터 그리드

### 메트릭 집계

실험 완료 후 메트릭을 집계하고 top-3 superpermutation을 추출:

```bash
python metrics.py
```

이 명령은:
- 모든 실행의 메트릭을 집계하여 `metrics/metrics_summary.csv`와 `metrics/metrics_summary.json`에 저장
- 각 (env_type, n, algo, hyperparams) 조합에 대해 top-3 shortest distinct superpermutation을 `metrics/` 디렉토리에 저장

## 로그 구조

각 실행은 `logs/` 디렉토리 아래에 다음 구조로 저장됩니다:

```
logs/
  {env_type}_n{n}_{algo_name}_seed{seed}_{hyperparams}/
    progress.csv          # 에피소드별 메트릭
    metadata.json         # 실행 메타데이터
    model/                # 저장된 모델 파라미터
    tensorboard/          # TensorBoard 로그
```
## 시각화
```bash
python generate_ci_report.py --log_root logs --report_dir report_ci
```

## 구현된 알고리즘

### RL 알고리즘 (Stable-Baselines3)
- **PPO**: Proximal Policy Optimization
- **MaskablePPO**: Action masking을 지원하는 PPO (WordEnv용)
- **A2C**: Advantage Actor-Critic

### 베이스라인
- **Random**: 무작위 정책
- **Greedy**: 탐욕적 정책 (환경별로 다른 휴리스틱 사용)

## 메트릭

각 실행에서 수집되는 메트릭:
- `final_length`: 최종 superpermutation 길이
- `success`: 성공 여부 (0/1)
- `coverage_ratio`: 커버된 순열 비율
- `episode_steps`: 에피소드 스텝 수
- `episode_return`: 에피소드 총 보상
- `duplicate_action_ratio`: 중복 액션 비율
- `best_final_length_so_far`: 지금까지의 최단 길이

## Canonicalization

Superpermutation은 relabeling symmetry를 고려하여 canonicalize됩니다:
- 첫 번째 길이 n의 순열 블록을 찾아 relabeling 매핑 생성
- 전체 시퀀스에 매핑 적용
- 이를 통해 동일한 superpermutation의 다른 표현을 통합

## 참고사항

- DQN 변형은 `word_cost` 환경에서 n=6일 때 계산 부담이 커서 스킵됩니다 (config.py의 SKIP_CONDITIONS 참조).
- 모든 실험 설정은 `config.py`에서 중앙 관리됩니다.
- 학습 알고리즘의 경우 모델 파라미터가 저장되어 나중에 로드할 수 있습니다.
- 성공한 에피소드 중 최단 길이 상위 5개만 저장됩니다 (callbacks.py의 top_k=5).

