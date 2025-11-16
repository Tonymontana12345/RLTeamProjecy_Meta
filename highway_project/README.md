# MetaDrive 강화학습 프로젝트

고정 랜덤 시드를 사용한 자율주행 강화학습 실험 프로젝트

## 📋 프로젝트 개요

### 목표
- **고정 시드 학습**: 하나의 맵(시드 1000)에서 PPO 알고리즘으로 학습
- **일반화 성능 평가**: 새로운 맵(시드 2000-2004)에서 성능 측정
- **과적합 분석**: 고정 시드 학습의 장단점 파악

### 핵심 특징
- ✅ **MetaDrive**: 절차적 생성 기반 자율주행 시뮬레이터
- ✅ **재현성**: 고정 시드로 동일한 환경 보장
- ✅ **모듈화**: 깔끔한 코드 구조
- ✅ **PPO 알고리즘**: Stable-Baselines3 사용

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 활성화 (이미 생성되어 있음)
source venv_pgdrive/bin/activate

# 패키지 설치 확인
pip list | grep metadrive
```

### 2. 환경 테스트

```bash
# 기본 데모 (모든 기능 테스트)
python quick_start.py

# 환경 데모만
python quick_start.py --demo

# 수동 제어 (키보드로 직접 운전)
python quick_start.py --manual
```

### 3. 학습 시작

#### 방법 1: 모드 선택 (기본)

```bash
# 빠른 테스트 (10,000 스텝, 약 5-10분)
python train.py --mode quick

# 고정 시드 학습 (100,000 스텝, 약 1-2시간)
python train.py --mode fixed

# 다중 시드 학습 (200,000 스텝)
python train.py --mode multi

# 알고리즘 선택
python train.py --mode fixed --algorithm ppo   # PPO (기본)
python train.py --mode fixed --algorithm sac   # SAC
python train.py --mode fixed --algorithm td3   # TD3
```

#### 방법 2: 실험 프로토콜 사용 (권장)

```bash
# 실험 1: 고정 시드 학습
python train.py --experiment exp1_fixed_seed --algorithm ppo

# 실험 2: 다중 시드 학습
python train.py --experiment exp2_multi_seed --algorithm sac

# 실험 3: 빠른 테스트
python train.py --experiment exp3_quick_test --algorithm td3
```

**실험 프로토콜의 장점**:
- ✅ 사전 정의된 설정으로 재현성 보장
- ✅ 일관된 실험 환경
- ✅ 간편한 실험 관리

### 4. 평가

```bash
# 학습된 모델 평가
python evaluate.py --model models/ppo_fixed_seed_1000.zip --episodes 20

# 결과 시각화
python visualize.py --results results/evaluation_results.json
```

---

## 📁 프로젝트 구조

```
pgdrive_project/
├── config.py                 # 설정 (시드, 하이퍼파라미터 등)
├── train.py                  # 학습 메인 스크립트
├── evaluate.py               # 평가 스크립트
├── visualize.py              # 결과 시각화
├── quick_start.py            # 빠른 시작 가이드
├── requirements.txt          # 필요 패키지
│
├── agents/
│   └── rl_agent.py          # RL 에이전트 팩토리 (PPO, SAC, TD3)
│
├── envs/
│   └── metadrive_env.py     # MetaDrive 환경 래퍼
│
├── utils/
│   └── path_utils.py        # 경로 유틸리티
│
├── models/                   # 학습된 모델 저장
├── logs/                     # TensorBoard 로그
└── results/                  # 평가 결과 (그래프, JSON)
```

---

## ⚙️ 설정

### 랜덤 시드 (config.py)

```python
FIXED_SEED = 1000                      # 단일 시드 학습용
TRAIN_SEEDS = [1409, 2824, 5506]      # 학습용 시드 (랜덤 선정, 3개)
TEST_SEEDS = [2679, 3286, 4657, 5012, 9935]  # 평가용 시드 (랜덤 선정, 5개)
```

### 환경 설정

```python
FIXED_SEED_ENV_CONFIG = {
    "start_seed": 1000,           # 시드
    "num_scenarios": 1,           # 1개 맵만 사용
    "map": 5,                     # 5개 블록 맵
    "random_traffic": False,      # 트래픽 고정 (재현성)
    "traffic_density": 0.1,       # 차량 밀도
    "decision_repeat": 5,         # 액션 반복
}
```

### PPO 하이퍼파라미터

```python
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
}
```

### 학습 설정

```python
FIXED_SEED_TRAINING = {
    "total_timesteps": 100000,    # 총 학습 스텝
    "save_freq": 10000,           # 모델 저장 주기
    "model_name": "ppo_fixed_seed_1000",
}

QUICK_TEST_TRAINING = {
    "total_timesteps": 10000,     # 빠른 테스트용
    "save_freq": 5000,
    "model_name": "ppo_quick_test",
}
```

---

## 📊 사용 예시

### 학습

```bash
# 고정 시드 학습
python train.py --mode fixed

# 학습 중 TensorBoard 모니터링 (다른 터미널)
tensorboard --logdir logs/
# 브라우저: http://localhost:6006
```

### 평가

```bash
# PPO 모델 평가
python evaluate.py --model models/ppo_fixed_seed_1000.zip

# SAC 모델 평가 (자동 감지)
python evaluate.py --model models/sac_fixed_seed_1000.zip

# TD3 모델 평가 (자동 감지)
python evaluate.py --model models/td3_fixed_seed_1000.zip

# 특정 시드만 평가
python evaluate.py --model models/sac_fixed_seed_1000.zip --seeds 1000 2000

# 에피소드 수 조정
python evaluate.py --model models/ppo_fixed_seed_1000.zip --episodes 50

# 렌더링과 함께 평가
python evaluate.py --model models/sac_fixed_seed_1000.zip --render
```

**알고리즘 자동 감지**: 모델 파일명에서 알고리즘을 자동으로 감지합니다!
- `ppo_*.zip` → PPO
- `sac_*.zip` → SAC  
- `td3_*.zip` → TD3

### 시각화

```bash
# 평가 결과 시각화
python visualize.py --results results/evaluation_results.json

# 학습 곡선 그리기
python visualize.py --training-curve logs/ppo_fixed_seed_1000

# 시드별 맵 구조 확인 (블록 다이어그램)
python visualize_maps.py

# 특정 시드만 확인
python visualize_maps.py --seeds 1000 2000

# 맵 난이도 분석
python visualize_maps.py --difficulty

# 결과 저장
python visualize_maps.py --save

# 실제 도로처럼 보이는 3D 스타일 맵
python visualize_maps_3d.py

# 특정 시드만 3D 스타일로
python visualize_maps_3d.py --seeds 1000 2000
```

---

## 🎯 실험 시나리오

### 실험 1: 고정 시드 학습

**목표**: 시드 1000에서 완벽한 성능 달성

```bash
python train.py --mode fixed
python evaluate.py --model models/ppo_fixed_seed_1000.zip
python visualize.py --results results/evaluation_results.json
```

**예상 결과**:
- 학습 시드(1000): 성공률 80-90%
- 테스트 시드(2000-2004): 성공률 40-60%
- **일반화 갭 발생** (과적합 가능성)

### 실험 2: 일반화 성능 분석

**목표**: 새로운 맵에서의 성능 측정

```bash
# 평가
python evaluate.py --model models/ppo_fixed_seed_1000.zip --episodes 50

# 결과 분석
python visualize.py --results results/evaluation_results.json
```

**분석 지표**:
- 평균 보상 (Mean Reward)
- 성공률 (Success Rate)
- 일반화 갭 (Generalization Gap)

---

## 📈 평가 지표

### 1. 성공률 (Success Rate)
- **정의**: 목적지 도달 비율
- **목표**: >80%

### 2. 평균 보상 (Mean Reward)
- **정의**: 에피소드당 누적 보상
- **목표**: >15

### 3. 일반화 갭 (Generalization Gap)
- **정의**: 학습 시드 성능 - 테스트 시드 평균 성능
- **목표**: <5 (작을수록 좋음)

---

## 🤖 알고리즘 비교

프로젝트는 **3가지 알고리즘**을 지원합니다:

### 1. PPO (Proximal Policy Optimization) - 기본

```bash
python train.py --mode fixed --algorithm ppo
```

**특징**:
- ✅ On-policy 알고리즘
- ✅ 안정적인 학습
- ✅ 연속/이산 액션 모두 지원
- ✅ 초보자 친화적
- ⚠️ 샘플 효율성: 중간

**적합한 경우**: 안정적인 학습, 일반적인 RL 문제

### 2. SAC (Soft Actor-Critic)

```bash
python train.py --mode fixed --algorithm sac
```

**특징**:
- ✅ Off-policy 알고리즘
- ✅ 매우 샘플 효율적
- ✅ 연속 액션에 최적화
- ✅ 탐험 우수 (엔트로피 최대화)
- ⚠️ 메모리 사용량 많음
- ⚠️ 하이퍼파라미터 민감

**적합한 경우**: 샘플 효율성 중요, 연속 액션 환경

### 3. TD3 (Twin Delayed DDPG)

```bash
python train.py --mode fixed --algorithm td3
```

**특징**:
- ✅ Off-policy 알고리즘
- ✅ 샘플 효율적
- ✅ 연속 액션 전용
- ✅ 안정적인 학습 (이중 Q-네트워크)
- ⚠️ SAC보다 탐험 약함

**적합한 경우**: 안정성과 효율성의 균형

### 알고리즘 비교 표

| 특징 | PPO | SAC | TD3 |
|------|-----|-----|-----|
| **타입** | On-policy | Off-policy | Off-policy |
| **샘플 효율성** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **안정성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **학습 속도** | 느림 | 빠름 | 빠름 |
| **메모리 사용** | 중간 | 많음 | 많음 |
| **초보자 친화** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### 알고리즘 선택 가이드

```bash
# 처음 시작하거나 안정성 중요
python train.py --mode fixed --algorithm ppo

# 빠른 학습 필요, 샘플 효율성 중요
python train.py --mode fixed --algorithm sac

# 안정성과 효율성의 균형
python train.py --mode fixed --algorithm td3
```

### 알고리즘별 예상 성능

| 알고리즘 | 학습 시간 | 최종 성공률 | 샘플 효율성 |
|---------|----------|------------|------------|
| **PPO** | 1-2시간 | 80-85% | 100K 스텝 |
| **SAC** | 30-60분 | 85-90% | 50K 스텝 |
| **TD3** | 40-80분 | 82-88% | 60K 스텝 |

*참고: 실제 성능은 환경과 하이퍼파라미터에 따라 다를 수 있습니다.*

---

## 🔧 커스터마이징

### 하이퍼파라미터 튜닝

`config.py` 수정:

```python
PPO_CONFIG = {
    "learning_rate": 1e-4,  # 기본 3e-4에서 감소
    "n_steps": 4096,        # 기본 2048에서 증가
    "batch_size": 128,      # 기본 64에서 증가
}
```

### 다른 시드 사용

```python
FIXED_SEED = 3000  # 시드 변경
```

### 맵 구조 변경

```python
FIXED_SEED_ENV_CONFIG = {
    "map": 7,  # 블록 수 변경
    # 또는
    "map": "SCSRXTS",  # 특정 블록 시퀀스
}
```

**블록 타입**:
- `S`: Straight (직선)
- `C`: Circular (커브)
- `r`: InRamp (진입로)
- `R`: OutRamp (출구)
- `O`: Roundabout (로터리)
- `X`: Intersection (교차로)
- `T`: TIntersection (T자 교차로)

---

## 🐛 트러블슈팅

### 문제 0: Success Rate 그래프가 안 보임

**증상**: `visualize.py` 실행 시 "Success Rate Across Seeds" 그래프가 비어있음

**원인**:
- 성공률이 0%이거나 매우 낮아서 막대가 보이지 않음
- 학습이 충분하지 않아 에이전트가 목적지에 도달하지 못함

**해결 방법**:

```bash
# 1. 평가 결과 확인
python debug_results.py --results results/evaluation_results.json

# 2. 학습 상태 확인
tensorboard --logdir logs/

# 3. 더 긴 학습 실행
python train.py --mode fixed  # 100K 스텝

# 4. 학습 완료 후 재평가
python evaluate.py --model models/ppo_fixed_seed_1000.zip
python visualize.py --results results/evaluation_results.json
```

**참고**: 수정된 `visualize.py`는 성공률이 0%일 때도 값을 표시합니다.

### 문제 1: MetaDrive 설치 실패

```bash
# 의존성 먼저 설치
pip install numpy gym panda3d shapely matplotlib pillow Cython wheel

# MetaDrive 설치
pip install --no-build-isolation git+https://github.com/metadriverse/metadrive.git
```

### 문제 2: 학습이 너무 느림

`config.py` 수정:

```python
FIXED_SEED_ENV_CONFIG = {
    "use_render": False,      # 렌더링 끄기
    "decision_repeat": 10,    # 기본 5에서 증가
}
```

### 문제 3: 메모리 부족

```python
PPO_CONFIG = {
    "n_steps": 1024,   # 기본 2048에서 감소
    "batch_size": 32,  # 기본 64에서 감소
}
```

---

## 📚 참고 자료

### 공식 문서
- [MetaDrive GitHub](https://github.com/metadriverse/metadrive)
- [MetaDrive 문서](https://metadrive-simulator.readthedocs.io/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

### 논문
- [MetaDrive: Composing Diverse Driving Scenarios for Generalizable RL](https://arxiv.org/abs/2109.12674)
- [PPO: Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)

---

## 🎓 프로젝트 체크리스트

### 기본 실험
- [ ] 환경 설치 및 테스트
- [ ] 수동 제어로 환경 이해
- [ ] 빠른 학습 테스트 (10K 스텝)
- [ ] 본격 학습 (100K 스텝)
- [ ] 평가 및 시각화

### 심화 실험
- [ ] 하이퍼파라미터 튜닝
- [ ] 다른 시드 실험
- [ ] 일반화 성능 분석
- [ ] 결과 리포트 작성

---

## 💡 팁

### 학습 모니터링

```bash
# TensorBoard 실행
tensorboard --logdir logs/

# 학습 중 모델 저장 확인
ls -lh models/
```

### 빠른 디버깅

```python
# quick_start.py에서 환경 테스트
python quick_start.py --demo

# 짧은 학습으로 코드 검증
python train.py --mode quick
```

### 결과 저장

```bash
# 평가 결과 저장
python evaluate.py --model models/ppo_fixed_seed_1000.zip --save my_results.json

# 시각화 결과 저장
python visualize.py --results results/my_results.json
```

---

## 📞 문의

프로젝트 관련 질문이나 이슈가 있으면:
1. `config.py`의 설정 확인
2. 로그 파일 확인 (`logs/`)
3. TensorBoard로 학습 과정 모니터링

---

## 📄 라이센스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

**Happy Learning! 🚗🎓**
