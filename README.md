# Policy Simulation — Cleaned

정책 문서를 IAD/IG 지식 그래프로 구조화하고, 이해관계자 페르소나를 구성한 뒤 Theory of Change(ToC)로 정책 전개를 시뮬레이션하는 실행 코드입니다. 기존 작업 폴더의 현재 소스를 기준으로 핵심 파이프라인, 분기 생성, 비교 실험, 재사용 가능한 유틸리티와 테스트를 정리했습니다.

기존 실행 결과, 정책 원문, Nemotron parquet, PDF 파싱 캐시, 가상환경, 개인 `.env`, 프롬프트 백업, 발표자료, 웹 데모, 정책별 일회성 배치 스크립트 및 별도 평가 프로젝트는 포함하지 않습니다. 국가별 필드 정의와 enum Python 모듈은 실행 의존성이므로 유지합니다. 새 실행 결과는 `outputs/` 아래 생성됩니다. 원래 작업 폴더는 보존합니다.

## 1. 구조

```text
Policy_Simulation_Cleaned/
├── main.py                   # 파이프라인 CLI
├── config/
│   ├── base.yaml             # 공통 실행 설정
│   └── policies.yaml         # 선택: baseline/ablation 입력 경로
├── graph/                    # 온톨로지, KG 추출·정규화·검증, 로컬/Zep reader
├── persona/                  # 데이터 기반 표집, 역할 정합화, 페르소나 스키마
├── stage1/                   # 정책 맥락, 페르소나 풀, 시나리오/고정 패널 구성
├── stage2_toc/               # ToC 순방향·역방향, 조건 분기, grounded Inputs
├── stage2/                   # 기존 시뮬레이션 및 ToC가 재사용하는 생성 모듈
├── models/                   # Pydantic 스키마, 프롬프트, KG 추출기
├── llm/                      # 모델 호출, 재시도·사용량 기록, PDF/Markdown 로더
├── baseline/                 # 정책 요약 + 페르소나 직접 예측
├── ablation/                 # Markdown baseline, KG 제외, ToC 제외 비교
├── zero_shot/                # 정책 문서만으로 직접 예측
├── dataset/processed/        # 국가별 Python 필드/enum만 포함
├── scripts/                  # 데이터 준비·패널·분기·결과 집계 유틸리티
├── tests/                    # 외부 API 없이 수행하는 회귀 테스트
└── requirements*.txt         # 필수/선택/개발 의존성
```

## 2. 설치와 입력 준비

모든 명령은 **이 README가 있는 폴더에서** 실행합니다. Python 3.11 이상을 사용합니다. 기존 로컬 정리본을 사용한다면 clone 대신 `Policy_Simulation_Cleaned` 폴더로 이동합니다.

```bash
git clone https://github.com/Jhyunee/Policy_Simulation_Framework.git
cd Policy_Simulation_Framework
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python main.py --help
```

`.env`에 `DEEPSEEK_API_KEY`를 채웁니다. 다른 공급자를 사용하려면 `config/base.yaml`의 `llm.url`, `llm.api_key_env`, `llm.model` 및 해당 공급자 설정을 함께 변경합니다. 예시 모델 이름은 기존 코드의 설정을 이어받은 값이며, 실제 계정에서 사용할 수 있는 모델로 지정하세요. `.env`는 새 폴더의 파일만 읽습니다. 프로세스에 이미 설정된 환경변수는 그대로 사용할 수 있습니다.

기본 PDF 파서는 `pypdf`입니다. `pdf.parser: opendataloader`를 사용하려면 `opendataloader-pdf`와 Java 11+가 필요합니다. Gemini 및 Zep도 선택 기능입니다.

```bash
# 해당 기능을 사용할 때만 설치
python -m pip install -r requirements-optional.txt
```

### 정책 문서

```text
dataset/<country>/<policy>/input_policy/policy.pdf
```

예를 들어 `--policy kor/example`은 `dataset/kor/example/input_policy/`를 읽습니다. 그 폴더에 PDF가 없으면 `.md` 또는 `.markdown`을 읽습니다. PDF와 Markdown이 함께 있으면 PDF가 우선합니다. 입력 문서는 직접 준비합니다.

```bash
mkdir -p dataset/kor/example/input_policy
# 위 폴더에 분석할 정책 PDF 또는 Markdown을 넣습니다.
```

### Nemotron 데이터: 기본 scope_definer 방식

기본 설정은 로컬 parquet를 DuckDB로 조회합니다. 아래 다운로드를 실행하거나 기존 parquet를 같은 경로에 배치합니다. 데이터는 정리본에 포함하지 않았습니다.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="nvidia/Nemotron-Personas-Korea",
    repo_type="dataset",
    local_dir="dataset/processed/nemotron_kor/Nemotron-Personas-Korea",
    allow_patterns=["data/*.parquet"],
)
PY
```

| `--policy` 국가 접두사 | 필드 패키지/저장 폴더 | 데이터셋 이름 |
|---|---|---|
| `kor` | `nemotron_kor` | `Nemotron-Personas-Korea` |
| `singapore` | `nemotron_sg` | `Nemotron-Personas-Singapore` |
| `usa` | `nemotron_usa` | `Nemotron-Personas-USA` |
| `fr` | `nemotron_fr` | `Nemotron-Personas-France` |
| `jp` | `nemotron_jp` | `Nemotron-Personas-Japan` |

예상 parquet 경로는 `dataset/processed/<폴더>/<데이터셋 이름>/data/*.parquet`입니다. `--policy`는 입력 경로, 국가별 sampler 설정, 출력 경로를 함께 바꿉니다. 표에 없는 국가는 데이터 스키마와 국가 매핑을 별도로 추가해야 합니다.

**parquet 없이 실행**하려면 `config/base.yaml`에서 `persona.sampler`를 `kg_gen`으로 바꿉니다. 이 방식은 KG의 행위자와 정책 맥락을 근거로 LLM이 페르소나를 생성하며, Nemotron에서 표집한 페르소나와는 다른 실험 조건입니다. `persona.kg_gen.personas_per_entity`로 후보 수를 조절합니다.

## 3. 기본 실행

처음에는 기본 설정의 3개 시나리오 중 1개만 실행해 입력과 출력을 점검합니다. 모든 생성 단계는 실제 LLM을 호출합니다.

### ① 온톨로지 및 지식 그래프

```bash
python main.py gen-ontology "정책의 시행 과정과 이해관계자별 효과 분석" \
  --policy kor/example \
  --out outputs/example/ontology/graph_ontology.json

python main.py build-kg "정책의 시행 과정과 이해관계자별 효과 분석" \
  --policy kor/example \
  --graph-ontology outputs/example/ontology/graph_ontology.json \
  --out outputs/example/kg/policy_graph.json
```

`build-kg`에서 `--graph-ontology`를 생략하면 온톨로지도 함께 생성합니다. `validation_report.json`이 생성되면 노드·엣지 검증 결과를 함께 확인합니다. 명시적으로 준 `--out` 경로는 자동 출력 경로보다 우선합니다.

### ② 페르소나·시나리오 생성

```bash
python main.py stage1 "정책의 시행 과정과 이해관계자별 효과 분석" \
  --policy kor/example \
  --policy-graph outputs/example/kg/policy_graph.json \
  --run-tag initial
```

출력되는 `run_dir`를 다음 단계에서 사용합니다. 핵심 전달 파일은 `stage1_result.json`, `simulation_personas.json`입니다. 아래 `STAGE1_RUN`은 **방금 출력된 실제 경로로 교체**합니다.

```bash
STAGE1_RUN='outputs/kor/example/stage1/<실제 실행 폴더>'

python main.py stage2-toc "정책 시행 후 1년간 예상되는 참여 규모와 성과" \
  --policy kor/example \
  --stage1-result "$STAGE1_RUN/stage1_result.json" \
  --scenarios 1-1 \
  --run-tag first
```

`--simulation-personas`를 생략하면 Stage 1 결과와 같은 폴더의 파일을 사용합니다. `--scenarios`는 1부터 시작하는 양끝 포함 범위입니다. 생략하면 준비된 전체 시나리오를 실행합니다.

### ③ 결과 확인

Stage 2 출력은 `outputs/kor/example/stage2_toc/<실행 폴더>/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `inner_schema.json` | 각 ToC 단계의 변수·단위·논의 항목 |
| `scenario_*.json` | 참여 페르소나, 단계별 발화·예측·집계 |
| `stage2_toc_result.json` | 전체 시나리오 결과 |
| `final_report.json` | 최종 집계 및 구현 격차 보고 |
| `run_log.json` | 설정, 입력 파일 경로, 소요 시간, 토큰 사용량 |

이 파일들은 실행 시 새로 만들어지며 정리본에는 저장되어 있지 않습니다.

## 4. 구현된 방법론

### 정책 구조화: IAD + Institutional Grammar

문서를 청크 단위로 처리하여 행위자·역할, 활동, 자격, 제약, 자원, 정보, 권한, 시점, 성과를 노드와 관계로 추출합니다. IAD의 position, boundary, choice, aggregation, information, payoff, scope 규칙과 IG의 의무·허용·금지 및 정의 관계를 프롬프트와 검증 규칙에 반영합니다. 노드 ID와 이름을 정규화하고 중복을 병합하며, 관계 참조와 행위자 분류를 검증합니다.

### 이해관계자와 페르소나

KG에서 행위자 역할을 고른 후 정책 맥락과 연결합니다. 기본 `scope_definer`는 LLM이 연령·학력 범위를 제안하고, DuckDB가 실제 데이터의 직업 후보를 조회한 뒤, LLM이 direct/plausible/weak 범위별 직업을 선택하는 흐름입니다. 필터링한 후보 풀에서 시나리오 참여자를 구성합니다. 기본 설정은 풀이 부족할 때 전체 Nemotron에서 무작위 표집을 허용하므로, 이 대체 표집을 금지하려면 `persona.scope_definer.fallback.enabled: false`로 바꾸고 관련 로그를 확인합니다.

`kg_gen`은 데이터 표집 대신 KG 기반 합성 후보를 생성합니다. 조직 행위자의 구조적 맥락과 개인 페르소나를 구분하고, `persona/simulation_schema.py`로 시뮬레이션에 전달할 속성을 정리합니다.

고정 역할 패널은 `persona.selection_mode: mechanism_panel`로 활성화합니다. `persona.mechanism_panel.roles`에 해당 정책의 역할·KG 매칭 키워드·제약·ToC 관련 단계를 명시하고, 역할별 후보 중 한 명씩 선택합니다. 모듈에 남아 있는 기본 역할은 의료정책 예제이므로 다른 정책에는 명시적인 역할 정의를 넣어야 합니다. 자동 refiner 패스는 현재 Stage 1에서 비활성화되어 있으며, 필요하면 `scripts/refine_scenario_personas.py`를 별도로 실행합니다.

### ToC 추론과 조건 분기

```text
Inputs → Activities → Outputs → Outcomes → Impact
       ←            ←         ←          ←
```

모델이 정책·분석 목표에 맞는 단계별 변수 스키마를 만들고, 같은 참여자들이 각 단계의 전개와 값을 추론합니다. 기본 설정은 순방향 결과의 Impact를 받아 역방향으로 필요한 조건을 추론합니다. `phase_interaction_mode: independent`는 같은 단계에서 서로의 초기 답을 보지 않고 응답하도록 합니다. `sequential`, `independent_revision`도 지원합니다.

분기 생성은 부모 경로의 이력을 조건으로 enabling / baseline / constraining 자식 단계를 생성합니다. 이는 경로 탐색을 위한 조건이며 발생 확률을 뜻하지 않습니다. 시뮬레이션 값은 모델·프롬프트·입력·표집 방식에 따른 생성 결과로 해석합니다.

### 재현과 비교

동일 비교에서는 정책 문서, 그래프, 참여자, 목표, `inner_schema.json`을 고정하고 실험 조건만 바꿉니다. 표집 seed는 후보 선택의 재현성을 돕지만 외부 LLM 응답까지 결정적으로 만들지는 않습니다. 평가용 실제값 `gt`는 비교용 입력이며 프롬프트의 예측 근거와 구분합니다. 기본 설정은 정리본의 소규모 시작점으로, 과거 실험 설정을 그대로 재현하지 않습니다.

## 5. 선택 실행

### 순방향/역방향 및 재개

```bash
# 순방향만
python main.py stage2-toc "분석 목표" \
  --stage1-result "$STAGE1_RUN/stage1_result.json" --no-backward --run-tag forward

# 역방향만: 순방향 Impact를 seed로 사용하지 않는 비교
python main.py stage2-toc "분석 목표" \
  --stage1-result "$STAGE1_RUN/stage1_result.json" --no-forward --run-tag backward

# 기존 스키마 고정 및 완료된 시나리오 재사용
python main.py stage2-toc "분석 목표" \
  --stage1-result "$STAGE1_RUN/stage1_result.json" \
  --inner-schema outputs/<이전 실행>/inner_schema.json \
  --resume-from outputs/<이전 실행> --run-tag resumed
```

위 선택 예제는 `--policy`를 생략했으므로 `outputs/stage2_toc/`에 저장됩니다. 정책별 하위 폴더를 원하면 `--policy kor/example`을 추가합니다. 재개는 같은 입력·스키마·조건의 실행에서 사용합니다.

추가 체인은 `stage2_toc.fwd_bwd_fwd_seq`, `fwd_bwd_fwd_bwd_seq`, `final_forward_refinement`로 켭니다. 순방향과 역방향의 독립 병렬 실행은 `sequential_backward: false`와 `parallel_forward_backward: true`를 함께 지정합니다. 시나리오 병렬화는 `--parallel-scenarios`로 켤 수 있습니다.

### 분기 트리

순방향을 포함해 완료된 Stage 2 실행 폴더를 지정합니다.

```bash
python scripts/generate_phase_branch_tree.py \
  --source-dir outputs/<완료된 Stage2 실행> \
  --scenario 1 \
  --output outputs/branching/pathway_branch_tree.json \
  --max-expansions 1
```

`--max-expansions 1`은 한 부모만 확장하는 확인 실행입니다. 제한을 빼고 같은 명령을 실행하면 저장된 체크포인트에서 나머지를 이어 생성합니다. 5단계에서 Inputs를 고정하고 이후 4단계를 모두 3갈래로 확장하면 전체 노드는 121개이므로 먼저 작은 범위로 확인합니다. `--branch-inputs`는 Inputs도 분기합니다.

### 문서 근거 기반 Inputs

```bash
python main.py grounded-inputs \
  --stage1-result "$STAGE1_RUN/stage1_result.json" \
  --policy-graph outputs/example/kg/policy_graph.json \
  --output outputs/example/grounded_input_state.json
```

`--policy` 옵션이 없는 명령이므로 다른 정책을 사용할 경우 `--config-path`로 그 정책의 `paths.input_policy_dir`가 지정된 설정을 전달합니다. `scripts/apply_grounded_inputs_to_schema.py`로 생성된 입력 상태를 기존 ToC 스키마에 적용할 수 있습니다.

### Baseline / Ablation

기본 파이프라인에는 필요하지 않습니다. 비교 실험 시 `config/policies.yaml`에 정책 Markdown과 새로 생성한 Stage 1의 `simulation_personas.json` 경로를 등록합니다. `config/policy_targets.json`도 직접 준비합니다. 아래 숫자는 형식 예시이며 실험의 실제 목표·관측값으로 바꿉니다.

```json
{
  "kor/example": {
    "simulation_target": "정책 시행 1년 후 참여자 수",
    "gt": {"participants": 100}
  }
}
```

| 실행 모듈 | 비교 조건 |
|---|---|
| `python -m zero_shot.predict` | 정책 Markdown만으로 직접 예측 |
| `python -m baseline.predict` | 정책 요약 + 기존 페르소나의 직접 예측 |
| `python -m ablation.baseline_md` | KG 없이 정책 텍스트 역할과 Nemotron 표집으로 구성한 baseline |
| `python -m ablation.no_kg_toc` | 위 baseline 실행의 참여자를 이용한 ToC |
| `python -m ablation.no_toc` | 기존 KG 페르소나를 이용한 직접 예측 |

각 모듈의 `--help`에서 필수 입력을 확인합니다. 예:

```bash
python -m zero_shot.predict --policies kor/example --repeats 1
python -m baseline.predict --policies kor/example --max-scenarios 1
python -m ablation.baseline_md --policy kor/example --scenarios 1
python -m ablation.no_kg_toc --policy kor/example \
  --baseline-run-dir outputs/ablation/baseline_md/<실제 실행> --scenarios 1
python -m ablation.no_toc --policy kor/example --scenarios 1
```

단일 ToC CLI에서도 `--simple-prediction`, `--policy-md-override`, Stage 1의 `--no-graph`, KG의 `--no-iadig`로 비교 조건을 구성할 수 있습니다. 조건별 정책 맥락과 페르소나 입력을 확인한 후 사용합니다.

## 6. 유틸리티와 검증

모든 유틸리티는 `python scripts/<파일명>.py --help`로 인자를 확인합니다.

| 용도 | 스크립트 |
|---|---|
| 국가별 enum 생성·데이터 값 확인 | `generate_nemotron_enums`, `nemotron_unique_values` |
| KG 범위 축소 | `scope_policy_graph`, `filter_policy_graph_scope` |
| 역할 패널 설계·재선정·분할 | `build_mechanism_panel_blueprint`, `reselect_existing_mechanism_panel`, `build_disjoint_mechanism_panels` |
| 페르소나 생성·정합화 | `generate_kg_personas`, `generate_md_personas`, `refine_scenario_personas`, `normalize_simulation_personas` |
| 동일 패널 반복 | `repeat_simulation_panel` |
| 스키마 앙상블·축약·Inputs 반영 | `generate_inner_schema_ensemble`, `select_compact_inner_schema`, `apply_grounded_inputs_to_schema` |
| 분기 생성·다음 단계 예측 | `generate_phase_branch_tree`, `predict_next_phase_from_run` |
| 집계·복구 | `build_results_csv`, `build_results_csv_exact`, `build_quantitative_report`, `recover_stage2_run`, `analyze_branch_tree_mape` |

CSV 비교 스크립트는 순차 실행과 backward-only 실행의 Outcomes 변수를 비교하는 기존 분석용 도구입니다. `build_results_csv_exact.py`는 정확한 필드명을 지정합니다. 다른 ToC 단계나 변수 체계에는 해당 도구의 집계 규칙을 먼저 확인하세요. 비활성화된 추가 체인의 열은 비어 있을 수 있습니다.

```bash
python -m pip install -r requirements-dev.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

정리 시 Python 3.13 환경에서 회귀 테스트 57개, CLI 도움말 검사 36개, 전체 Python 구문 검사를 통과했습니다. 원본 저장소 밖의 임시 폴더로 정리본을 복사한 상태에서도 테스트 57개가 통과했습니다. 외부 LLM 호출, 실제 정책 데이터로의 전체 생성, Gemini/Zep/OpenDataLoader 연동은 이 검증에 포함하지 않았습니다.

새 국가를 추가할 때는 `dataset/processed/nemotron_<국가>/fields.py`와 enum을 만들고 `main.py`, `ablation/common/io.py`의 국가 매핑을 함께 갱신합니다. `kg_gen`을 쓸 경우 `stage1/modules.py`의 언어·국가 매핑도 맞춥니다.
