# motion_person 모듈 경량화 및 리팩토링 검토

2026년 10월 12일 현재 작업 폴더 기준. 설치한 python-clean-architecture 스킬의 기존 심층 검토를 이어서 적용했다. 이번 범위는 다른 프로젝트에 붙일 때의 기능 경계, 의존성 용량, 실행 메모리, 장시간 운영과 유지보수다. 구현 코드는 변경하지 않았다.

**판단: 리팩토링 방향은 적절하다. 모듈 자체의 코드 용량은 작지만, 영상 분석 실행은 가볍지 않다. 우선 줄일 것은 필수로 결합된 실행 이력과 장시간 누적되는 추적 상태이며, 더 많은 계층이나 프레임워크를 추가할 필요는 없다.**

전체 테스트는 **227개 통과**했고 `git diff --check`도 통과했다. 이전 리팩토링 검토의 세 문제는 모두 수정됐다. 해당 과거 문서는 당시 상태의 기록이며, 현재 판정은 이 문서를 따른다.

## 현재 구조와 개선된 점

현재 패키지의 Python 파일 22개는 다음 역할을 가진다. 루트 호환 진입점, 테스트와 개발 도구의 전체 분류는 앞선 [심층 검토의 파일 목록](clean_architecture_review.md)에 있다.

| 역할 | 파일 |
| --- | --- |
| 공개 API·데이터 계약 | `__init__.py`, `contracts.py` |
| 설정·CLI 조립 | `config.py`, `cli.py` |
| 실행·공통 분석 | `engine.py`, `processor.py` |
| 입력 어댑터 | `video_source.py`, `webcam_source.py` |
| 분석·판정 | `frame_processor.py`, `object_detector.py`, `tracking_adapter.py`, `motion_detector.py`, `event_detector.py` |
| 사건 수집·사진 선택·저장 | `event_capture.py`, `photo_selection.py` |
| 실행 진단 | `run_logger.py` |
| 미리보기·표현 | `webcam_preview.py`, `overlay_renderer.py`, `console_messages.py` |
| 기존 개별 캡처 경로 | `main.py`, `capture_manager.py`, `object_state.py` |

실제 호출 흐름은 다음과 같다.

- Python API 또는 사건 CLI → MotionEngine → FrameProcessor → YOLO·MOG2·EventDetector
- FrameAnalysis → EventCaptureEngine → photo_selection → 원본 JPEG와 사건 메타데이터 → on_event
- MotionEngine → RunLogger 및 선택적 화면 표시
- 기존 개별 캡처 main → 별도 분석 반복문 → CaptureManager 및 ObjectStateStore

좋아진 부분은 다음과 같다.

- [contracts.py:329](../motion_person/contracts.py#L329)의 FrameAnalysis는 분석 결과를 명시적으로 전달한다. 프레임 배열을 추가로 복사하는 구조가 아니므로 이 DTO를 없애서 얻는 메모리 이익은 작다.
- [processor.py:16](../motion_person/processor.py#L16)의 주입 경로와 [photo_selection.py:10](../motion_person/photo_selection.py#L10)의 작은 Protocol은 테스트와 변경 범위를 줄인다. 런타임 컨테이너나 복잡한 상속 계층이 추가된 것이 아니다.
- [event_capture.py:156](../motion_person/event_capture.py#L156)의 사건 관측 집합은 사진 제거와 독립적으로 유지된다. 이전 사건의 사전 사진을 현재 사건 증거로 계산하는 문제도 수정됐다.
- [event_capture.py:103](../motion_person/event_capture.py#L103)의 새 필드는 마지막에 기본값을 두어 기존 EventBundle 생성자와 호환된다. 각각을 확인하는 회귀 테스트가 추가됐다.
- 사진 버퍼는 시간 범위와 JPEG 바이트 예산으로 제한된다. `_pending`은 정상 반환 시 제거되는 완료 묶음 복구 대기열이며, 모든 사건을 계속 모으는 이력 저장소가 아니다.

## 우선 보완할 문제

### P2 보조 실행 로그의 실패가 핵심 완료 전달을 막음

**위치:** [engine.py:138](../motion_person/engine.py#L138), [engine.py:162](../motion_person/engine.py#L162), [engine.py:258](../motion_person/engine.py#L258).

**원칙:** P1 High Cohesion, P2 Low Coupling, P7 Keep Things Simple. **패턴:** 선택적 진단 함수와 기존 완료 콜백.

현재 `capture=True`이면 사진 수집뿐 아니라 run_config.json·events.jsonl·summary.json 저장과 tracker.yaml 복사가 필수다. 사진은 유지하면서 실행 이력만 끄는 옵션은 없다. `capture=False`는 사진과 사건 콜백까지 끈다.

완료 전달에서 logger.write가 on_event보다 먼저 실행된다. EVENT_COMPLETED 로그에만 LogWriteError를 주입한 재현 결과는 다음과 같다.

- 사건 manifest 1개와 사진들은 이미 정상 저장됨.
- summary의 event_count는 1.
- on_event 호출은 0회이며 실행은 LogWriteError로 종료.
- 입력 자원은 정상 해제됨.

이것은 이번 DTO 리팩토링에서 새로 생긴 문제는 아니다. 기존 PoC의 엄격한 로그 계약이 라이브러리에도 결합되어 있는 문제다. 호스트 프로젝트에서 완료 콜백으로 후속 처리를 시작하면, 보조 이력 파일의 실패 때문에 저장된 사건을 전달받지 못한다.

라이브러리에는 별도의 실행 진단 옵션을 두고 기본적으로 끄는 것을 권장한다. 사진·event.json 저장 실패는 계속 핵심 오류로 취급한다. 선택적 진단 실패는 완료 콜백을 막지 않고, 제한된 오류 카운터나 호스트 로거로 보고한다. 시작·종료·tracker 설정 복사까지 함께 분리해야 하므로 전달 순서만 바꾸는 것으로 전부 해결되지는 않는다.

아래는 선택적 로그 실패와 핵심 전달을 분리하는 개념 예시다. 새 함수와 옵션은 현재 API에 존재하지 않는다.

```python
def write_optional_event(logger, bundle) -> bool:
    if logger is None:
        return True
    try:
        logger.write("EVENT_COMPLETED", bundle.to_dict())
    except LogWriteError:
        return False
    return True

# 실패 결과는 무제한 목록 대신 작은 진단 카운터 등에 반영한다.
diagnostic_ok = write_optional_event(logger, bundle)
if on_event is not None and callbacks:
    on_event(bundle)
```

기존 CLI의 엄격한 로그 모드는 호환 옵션으로 유지할 수 있다. 호스트 콜백 자체의 예외 처리 계약은 별도로 보존한다.

### P2 장시간 실행하면 사라진 추적 ID의 메모리가 계속 쌓임

**위치:** [object_detector.py:63](../motion_person/object_detector.py#L63), [object_detector.py:339](../motion_person/object_detector.py#L339), [object_detector.py:347](../motion_person/object_detector.py#L347).

**원칙:** P6 Start with the Data, P7 Keep Things Simple. **패턴:** 수명과 크기가 제한된 추적 상태 캐시.

`_identity_ids`, `_tracker_classes`, `_observation_hits`는 새 추적 ID마다 증가하고, 해당 객체가 사라져도 제거되지 않는다. 이 경로는 실제 MotionEngine의 추적 모드에서도 사용된다. JPEG 버퍼의 64 MiB 제한은 이 세 사전에 적용되지 않는다.

모델을 실행하지 않고 정상 박스 정리 함수에 매번 다른 ID를 전달한 합성 재현에서 다음 결과를 얻었다. tracemalloc으로 측정한 이 bookkeeping의 Python 할당 증가이며, 전체 프로세스 RSS나 실제 카메라의 ID 생성 속도가 아니다.

| 서로 다른 ID 수 | 각 사전의 항목 수 | 유지되는 Python 할당 증가 |
| --- | --- | --- |
| 10,000 | 각 10,000 | 약 4.12 MiB |
| 20,000 | 각 20,000 | 약 8.17 MiB |

현재 보이는 객체와 추적기가 다시 연결할 수 있는 최근 객체만 남기는 정책이 필요하다. 오래된 항목의 만료와 최대 상태 수를 두되, 가림 복구 및 관측 확인 횟수와 종류별 ID 분리 계약을 유지해야 한다. 웹캠의 원본 프레임 번호는 건너뛸 수 있으므로 만료 기준도 추적기 갱신 횟수와 맞춰야 한다.

캐시를 지우기 시작하면 ID 생성 방식도 바꿔야 한다. 현재 `len(_identity_ids) + 1`은 항목 삭제 뒤 ID를 재사용할 수 있다. 활성 상태 수와 독립적인 증가 카운터를 둔다.

```python
# 제안: __init__에서 self._next_identity_id = 0
if key not in self._identity_ids:
    self._next_identity_id += 1
    self._identity_ids[key] = self._next_identity_id
```

이력 조회가 필요하면 호스트의 저장소에 완료 사건을 기록한다. 분석기 안에서 모든 과거 ID를 영구 보관할 필요는 없다.

## 무게 측정

아래 수치는 현재 macOS 개발 환경의 측정값이다. 다른 OS·장치·입력 크기의 보장값이 아니다.

### 설치 용량

자체 Python 소스 22개는 합계 **249,132바이트, 약 243 KiB**다. pyproject의 패키지 범위는 motion_person이므로 저장소의 가상환경·예제 영상·개발 문서 전체가 모듈 설치물에 포함되는 것은 아니다. 모델도 별도 파일이며 현재 yolo11n.pt는 약 5.35 MiB다.

설치된 배포판의 등록 파일 크기를 합산했다. 아래 여섯 항목만으로 약 **722 MiB**다. 다운로드 크기·모든 전이 의존성의 합계·실행 RAM을 뜻하지 않는다.

| 배포판 | 설치 버전 | 등록 파일 크기 |
| --- | --- | --- |
| torch | 2.14.1 | 552.7 MiB |
| opencv-python | 5.0.0.93 | 119.3 MiB |
| numpy | 2.5.3 | 31.1 MiB |
| torchvision | 0.29.1 | 8.5 MiB |
| ultralytics | 8.4.174 | 8.4 MiB |
| lap | 0.5.13 | 1.6 MiB |

근거는 [pyproject.toml:11](../pyproject.toml#L11)과 실제 설치 메타데이터다. 설치된 Ultralytics는 torch·torchvision·OpenCV·matplotlib 등을 필수로 요구한다. 이력 클래스나 테스트 extra의 matplotlib를 지우는 것만으로 이 의존성들을 없앨 수 없다. 같은 호스트 환경에서 이미 사용하는 호환 패키지는 보통 공유하지만, 별도 프로세스의 모델 메모리는 별도 문제다.

### import 및 실제 영상 처리

각 import는 새 Python 프로세스에서 한 번씩 측정했다. RSS는 프로세스 생애 최대값이며 현재 상주 메모리와 다르다.

| 동작 | 측정 시간 | 최대 RSS |
| --- | --- | --- |
| 빈 Python 기준 | — | 16.1 MiB |
| import motion_person | 0.003초 | 16.9 MiB |
| from motion_person import MotionEngine | 0.307초 | 57.5 MiB |
| from ultralytics import YOLO, 모델 로드 전 | 1.082초 | 228.0 MiB |

실제 yolo11n.pt와 caviar_stop_resume.mp4의 첫 3초, 75프레임을 CPU에서 처리했다. 추적은 공개 API 기본값, display=False와 pace=fast를 사용했다.

| 실행 | 전체 실행 시간 | 최대 RSS | 분석 처리 FPS |
| --- | --- | --- | --- |
| capture=False | 6.154초 | 410.8 MiB | 21.5 |
| capture=True | 4.975초 | 436.9 MiB | 25.3 |

표본이 짧고 캐시·초기화 조건이 달라 두 행의 차이를 캡처 비용으로 해석할 수 없다. FPS에는 모델 초기화와 영상 열기 시간이 포함되지 않는다. 캡처 실행의 JPEG 버퍼 최대값은 약 248 KiB, 실행 이력은 약 5 KiB였고, 이 짧은 구간에서 완성된 사건은 없어 사진 묶음 디스크 쓰기 비용은 측정하지 않았다.

저장과 화면을 모두 끈 실행에서도 약 411 MiB를 사용했다는 점이 핵심이다. 이력만 제거해서 수십 MiB 수준의 영상 분석 모듈이 되지는 않는다. 반대로 64 MiB 버퍼 설정은 시작할 때 64 MiB를 미리 할당한다는 뜻도 아니다.

## 핵심으로 남길 기능과 선택 기능

| 기능 | 권장 경계 | 이유 |
| --- | --- | --- |
| YOLO 탐지·MOG2·움직임 판정 | 핵심 유지 | 현재 모듈의 주요 기능 |
| 사건 시작·종료·분할 및 사전/사후 맥락 | 핵심 유지 | 후속 분석에 필요한 사진 구간 결정 |
| 제한된 JPEG 버퍼·사진 선별 | 핵심 유지 | 과거 전체 영상을 보관하지 않고 증거를 제공 |
| 원본 사진·event.json·완료 콜백 | 핵심 유지 | 사진과 시각·객체·누락 정보를 함께 전달 |
| 임시 폴더 공개·저장 실패 처리·자원 해제 | 핵심 유지 | 불완전한 묶음을 성공으로 전달하지 않음 |
| FrameAnalysis·EventBundle·작은 Protocol | 유지 | 비용보다 계약 명확성과 테스트 이익이 큼 |
| 메모리의 RunSummary와 숫자 카운터 | 유지 | 작은 반환 데이터이며 전체 이력 목록이 아님 |
| run_config.json·events.jsonl·summary.json·tracker.yaml 복사 | 선택, 모듈 기본값은 꺼짐 권장 | 재현·디버그 목적. 현재는 캡처와 강제 결합 |
| 미리보기·오버레이·마스크 화면 | CLI/개발용 선택 | 현재도 display=False로 렌더링 경로를 끌 수 있음 |
| ObjectStateStore 전체 객체 이력·snapshot | 기존 CLI 전용 또는 필요 시 선택 | MotionEngine 경로에서는 사용하지 않음 |
| DB·장기 보관·보관 기간·알림·클라우드 작업 | 호스트 책임 | 해당 기능은 현재 모듈에 없으며 새로 넣을 필요 없음 |

event.json은 실행 이력 파일과 구분해야 한다. 사진의 시각·좌표·종류와 증거 누락 정보를 연결하는 최소 manifest다. 호스트에 같은 메타데이터를 전달하는 저장 어댑터를 실제로 구현할 때까지 삭제를 권장하지 않는다.

[event_capture.py:405](../motion_person/event_capture.py#L405)는 각 저장 파일을 flush·fsync한다. 느린 저장 장치에서는 지연 원인이 될 수 있지만, 이번 측정으로 병목이라고 확인하지는 않았다. 영속성 계약을 먼저 정하고 조정할 사항이다.

## 유지보수와 추가 경량화 순서

1. **실행 진단을 캡처에서 분리한다.** 현재는 사진을 유지하며 이력만 끄는 API가 없다. 새 diagnostics 옵션이나 선택적 로거 정도면 충분하고, 별도 이벤트 버스나 DI 프레임워크는 필요하지 않다.
2. **추적 상태의 수명을 제한한다.** 장시간 웹캠 운영 전 우선순위가 높다. 완료 묶음 복구 대기열과 사건 단위 관측 집합을 불필요한 이력으로 보고 제거하지 않는다.
3. **라이브러리 실행 기본값을 창 없이 정한다.** 현재 두 팩토리의 display 기본값은 True다. 사용자는 지금도 display=False로 호출할 수 있다. CLI의 표시 기본값과 라이브러리 기본값을 구분하는 정도면 충분하다.
4. **호스트가 이미 분석기를 갖고 있을 때만 주입 경로를 연결한다.** [processor.py:16](../motion_person/processor.py#L16)은 주입을 받지만 [engine.py:147](../motion_person/engine.py#L147)은 매 실행 새 FrameProcessor를 만들고 모델을 로드한다. 모델 재사용 또는 호스트 탐지 결과 입력의 필요가 확인되면 이 조립 경계 한 곳을 연다. 여러 카메라에서 persist=True 추적 상태를 한 객체로 무조건 공유하면 서로의 상태가 섞일 수 있으므로 카메라별 상태는 분리한다.
5. **기존 분석 반복문과 설정의 중복을 줄인다.** main의 개별 사진 모드와 engine의 사건 모드는 유지보수 경로가 둘이다. 기존 모드 사용 여부를 확인해 공통 FrameProcessor를 쓰거나 기존 모드를 호환/예제 경로로 격리한다. ObjectStateStore를 지우는 것만으로 현재 MotionEngine의 실행 비용이 줄지는 않는다.

핵심 경계는 **프레임 분석 → 사건 사진 생성 → 완료 결과 전달**이면 충분하다. 저장소가 하나인 동안 사건 정책·압축기·저장소·발행기 등을 모두 별도 계층으로 쪼개는 작업은 우선순위를 낮춰도 된다. 실제 호스트 저장소 교체가 필요할 때 작은 저장 경계를 추가한다.

설치 용량을 훨씬 줄여야 하는 별도 요구가 생기면 추론 백엔드 의존성이 대상이다. 호스트의 기존 분석기 재사용을 먼저 검토하고, 다른 백엔드로 바꾸려면 모델 출력·추적·정확도·배포 호환성을 별도로 검증한다. 핵심 기능을 유지하는 이번 정리에서 즉시 바꿀 대상은 아니다.

## 검증 범위

- 전체 테스트 227개 통과 및 git diff --check 통과.
- 기존 누락 표시와 생성자 호환성 회귀 테스트 확인.
- 저장 성공 뒤 보조 로그 실패에 따른 완료 콜백 누락을 합성 입력으로 재현.
- 탐지기 추적 사전의 누적 메모리를 서로 다른 ID 20,000개로 측정.
- 설치 배포판 파일 용량, 새 프로세스 import, 실제 CPU 영상 분석 측정.
- 성능 표본은 3초 영상이며, 실제 웹캠·다중 카메라·장시간 메모리 안정성·완료 사진 디스크 성능은 검증하지 않음.
- 구현 삭제·리팩토링·새 API 도입은 수행하지 않음.
