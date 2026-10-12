# motion_person 리팩토링 검토

2026년 10월 12일 현재 미커밋 리팩토링을 기준으로 검토했다. 기존 심층 검토 기준을 이어서 적용했다.

읽기와 테스트 가능성은 개선됐다. FrameAnalysis로 결과를 명확히 하고, FrameProcessor에 의존성 주입 경로를 열고, 사진 선별을 소비자 측 Protocol로 분리한 방향은 적절하다. 앞서 발견한 구현과 호출부의 불일치도 해소되어 **전체 223개 테스트가 통과**한다. 추가 합성 입력에서는 아래 3개 문제를 재현했다.

## 현재 구조

- CLI 또는 공개 API → MotionEngine → FrameProcessor → 탐지 및 전경 분석과 EventDetector
- 분석 결과 FrameAnalysis → EventCaptureEngine → 사진 선별 규칙 → JPEG 및 JSON 저장
- 웹캠 미리보기 → FrameProcessor → 화면 표시
- 기존 개별 캡처 main → 별도 프레임 분석 순서 → CaptureManager

현재 변경한 Python 파일 10개는 아래와 같이 분류할 수 있다.

| 계층 또는 역할 | 파일 |
| --- | --- |
| 공개 API와 공유 DTO | motion_person/__init__.py 및 contracts.py |
| 분석 및 실행과 표현 | motion_person/processor.py 및 engine.py 및 webcam_preview.py |
| 사건 규칙과 저장이 혼합됨 | motion_person/event_capture.py |
| 독립 선별 규칙 | motion_person/photo_selection.py |
| 회귀 및 호출 계약 테스트 | tests/test_engine.py 및 test_event_capture.py 및 test_webcam_preview.py |

기존 main의 분석 반복문과 API 및 미리보기의 실행 수명 관리는 아직 별도다. 사건 상태와 압축 버퍼 및 디스크 저장도 같은 엔진에 남아 있다. 현재 단계는 도메인과 인프라의 완전한 분리까지 진행한 상태는 아니다.

## 개선된 점

- [contracts.py:329](../motion_person/contracts.py)의 FrameAnalysis는 packet과 objects 및 motion과 decision을 명시적으로 전달한다. 분석 결과에서 DISABLED CaptureResult를 제거하고 표시할 때 만드는 방식이 책임에 맞다.
- [processor.py:16](../motion_person/processor.py)에 detector와 motion 및 events를 주입할 수 있다. 테스트가 분석기 생성 후 속성을 직접 덮어쓰는 부담이 줄었다.
- [photo_selection.py:10](../motion_person/photo_selection.py)의 PhotoCandidate는 필요한 인터페이스를 소비자 측에서 정의한다. event_capture의 비공개 _Sample 타입을 import하던 의존성이 제거됐다.
- [event_capture.py:218](../motion_person/event_capture.py)의 완료 묶음 회수와 [engine.py:233](../motion_person/engine.py)의 복구 호출이 실제로 연결됐다. 다음 사진 실패 때 완료 묶음을 회수하는 기존 회귀 테스트도 통과한다.

## 수정이 필요한 사항

### P2 메모리에서 제거된 객체가 누락 기록에서도 사라짐

**위치:** [event_capture.py:361](../motion_person/event_capture.py).

**원칙:** P6 Start with the Data 및 P1 High Cohesion. **패턴:** 사건 관측 메타데이터와 제거 가능한 사진 버퍼를 분리.

observed를 현재 part.frames에서만 계산한다. 메모리 예산 때문에 프레임이 제거되면 그 프레임에만 존재했던 객체는 실제 관측 목록에서도 빠진다. omitted_moving_objects는 사진 수 제한으로 빠진 일부 객체는 표시하지만 메모리 제거로 사라진 객체를 표시하지 못한다.

3.0초부터 3.4초까지 ID 1과 99와 100과 101과 102를 각각 관측하고 max_buffer_bytes=3600으로 실행했다. 저장된 움직임 증거는 ID 1과 102였고, 누락 목록은 비어 있었다. ID 99와 100과 101의 증거가 없다는 사실이 기록되지 않는다.

각 part가 시작한 뒤 조건을 충족한 객체 키를 별도의 set에 누적한다. 사진을 제거해도 이 집합은 유지한다. 다음 코드는 경계의 예시다.

```python
# _Part의 필드
observed_moving_objects: set[ObjectKey] = field(default_factory=set)

# 유효한 현재 관측을 활성 part에 반영할 때
part.observed_moving_objects.update(moving_keys)

# 종료 시 사진 버퍼가 아닌 관측 이력과 비교
omitted = part.observed_moving_objects - covered
```

서로 다른 객체의 유일한 사진 여러 개를 강제로 제거해야 하는 작은 메모리 예산으로 검증하는 테스트가 필요하다.

### P2 이전 사건의 사진이 현재 사건의 움직임 증거로 계산됨

**위치:** [event_capture.py:363](../motion_person/event_capture.py), [photo_selection.py:39](../motion_person/photo_selection.py).

**원칙:** P6 Start with the Data 및 P2 Low Coupling. **패턴:** 명시적인 사건 구간에 따른 증거 투영.

observed는 trigger 이후로 제한하지만 covered는 선택된 사전 사진도 포함한다. 같은 track_id가 이전 사건과 현재 사건에 나타나면, 이전 움직임 사진만으로 현재 움직임이 보존됐다고 계산할 수 있다. 선별 정책의 uncovered 계산도 같은 문제를 가진다.

quiet_sec=1 및 max_images=3으로 첫 사건의 1초에 ID 99를 관측하고, 두 번째 사건의 4초에 ID 1 및 4.1초에 ID 99를 관측했다. 두 번째 묶음의 사진은 1초와 4초 및 5.1초다. 4.1초 증거가 없는데 omitted_moving_objects는 비어 있다.

사진은 사전 맥락으로 유지하되, 현재 사건의 움직임 증거 충족 여부는 현재 part의 활성 구간에서만 계산한다.

```python
evidence_start = max(part.start, part.trigger)
covered = {
    key
    for sample, _ in selected
    if sample.timestamp >= evidence_start
    for key in sample.moving_objects
}
```

photo_selection의 이미 선택된 사진에 따른 uncovered 갱신에도 동일한 구간 조건을 적용한다. 단순히 ID가 같은 사진이 있다는 조건과 현재 사건의 움직임 증거가 있다는 조건을 구분해야 한다.

### P2 공개 EventBundle 생성자의 호환성이 깨짐

**위치:** [event_capture.py:100](../motion_person/event_capture.py).

**원칙:** P6 Start with the Data. **패턴:** 공개 DTO의 기존 필드 순서와 기본값을 통한 호환성 유지.

omitted_moving_objects가 필수 필드로 기존 필드 중간에 삽입됐다. EventBundle은 패키지에서 공개하는 타입이다. 기존 키워드 인자로 생성하면 새 필드가 없다는 TypeError가 발생하고, 위치 인자의 기존 순서도 달라진다.

기존 필드의 순서는 유지하고, 새 필드를 맨 끝에 기본값과 함께 추가한다. 아래는 마지막 필드들의 배치를 보여주는 예시다.

```python
@dataclass(frozen=True)
class EventBundle:
    # 앞의 기존 필드들은 그대로 유지
    images: tuple[EventImage, ...]
    analysis_roi: tuple[float, float, float, float] | None
    manifest_path: Path
    omitted_moving_objects: tuple[tuple[str, int | None], ...] = ()
```

내부 생성 코드도 새 필드를 키워드 인자로 전달한다. 이전 키워드 및 위치 인자로 생성하는 호환 테스트를 추가하면 이 문제가 반복되지 않는다.

## 다음 구조 개선 우선순위

### 공통 분석과 실행 흐름

[main.py:442](../motion_person/main.py)은 여전히 분석 패킷과 YOLO 및 MOG2와 판정을 직접 실행한다. [engine.py:192](../motion_person/engine.py)은 FrameProcessor를 사용한다. P1 High Cohesion과 P7 Keep Things Simple 관점에서 공통 분석을 먼저 재사용하고, 다음에 자원 생성과 표시 및 재생 대기와 종료 처리를 RunSession으로 모으는 것이 좋다.

**패턴:** 공통 실행 함수와 입력 및 캡처 처리기의 합성.

```python
analysis = processor.process(raw, index, elapsed)
capture.consume(analysis)
preview.show(analysis)
```

개별 사진의 쿨다운과 사건 묶음의 종료 규칙은 각 캡처 처리기에 남긴다. 이 코드는 목표 호출 흐름의 예시이며 consume과 show의 새 인터페이스는 현재 존재하지 않는다.

### 사건 정책과 저장 어댑터

[event_capture.py:226](../motion_person/event_capture.py)와 [event_capture.py:354](../motion_person/event_capture.py)는 사건 경계와 파일 저장을 함께 관리한다. P1 High Cohesion 및 P3 Depend on Abstractions 관점에서 정책은 관측 메타데이터를 받아 결정을 반환하고, 응용 계층이 압축과 저장을 실행하도록 나눌 수 있다.

**패턴:** EpisodePolicy와 작은 저장 포트.

```python
actions = episode_policy.advance(observation)
if actions.capture_frame:
    buffer.add(encoder.encode(frame))
if actions.closed_part is not None:
    bundle = bundle_store.publish(actions.closed_part, buffer)
```

이 코드는 책임 경계의 예시다. 사건별 사진 소유권과 메모리 예산 및 완료 묶음 전달 규칙을 보존하는 것이 클래스 이름이나 폴더 이동보다 우선이다.

### 탐지기의 의존성 주입 타입

[processor.py:17](../motion_person/processor.py)의 주입 타입은 여전히 ObjectDetector와 MotionDetector라는 구체 클래스다. 실행 시 테스트 대역을 주입할 수는 있지만, 정적 타입 검사에서는 해당 구체 클래스를 요구한다. P3 Depend on Abstractions 관점에서 소비자가 필요한 동작을 Protocol로 선언할 수 있다.

**패턴:** 작은 DetectorPort와 생성 함수.

```python
class DetectorPort(Protocol):
    def detect(self, frame: np.ndarray) -> tuple[ObjectDetection, ...]: ...

def analyze_objects(detector: DetectorPort, frame: np.ndarray):
    return detector.detect(frame)
```

load와 prepare_tracking 및 warmup이 필요한 초기화 단계의 계약은 별도로 명확히 한다. 기존 편의 생성자는 유지하면서 조립 함수에서 실제 탐지기들을 만드는 전환 방식이 가능하다.

## 검증

- 전체 테스트 223개 통과.
- git diff --check 통과.
- 메모리 제거 및 이전 사건 사전 사진에 따른 누락 기록 문제 재현.
- 새 필드를 제외한 기존 EventBundle 키워드 생성 인자에서 TypeError 재현.
- 실제 카메라 및 새 모델 추론과 정적 타입 검사는 실행하지 않음.
- 이번 작업은 검토이며 구현과 테스트 파일을 변경하지 않음.
