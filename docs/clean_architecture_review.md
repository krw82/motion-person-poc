# motion_person 클린 아키텍처 검토

2026년 10월 12일 현재 작업 폴더를 기준으로 한 심층 검토다. [MKToronto의 스킬](https://github.com/MKToronto/python-clean-architecture-codex)을 설치하고, 설치본의 7개 설계 원칙과 코드 품질 및 오류 처리 지침을 적용했다.

이 프로젝트는 영상 처리 CLI와 Python 라이브러리다. 웹 API용 Routers와 Operations와 Database 폴더를 그대로 만드는 대신, **사건 규칙과 실행 흐름을 입력 장치 및 파일 저장으로부터 분리하는 것**이 핵심이다. 현재는 기능별 모듈화가 이루어졌지만, 핵심 규칙을 저장 장치나 탐지 구현과 독립적으로 바꾸기 어려운 구조다.

검토 대상은 원본 Python 파일 67개와 빌드 복사본 21개다. 외부 의존성이 설치된 가상환경은 프로젝트 코드에서 제외했다. 모든 파일의 역할 분류는 마지막 목록에 있다. 구현 파일은 이번 검토에서 수정하지 않았다.

## 현재 의존성 구조

실제 호출 흐름은 다음과 같다.

- CLI → 기존 main → 직접 구성한 YOLO와 MOG2 및 EventDetector → CaptureManager → JPEG 파일
- CLI 또는 Python API → MotionEngine → FrameProcessor → EventCaptureEngine → JPEG와 JSON 파일
- 웹캠 미리보기 CLI → 별도 실행 루프 → FrameProcessor → OpenCV 화면
- 공통 계약 → Config와 탐지 및 저장과 화면 모듈
- photo_selection → event_capture의 비공개 _Sample 타입을 참조하지만, 현재 event_capture에서는 호출하지 않음

사건 종료와 사진 수집 정책이 EventCaptureEngine 내부의 JPEG 인코딩 및 폴더 공개에 직접 연결된다. FrameProcessor는 자신이 사용할 탐지기들을 직접 생성한다. 기존 main은 FrameProcessor를 사용하지 않고 분석 순서를 다시 구현한다. 규칙 변경이나 오류 처리 변경이 한 곳에서 끝나기 어렵다.

목표 의존성은 다음처럼 나눌 수 있다.

- CLI와 공개 MotionEngine Facade → RunSession
- RunSession → FrameAnalyzer와 EpisodePolicy 및 저장 포트
- FrameAnalyzer → 탐지 및 전경 분석 포트
- JPEG 파일 저장과 YOLO 및 OpenCV 입력 어댑터 → 응용 계층의 포트를 구현
- 구성 함수 → 구체 어댑터를 만들어 RunSession에 전달

여기서 포트는 필요한 메서드만 선언하는 작은 Protocol이다. 도메인 규칙은 사건 시작과 종료 및 분할을 결정하고, 응용 계층은 그 결정을 실행한다. 어댑터는 실제 카메라와 파일 시스템 및 외부 라이브러리를 다룬다.

## 유지할 수 있는 부분

- [contracts.py:160](../motion_person/contracts.py)은 Box와 FramePacket 및 FrameDecision을 frozen dataclass로 정의한다. 데이터 전달 계약을 분명히 한 기반은 유지할 가치가 있다. NumPy 배열의 변경 가능성까지 frozen이 막는 것은 아니므로, 기존 원본 픽셀 보존 테스트도 계속 필요하다.
- [event_detector.py:38](../motion_person/event_detector.py)은 마스크 증거 계산과 [event_detector.py:66](../motion_person/event_detector.py)의 조건 판정을 함수로 분리한다. 판정은 저장이나 화면 표시를 하지 않는다.
- [frame_processor.py:27](../motion_person/frame_processor.py)의 면적 보정과 좌표 변환은 계산 규칙으로 독립되어 있다. 영상 전처리에서 NumPy와 OpenCV를 사용하는 것 자체를 제거할 필요는 없다.
- [tracking_adapter.py:13](../motion_person/tracking_adapter.py)은 Ultralytics의 종류별 추적 연결을 어댑터로 분리한다. 실제 추적기를 쓰는 테스트는 외부 라이브러리 호환성 검증에 유용하다.
- [__init__.py:21](../motion_person/__init__.py)의 지연 import는 의존성이 없을 때 도움말과 설치 진단을 사용할 수 있도록 한다.
- 루트의 동일 이름 파일들은 대부분 7줄 또는 10줄의 호환 진입점이다. 이 파일들을 중복된 분석 구현으로 계산해서 제거하면 기존 사용법을 깨뜨릴 수 있다.
- 예외를 사용자 정의 타입으로 나누고, 원본 사진과 표시용 프레임을 구분하며, 저장 성공 뒤 카운터를 갱신하는 계약은 잘 정리되어 있다.

## 발견 사항과 개선 예시

아래 코드는 경계를 설명하기 위한 제안이다. 새로 제안한 타입과 모듈은 현재 구현에 존재하지 않는다. Python 3.11에서 사용할 수 있는 구문을 사용했다.

### 우선 해결할 문제 현재 구현과 호출 계약이 어긋남

**위치:** [engine.py:230](../motion_person/engine.py), [event_capture.py:186](../motion_person/event_capture.py), [photo_selection.py:26](../motion_person/photo_selection.py), [tests/test_event_capture.py:230](../tests/test_event_capture.py).

**원칙:** P2 Low Coupling, P3 Depend on Abstractions. **패턴:** Protocol로 호출 계약을 명시하고, 한 변경의 구현과 호출부 및 회귀 테스트를 함께 검증.

현재 engine은 오류 처리 중 drain_completed()를 호출하지만 EventCaptureEngine에는 해당 메서드가 없다. 새 선별 모듈이 기대하는 moving_objects와 motion_starts도 현재 _Sample에 없다. 사건 캡처 구현은 기존 _select()를 사용한다. 이전 수정 후 223개 통과했던 시점과 현재 파일 상태가 다르다.

현재 전체 테스트는 **214개 통과와 9개 실패**다. 실패한 9개는 완료 묶음 회수 및 새 객체 증거 보존에 관련된다. 구현이 이전 상태가 된 원인은 이번 검토만으로 확정할 수 없다.

먼저 구현과 호출부를 같은 계약으로 맞춘 뒤 구조 변경을 진행해야 한다. 공개 반환값과 오류 후 회수 규칙을 테스트하고, 포트에 타입을 붙이면 메서드 누락을 정적 검사로도 확인할 수 있다.

```python
from typing import Protocol

class EventCapturePort(Protocol):
    def process(self, packet: FramePacket,
                decision: FrameDecision) -> tuple[EventBundle, ...]: ...
    def flush(self, reason: str) -> tuple[EventBundle, ...]: ...
    def drain_completed(self) -> tuple[EventBundle, ...]: ...

def recover_published(capture: EventCapturePort) -> tuple[EventBundle, ...]:
    return capture.drain_completed()
```

완료된 묶음의 전달 대기 상태는 미완성 사진 버퍼와 분리한다. 다음 구간이 실패해도 이전에 공개한 묶음을 회수할 수 있어야 한다. 다만 실패한 외부 콜백을 무조건 재시도하면 중복 처리가 생길 수 있으므로, 프로세스 내부 전달 횟수와 외부 시스템의 처리 보장은 구분해야 한다.

### 구조상 높은 우선순위 사건 규칙과 파일 저장이 같은 클래스에 있음

**위치:** [event_capture.py:201](../motion_person/event_capture.py), [event_capture.py:280](../motion_person/event_capture.py), [event_capture.py:348](../motion_person/event_capture.py), [capture_manager.py:171](../motion_person/capture_manager.py).

**원칙:** P1 High Cohesion, P2 Low Coupling, P3 Depend on Abstractions. **패턴:** 사건 정책과 저장 어댑터의 합성, 파일 저장 포트.

EventCaptureEngine은 사건 종료 시각과 분할을 계산하고, 후보를 보관하고, JPEG를 인코딩하고, 대표 사진을 고르고, JSON과 파일을 쓰고, 임시 폴더를 공개한다. CaptureManager도 쿨다운과 JPEG 저장을 함께 담당한다. 저장의 실패가 사건 상태 처리에 영향을 주는 경계가 복잡하다.

사건 정책은 관측 메타데이터만 받아 수집 및 종료 결정을 반환하도록 한다. JPEG 인코딩과 임시 파일 및 폴더 공개는 저장 어댑터가 담당한다. 압축 버퍼는 응용 계층에서 예산을 관리하고, 완료된 묶음을 응용 계층에서 전달한다.

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class EpisodeActions:
    capture_frame: bool
    close_reason: str | None

class EpisodePolicy(Protocol):
    def advance(self, evidence: FrameEvidence) -> EpisodeActions: ...

class BundleStore(Protocol):
    def publish(self, plan: BundlePlan,
                photos: tuple[EncodedPhoto, ...]) -> EventBundle: ...
```

이렇게 나누면 사건 경계는 카메라나 디스크 없이 테스트하고, 저장 어댑터는 파일 공개와 실패 복구를 따로 테스트할 수 있다. 처음부터 사건 상태마다 새 클래스를 만들기보다는, 경계 계산과 상태 갱신을 몇 개의 명확한 함수로 정리하는 편이 이 규모에 맞다. 범용 CRUD 저장소를 만들 필요도 없다.

### 중요 분석 및 실행 흐름이 여러 경로에 중복됨

**위치:** [main.py:393](../motion_person/main.py), [main.py:442](../motion_person/main.py), [engine.py:111](../motion_person/engine.py), [webcam_preview.py:61](../motion_person/webcam_preview.py).

**원칙:** P1 High Cohesion, P5 Separate Creation from Use, P7 Keep Things Simple. **패턴:** 공통 실행 함수와 작은 Protocol, 캡처 처리기를 Callable로 전달.

기존 main의 반복문은 174줄이고 main 함수는 232줄이다. MotionEngine.start는 164줄이며 별도의 자원 생성과 실행 및 정리 흐름을 가진다. 웹캠 미리보기도 자체 반복문이 있다. 기존 main의 YOLO와 MOG2 및 판정 순서는 FrameProcessor와 중복된다.

개별 사진 저장과 사건 묶음 저장의 규칙은 서로 다르므로 그대로 구분한다. 공통화할 부분은 프레임 읽기와 분석 및 표시와 재생 대기와 종료 처리다. 입력별 타임라인과 웹캠의 최신 프레임 정책은 입력 어댑터에 남긴다.

```python
from typing import Callable, Protocol

class FrameSource(Protocol):
    def read(self) -> InputFrame | None: ...

class Analyzer(Protocol):
    def analyze(self, frame: InputFrame) -> FrameAnalysis: ...

def analyze_frames(source: FrameSource, analyzer: Analyzer,
                   consume: Callable[[FrameAnalysis], None]) -> None:
    while (frame := source.read()) is not None:
        consume(analyzer.analyze(frame))
```

이 예시는 공통 분석 흐름만 보여준다. 실제 RunSession에는 중단과 입력 완료 확인 및 표시와 재생 속도 정책을 넣고, 현재의 각 종료 코드와 로그 및 성능 지표를 보존하는 테스트가 필요하다. 거대한 모드 플래그 함수나 공통 상속 기반 클래스를 만드는 방향은 피하는 편이 좋다.

### 중요 객체 생성과 분석 사용이 묶여 있어 테스트가 내부 교체에 의존함

**위치:** [processor.py:16](../motion_person/processor.py), [engine.py:147](../motion_person/engine.py), [tests/test_engine.py:20](../tests/test_engine.py), [tests/test_object_tracking.py:178](../tests/test_object_tracking.py).

**원칙:** P2 Low Coupling, P3 Depend on Abstractions, P5 Separate Creation from Use. **패턴:** 생성자 주입과 composition root, 외부 탐지 구현을 Adapter로 격리.

FrameProcessor가 ObjectDetector와 MotionDetector 및 EventDetector를 직접 만든다. MotionEngine도 구체 입력과 저장 및 창 클래스를 start 내부에서 선택한다. 테스트는 모듈의 클래스 이름을 monkeypatch하거나 _model과 _loaded를 직접 바꿔야 한다.

교체가 필요한 탐지와 입력 및 저장 경계에만 작은 Protocol을 두고, 조립 함수에서 실제 구현을 만든다. 순수 판정 함수를 모두 인터페이스로 바꿀 필요는 없다.

```python
class FrameAnalyzer:
    def __init__(self, detector: DetectorPort,
                 foreground: ForegroundPort,
                 rules: EventDetector) -> None:
        self.detector = detector
        self.foreground = foreground
        self.rules = rules

# 조립 함수에서 한 번 구성
analyzer = FrameAnalyzer(
    detector=YoloDetector(analysis_config),
    foreground=Mog2Foreground(analysis_config),
    rules=EventDetector(analysis_config),
)
```

이후 단위 테스트는 가짜 탐지기를 인자로 전달한다. 실제 Ultralytics 추적기 검증은 별도의 통합 테스트로 계속 유지한다. DI 프레임워크를 추가하지 않아도 생성자 인자와 구성 함수로 충분하다.

### 중요 설정 타입에 CLI와 실행 및 파일 시스템 책임이 섞임

**위치:** [config.py:19](../motion_person/config.py), [config.py:210](../motion_person/config.py), [config.py:347](../motion_person/config.py).

**원칙:** P1 High Cohesion, P2 Low Coupling, P6 Start with the Data. **패턴:** 목적별 frozen dataclass와 입력 어댑터.

Config에는 42개 필드가 있고 탐지 및 추적과 마스크와 입력 경로 및 저장과 표시 및 실행 속도 설정이 함께 있다. 같은 모듈이 argparse 파싱과 값 검증 및 파일 존재 확인과 출력 폴더 쓰기 검사도 한다. 특히 validate_config가 출력 폴더를 만들고 .write_probe 파일을 쓴다.

분석 설정과 캡처 설정 및 실행 옵션 정도로 나누고, 값 검증과 자원 준비를 분리한다. 이미 존재하는 EventConfig는 유지할 수 있다. 기존 Config는 전환 과정에서 호환 타입으로 두고 새 타입으로 변환하면 된다.

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class AnalysisConfig:
    confidence: float = 0.4
    tracking: bool = True

@dataclass(frozen=True)
class RuntimeOptions:
    display: bool = True
    pace: Literal["realtime", "fast"] = "realtime"

def validate_analysis(config: AnalysisConfig) -> None:
    if not 0 <= config.confidence <= 1:
        raise ConfigError("confidence must be within [0, 1]")
```

출력 폴더와 쓰기 가능 여부는 실행 준비 함수에서 확인한다. 값만 검증하는 함수가 파일 시스템 상태를 바꾸지 않게 하는 것이 목적이다.

### 중요 프레임 분석 결과가 위치 기반 튜플과 표시용 상태로 전달됨

**위치:** [processor.py:33](../motion_person/processor.py), [processor.py:43](../motion_person/processor.py), [engine.py:192](../motion_person/engine.py).

**원칙:** P1 High Cohesion, P6 Start with the Data. **패턴:** 명시적인 결과 dataclass.

FrameProcessor는 packet과 objects와 motion과 decision 및 DISABLED CaptureResult를 순서대로 반환한다. 분석기가 저장을 하지 않았다는 표시용 값을 만들고, 모든 호출자는 다섯 값의 순서를 알아야 한다. 프레임 결과의 타입 힌트도 없어 변경을 확인하기 어렵다.

분석 결과는 이름 있는 DTO로 반환하고, 저장 상태와 미리보기 문구는 캡처 처리기 또는 표현 계층에서 만들도록 한다.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class FrameAnalysis:
    packet: FramePacket
    objects: tuple[ObjectDetection, ...]
    motion: MotionResult
    decision: FrameDecision

analysis = analyzer.analyze(frame)
capture.consume(analysis)
preview.show(analysis)
```

의미 있는 이름을 통해 새로운 필드를 추가하더라도 모든 튜플 분해 코드를 동시에 바꾸는 부담이 줄어든다.

### 제안 자원 수명과 종료 요약 처리를 실행 본문에서 분리

**위치:** [engine.py:233](../motion_person/engine.py), [main.py:738](../motion_person/main.py), [webcam_preview.py:116](../motion_person/webcam_preview.py), [run_logger.py:211](../motion_person/run_logger.py).

**원칙:** P1 High Cohesion, P5 Separate Creation from Use. **패턴:** Context Manager와 ExitStack, 실행 결과 DTO.

현재 종료 처리는 입력과 창 및 로거 각각에 대한 try와 close 호출을 여러 실행 경로에 구현한다. 이미 원래 오류를 보존하고 나머지 자원도 정리하려는 계약은 있으므로, try/finally를 일괄 교체하기보다 그 계약을 자원 어댑터에 옮기는 것이 좋다.

```python
from contextlib import ExitStack

def run_session(options: RuntimeOptions) -> RunSummary:
    with ExitStack() as stack:
        source = stack.enter_context(open_source(options))
        preview = stack.enter_context(open_preview(options))
        journal = stack.enter_context(open_journal(options))
        return execute_session(source, preview, journal)
```

새 context manager는 정상 종료와 초기화 실패 및 처리 오류와 close 오류에서 기존 로그와 종료 코드 및 최초 예외 보존 규칙을 지켜야 한다. 정리 오류가 최초 처리 오류를 가리는 문제는 별도 테스트로 검증한다.

### 제안 사진 선별 규칙이 비공개 구현 타입을 참조함

**위치:** [photo_selection.py:7](../motion_person/photo_selection.py), [event_capture.py:121](../motion_person/event_capture.py).

**원칙:** P2 Low Coupling, P3 Depend on Abstractions. **패턴:** 소비자가 선언하는 작은 Protocol.

photo_selection은 TYPE_CHECKING 아래에서 event_capture._Sample을 가져온다. 런타임 순환 import라고 볼 수는 없지만, 정책 모듈이 저장 엔진의 비공개 타입에 의존한다. 현재는 요구 필드 자체도 서로 맞지 않는다.

파일 바이트를 선별 규칙에 노출할 필요는 없다. 선별에 필요한 메타데이터만 정의하고 선택 결과는 프레임 번호로 반환할 수 있다.

```python
from typing import Protocol

class PhotoCandidate(Protocol):
    frame_index: int
    timestamp: float
    matched: bool

    @property
    def moving_objects(self) -> frozenset[tuple[str, int | None]]: ...
    @property
    def motion_starts(self) -> frozenset[tuple[str, int | None]]: ...
    @property
    def score(self) -> tuple[int, float]: ...
```

구체 샘플은 이 형태를 만족하면 된다. 작은 공개 dataclass를 공유 모델 모듈에 두는 방식도 가능하다. 실제 선택 호출 연결은 현재 구현 정합성을 맞추는 작업과 함께 진행해야 한다.

## 설계 원칙 평가

| 원칙 | 현재 판단 | 근거 |
| --- | --- | --- |
| P1 High Cohesion | 개선 필요 | 사건 규칙과 저장 혼합 및 큰 실행 함수 |
| P2 Low Coupling | 개선 필요 | Config 전체 공유와 구체 클래스 및 비공개 샘플 타입 의존 |
| P3 Depend on Abstractions | 개선 필요 | 입력과 탐지 및 저장의 교체 경계에 Protocol 부재 |
| P4 Composition over Inheritance | 대체로 양호 | 깊은 실행 클래스 상속이나 mixin 구조가 없고 객체 합성 사용 |
| P5 Separate Creation from Use | 개선 필요 | 분석 및 실행 내부에서 구체 자원 생성 |
| P6 Start with the Data | 일부 양호 | frozen DTO는 좋지만 큰 Config와 위치 기반 결과 및 dict 상태 존재 |
| P7 Keep Things Simple | 일부 양호 | 실제 문제는 과도한 프레임워크보다 중복 실행 흐름과 혼합 책임 |

## 코드 품질 검토

런타임 패키지의 중첩 함수와 메서드를 포함한 함수 정의는 189개이며, 반환 타입 힌트가 없는 정의는 53개다. except Exception 처리기는 28개다. 이는 개선 후보를 찾는 지표이며, 각각을 버그 수로 해석하면 안 된다. 외부 라이브러리 예외를 명시적인 도메인 오류로 감싸거나 정리 후 재발생시키는 처리는 필요한 경우가 있다.

| 규칙 | 검토 결과 |
| --- | --- |
| 1 타입에 맞는 값 표현 | 상태 Literal은 있으나 종료 사유와 모드 문자열은 여러 곳에서 반복됨 |
| 2 명확한 이름 | 긴 실행 함수의 data와 info 및 result와 축약 변수는 DTO 도입과 함께 개선 |
| 3 동작을 바꾸는 플래그 | capture와 display 같은 사용자 옵션은 유효함 분기된 내부 실행 경로를 줄이는 것이 우선 |
| 4 깊은 중첩 | start와 main 및 사건 처리에 책임 추출 후보가 있음 |
| 5 Tell Dont Ask | 실행 계층이 상태와 카운터를 많이 읽음 요약 DTO로 정리 가능 |
| 6 병렬 데이터 구조 | 탐지기의 identity와 클래스 및 관측 횟수 dict들은 같은 ID의 상태를 공유함 필요 시 TrackState로 묶기 |
| 7 지나치게 많은 책임 | EventCaptureEngine과 실행 진입점이 주요 대상 |
| 8 다른 객체 데이터에 대한 과도한 접근 | 메타데이터 변환과 표시용 변환은 정당한 경우가 많음 일괄 이동 불필요 |
| 9 도메인 값 표현 | Box와 EventConfig 등 값 객체 기반이 있음 모든 숫자를 새 클래스로 감쌀 필요 없음 |
| 10 사용되지 않는 코드 | photo_selection은 현재 미연결이며 webcam_preview의 옛 분석 import가 남아 있음 |
| 11 의미 있는 상수 | 임계값은 Config로 정리됨 상태 및 종료 사유의 일관성 개선 가능 |
| 12 광범위한 예외 포착 | 어댑터 변환 및 최초 오류 보존 처리는 유지하고 불필요한 내부 포착은 줄이기 |
| 13 자원 수명 | 파일은 with를 사용하나 카메라와 GUI 및 로거는 수동 close 중심 |
| 14 변경 가능한 기본 인자 | list와 dict 같은 기본 인자 문제는 확인되지 않음 |
| 15 wildcard import | 확인되지 않음 |
| 16 범용 패키지 이름 | utils 같은 포괄 패키지 없이 기능별 이름 사용 |
| 17 구조와 디렉터리 | 평면 기능별 구조로 책임 경계가 불명확함 경계를 나눈 후 파일 배치 |
| 18 isinstance 기반 동작 선택 | 대부분 입력 검증이나 라이브러리 변환 용도임 일괄 제거 불필요 |
| 19 과도하게 많은 상태 | Config 42개 필드와 탐지 및 사건 엔진의 여러 상태 묶음이 개선 후보 |
| 20 비대칭 이름 | VideoSource.open은 VideoInfo 웹캠 open은 크기 튜플 반환 응용 입력 DTO로 맞추기 |
| 21 오해할 수 있는 이름 | validate_config의 출력 폴더 생성과 쓰기 검사는 이름에 드러나지 않음 |
| 22 초기화 호출 순서 | load와 prepare_tracking 및 warmup과 open과 start 순서를 조립 경계에서 보장 |

## 개선 순서와 완료 조건

1. 현재 1번 및 3번 수정의 구현과 호출 계약을 먼저 맞춘다. 전체 223개 테스트가 통과하고 오류 후 완료된 묶음이 중복 전달되지 않아야 한다.
2. FrameAnalysis DTO와 탐지기 생성자 주입을 도입한다. 원본 픽셀과 ROI 및 추적 확인과 프레임당 추론 1회 계약을 보존한다.
3. 기존 main과 API 및 미리보기의 공통 분석을 합친 뒤 실행 수명 관리를 RunSession으로 모은다. 개별 저장의 쿨다운과 사건 묶음의 종료 규칙은 각 처리기에 남긴다.
4. EpisodePolicy와 JPEG 파일 저장을 나눈다. 사건 경계 테스트는 인코더나 파일 없이 실행되고 파일 공개 테스트는 임시 폴더에서 실행되어야 한다.
5. 설정과 포트 및 어댑터의 위치를 정리한다. 패키지 외부 import와 기존 루트 진입점 및 CLI 도움말 호환성도 검증한다.

현재 규모에는 소수의 Protocol과 dataclass 및 조립 함수가 적절하다. DI 컨테이너와 이벤트 버스 및 범용 Repository와 CQRS를 추가할 근거는 없다. 콜백 한 개로 충분한 완료 전달은 그대로 사용할 수 있다. 상태 문자열의 Enum 전환은 외부 JSON 스키마와 CLI 호환성을 확인하면서 점진적으로 진행한다.

## 검증 상태

명령은 프로젝트의 기존 가상환경에서 실행했다.

```text
.venv/bin/python -m pytest -q
214 passed, 9 failed
```

실패 목록은 다음과 같다.

- test_completed_callback_survives_next_photo_error_and_preserves_original_error 두 경우
- test_published_bundle_can_be_recovered_once_after_next_part_encoding_fails
- test_brief_new_moving_object_between_ticks_survives_photo_selection 두 경우
- test_brief_motion_resume_of_same_track_is_saved_between_ticks
- test_photo_limit_reports_moving_objects_without_saved_evidence
- test_memory_pressure_keeps_brief_object_evidence_before_replaceable_context
- test_one_off_grid_photo_can_preserve_multiple_new_objects

실제 카메라나 새 모델 추론을 이번 아키텍처 검토에서 실행하지 않았다. 구조 개선이 정확도나 실제 시간 성능을 높였다는 주장은 하지 않는다. 기존에 수정 대상으로 선택하지 않은 사전 및 사후 맥락 완전성 문제는 이번 검토에서도 구현을 변경하지 않았다.

## 후속 조치 (2026년 10월 12일)

검토 후 리팩토링을 진행했다. 개선 순서 1번과 2번을 적용했고 3번 이하는 적용하지 않았다.

- 계약 불일치 해소: EventCaptureEngine에 drain_completed와 완료 묶음 전달 대기열을 추가하고, _Sample이 moving_objects와 motion_starts를 계산하며, 사건 묶음이 omitted_moving_objects를 보고한다. photo_selection의 select_photos와 eviction_candidate가 실제 선택과 제거에 연결되었다. 전체 223개 테스트가 통과한다.
- FrameAnalysis DTO와 FrameProcessor 생성자 주입을 도입했다. DISABLED 표시 값은 분석기에서 표현 계층으로 옮겼다.
- 알려진 남은 문제: MAX_DURATION 분할 후 quiet 시점을 넘는 프레임 공백이 있으면 해당 에피소드가 종료 신호 없이 끝날 수 있다(기존 동작). omitted_moving_objects는 자기 사건 트리거 이후 프레임만 대상으로 계산한다.

후속 검토가 지적한 3건의 P2 문제도 수정했다. 사건 관측 이력을 _Part의 observed_moving_objects 집합에 프레임 단위로 누적해 메모리 제거와 무관하게 누락을 기록하고, 증거 충족 판정은 max(start, trigger) 이후 사진만 인정하도록 _finish와 select_photos의 윈도를 맞췄으며, EventBundle의 omitted_moving_objects는 기존 필드 순서를 유지한 채 마지막 필드로 기본값과 함께 이동했다. 세 시나리오의 재현과 수정 검증 테스트 4개를 추가해 전체 227개 테스트가 통과한다. 다음 구조 개선(공통 실행 흐름 합성, EpisodePolicy와 저장 분리, 탐지 주입 Protocol)은 별도 라운드로 남겨둔다.

## 전체 Python 파일 분류

역할 분류는 파일을 이동하라는 지시가 아니라 현재 책임을 보여준다. 생성 복사본 21개는 원본과 비교했으며, engine은 현재 오류 회수 호출 추가 때문에 다르고 processor는 끝의 빈 줄만 다르다. 나머지 19개는 원본과 바이트가 같다. 생성 복사본을 별도의 유지보수 대상 구현으로 다루지 않는다.

| 파일 | 현재 계층 또는 역할 | 내용 |
| --- | --- | --- |
| [build/lib/motion_person/__init__.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/__init__.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/capture_manager.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/capture_manager.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/cli.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/cli.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/config.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/config.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/console_messages.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/console_messages.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/contracts.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/contracts.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/engine.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/engine.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/event_capture.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/event_capture.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/event_detector.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/event_detector.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/frame_processor.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/frame_processor.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/main.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/main.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/motion_detector.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/motion_detector.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/object_detector.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/object_detector.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/object_state.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/object_state.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/overlay_renderer.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/overlay_renderer.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/processor.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/processor.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/run_logger.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/run_logger.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/tracking_adapter.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/tracking_adapter.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/video_source.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/video_source.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/webcam_preview.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/webcam_preview.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [build/lib/motion_person/webcam_source.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/build/lib/motion_person/webcam_source.py) | 생성 복사본 | 대응 패키지 모듈의 빌드 산출물 |
| [capture_manager.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/capture_manager.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [cli.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/cli.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [config.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/config.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [conftest.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/conftest.py) | 테스트 설정 | 기존 루트 import 경로 구성 |
| [console_messages.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/console_messages.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [contracts.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/contracts.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [event_detector.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/event_detector.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [frame_processor.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/frame_processor.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [main.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/main.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [motion_detector.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_detector.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [motion_person/__init__.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/__init__.py) | 공개 API | 지연 import와 공개 이름 |
| [motion_person/capture_manager.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/capture_manager.py) | 도메인 및 저장 혼합 | 쿨다운과 JPEG 파일 저장 |
| [motion_person/cli.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/cli.py) | 입력 및 조립 | CLI 입력과 실행 모드 구성 |
| [motion_person/config.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/config.py) | 혼합 | 설정 DTO와 CLI 파싱 및 파일 시스템 준비 |
| [motion_person/console_messages.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/console_messages.py) | 표현 | 한국어 오류 메시지 |
| [motion_person/contracts.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/contracts.py) | 공유 계약 | 영상 DTO 및 오류와 CLI 종료 코드 |
| [motion_person/engine.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/engine.py) | 응용 및 인프라 혼합 | 공개 Facade와 자원 생성 및 실행 루프 |
| [motion_person/event_capture.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/event_capture.py) | 도메인 및 저장 혼합 | 사건 경계와 버퍼 및 JPEG 저장 |
| [motion_person/event_detector.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/event_detector.py) | 도메인 규칙 | 객체별 움직임 조건과 프레임 판정 |
| [motion_person/frame_processor.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/frame_processor.py) | 영상 전처리 | 리사이즈와 ROI 및 좌표 변환 |
| [motion_person/main.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/main.py) | 응용 및 인프라 혼합 | 기존 개별 캡처 실행과 생성 및 로그 |
| [motion_person/motion_detector.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/motion_detector.py) | 영상 알고리즘 어댑터 | OpenCV MOG2와 마스크 후처리 |
| [motion_person/object_detector.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/object_detector.py) | 외부 라이브러리 어댑터 | YOLO 및 추적 설정과 관측 확인 |
| [motion_person/object_state.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/object_state.py) | 도메인 상태 | 추적 객체의 관측 및 저장 이력 |
| [motion_person/overlay_renderer.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/overlay_renderer.py) | 표현 및 GUI 어댑터 | 픽셀 렌더링과 OpenCV 창 수명 |
| [motion_person/photo_selection.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/photo_selection.py) | 선별 규칙 | 사진 선별 및 제거 정책 현재 미연결 |
| [motion_person/processor.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/processor.py) | 응용 | 프레임 분석 순서와 구체 탐지기 생성 |
| [motion_person/run_logger.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/run_logger.py) | 저장 어댑터 | 파일 기반 JSON 및 JSONL 로그 |
| [motion_person/tracking_adapter.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/tracking_adapter.py) | 외부 라이브러리 어댑터 | 종류별 Ultralytics 추적기 연결 |
| [motion_person/video_source.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/video_source.py) | 입력 어댑터 | OpenCV 파일 입력과 시간 계산 |
| [motion_person/webcam_preview.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/webcam_preview.py) | 입력 및 응용 혼합 | 웹캠 CLI와 미리보기 실행 루프 |
| [motion_person/webcam_source.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/motion_person/webcam_source.py) | 입력 어댑터 | 카메라 읽기 스레드와 최신 프레임 |
| [object_detector.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/object_detector.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [object_state.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/object_state.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [overlay_renderer.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/overlay_renderer.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [run_logger.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/run_logger.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [tests/__init__.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tests/__init__.py) | 테스트 | 테스트 패키지 |
| [tests/test_capture_manager.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tests/test_capture_manager.py) | 테스트 | 계약 및 회귀 또는 실제 추적기 검증 |
| [tests/test_cli.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tests/test_cli.py) | 테스트 | 계약 및 회귀 또는 실제 추적기 검증 |
| [tests/test_config.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tests/test_config.py) | 테스트 | 계약 및 회귀 또는 실제 추적기 검증 |
| [tests/test_engine.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tests/test_engine.py) | 테스트 | 계약 및 회귀 또는 실제 추적기 검증 |
| [tests/test_event_capture.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tests/test_event_capture.py) | 테스트 | 계약 및 회귀 또는 실제 추적기 검증 |
| [tests/test_event_detector.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tests/test_event_detector.py) | 테스트 | 계약 및 회귀 또는 실제 추적기 검증 |
| [tests/test_motion_detector.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tests/test_motion_detector.py) | 테스트 | 계약 및 회귀 또는 실제 추적기 검증 |
| [tests/test_object_tracking.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tests/test_object_tracking.py) | 테스트 | 계약 및 회귀 또는 실제 추적기 검증 |
| [tests/test_overlay_modes.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tests/test_overlay_modes.py) | 테스트 | 계약 및 회귀 또는 실제 추적기 검증 |
| [tests/test_reference_math.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tests/test_reference_math.py) | 테스트 | 계약 및 회귀 또는 실제 추적기 검증 |
| [tests/test_tracking_improvements.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tests/test_tracking_improvements.py) | 테스트 | 계약 및 회귀 또는 실제 추적기 검증 |
| [tests/test_video_source.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tests/test_video_source.py) | 테스트 | 계약 및 회귀 또는 실제 추적기 검증 |
| [tests/test_webcam_preview.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tests/test_webcam_preview.py) | 테스트 | 계약 및 회귀 또는 실제 추적기 검증 |
| [tools/build_demo_videos.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tools/build_demo_videos.py) | 개발 도구 | 데이터 준비 및 실행 또는 평가 보고서 |
| [tools/build_public_test_report.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tools/build_public_test_report.py) | 개발 도구 | 데이터 준비 및 실행 또는 평가 보고서 |
| [tools/build_tracking_improvement_report.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tools/build_tracking_improvement_report.py) | 개발 도구 | 데이터 준비 및 실행 또는 평가 보고서 |
| [tools/download_public_samples.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tools/download_public_samples.py) | 개발 도구 | 데이터 준비 및 실행 또는 평가 보고서 |
| [tools/evaluate_public_samples.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tools/evaluate_public_samples.py) | 개발 도구 | 데이터 준비 및 실행 또는 평가 보고서 |
| [tools/evaluate_tracking_tests.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tools/evaluate_tracking_tests.py) | 개발 도구 | 데이터 준비 및 실행 또는 평가 보고서 |
| [tools/prepare_public_samples.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tools/prepare_public_samples.py) | 개발 도구 | 데이터 준비 및 실행 또는 평가 보고서 |
| [tools/prepare_tracking_samples.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tools/prepare_tracking_samples.py) | 개발 도구 | 데이터 준비 및 실행 또는 평가 보고서 |
| [tools/run_event_bundle_tests.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tools/run_event_bundle_tests.py) | 개발 도구 | 데이터 준비 및 실행 또는 평가 보고서 |
| [tools/run_public_samples.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tools/run_public_samples.py) | 개발 도구 | 데이터 준비 및 실행 또는 평가 보고서 |
| [tools/run_tracking_improvement_tests.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tools/run_tracking_improvement_tests.py) | 개발 도구 | 데이터 준비 및 실행 또는 평가 보고서 |
| [tools/run_tracking_tests.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tools/run_tracking_tests.py) | 개발 도구 | 데이터 준비 및 실행 또는 평가 보고서 |
| [tools/verify_event_package.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tools/verify_event_package.py) | 개발 도구 | 데이터 준비 및 실행 또는 평가 보고서 |
| [tracking_adapter.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/tracking_adapter.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [video_source.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/video_source.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [webcam_preview.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/webcam_preview.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
| [webcam_source.py](/Users/jeong-woncheol/AIO/AIO_TEAM_PROJET_1/idea/motion_person_poc/webcam_source.py) | 호환 진입점 | 같은 이름의 패키지 모듈로 위임 |
