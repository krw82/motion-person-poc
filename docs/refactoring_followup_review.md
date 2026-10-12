# 리팩터링 후속 검토 보고서

- **검토 대상**: 커밋 `70e07a4` 이후 작업 트리의 변경(`engine.py`, `event_capture.py`, `contracts.py`, `__init__.py`, `processor.py`, `webcam_preview.py`, 신규 `photo_selection.py`, 테스트 3종)
- **검토 기준일**: 2026-10-12
- **문서 성격**: 하위 검토자(엔진 / 캡처+DTO / 사진 선별 / 미리보기 / 테스트 / 무게·회귀 확인)가 독립 검증한 결과의 취합. 초판 독자 검토 지적(심각도 통일, 우선순위 근거, 기본 설정·호환성·사후 회수 등 미답변 항목)을 반영해 갱신했다. 실행으로 확인된 사실은 출처를 밝히고, 확인되지 않은 주장은 **미확인**으로 명시한다. 갱신 시 본 취합 단계에서 직접 실행한 보충 확인도 있다(§6 실행 목록).
- **전체 테스트 게이트**: `.venv/bin/python -m pytest tests -q` → **227 passed, exit 0** (워크플로우가 전달한 게이트 결과)

---

## 1. 총평 — 질문 ① "방금 한 리팩터링이 괜찮은가"

**결론: 괜찮다 — 네 그룹 모두 "실제 개선" 판정이며, 전체 테스트 227건이 통과해 회귀는 없다. 단, 이전 검토의 구조적 우려 2건(high)은 그대로이고 이번 변경 범위에서 새로 확인된 결함 2건(medium)이 있으므로 조건이 붙는다.**

**분류 기준(발견 항목 중복 집계 방지)**: 이 문서의 발견 목록은 "이번 변경 범위에서 새로 확인된 결함"(F1, F8 — medium)과 "이전 검토 지적의 지속"(P1, P2 — high)으로 나뉜다. 발견 목록의 F2는 §4 P1과 논거까지 동일한 **동일 문제의 중복 항목**이므로(§3 F2 참조) 심각도는 P1 기준 high로 통일해 읽고, "새 결함" 집계에는 넣지 않는다.

실제로 좋아진 것(모두 실행·grep·테스트로 확인):

| 변경 | 확인 수단 |
|---|---|
| 5-튜플 언패킹 `(packet, objects, motion, decision, disabled)`가 `FrameAnalysis` 속성 접근으로 전면 전환. 패키지 내 호출부 2곳(`engine.py:192`, `webcam_preview.py:92`) 모두 반영, 구 `disabled` 변수 잔재 없음 | grep 전수 확인 |
| 사진 선별 정책이 stdlib만 쓰는 독립 모듈 `photo_selection.py`(77줄)로 분리 — 인코딩·디스크 I/O와 실제 격리 | 임포트 검사: `sys.modules`에 cv2/numpy/ultralytics 미로드 |
| 오류 경로의 회복 전달 계약 신설: 이미 게시된 번들을 `drain_completed()`로 콜백과 함께 전달, 전달 실패 시 `error.add_note`로 원본 예외 보존(`engine.py:229-235`) | 신규 테스트 `test_completed_callback_survives_next_photo_error_and_preserves_original_error`가 콜백 성공/실패 두 경우로 검증 |
| `omitted_moving_objects`가 프레임 단위로 누적되어 메모리 예산 초과 제거 후에도 관측 이력 유지 — 직전 리팩터링 검토 지적의 수정 | `tests/test_event_capture.py:315-329` 고정 |
| 미리보기의 죽은 탐지기 import 4개 제거(`EventDetector`·`make_packet`·`MotionDetector`·`ObjectDetector`) — 미리보기→분석 내부 의존 단절 | `git diff HEAD` 확인 |

**권장 후속 조치 우선순위** — 정렬 기준: ① 심각도(high → medium), ② 같은 심각도 안에서는 노출 범위가 넓고 회수 불가능한 것부터. medium 2건이 뒤에 오는 이유는 발생 조건이 제한적이고(Ctrl+C 인터럽트 시 / `max_images=2` 명시 설정 시) 사후 회수 여지가 남아 있기 때문이다(각 항목 근거 참조).

1. **P1 (high) — 사진 저장-실행 이력 강제 결합 + 로그 실패 시 완료 콜백 누락**. 모든 `capture=True` 실행에 구조적으로 적용되는 결함이며, 임베딩 사용자(질문 ②)에게 가장 직접적이다. 조치 방향: 실행 이력을 선택화(모듈 기본 off)하고, 선택적 로그 실패가 완료 콜백을 막지 않게 `deliver` 순서 분리(§4 P1).
2. **P2 (high) — 사라진 추적 ID 상태의 무한 메모리 누적**. 장시간 웹캠 운영에서 무증상으로 악화(이전 검토 합성 측정: 상이한 ID 20,000개 ≈ 8.17 MiB, §4 P2). 조치 방향: 오래된 항목 만료·최대 상태 수 + 증가 카운터 기반 ID.
3. **F1 (medium) — KeyboardInterrupt 경로 번들 유실**. 발생 조건이 사용자 인터럽트로 제한적이고 파일은 디스크에 남아 수동 회수 가능(§3 F1). 조치 방향(제안): KI 핸들러에도 except Exception 블록과 동일한 drain 구조 적용.
4. **F8 (medium) — `max_images=2`에서 증거 프레임 누락**. **기본 설정에서는 발생하지 않는다** — `EventConfig.max_images` 기본값은 6(`event_capture.py:30`, 본 검토 직접 확인), MotionEngine은 `events` 미주입 시 기본 `EventConfig()`를 쓰고(`engine.py:67`) Config/CLI에는 `max_images` 필드가 없어(grep 무결과) `MotionEngine(..., events=EventConfig(max_images=2))`를 명시 준 경우에만 해당. 조치 방향(제안): §3 F8 참조.

---

## 2. 파일 그룹별 판정·좋아진 점

### 2.1 엔진 (`engine.py`) — 개선 (조건부)

**판정**: 이번 변경은 실제 개선. 단 KI 경로 회복 부재(F1)와 실행 이력 강제 결합(P1)이 그대로라 "괜찮은가"에는 조건이 붙음.

- `FrameAnalysis` 도입으로 `processor.process`의 5-튜플 언패킹 제거가 완전. 호출부는 `engine.py:192`, `webcam_preview.py:92` 두 곳뿐이고 모두 속성 접근 사용.
- 오류 경로 회복이 계약적으로 정리됨: `engine.py:229-235`는 이미 게시된 번들만 `drain_completed()`로 콜백과 함께 전달하고, 전달 실패 시 `error.add_note`로 원본 예외 보존. `finally`의 flush는 `error`가 있으면 콜백 없이 꼬리만 저장(`:240`)해 "오류 꼬리를 콜백으로 승격하는" 과잉 복구 방지.
- 실행 확인: `./.venv/bin/python -m pytest tests/test_engine.py -p no:cacheprovider -q` → **18 passed (0.52s)**.
- `finally` 블록의 flush·자원 close·summary 기록이 기존 `error`를 덮어쓰지 않도록 `if error is None` 가드 유지, summary는 오류 시에도 항상 기록(`:258-263`).

### 2.2 캡처+DTO (`event_capture.py`·`contracts.py`·`__init__.py`) — 개선

**판정**: 실제 개선. 무게 관점 증가분(본 취합 단계 실측: `git diff HEAD --numstat` → event_capture.py +54/−36줄, contracts.py +14/−0줄, __init__.py +2/−1줄, 신규 photo_selection.py 77줄 `wc -l` 확인, 신규 의존성 0). *주의: 이 그룹 검토자는 검토 전용 제약으로 전체 스위트를 직접 실행하지 않음(not run) — 결론은 정적 검토와 임포트 검사에 근거(전체 게이트는 §6 참조).*

- `FrameAnalysis`(`contracts.py:328-339`)가 저장 상태·표시 문구 없는 순수 분석 DTO 계약을 지킴 — 두 표현 계층 소비자가 자기 쪽에서 `CaptureResult("DISABLED", ...)` 생성(`engine.py:213` 부근, `webcam_preview.py:103`).
- "저장됐는데 완료 전달 누락" 회복 메커니즘: `_finish`가 발행 묶음을 `_pending`에 쌓고(`event_capture.py:401`), `process()`/`flush()`는 반환 직전 `_mark_delivered`로 정리(`:208`, `:310`), 오류로 전달이 끊긴 묶음은 `drain_completed()`로 회수(`:219-222`, `engine.py:233`이 소비).
- `omitted_moving_objects`가 `_Part.observed_moving_objects`에 프레임 단위 누적(`:295-296`) — 메모리 예산 초과 제거 후에도 관측 이력 유지. `docs/refactoring_review.md:41` 지적의 수정.
- `EventBundle` 새 필드가 기본값 `()`을 가진 마지막 필드로 추가돼 위치·키워드 생성자 호환 유지(`event_capture.py:103`, `tests/test_event_capture.py:349-365`). `docs/refactoring_review.md:90` 지적의 수정.
- `__init__.py` 지연 export(`__getattr__`) 실행 검증 완료(형상·필드·AttributeError 가드 정상, 파일 생성 없음).

### 2.3 사진 선별 (`photo_selection.py`·`processor.py`) — 개선 (보완 1건 + 호환성 주의)

**판정**: 담당 그룹 범위에서 실제 개선. 단 `max_images=2`에서 증거 프레임 누락 결함이 실행으로 확인됨(F8) — 그 점만 보완하면 됨.

- 선별 정책(`select_photos`·`eviction_candidate`)이 인코딩·디스크 I/O에서 완전히 분리된 순수 함수로 추출. 실행 확인: 임포트 후 `sys.modules`에 cv2/numpy/ultralytics 미로드.
- `processor.process`가 표현용 더미 `CaptureResult("DISABLED", ...)` 생성 책임을 벗어남(`processor.py:47`). 마커는 표현 계층(`webcam_preview.py:103`, `engine.py:213`)이 자체 생성하며 `FrameAnalysis` docstring(`contracts.py:328-334`)이 경계 명시.
- `FrameProcessor`가 `detector/motion/events`를 키워드 전용 인자로 주입받음(`processor.py:16-19`) — 테스트가 `SimpleNamespace` 페이크 주입으로 모델 없이 검증(`tests/test_webcam_preview.py:117`).
- 담당 그룹 관련 테스트 73개(55+18) 통과.

**외부 소비자 호환성(질문 ② 관련, 주의)**: `process()`의 반환 타입 전환(5-튜플 → `FrameAnalysis`)은 **호환 장치 없는 breaking change**다. 본 취합 단계에서 `contracts.py:328-339`를 직접 읽어 확인한 결과 `FrameAnalysis`는 4개 필드만 가진 frozen dataclass로 `__iter__`·`__getitem__` 등 튜플 호환 프로토콜이 없고, `processor.py:36-47`은 `FrameAnalysis`만 반환하며 별칭 반환·변환 헬퍼도 없다. 즉 HEAD에서 `packet, objects, motion, decision, disabled = processor.process(...)` 형태로 소비하던 외부 코드가 있었다면 이 변경으로 깨진다. `EventBundle` 생성자는 호환 회귀 테스트로 보호한 것(§2.2)과 대비되는 차이다. 완화 요소: 이 변경은 커밋 전 작업 트리 상태이고(git status에서 해당 파일들 modified) 패키지 버전은 0.x(pyproject.toml `version = "0.2.0"`)이라, 배포 전 단계에서 호환 장치 없이 전환하는 것 자체는 정상적 선택이다. 다만 0.2.0 배포물에 5-튜플 API가 포함돼 있었는지는 저장소에서 확인할 수 없어(미확인), 이미 배포된 버전을 소비 중인 사용자가 있다면 릴리스 노트에 반환 타입 파괴를 명시해야 한다.

### 2.4 미리보기 (`webcam_preview.py`) — 개선

**판정**: 명확한 개선. 132줄짜리 단일 책임 CLI 진입점으로 과설계 없음. *이 그룹 검토자는 스위트를 실행하지 않음(제약) — 테스트 코드 diff 정독으로 신규 API와 정합함을 확인.*

- HEAD에서 임포트만 되고 쓰이지 않던 `EventDetector`·`make_packet`·`MotionDetector`·`ObjectDetector` 4개 제거(`webcam_preview.py:18-25`) — 미리보기가 `FrameProcessor` 한 곳에만 의존.
- 위치 기반 5-튜플 언팩 → `analysis.objects` 등 명명 접근. 소비부가 한 곳뿐이라 리팩터링 안전.
- 미리보기 선택성 3단 구조 유지 확인: `__init__.py` `__getattr__` 지연 임포트, `cli.py:201` 함수 내 지연 임포트, `--no-display` 진단 모드. 역방향 의존(엔진→미리보기) 없음(grep 확인). `webcam_preview`는 engine/event_capture/run_logger/photo_selection을 임포트하지 않음(`webcam_preview.py:18-25`).
- 특별 확인 사항인 프레임 지연·드롭 처리는 이번 변경에 미포함(`git diff HEAD -- motion_person/webcam_source.py` → 0줄). 드롭 내성은 `FrameProcessor`가 인덱스 스킵을 허용하는 기존 계약(`processor.py:37-40`, 비감소·시간 역행만 거부)으로 유지되고 `frame_age_ms` 지표(`webcam_preview.py:99`)도 그대로 — 나빠진 것 없음.

### 2.5 테스트 3종 — 리팩터링을 실제로 검증

**판정**: 이번 테스트 변경은 리팩터링을 실제로 검증함. *이 그룹 검토자는 절대 규칙(읽기 전용)으로 pytest를 직접 실행하지 못함 — 정적 검증(코드 경로 추적) 기반. 전체 스위트 통과는 별도 게이트로 확인(§6).*

- `tests/test_engine.py:134-168` 신규 통합 테스트가 drain/deliver 회복·`add_note` 보존·리소스 해제·EVENT_COMPLETED 로그 1회를 파라미터화(`callback_fails`)로 검증 — 직전 검토 우선 보완 ①이 정말 고쳐졌음을 증명.
- `tests/test_event_capture.py:225-235`가 회복 번들이 정확히 한 번만 전달됨(재호출·`flush('ERROR')` 모두 빈 튜플)을 단위 검증 — 이중 전송 방지.
- 사진 선별 신규 의미론을 양방향 검증: 짧은 새 객체는 바이트 정합(`:251-253`), 연속 움직임 과샘플링 부재(`:264-269`).
- `EventBundle` 키워드·위치 생성자 호환 회귀 테스트(`:349-365`) — "다른 프로젝트에 모듈로 붙이기" 질문에 대한 안전장치(*단, `process()` 반환 타입은 대상 아님 — §2.3 참조*).
- 삭제된 테스트 함수 없음. 기존 오류 경로 단언 유지.

---

## 3. 발견 사항

상태 표기: **확인됨** = 하위 검토자가 코드 정독·grep·인라인 실행·전용 테스트로 검증(전달받은 결과), 또는 본 취합 단계의 직접 확인(§6). **미확인** = 검증 불가 명시. 위치는 `파일:줄` 원문 유지.

### 개요

| # | 위치 | 심각도 | 상태 | 요지 |
|---|---|---|---|---|
| F1 | `motion_person/engine.py:225` | medium | 확인됨 | KI 경로에서 게시된 번들이 콜백·로그 없이 유실(파일은 디스크에 잔존) |
| F2 | `motion_person/engine.py:162` | **high — §4 P1과 동일 문제** | 확인됨 (부차 서술 1건 미확인) | 실행 이력 강제 결합 + 로그 실패 시 콜백 누락 경로 |
| F3 | `motion_person/engine.py:241` | low | 확인됨 | finally 이차 예외가 무기록으로 버려짐 |
| F4 | `motion_person/engine.py:213` | low | 확인됨 | DISABLED 더미 CaptureResult 리터럴 |
| F5 | `motion_person/photo_selection.py:40-42` · `event_capture.py:362-365` | low | 확인됨 | evidence_start 규칙 3곳 중복 |
| F6 | `motion_person/event_capture.py:211-217` 대비 `231-236` | low | 확인됨 | discard() 후 재시작 피드 거부 — 주석과 계약 불일치 |
| F7 | `motion_person/event_capture.py:238-239` 등 | low | 확인됨 | 트래커 ID 스위치 오탐: 간격 외 샘플 + 구 ID omitted |
| F8 | `motion_person/photo_selection.py:36` | medium | 확인됨 | max_images=2에서 증거 프레임 누락 (기본 설정 미해당) |
| F9 | `motion_person/photo_selection.py:33` | low | 확인됨 | 빈 frames에 IndexError(-O에서 가드 소멸) |
| F10 | `motion_person/photo_selection.py:7` | low | 확인됨 | ObjectKey 타입 중복 정의 |
| F11 | `motion_person/processor.py:16-19` | low | 확인됨 | DI 애너테이션이 구체 클래스인데 실제는 덕 타이핑 허용 |
| F12 | `motion_person/webcam_preview.py:18` | low | 확인됨 | 죽은 import 2개 잔존 |
| F13 | `motion_person/webcam_preview.py:103` | low | 확인됨 | DISABLED 마커 생성 리터럴 4곳 중복 |
| F14 | `tests/test_event_capture.py:368` | low | 확인됨 | photo_selection 직접 테스트 얇음 |
| F15 | `tests/test_webcam_preview.py:122` | low | 확인됨 | 프로덕션 마커 생성 경로 무검증 |

> **F2와 P1의 관계**: F2는 하위 발견 검토가 medium으로 분류해 보고한 항목이지만, 논거(`engine.py:129`·`:162-166`, `deliver`의 `logger.write` 선행, `run_logger.py:188-196`)가 §4 P1("사진 저장-실행 이력 강제 결합", 이전 검토 추적 high)과 사실상 동일한 문제다. 본 문서는 이 둘을 하나의 문제로 보고 심각도를 high로 통일하며, F2를 "새 결함" 집계에서 제외한다(§1 분류 기준).

### 상세

**F1 (medium, 확인됨) — `motion_person/engine.py:225`: KeyboardInterrupt 경로 번들 유실.**
KI 발생 시 게시 직후 인터럽트된 번들이 콜백과 EVENT_COMPLETED 로그 없이 유실됨. 실행 재현(임시 디렉터리에서 `_finish` 원본 실행 후 KI 주입): `flush('USER_INTERRUPT')` 0건, `_pending`에 1건 잔존, `manifest_path.is_file()` True 및 이미지 전부 존재. 원인: `engine.py:225-226`의 KI 핸들러는 reason만 설정하고 `drain_completed()` 호출처는 `engine.py:233`(except Exception 블록)이 유일(grep 확인), `finally`(`:238-240`)는 flush만 호출. `event_capture.py:203-207`의 `except Exception`이 KI(BaseException)를 잡지 않아 `_mark_delivered`(`:208`)가 건너뛰어짐.
**"유실"의 실질 의미와 사후 회수**: 재현 결과 번들의 manifest(event.json)와 이미지는 디스크에 온전히 존재한다 — 원자 게시(staging→rename, `event_capture.py:393`)까지 완료된 상태다. 따라서 이것은 콜백·로그 기록 관점의 유실이며, 사용자가 출력 디렉터리(`output_dir/run_id` 아래 사건 폴더들)를 직접 조사하면 사진 자체는 사후 회수 가능하다. 다만 그런 미전달 번들을 찾아주는 스캔·복구 기능은 패키지에 확인되지 않았고(이번 검토에서 확인된 회수 경로는 `drain_completed()`뿐, KI 경로에선 미호출), EVENT_COMPLETED 로그·콜백 기록은 영구적으로 남지 않아 방치하면 조용히 잊히는 상태가 된다.
**수정 방향(제안, 미검증)**: KI 핸들러(`engine.py:225-226`)에도 except Exception 블록(`:229-235`)과 동일한 `deliver(captures.drain_completed())` 구조(전달 실패 시 `add_note` 보존 포함)를 적용한다. `event_capture.process()`의 예외 포착 계층을 BaseException으로 넓히는 대안보다 엔진 핸들러 한 곳의 수정으로 국소적으로 끝난다.
한계: `MotionEngine.start()` 전체를 실제 Ctrl+C로 구동하지는 않음 — 엔진 라우팅은 코드 정독, 캡처 메커니즘·flush 거동은 인라인 재현으로 확인.

**F2 (high로 통일, 확인됨 / 부차 서술 미확인) — `motion_person/engine.py:162`: 실행 이력 강제 결합. §4 P1과 동일 문제.**
`engine.py:129`(`run_dir = ... if self.capture else None`)과 `:162-166`(`if run_dir is not None: logger = RunLogger(...)`) — 이력만 끄는 옵션이 없음. `Config.log_dir`(`config.py:26`)은 위치 변경용이며 engine 경로는 이조차 안 씀. `engine.py:115-116`의 "on_event requires capture=True" 때문에 콜백 사용 자체가 JSONL 로깅과 구조적으로 결합. 또한 `deliver`(`:138-143`)가 `logger.write`를 `on_event`보다 먼저 실행하므로, 정상 run의 finally flush 전달(`:240`)에서 `LogWriteError`가 나면(`run_logger.py:188-196`이 쓰기 실패 시 예외 발생, 인라인 검증 확인) 꼬리 사진은 저장됐지만 콜백은 미호출인 채 run이 에러 종료(`:241-243`, `:275-276`). **미확인**: "임베딩 사용자의 '실행 이력 저장 제외' 요구와 직결"이라는 부차 서술은 README.md와 docs/*.md 전체에서 '임베딩' 관련 요구 grep 무결과 — 저장소에서 확인 불가한 외부 요구사항 문맥으로, 이번 검증 범위 밖임. 코드 수준 주장(강제 결합, 콜백 차단 경로)은 모두 성립.

**F3 (low, 확인됨) — `motion_person/engine.py:241`: finally 이차 예외 무기록.**
`engine.py:239-250`의 두 except 블록이 `if error is None: error, reason = exc, ...` 가드뿐이라 본체 try에서 이미 error가 있으면 flush/close 이차 예외가 어떤 변수·로그에도 남지 않고 버려짐. 동일 가드 패턴이 `:258-272` 로거 경로에도 세 번 더 존재. `add_note`는 `:235`에만 있음. 원본 예외는 `:275-276` `raise error`로 보존·재발생하므로 숨김은 아니나, 오류 시 꼬리 저장 실패 여부를 사후에 알 방법이 없음.

**F4 (low, 확인됨) — `motion_person/engine.py:213`: DISABLED 더미 CaptureResult.**
`CaptureResult("DISABLED", None, elapsed, 0.0, None)` 리터럴이 엔진(`:213`)과 미리보기(`webcam_preview.py:103`)에 존재하나, `overlay_renderer.py:204`(event_capture)와 `:215`(preview) 브랜치는 capture 인자를 읽기 전에 return하며 `engine.py:199-206`의 metrics 때문에 세 번째 브랜치(`:244-245`)에 도달하지 않음. 실제 capture 소비자는 `main.py:528`(CaptureManager 결과)뿐. `contracts.py:329-334`가 이 패턴을 문서화한 의도적 결정이므로 존재 자체가 과설계는 아니나, `capture: CaptureResult | None` 계약이면 두 리터럴이 모두 사라짐(F13 참조). *경로 정정: 인용 줄번호가 가리키는 파일은 모두 `motion_person/` 패키지 사본(루트 `webcam_preview.py`는 10줄 shim 등으로 내용이 다름). 줄번호 자체는 정확.*

**F5 (low, 확인됨) — `motion_person/photo_selection.py:40-42` · `event_capture.py:362-365`: evidence_start 규칙 중복.**
`evidence_start = max(start, trigger)` 계산이 photo_selection.py 내 두 곳(`:35` 인라인, `:40` 정의) + event_capture.py 한 곳(`:363`)에 존재하며 공유 상수·헬퍼 없음(grep 확인). `event_capture.py:360`이 `select_photos(start=part.start, trigger=part.trigger)`로 호출하므로 현재는 같은 값 — 당장 부정합 없음. `tests/test_event_capture.py:332-346`(사전 맥락 사진 비커버)과 `:368-385`(select_photos 단독)가 양쪽을 고정하나, 한쪽 임계값만 바꾸면 covered/omitted와 사진 선별이 소리 없이 어긋나는 유지보수 리스크 존재.

**F6 (low, 확인됨) — `motion_person/event_capture.py:211-217` 대비 `231-236`: discard() 후 재시작 피드 거부.**
가드는 `process()`가 아니라 내부 위임 대상 `_process()`(`:229-236`)에 있으며 `packet.frame_index <= self._last_index` 또는 시각 역행 시 ContractError. 인라인 실행 재현: index 500/time 10.0 처리 → `discard()` → index 0/time 0.5 투입 → `ContractError` 재생. `discard()` 직후 상태는 `_moving_keys=frozenset()`, `_last_index=500`, `_last_time=10.0`, `_last_sample_time=10.0` — `:215-216` 주석("새 엔진과 동일")이 명시적으로 가정한 '복구된 피드'의 인덱스 0 재시작은 수용 불가. `discard()`는 `_part`·`_continuation`·`_ring`도 정리하며(`:213-214`), 초기화하지 않는 것은 `_last_index`/`_last_time`/`_last_sample_time`(`:284` 샘플링 주기 판정에 사용 — 복구 첫 프레임의 즉시 샘플링을 막음). 패키지 내부에는 discard 후 같은 인스턴스로 프레임을 계속 먹이는 경로가 없어 실영향 없고, public 메소드인 `discard()`로 인스턴스를 재사용하는 외부 소비자만 주석에 오도될 수 있음.

**F7 (low, 확인됨) — `motion_person/event_capture.py:238-239`, `284-286`, `362-367`: 트래커 ID 스위치 오탐.**
`moving_keys = frozenset((class_name, track_id) ...)`(`:238-239`)의 `starts`가 비어 있지 않으면 샘플 간격과 무관하게 샘플이 강제(`:284-286`)되고, 구 ID는 관측 집합에 남아 `omitted = observed - covered`(`:362-367`)로 누락 보고됨. 실제 트래커(ByteTrack/BoT-SORT, `tracking_adapter.py:12-58`)는 가림 후 재등장 시 새 ID 부여 — `object_state.py:1` docstring 자체가 ID 불안정성을 인정. 실행 재현 2건: (1) ID 스위치 시 0.1초 간격 외 샘플 강제 + 그 샘플이 실제 선택됨 + 다음 정규 샘플 타이밍 밀림, (2) 물리 객체 B가 사진에 신 ID로 담겼는데도 옛 ID가 `omitted_moving_objects`에 보고됨. ID 스위치 오탐 시나리오 테스트 부재(grep 무결과). 설계상 트레이드오프에 가깝고(엔진이 새 track_id의 움직임 시작을 실제 새 객체와 구분 불가) 보수적 설계이나, ID 스위치가 빈번한 환경에서 간격 외 샘플·오탐 omitted가 반복될 수 있음.

**F8 (medium, 확인됨) — `motion_person/photo_selection.py:36`: max_images=2에서 증거 프레임 누락.**
양끝 프레임(before+end)이 예산 전부를 차지해(`:33-34`가 max_images와 무관하게 양끝 추가, `:36`의 `if max_images > 2:` 때문에 start 역할 선택 생략, `:44`·`:51-58` 후속 채움도 진입 불가) 움직임 객체가 있는 트리거(증거) 프레임이 선택에서 빠지고 `omitted_moving_objects`로만 보고됨. 실행 확인: `select_photos([t=0 미매칭, t=1 매칭+moving={("person",1)}, t=2·3 미매칭], start=0, trigger=1, last_motion=1, max_images=2)` → `[(0, ('before',)), (3, ('end',))]`, omitted=`[('person', 1)]`. `event_capture.py:48`이 `2 <= max_images <= 6`을 허용하므로 legal 설정에서 발생.
**기본 설정 해당 여부(본 취합 단계 직접 확인)**: 해당 안 함. `EventConfig.max_images` 기본값은 6(`event_capture.py:30`), MotionEngine은 `events` 미주입 시 기본 `EventConfig()` 사용(`engine.py:67` `self.event_config = events or EventConfig()`), `Config`·CLI에는 `max_images` 필드 자체가 없음(grep 무결과). 즉 기본·CLI 경로에서는 발생하지 않으며 `MotionEngine(..., events=EventConfig(max_images=2))`를 명시한 임베딩 사용자만 영향을 받는다. max_images=3이면 트리거 프레임 포함(실행 확인) — 결함은 max_images=2에 한정.
**수정 방향(제안, 미검증)**: (a) 선택 우선순위를 재조정해 start(트리거) 역할에 최소 1석을 보장(예: before → start → end 순서로 배정)하거나, (b) 허용 하한을 3으로 상향(`event_capture.py:48`·`:49` 검증 변경 + 오류 안내). (b)는 공개 검증 계약의 변경을 수반하므로 (a)를 권장한다.

**F9 (low, 확인됨) — `motion_person/photo_selection.py:33`: 빈 frames에 IndexError.**
`frames[0]` 즉시 접근으로 빈 시퀀스에서 `IndexError`(실행 확인). 유일한 사내 호출처는 `_finish`의 assert(`event_capture.py:359`)인데 `python -O`에서 사라지고(실행 확인), 이 모듈은 타 프로젝트가 직접 import할 수 있는 공개 순수 함수임. 대조적으로 `eviction_candidate([])`는 None 정상 반환 — 두 함수의 계약 비대칭.

**F10 (low, 확인됨) — `motion_person/photo_selection.py:7`: ObjectKey 타입 중복.**
`ObjectKey = tuple[str, int | None]` 정의는 photo_selection.py 내부 3곳(7, 15, 18행)에서만 쓰이고 import는 0회. event_capture.py에 동일 인라인 어노테이션 3곳(`:103` omitted_moving_objects, `:133` motion_starts, `:186` _moving_keys). 구조적 타이핑(`_Sample`이 `PhotoCandidate` Protocol 충족) 때문에 키 타입이 두 모듈에 독립 표기되어 변경 시 따로 놀 수 있음.

**F11 (low, 확인됨) — `motion_person/processor.py:16-19`: DI 애너테이션과 실제 계약 불일치.**
세 DI 키워드 인자의 타입 애너테이션이 구체 클래스(`ObjectDetector | None` 등 — `ObjectDetector`는 `object_detector.py:417`의 `PersonDetector` 별칭)인데, 실제 테스트는 `SimpleNamespace(detect=...)` 덕 타이핑 주입(`tests/test_webcam_preview.py:117-118`, `:153`). 인라인 검증으로 애너테이션이 런타임에 강제되지 않고 덕 타이핑 주입이 실제 동작함을 확인. Protocol 전환 또는 docstring 명시 권장. (*용어 정정: 이전 지적의 '주석'은 타입 애너테이션이며, `processor.py:16-19`에는 코멘트도 덕 타이핑 허용 docstring도 없음 — 클래스 docstring은 `:14` 한 줄.*)

**F12 (low, 확인됨) — `motion_person/webcam_preview.py:18`: 죽은 import 2개 잔존.**
`validate_analysis_values`(`:18`)와 `ContractError`(`:20`)가 임포트문에만 존재하고 본문 사용 없음(grep 확인). HEAD에서도 미사용이었고(`git show HEAD` 확인), 이번 변경이 같은 성격의 죽은 import 4개를 제거하면서 이 2개는 남김.

**F13 (low, 확인됨) — `motion_person/webcam_preview.py:103`: DISABLED 마커 생성 리터럴 4곳 중복.**
`CaptureResult("DISABLED", None, elapsed, 0.0, None)` 규약(path=None·cooldown=0.0·sequence=None)이 `webcam_preview.py:103`, `engine.py:213`, `tests/test_webcam_preview.py:123`, `tests/test_overlay_modes.py:81`에 위치인자로 반복. `contracts.py:330-333`의 설계 의도(표현 계층 자체 생성)는 맞지만 `CaptureResult.disabled(elapsed)` 팩토리 하나로 모을 여지(팩토리 부재 grep 확인).

**F14 (low, 확인됨) — `tests/test_event_capture.py:368`: photo_selection 직접 테스트 얇음.**
photo_selection을 import하는 테스트는 이 파일 한 곳뿐이고 `select_photos` 직접 호출 1건(`:377-378`), `eviction_candidate` 직접 테스트 없음(간접 커버는 메모리 압박 2건 `:286-329`). 그 1건도 event_capture의 private `_Sample`를 import(`:11`)해 `PhotoCandidate` Protocol의 구조적 타이핑 이점(소비자 자료구조로 독립 테스트)을 살리지 못함. 선별 정책(tie-break·첫 움직임 보호·motion_starts 보호·중앙 근접 우선)의 직접 고정 부족.

**F15 (low, 확인됨) — `tests/test_webcam_preview.py:122`: 프로덕션 마커 생성 경로 무검증.**
기존 `disabled.status == DISABLED ...` 단언들이 삭제되고 테스트가 자체 마커(`:123`)로 대체 — 렌더 시점 마커 생성 라인(`webcam_preview.py:103`, `engine.py:213`)을 실행하는 테스트가 없음(`webcam_preview.py:103`은 main() 루프 안인데 테스트의 유일한 main() 호출은 루프 전 실패로 도달 불가; `engine.py:213`은 모든 MotionEngine 테스트가 display=False라 미실행). 세 번째 삭제(`:160`의 DISABLED 확인 → `.decision.frame_index == 9` 검증 교체)도 확인. 완화 요소: 저장 금지·원본 불변 단언(`:127`, `:149`)은 유지되고, 마커를 소비하는 overlay_renderer의 DISABLED 분기(`overlay_renderer.py:174-175`)는 자체 생성 마커 테스트로 여전히 검증 — 실효 손실은 제한적.

---

## 4. 이전 지적 해소 현황 — 전체 대비 요약 및 2건(high) 상태

### 4.0 이전 검토 지적 전체 대비 해소 현황

직전 검토 문서 2종의 **결함성 지적은 총 5건**이며, 그 해소 현황은 다음과 같다(본 취합 단계에 두 문서 전문을 직접 읽어 대조):

| 출처 | 지적 | 현재 상태 |
|---|---|---|
| `docs/refactoring_review.md` "수정이 필요한 사항" (3건) | ① 메모리 제거된 객체가 누락 기록에서도 사라짐(`:41`) | **해소** — `omitted_moving_objects` 프레임 단위 누적(`event_capture.py:295-296`), `tests/test_event_capture.py:315-329` 고정 |
| 〃 | ② 이전 사건 사진이 현재 사건 증거로 계산됨(`:60-68`) | **해소** — `evidence_start = max(start, trigger)` 도입(§3 F5의 규칙), `tests/test_event_capture.py:332-346` 고정 |
| 〃 | ③ 공개 EventBundle 생성자 호환 깨짐(`:84-90`) | **해소** — 새 필드를 마지막에 기본값과 추가(`event_capture.py:103`), 회귀 테스트 `:349-365` |
| `docs/module_weight_review.md` "우선 보완할 문제" (2건) | ① 보조 실행 로그 실패가 핵심 완료 전달을 막음(`:42-57`) | **미해소 (high)** — 아래 P1 |
| 〃 | ② 장시간 실행 시 사라진 추적 ID 메모리 누적(`:81-96`) | **미해소 (high)** — 아래 P2 |

참고: `docs/module_weight_review.md:7`은 "이전 리팩토링 검토의 세 문제는 모두 수정됐다"고 이미 기록하고 있으며(당시 테스트 223 → 227로 증가한 것은 이 3건 수정과 회귀 테스트 추가에 해당), 본 검토가 그 해소를 테스트 위치까지 재확인했다. 두 문서의 **구조 권고**(결함 아님 — refactoring_review "다음 구조 개선 우선순위" 3건: 공통 분석 흐름 통합·사건 정책/저장 분리·탐지기 Protocol化, module_weight_review "경량화 순위" 5건 중 3건: display 기본값 전환·호스트 주입 경로 연결·main 중복 해소)은 모두 미착수다. 이 중 탐지기 Protocol化의 잔여가 이번 F11로 재지적됐고, "engine이 매 실행 새 FrameProcessor를 만든다"는 경량화 순위 4번의 전제는 본 취합 단계에 `engine.py:147`(`processor = FrameProcessor(self.config, run_id)`)에서 여전히 성립함을 확인했다.

### 4.1 P1. 사진 저장-실행 이력 강제 결합 — **미해소 (high, 확인됨)** (= §3 F2)

두 측면 모두 현재 작업 트리에서 그대로다:

- **이력만 끄는 옵션 없음**: `engine.py:129`로 capture=True이면 항상 run_dir 생성, `:162-168`의 `if run_dir is not None:` 블록이 RunLogger 생성+start+tracker.yaml 복사(`:167-168`)를 무조건 실행. `git diff HEAD`에서 이 logger 생성부와 `deliver` 함수(`:138-143`)는 diff에 없음(HEAD와 바이트 동일). Config에 로그 비활성화 필드 없음(`log_dir`는 경로 변경용, `config.py:26`), video/webcam 팩토리 시그니처(`engine.py:89-103`)에도 로그 파라미터 없음. `on_event`는 capture=True 요구(`:115-116`)라 콜백 사용 자체가 로거 존재와 결합.
- **로그 오류 시 사진 저장·콜백 미호출**: `deliver()`가 `logger.write("EVENT_COMPLETED")`를 `on_event`보다 먼저 호출(`:140-143`). `run_logger.py:188-196`은 쓰기 실패 시 LogWriteError 발생(설계 의도: 모듈 docstring "로그 쓰기 실패는 재현성 손실"). 경로: `captures.process()`가 `_finish`로 번들 게시 후 `_mark_delivered`로 `_pending`에서 제거 → deliver 내 logger.write 실패 시 on_event 건너뛰고 예외 전파 → except의 `drain_completed()`는 `_pending`만 반환하므로 이미 제거된 번들 복구 불가 → finally의 flush는 callbacks=False로 콜백 없이 저장만 → `raise error`로 종료(`:275-276`). **결과는 이전 검토와 동일한 "사진은 저장됐지만 완료 콜백 미호출"**. `grep -rn "LogWriteError" tests/` → 0건, 해당 경로 테스트 부재.
- **부분적 인접 개선(참고)**: 신규 `_pending`/`drain_completed` 메커니즘과 except 핸들러의 재전달(`engine.py:229-235`)은 같은 process() 호출 내 '캡처 인코딩 오류'가 난 경우의 콜백 회복을 다룸(신규 테스트는 cv2.imencode 실패 경로만 검증). 그러나 로그 쓰기 오류 트리거와 이력 off 옵션 자체는 손대지 않음 — 미해소 판정.

### 4.2 P2. 사라진 추적 ID 상태의 무한 메모리 누적 — **미해소 (high, 확인됨)**

- `git diff HEAD --stat -- motion_person/object_detector.py motion_person/tracking_adapter.py motion_person/object_state.py motion_person/run_logger.py` → 출력 없음(exit 0). 추적 상태 파일들은 이전 검토가 본 상태(커밋 70e07a4)와 동일.
- `object_detector.py:63-66`의 세 사전(`_identity_ids`, `_tracker_classes`, `_observation_hits`)이 그대로 있고, `_identity_id()`(`:347-357`)는 새 (class_id, tracker_id) 키마다 항목 삽입(`:356`), `:340-344`는 관측마다 누적. 파일 전체 grep 결과 이 세 사전에 대한 제거·만료·상한 연산 없음.
- **정량 근거(출처: 이전 검토 `docs/module_weight_review.md:89-95`의 합성 재현 — 모델 미실행, 박스 정리 함수에 상이한 ID를 반복 주입하고 tracemalloc으로 측정한 Python 할당 증가)**: 상이한 ID 10,000개 ≈ 4.12 MiB, 20,000개 ≈ 8.17 MiB. 전체 프로세스 RSS나 실제 카메라의 ID 생성 속도는 아니며, 본 검토에서 재측정하지는 않았다(not run) — 다만 append-only 구조 자체는 이번에 재확인했다.
- 부수 확인: `EventCaptureEngine`의 메모리 상한 `_bound_memory`(`event_capture.py:341-355`)은 JPEG 버퍼에만 적용되고 탐지기 사전에는 미적용. `ObjectStateStore`(`object_state.py:11`, `motion_person/main.py:420`에서 사용)도 track_id마다 사전 누적에 삭제 경로 없음.
- 이전 검토의 권고(만료·최대 상태 수 + 증가 카운터 기반 ID, `docs/module_weight_review.md:96-105`)는 구현되지 않음.

---

## 5. 모듈 무게 — 질문 ② "모듈로 붙일 때 너무 무겁지 않을까" · 질문 ③ "핵심 외에 뺄 것"

### 5.1 질문 ② 결론

**무겁다면 그 원인은 붙이는 코드가 아니라 의존성이다.** 자체 코드는 243KiB(22파일 249,132바이트, `wc -c` 재확인)뿐이고, 코드를 아무리 잘라도 설치 용량(이전 측정 총 722MiB; 이번 du로 구성만 재측정 — §5.4)은 거의 줄지 않는다. 실질 경량화는 **ultralytics+lap을 선택 의존성(extra)으로 옮기는 것뿐**인데, lazy import(`object_detector.py:100` — `from ultralytics import YOLO`가 load() 내부에만 존재)와 `FrameProcessor`의 탐지기 주입(`processor.py:16-25`) 덕분에 구조상 이미 가능하다. 호스트가 자체 분석기를 주입하면 torch 스택 전체를 설치하지 않아도 된다. `import motion_person` 자체는 가볍다(지연 export, 이번 실측 RSS 16.7MiB — 이전 측정 16.9MiB와 동일 규모).
임베딩 시 API 호환성 주의 1건: `process()` 반환 타입 전환은 호환 장치 없는 breaking change다(상세·근거는 §2.3 — `FrameAnalysis`에 튜플 호환 프로토콜 없음, 변경은 커밋 전 상태).

### 5.2 질문 ③ 결론 — 임베딩 시 제외 추천

| 우선순위 | 대상 | 조치 |
|---|---|---|
| **1순위** | 기존 개별 캡처 경로 (`main.py`, `capture_manager.py`, `object_state.py` — 약 52KiB, 패키지의 21%) | MotionEngine 사건 경로에서 전혀 사용되지 않음(`main.py`만 사용, engine은 import 안 함). 임베딩 모듈에서 제외할 1순위. 저장소 CLI 호환(`python main.py` 사용법) 때문에 즉시 삭제보다는 레거시/예제 경로로 격리하거나 공통 FrameProcessor로 수렴 권장 |
| **2순위** | 실행 이력 저장 (`run_logger.py` — 약 9.7KiB + engine의 tracker.yaml 복사) | 재현·디버그용 부가 기능인데 사진 저장(capture=True)과 강제 결합돼 있어(P1 미해소) 우려의 핵심. 모듈 기본값은 끄고 CLI 엄격 모드로 유지, 선택 로그 실패가 완료 콜백·실행 성공을 침식하지 않게 분리 필요. 런타임 메모리 영향은 이전 검토의 3초 영상 표본에서 실행 이력 약 5 KiB로 측정된 바 있다(`docs/module_weight_review.md:148` — 사건 완성 없는 짧은 표본이라 일반화 불가, 이번 검토 재측정은 not run) |
| **3순위** | 웹캠 미리보기·오버레이·간편 CLI (`webcam_preview.py` 약 6.9KiB, `overlay_renderer.py` 약 19.2KiB, cli 등 약 20KiB) | 저장 없는 개발·시연 도구. CLI/진단 extra로 분리. 오버레이는 engine이 모듈 최상위에서 import하므로(`engine.py:20`) lazy import 전환으로 headless 실행에서 미로드되게 하는 정도가 적당 |

**호스트 담당 영역은 이미 비워둠**: 알림·장기 보관·클라우드 후속 처리는 구현돼 있지 않아 호스트 영역으로 남겨둔 상태 — 과설계가 아니다. 루트 호환 진입점 심 17개는 위임용 shim일 뿐 설치물에 포함되지 않아(`pyproject.toml:19-20` packages.find 확인) 설치 무게 0바이트, 제거하면 레거시 호출이 깨지므로 유지.

### 5.3 분류 표

| 영역 (파일) | 규모 | 분류 | 근거·비고 |
|---|---|---|---|
| 공개 API·데이터 계약 (`contracts.py`, `__init__.py`) | 약 10.7KiB | **계약 유지(keep)** | 임베딩 계약 자체. frozen DTO와 오류 코드 테이블은 호스트 통합에 필요한 최소 표면. `__init__.py`는 `__getattr__` 지연 export만 |
| 프레임 분석 오케스트레이션 (`processor.py`) | 2.4KiB | **핵심(core)** | 카메라·창·저장 없는 공통 분석 경로. detector·motion·events 생성자 주입 지점 |
| YOLO 탐지·실행 내 추적 (`object_detector.py`, `tracking_adapter.py`) | 약 23.4KiB | **핵심(core, 수명 제한 보완 필요)** | 핵심이나 추적 ID 상태 무한 누적(P2)이 남아 장시간 웹캠 운영 전 만료·상한 필요 |
| MOG2 전경 분리·움직임 판정 (`motion_detector.py`, `event_detector.py`, `frame_processor.py`) | 약 28.7KiB | **핵심(core)** | 디스크·화면 의존 없는 순수 판정·전처리 계층 |
| 사건 분할·유한 사진 버퍼·묶음 저장 (`event_capture.py`) | 약 18.4KiB | **핵심(core)** | 사건 경계·10초 분할·64MiB 상한 버퍼·원자 공개. 사건 정책과 디스크 저장이 한 클래스에 있어 후속 분리 후보 |
| 사진 선별·제거 정책 (`photo_selection.py`) | 3.4KiB (77줄) | **핵심(core)** | stdlib만 쓰는 디스크·인코딩 독립 선별 규칙. 이번 리팩터링에서 실제 연결 확인. F8 결함 보완 필요(기본 설정 미해당) |
| 실행 엔진·입력 어댑터 (`engine.py`, `video_source.py`, `webcam_source.py`) | 약 33.9KiB | **핵심(core)** | 공개 Facade. 라이브러리 기본 display=True는 headless가 자연스러우므로 기본값 전환 권장(현재도 display=False 호출 가능) |
| 실행 이력 저장 (`run_logger.py` + engine의 tracker.yaml 복사) | 약 9.7KiB | **선택 분리(optional)** | §5.2 2순위 참조 |
| 웹캠 미리보기 (`webcam_preview.py`) | 약 6.9KiB (132줄) | **선택 분리(optional)** | §5.2 3순위 참조 |
| 오버레이 렌더링·시연 창 (`overlay_renderer.py`) | 약 19.2KiB | **선택 분리(optional)** | 실행 경로는 display=False·overlay='none'으로 이미 끌 수 있으나 engine이 최상위 import — lazy 전환 권장 |
| 간편 CLI (`cli.py`, `console_messages.py`, 스크립트) | 약 20KiB | **선택 분리(optional)** | 호스트는 자체 진입점 사용 |
| 기존 개별 캡처 경로 (`main.py`, `capture_manager.py`, `object_state.py`) | 약 52KiB (패키지의 21%) | **선택 분리(optional, 제외 1순위)** | §5.2 1순위 참조 |
| 루트 호환 진입점 심 17개 (+build/dist 산출물) | 4,638바이트 | **계약 유지(keep)** | 위임 shim. 설치물 미포함. build/lib의 낡은 복사본 284K는 git 미추적 |
| ultralytics==8.4.174 + lap (torch·torchvision 전이 포함) | torch 583M 등 | **선택 분리(의존성 extra 전환)** | §5.4 참조 |
| opencv-python·numpy | cv2 119M·numpy 34M | **계약 유지(keep)** | 영상 디코드·MOG2·JPEG 인코딩이 cv2 그 자체 — core에서 제거 불가 |

### 5.4 의존성 무게 논의

- **측정(이번 검토, `.venv` python3.12 site-packages `du -sh`)**: torch 583M · torchvision 9.3M · ultralytics 9.5M · lap 1.6M · cv2 119M · numpy 34M. 총량 722MiB는 이전 검토 측정치(`docs/module_weight_review.md:117-126`, 배포판 등록 파일 크기 합산 — torch 552.7MiB·opencv 119.3MiB 등으로 이번 du와 등록 크기 기준 차이만 있음)이며, 이번엔 구성 요소만 재측정(총합 재측정은 not run).
- **ultralytics+lap → extra 전환**: `pyproject.toml:11`이 필수 의존성으로 두고 있으나, `from ultralytics import YOLO`는 `load()` 내부에만 있고(`object_detector.py:100`, import 실패는 ModelLoadError로 안내 메시지와 함께 변환 `:100-105`), README도 ultralytics/torch 미설치 환경에서 테스트 동작을 문서화함(`README.md:600-601`). FrameProcessor 주입과 합쳐 호스트 자체 탐지기 경로가 이미 열려 있음.
- **cv2·numpy는 불가결**: 9개 런타임 모듈이 최상위 `import cv2` 사용 — 영상 디코드·MOG2·JPEG 인코딩이 cv2 자체라 core 경로에서 제거 불가. 다만 `opencv-python` 이름 고정은 headless opencv 사용 호스트와 충돌 여지가 있어(README `:430-431`이 혼용 금지) 공급자를 허용하는 패키징 조정 검토 가치 있음.
- **런타임 메모리**: 이번 검토에서는 pytest·런타임 피크 메모리 측정을 수행하지 않았음(not run). `import motion_person` RSS 16.7MiB(`/usr/bin/time -l` 이번 실측)만 확인. 참고로 이전 검토는 3초 영상 표본에서 저장·화면 모두 꺼도 약 411MiB를 측정한 바 있다(`docs/module_weight_review.md:143-150`) — 이 수치의 재검증은 이번 범위 밖.

---

## 6. 테스트 게이트 결과 및 실행 목록

| 항목 | 명령 | 결과 | 비고 |
|---|---|---|---|
| 전체 게이트 | `.venv/bin/python -m pytest tests -q` | **227 passed, exit 0** | 워크플로우가 전달한 게이트 결과(본 취합 단계에서 재실행하지 않음) |
| 엔진 그룹 보조 실행 | `./.venv/bin/python -m pytest tests/test_engine.py -p no:cacheprovider -q` | **18 passed (0.52s)** | 신규 회복 전달 테스트 포함 |
| 사진 선별 그룹 | (그룹 관련 테스트 73개 = 55+18) | 통과 | 그룹 검토자 보고 |
| 임포트·인라인 검증 | `.venv/bin/python -c` 인라인 (파일 생성 없음) | 전부 재현 성공 | photo_selection 무거움 검사, `__init__` 지연 export, KI 번들 유실, discard 재시작 거부, ID 스위치 오탐 2건, LogWriteError 발생, select_photos 빈 입력/max_images=2, du·wc 측정 등 |
| 일부 그룹의 스위트 직접 실행 | — | **not run** | 캡처+DTO·미리보기·테스트 3종 검토자는 검토 전용 읽기 전용 제약으로 실행 못함 — 정적 검토·코드 경로 추적으로 대체(전체 게이트가 별도로 통과해 실제 커버는 확보) |

**본 취합(갱신) 단계에서 직접 실행한 보충 확인**(모두 읽기 전용 명령, 파일 생성 없음):

- `grep -n "max_images" motion_person/event_capture.py motion_person/config.py` → 기본값 6(`event_capture.py:30`)·검증 `:47-49` 확인, config.py에는 필드 없음
- `grep -n "EventConfig" motion_person/engine.py` → `:67` `events or EventConfig()` 확인
- `motion_person/processor.py`·`contracts.py:325-345`·`event_capture.py:20-59` 전문 읽기 → `FrameAnalysis`에 튜플 호환 프로토콜 없음, `process()` 반환 타입 확인
- `git diff HEAD --numstat -- motion_person/event_capture.py motion_person/contracts.py motion_person/__init__.py` → +54/−36, +14/−0, +2/−1
- `grep -n "FrameProcessor" motion_person/engine.py` → `:147`에서 매 실행 생성 확인
- `wc -l` photo_selection.py 77줄, `sed` pyproject.toml(version 0.2.0), `docs/refactoring_review.md`·`docs/module_weight_review.md` 전문 대조

---

## 7. 이번 검토의 한계

1. **취합 문서의 성격**: 본 문서는 하위 검토자의 독립 검증 결과를 취합한 것이다. 갱신 시 보충 확인(§6 하단) 외에는 게이트·재현 명령을 재실행하지 않았고, 유일한 쓰기는 이 문서다.
2. **미확인 주장**: "임베딩 사용자의 '실행 이력 저장 제외' 요구"는 저장소 문서(README.md, docs/*.md)에서 근거를 찾지 못함 — 외부 요구사항 문맥으로 추정되며 검증 범위 밖(F2 참조). 또한 0.2.0 배포물에 구 5-튜플 API가 포함됐는지도 저장소에서 확인 불가(§2.3).
3. **수치의 출처 구분**: 설치 총량 722MiB·추적 ID 20,000개 ≈ 8.17MiB(§4.2 인용)·런타임 약 411MiB·실행 이력 약 5KiB는 모두 이전 검토(`docs/module_weight_review.md`) 수치 — 이번에는 du 구성 요소·RSS 16.7MiB·코드 크기·diff 줄 수만 실측하고 나머지는 재측정하지 않았다(not run).
4. **F1(KI) 재현의 범위**: `MotionEngine.start()` 전체를 실제 Ctrl+C로 구동하지는 않음 — 엔진 라우팅은 코드 정독, 이벤트 캡처 메커니즘·flush 거동은 인라인 재현으로 확인.
5. **그룹별 제약**: 검토 전용 읽기 전용 제약으로 일부 그룹은 전체 스위트를 직접 실행하지 못함(§6). 런타임 메모리·성능은 이번 검토에서 측정하지 않음.
6. **빌드 산출물**: `build/lib`에 낡은 소스 복사본 21개(284K)가 있으나 git이 추적하지 않아(`.gitignore:30-32`, `git ls-files` 빈 결과) 검토 대상에서 제외.
7. **수정 방향 제안의 지위**: §1·§3의 "수정 방향(제안)" 항목(F1의 drain 추가, F8의 우선순위 재조정 또는 하한 상향)은 코드 근거 위의 의견 제안이며 구현·검증은 하지 않았다.
