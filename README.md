# motion_person_poc 사용 안내

동영상에서 사전학습 YOLO11n으로 선택한 객체(기본 사람)를 탐지하고, 같은 프레임의 MOG2 전경 마스크를
객체 박스 영역과 결합하여 움직임 조건을 만족하는 순간의 원본 프레임을 JPG로 저장하는
캡처 PoC(Proof of Concept, 개념 검증) 프로그램이다. 실행 장치는 CPU이며 고정 카메라 영상을 전제로 한다.
영상 창의 문구는 영어로, 간편 CLI의 안내와 이 문서는 한국어로 제공한다 (SPEC.md 17.3).

## 저장소 범위와 현재 상태

이 저장소는 영상 이벤트 감지 모듈의 소스, 개발 명세, 테스트, 재현 도구와 공개 예제 영상을 포함한다.
핵심 파일 분석 기능은 구현됐고 표시 선택·간편 CLI·웹캠 미리보기를 포함한 로컬 자동 테스트 163개를 통과했다.
웹캠은 저장 없는 테스트용 미리보기로 지원한다. RTSP 입력과 실카메라 장시간 운용은 아직 구현·검증하지 않았다.
실제 Windows 실행 검증도 남아 있다. MOG2는 영상 변화 단서이며 걷기·기어다니기 행동 분류가 아니다.

사용자 영상, 캡처, 실행 로그, 가상환경과 모델 가중치는 포함하지 않는다.
`examples/`에는 이용 조건을 확인한 공개 입력 영상 두 개와 처리 결과 두 개를 포함했다.
영상 보기·실행 명령·검증 결과는 [예제 안내](examples/README.md), 출처와 이용 조건은
[예제 저작자 표시](examples/ATTRIBUTION.md)에 있다.
보고서의 실행 결과 경로는 로컬 재현 시 생성되는 파일 위치다.
공개 데이터의 다운로드 URL·체크섬은 `test_data/public_sources/manifest.json`에 포함했다.
실제 입력에 카메라 화면만 들어오면 `--roi`를 생략하고 전체 화면을 분석한다.

## macOS / Linux 빠른 시작

```bash
git clone https://github.com/krw82/motion-person-poc.git
cd motion-person-poc
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.in
.venv/bin/python -c "from ultralytics import YOLO; YOLO('models/yolo11n.pt')"
```

Python 3.11 이상을 사용한다. 설치한 다음 가상환경 활성화 없이 시작할 수 있다.

설치 직후 저장소에 포함된 영상으로 실행할 수 있다.

```bash
./motion video examples/videos/caviar_stop_resume.mp4
./motion video examples/videos/dog_walker_tokyo.mp4 --objects 사람 개
```

| 하고 싶은 일 | 명령 |
| - | - |
| 메뉴에서 선택 | `./motion` |
| 웹캠 화면 확인 (저장 없음) | `./motion webcam` |
| 영상 파일 분석 | `./motion video "영상 파일 경로"` |
| 영상 경로만으로 분석 | `./motion "영상 파일 경로"` |
| 설치·모델 확인 (카메라를 열지 않음) | `./motion doctor` |

영상 파일은 원하는 위치에 두면 된다. 공백이 있는 경로는 따옴표로 감싼다.
`./motion video`만 입력하면 경로를 물어보며, 이 입력란에는 Finder에서 파일을 끌어다 놓아도 된다.
상호작용이 없는 실행 환경에서 `./motion`만 호출하면 도움말을 표시한다.

기본 대상은 사람이다. 사람과 개를 함께 선택하거나 캡처 간격을 바꾸는 예:

```bash
./motion webcam --objects 사람 개
./motion video "videos/demo.mp4" --objects 사람 개 --every 0.5 --mask
./motion video "videos/demo.mp4" --no-display --fast
```

간편 영상 명령은 **추적 켜짐, 객체별 최소 저장 간격 0.5초, 탐지 신뢰도 0.4,
기준 전경 면적 1000픽셀**로 시작한다. 움직임·탐지·추적 확인 조건을 모두 만족해야 캡처하므로
매초 2장이 보장되지는 않는다. 여러 객체가 같은 프레임에서 저장 조건을 만족해도
원본 전체 JPG 한 장을 저장한다. `사람`과 `아기`는 모두 모델의 `person` 종류에 해당한다.

`--objects`는 `person dog` 같은 모델의 영문 이름도 지원한다. `--camera 1`로 웹캠을 바꾸고,
`--output "저장 폴더"`로 영상의 캡처 저장 위치를 지정한다.
자세한 옵션은 `./motion webcam --help`, `./motion video --help`에서 확인한다.
모델·기본 결과 폴더는 모듈 위치를 기준으로 찾고, 직접 입력한 상대 경로는 실행한 폴더 기준이다.

입력과 모델은 사전에 준비하며 분석 중 외부 API를 호출하지 않는다.
캡처는 `captures/<run_id>/`, 로그는 `logs/<run_id>/`에 저장한다.
시작 시 실제 저장 폴더를 표시하고, 종료 시 처리 프레임과 캡처 수를 알려준다.
기존 `main.py`의 옵션과 기본값도 유지한다. ROI·추적기 등 상세 조정에는 아래 기존 명령을 사용한다.

```bash
.venv/bin/python main.py --video videos/demo.mp4 --classes person dog --track --capture-scope object --cooldown-sec 0.5 --confidence 0.4 --min-object-motion-pixels 1000 --show-mask
```

## Windows 간편 실행

기존 Windows 설치 절차로 `.venv`와 모델을 준비한 다음 PowerShell에서 실행한다.

```powershell
.\motion.cmd
.\motion.cmd webcam
.\motion.cmd video "C:\Videos\걷는 사람.mp4"
.\motion.cmd doctor
```

`motion.cmd`는 모듈의 `.venv\Scripts\python.exe`를 우선 사용한다.
Windows 실행과 카메라 화면은 아직 실제 검증하지 않았다.

자동 테스트에는 입력 영상이나 모델 가중치가 필요하지 않다.

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
```

GitHub Actions도 같은 테스트를 Python 3.11 / Ubuntu에서 실행한다.

## 웹캠으로 저장 없이 확인하기

```bash
./motion webcam
```

Windows에서는 `.\motion.cmd webcam`을 사용한다.
카메라 접근 권한을 허용하고 화면의 `PERSON #1` 박스를 확인한다.
기본 화면은 거울처럼 좌우 반전하며 `--no-mirror`로 끌 수 있다.

- 녹색 박스: 선택한 객체 탐지. 움직임 조건을 만족하지 않아도 표시한다.
- 노란 박스: 그 객체의 MOG2 전경 조건 충족. 실제 행동 분류는 아니다.
- 회색 PENDING: 캡처 조건에 쓰기 전 관측 확인 중.
- 오른쪽 DETECTED / MOVING: 현재 탐지·전경 조건 충족 객체 수.
- FPS / AGE: 처리 속도와 프레임을 받아 표시 준비할 때까지의 경과 시간.
- ESC, q 또는 창 닫기로 종료한다. 처음 2초는 배경 초기 학습이다.

이 모드는 **사진·영상·실행 로그를 저장하지 않는다.** 추적 설정 YAML만 임시 파일로 준비하고 종료 시 지운다.
추론 중 카메라 입력을 최신 한 장으로 교체해 오래된 프레임을 쌓지 않으며,
배경 초기 학습·표시는 프레임 번호/FPS 대신 실제 경과 시간을 쓴다.
카메라/드라이버 자체의 지연이나 30 FPS 추론을 보장하는 것은 아니다.
`--camera 1`은 다른 장치 선택, `--mask`는 별도 움직임 마스크 창,
`--seconds 30`은 30초 후 자동 종료, `--imgsz 960`은 추론 크기 비교다.
실제 웹캠 기능은 독립 `webcam_preview.py`이며 기존 `main.py --video` 저장 실행은 유지한다.
macOS가 `not authorized to capture video`를 반환하면 실행 앱의 카메라 권한을 허용한 뒤
다시 실행한다. 해당 환경의 카메라 권한이 없어 이번 구현에서 실제 웹캠 영상 표시는 아직 확인하지 못했다.
권한을 받는 앱은 실행 위치에 따라 Terminal 또는 Codex 등으로 표시될 수 있다.
CLI·웹캠 검증 범위와 실제 파일 실행 결과는 [간편 CLI 테스트 보고서](docs/cli_webcam_test_report.md)에 있다.

## 화면의 박스·강조 표시 선택하기

노란 테두리는 움직임 조건을 만족한 객체의 화면 강조다. 캡처 저장 순간뿐 아니라
그 조건을 만족하는 동안 계속 표시하며 쿨다운 중에도 표시한다.
저장 JPG는 항상 박스·문구가 없는 원본 전체 프레임이다.

| 선택값 | 표시 내용 |
| - | - |
| `--overlay full` | 기존 전체 표시: 객체 박스·번호, 빨간 전경 박스, 노란 움직임 강조, 상태 패널 (기본값) |
| `--overlay objects` | 객체 박스·번호와 상태 패널. 노란 강조·MATCH 문구와 빨간 전경 박스 숨김 |
| `--overlay none` | 영상 창에 분석 영상만 표시. 박스·라벨·상태 패널·범례 모두 숨김 |

```bash
./motion webcam --overlay objects
./motion webcam --overlay none
./motion video "videos/demo.mp4" --overlay objects
./motion video "videos/demo.mp4" --overlay none
```

어떤 값을 골라도 탐지·추적·움직임 판정·캡처 저장 간격과 로그 동작은 같다.
`--mask`로 켜는 별도 마스크 창은 이 선택과 독립적이다.
기존 `main.py`, `webcam_preview.py`에서도 같은 `--overlay` 옵션을 사용할 수 있다.
Python 호출에서는 `Config(overlay_mode="objects")`로 설정하고 렌더러를 직접 쓰는 경우
`render_overlay(..., overlay_mode="objects")`로 전달한다.

## 객체 종류와 추적 번호를 보면서 테스트하기

`--classes person dog`로 대상 종류를 선택하고 `--track`으로 객체별 번호를 유지한다.
기존 `main.py` 기본 명령의 사람 전용 탐지와 전체 쿨다운 동작은 유지된다. 현재 Mac에 준비된 한 명 통행 영상:

```bash
.venv/bin/python main.py --video videos/public_tests/cdnet_single_pass.mp4 --classes person dog --track --capture-scope object --cooldown-sec 0.5 --show-mask --debug-decisions
```

화면에는 `PERSON #1`, `DOG #2` 같은 종류와 실행 내 번호가 표시된다. 추적 모드의 상태
패널은 오른쪽에 배치해 객체 번호를 가리지 않는다. 번호는 실행할 때마다 새로 시작한다.

사람과 개가 함께 나오는 실제 산책 영상:

```bash
.venv/bin/python main.py --video videos/public_tests/dog_walker_tokyo.mp4 --classes person dog --track --capture-scope object --cooldown-sec 0.5 --confidence 0.4 --min-object-motion-pixels 1000 --show-mask --debug-decisions
```

카메라 움직임과 가림이 있는 영상이라 종류·추적·로그 기능을 확인하는 용도다.
개만 선택하려면 `--classes dog`, 사람만 선택하려면 `--classes person`으로 바꾼다.
모델에 없는 종류 이름은 시작 단계에서 오류로 알려준다.

| 옵션 | 의미 |
| - | - |
| `--classes person dog` | 모델의 영문 종류 이름 선택. 기본 `person` |
| `--track` | 프레임 사이 추적 번호 유지. 기본 비활성 |
| `--tracker bytetrack` | 기본 추적기. `botsort`도 선택 가능 |
| `--tracking-profile stable` | 기본 추적 모드. 한 번의 YOLO 결과를 종류별 독립 추적기에 연결 |
| `--track-buffer 60` | 놓친 추적을 유지하는 업데이트 횟수. 30 FPS 전 프레임 처리 시 약 2초 |
| `--track-new-threshold 0.4` | 새 번호 생성 신뢰도 하한. 기존 번호 연결 하한과 별도 |
| `--track-min-hits 3` | 신뢰도 하한을 통과한 관측이 3회 누적돼야 캡처 가능 |
| `--tracker botsort --reid` | YOLO 자체 특징으로 외형 연결을 비교. 고정 카메라 보정은 끔 |
| `--roi 0 0.285 1 0.755` | 정규화 좌표로 분석 영역 선택. 이 값은 제공된 5077 영상용 예시 |
| `--capture-scope object` | 객체별 쿨다운. `--track` 필수. 기본 `frame`은 전체 쿨다운 |
| `--confidence 0.4` | 모든 선택 종류의 신뢰도 하한. 기존 `--person-confidence` 별칭 |
| `--min-object-motion-pixels 1000` | 공통 전경 면적 기준. 기존 `--min-person-motion-pixels` 별칭 |
| `--min-object-motion-ratio 0.03` | 박스 내부 전경 비율 기준. 기존 옵션 별칭 |

새 객체는 다른 객체의 쿨다운 중에도 저장을 시작할 수 있다. 여러 객체가 같은 프레임에서
저장 대상이어도 **원본 전체 JPG 한 장**만 저장한다. 객체 crop을 저장하는 기능은 아니다.
`saved_track_ids`는 해당 저장을 발생시킨 번호이고 `matched_objects`는 전경 조건을
충족한 모든 객체다. JPG에 다른 객체도 함께 보일 수 있어 실제 사진에 등장한 횟수와
객체별 저장 트리거 수는 다르다.

종료 후 `logs/<run_id>/summary.json`의 `objects`에는 번호별 종류, 첫/마지막 탐지 시간,
관측·전경 조건 충족 프레임 수, 저장 트리거 수와 마지막 저장 시간이 남는다. 매 프레임
결과는 `--debug-decisions`를 켜고 `events.jsonl`의 `objects`에서 확인한다. 기존
`persons`/`matched_persons` 필드는 사람만 포함한다.

추적 옵션은 기본적으로 종류별 연결을 분리한다. YOLO 추론은 프레임당 한 번이며 탐지가
없는 종류도 추적 시간을 갱신한다. 회색 `PENDING` 박스는 아직 필요한 관측 횟수를 채우지
않은 번호이고 캡처를 발생시키지 않는다. `confirmed`, `observation_hits`, 거부 사유
`TRACK_PENDING`으로 이를 로그에 남긴다. 관측 횟수는 사용자 신뢰도 기준을 통과한 박스만
누적한다. 확정된 번호는 잠깐 놓쳤다가 복구돼도 확정 상태를 유지한다.

`--tracking-profile baseline`은 이전 방식과 비교할 때 사용한다. 이 CLI는 추적기를 함께
쓰는 이전 연결 방식, buffer 30, 새 번호 하한 0.25, 관측 확인 1회로 설정한다. 그 뒤 개별
옵션을 지정하면 덮어쓴다. 실제 적용된 값은 `run_config.json`과 실행 폴더의 `tracker.yaml`에
기록한다. `--reid`는 `--track --tracker botsort`와 함께 써야 한다.

분석 ROI는 원본의 비율 좌표 `X1 Y1 X2 Y2`다. YOLO와 MOG2가 같은 영역을 분석하고,
로그의 `original_box_xyxy`에는 영역의 시작 위치와 리사이즈를 반영한다. 원본 전체 JPG는
그대로 저장한다. ROI를 사용한 표시 영상은 분석 영역을 보여주므로 캡처와 화면 크기가
다를 수 있다. ROI를 바꾸면 분석 면적에 따라 최소 전경 픽셀 값도 다시 보정한다.
실제 카메라 입력에 여백이 없다면 ROI를 지정할 필요가 없다.

Python의 `ObjectStateStore.objects[track_id]`로 객체 상태를 읽을 수 있다. 종료 요약 JSON의
키는 문자열이라 `summary['objects']['1']`처럼 조회한다. `motion_qualified`는 MOG2
전경 결합 조건 충족이며 실제 속도나 정지를 확정하지 않는다. `visible`은 마지막 처리
프레임에서 추적 결과에 관측됐는지 뜻한다.

`track_id`는 종류와 원래 추적기 번호의 조합을 실행 내 번호로 매핑한 값이다. 원래 번호는
`tracker_id`에 남긴다. 종류가 바뀐 결과를 같은 객체 변수에 섞지 않는다. 가림, 탐지 누락,
퇴장·재입장이나 편집 컷에서는 번호가 달라지거나 잘못 연결될 수 있으므로 번호 수를
실제 사람·개 수로 해석하지 않는다.

산책 영상 출처: Nesnad,
[Pro dog walker - Tokyo - 2024 Nov 1](https://commons.wikimedia.org/wiki/File:Pro_dog_walker_-_Tokyo_-_2024_Nov_1.webm),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
전체 화면·디코드 프레임을 유지하고 오디오를 제외해 MP4로 변환했다. 출처와 해시는
`test_data/tracking_sample_manifest.json`에 기록한다. 재준비는
`.venv/bin/python tools/prepare_tracking_samples.py`로 실행하며 준비 도구는
`imageio-ffmpeg`를 사용해 프레임 수를 교차 확인한다. 추적 런타임의 `lap`은
`requirements.in`에 추가했다.

추적·사용자 영상 결과는 [객체 추적 테스트 보고서](docs/object_tracking_test_report.md)에 있다.
종류별 연결·관측 확인·분석 ROI를 추가한 최신 비교와 직접 실행 명령은
[추적 개선 테스트 보고서](docs/object_tracking_improvement_report.md)에 있다.
추가 사용자 영상 5396의 ROI·추론 크기 비교는 [5396 영상 테스트 보고서](docs/tiktok_7605040600257285396_test_report.md)에 있다.

공개 영상 9회 실행, 6,242프레임의 실제 결과는
[공개 데이터셋 테스트 보고서](docs/public_dataset_test_report.md)에 있다. macOS CPU에서
저장 계약과 자동 테스트 89개를 확인했다. 기본 CAVIAR 움직임 구간 검출은 2/7로 부족하며
Windows 실행은 아직 검증하지 않았다.

확인한 공개 영상으로 실행하려면 프로젝트 루트에서 다음 명령을 사용한다.

```bash
.venv/bin/python main.py --video videos/public_tests/cdnet_pedestrians.mp4 --model models/yolo11n.pt --no-display --pace fast
```

## 1. 바로 실행 (기본 명령)

모델 가중치(`models/yolo11n.pt`)와 시연 영상(`videos/demo.mp4`)을 먼저 준비한 뒤 프로젝트
루트에서 실행한다. 가상환경 활성화 없이 실행 파일을 명시하는 방식을 기본으로 한다.

```powershell
.\.venv\Scripts\python.exe main.py --video videos/demo.mp4 --model models/yolo11n.pt
```

```bash
# macOS / Linux 로컬 개발 환경 예시
.venv/bin/python main.py --video videos/demo.mp4 --model models/yolo11n.pt
```

- 창 이름 `Motion Person PoC`가 열리고 사람 박스(녹색), 전경 영역(빨강), 적격 사람(노란
  강조)과 상태 문구가 표시된다.
- 조건을 만족하면 원본 해상도 JPG가 `captures/<run_id>/`에 저장된다. 박스나 글자는 섞이지
  않는다.

종료 방법과 종료 코드:

| 조작 | 결과 | 종료 코드 |
| - | - | - |
| ESC 또는 q 키 | 사용자 중단 (USER_STOP), 정상 종료 | 0 |
| 창 닫기 버튼 | 창 속성 확인 후 USER_STOP 정상 종료 | 0 |
| 영상 끝 (EOF) | 정상 종료 | 0 |
| Ctrl+C | 사용자 인터럽트 (USER_INTERRUPT) | 130 |

오류 종료 코드: 설정 오류 2, 영상 입력 오류 3, 모델·추론 오류 4, 저장·로그 오류 5, 화면 표시
오류 6 (전체 목록은 SPEC.md 21절). 표시 오류 안내는 `--no-display` 재실행이다. 다시 실행하면
배경 모델과 쿨다운이 초기화된 새 `run_id`로 시작한다.

## 2. 요구 환경과 설치

- 대상 환경: Windows 10 / Windows 11 64비트, Python 3.11
- 직접 의존성: `opencv-python`, `numpy`, `ultralytics` (requirements.in)
- GUI 시연에 `opencv-python`을 사용하므로 `opencv-python-headless`를 같은 환경에 함께
  설치하지 않는다.
- 런타임의 영상 분석과 저장에는 외부 API 호출이 없다. 패키지와 모델 다운로드는 인터넷 연결
  환경에서 미리 완료한다.

### 2.1 설치 (PowerShell)

가상환경을 활성화하지 않고 실행 파일을 명시하므로 실행 정책 변경은 필요 없다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.in
.\.venv\Scripts\python.exe -c "import cv2, numpy, ultralytics; print(cv2.__version__, numpy.__version__, ultralytics.__version__)"
```

### 2.2 모델 준비

`models` 폴더를 만들고 공식 YOLO11n Detection 가중치를 `yolo11n.pt` 이름으로 둔다. 다음
명령으로 가중치를 `models/yolo11n.pt`에 준비한다.

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; Path('models').mkdir(exist_ok=True); from ultralytics import YOLO; YOLO('models/yolo11n.pt')"
```

시연 명령은 `models` 경로를 사용하며 파일이 없으면 시작 단계에서 중단한다
(MODEL_NOT_FOUND, 종료 코드 4).

### 2.3 버전 고정

`requirements.lock.txt`는 대상 노트북에서 모델 로드, 영상 읽기와 캡처 테스트가 통과한
설치를 기준으로 **검증 후 생성**한다. 지금은 포함되어 있지 않다.

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements.lock.txt
```

팀원은 동일한 Python 버전과 운영체제 계열에서 `requirements.lock.txt`로 재현한다. 다른
운영체제나 CPU 아키텍처에는 같은 wheel이 없는 경우가 있으므로 별도 검증 환경으로 기록한다.
문서의 예시를 실제 검증한 패키지 조합으로 표시하지 않는다.

## 3. 실행 모드와 CLI 옵션

### 3.1 실행 모드 4가지

기본 시연 (실시간 재생, 창 표시):

```bash
python main.py --video videos/demo.mp4 --model models/yolo11n.pt
```

마스크 표시 — `Valid Foreground Mask` 창이 추가로 열린다:

```bash
python main.py --video videos/demo.mp4 --show-mask
```

화면 없는 빠른 테스트 — 재생 대기 없이 전 프레임을 순서대로 처리한다. 종료는 EOF 또는
Ctrl+C:

```bash
python main.py --video videos/demo.mp4 --no-display --pace fast
```

기준값 비교 — 면적·비율 기준을 바꿔 실행한다. 로그에는 입력 기준값과 실제 분석 크기로 보정한
값이 모두 기록된다:

```bash
python main.py --video videos/demo.mp4 --min-person-motion-pixels 2000 --min-person-motion-ratio 0.05
```

`--min-person-motion-pixels`는 기준 960 × 540 해상도에서의 값이고, 실제 분석 크기가 다르면
면적 비율로 자동 보정된다 (예: 640 × 360에서 3000 → 1334).

### 3.2 CLI 옵션 전체 (SPEC.md 22.5)

| 옵션 | 타입 | 기본값 |
| - | - | - |
| `--video` | 경로 | 필수 |
| `--model` | 경로 | `models/yolo11n.pt` |
| `--analysis-width` | 정수 | 960 |
| `--imgsz` | 정수 | 640 |
| `--classes` | 영문 종류 목록 | person |
| `--track` | 플래그 | false |
| `--tracker` | bytetrack 또는 botsort | bytetrack |
| `--tracking-profile` | stable 또는 baseline | stable |
| `--track-buffer` | 양의 정수 | stable 60 / baseline 30 |
| `--track-new-threshold` | 0~1 실수 | stable 0.4 / baseline 0.25 |
| `--track-min-hits` | 양의 정수 | stable 3 / baseline 1 |
| `--reid` | 플래그 | false |
| `--roi` | 0~1 실수 네 개 | 전체 화면 |
| `--capture-scope` | frame 또는 object | frame |
| `--person-confidence` | 실수 | 0.60 |
| `--min-person-motion-pixels` | 정수 | 3000 |
| `--min-person-motion-ratio` | 실수 | 0.03 |
| `--warmup-sec` | 실수 | 2.0 |
| `--cooldown-sec` | 실수 | 3.0 |
| `--capture-dir` | 경로 | `captures` |
| `--log-dir` | 경로 | `logs` |
| `--show-mask` | 플래그 | false |
| `--no-display` | 플래그 | false |
| `--pace` | realtime 또는 fast | realtime |
| `--fallback-fps` | 실수 | 없음 |
| `--debug-decisions` | 플래그 | false |

MOG2와 morphology 세부 설정(mog2_history, mog2_var_threshold, 커널 크기 등)은 `config.py`에서
관리하며 CLI로 노출하지 않는다. README와 CLI의 옵션 이름은 동일하게 유지한다. `--pace`는
realtime(원본 속도에 맞춰 가능한 범위에서 대기, 팀 시연용)와 fast(대기 없이 전 프레임 처리,
처리속도 측정과 회귀 테스트용) 중 하나다.

## 4. 결과 확인

실행마다 UTC 시각과 짧은 UUID 조합의 `run_id`(예: `20261008T001800123456Z_a13f90c2`)가
만들어지고, 결과는 아래 구조로 남는다. 별도 DB나 서버 전송, 캡처별 sidecar 파일은 없다.

```text
captures/<run_id>/event_<저장시각UTC>_f<프레임번호>_c<저장순서>.jpg
logs/<run_id>/run_config.json    # 실제 사용 설정과 패키지 버전
logs/<run_id>/events.jsonl       # 실행, 상태 변화, 캡처, 오류 이벤트
logs/<run_id>/summary.json       # 종료 사유, 처리 수, 속도, 캡처 수 요약
```

- JPG 파일명의 시각은 실제 저장 시각(UTC)이고 `f00000150`은 입력 영상 프레임 번호 150,
  `c000001`은 실행 내 성공 저장 순서다. 영상 시간은 로그 필드로 확인한다.
- JPG는 원본 프레임 전체를 원본 해상도로 저장하며 박스·라벨·상태 글자가 없다. 사람 crop이나
  분석 프레임 확대본이 아니다.

`events.jsonl`의 대표 이벤트: RUN_START, VIDEO_OPENED, MODEL_READY, STATUS_CHANGED,
FRAME_DECISION(`--debug-decisions` 시 매 프레임), CAPTURE_SAVED, METRICS, WARNING, ERROR,
RUN_END.

저장 성공 시 `CAPTURE_SAVED` 이벤트에 판정 근거가 남는다. 주요 필드:

| 필드 | 의미 |
| - | - |
| `video_time_sec` / `frame_index` | 저장된 프레임의 영상 시간과 프레임 번호 |
| `status` | 저장 시점 분석 상태 (예: MOVING_PERSON) |
| `capture_sequence` / `capture_path` | 실행 내 저장 순서와 JPG 경로 |
| `effective_min_person_motion_pixels` / `min_person_motion_ratio` | 실제 적용된 기준값 (보정 포함) |
| `matched_persons[]` | 적격 사람별 `confidence`, `motion_pixels`, `motion_ratio`, 분석/원본 박스 좌표 |

`matched_persons`의 `detection_index`는 해당 프레임 안에서만 유효한 번호다. 다른 캡처의 같은
index를 같은 사람으로 연결하지 않는다.

## 5. 화면 상태 읽기

OpenCV 기본 폰트는 한국어를 표시하지 않으므로 시연 화면 문구는 영어로 고정하고 문서와 로그
설명은 한국어로 제공한다.

분석 상태 6종 (우선순위 순):

| 상태 | 의미 |
| - | - |
| WARMUP | 초기 배경 학습 중 (기본 2초). 캡처 없음 |
| MOVING_PERSON | 적격 사람이 한 명 이상 — 사람 박스 안에서 충분한 배경 변화가 검출됨. 실제 보행·이동 속도를 확정하는 표시가 아니다 |
| PERSON_AND_MOTION_UNMATCHED | 사람도 있고 유효 전경도 있으나 기준 미달 (사람 바깥 변화, 작은 움직임, 비율 미달 포함) |
| PERSON_ONLY | 사람은 있고 유효 전경은 없음 |
| MOTION_ONLY | 사람은 없고 유효 전경은 있음 |
| IDLE | 사람과 유효 전경 모두 없음 |

저장 상태는 분석 상태와 별개다. 후보 없음 `CAPTURE NONE`, 후보이나 쿨다운 `COOLDOWN 2.1s`
형식, 저장 성공 `CAPTURE SAVED`. 저장 오류는 로그로 남기고 실행이 중단된다. MOVING_PERSON과
COOLDOWN이 동시에 표시되는 것은 정상이다 (조건은 만족했지만 저장 횟수 제한).

쿨다운은 실행 전체에서 하나이며 영상 시간 기준 3초다. 후보가 계속 유지되면 3초마다 저장될 수
있다. 이는 저장 주기 제한이지 같은 행동을 하나의 이벤트로 묶는 기능이 아니다.

## 6. 테스트 실행

단위 테스트는 면적·좌표·시간 계산이 핵심이라는 이유로 합성 마스크와 합성 프레임 기반 경계
테스트로 구성되어 있다 (SPEC.md 27.1). 실제 YOLO 모델을 로드하지 않는다. `object_detector`는
lazy import 구조이므로 ultralytics/torch가 없는 환경에서도 테스트가 동작한다.

```powershell
# Windows
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -v
```

```bash
# macOS / Linux 로컬 venv
.venv/bin/python -m pytest tests -v
```

제약: 합성 마스크 테스트는 결합 알고리즘(면적·비율·쿨다운·좌표 변환·계약 예외)을 검증하며
실제 영상에서의 person 검출 성공이나 MOG2 배경 학습 품질을 증명하지 않는다 (SPEC.md 27.2).
실행 성능과 검출 성능은 실제 영상 수동 시연(§27.3)과 테스트 세트로 평가한다.

## 7. 테스트 영상 세트와 평가 자료

### 7.1 필수 테스트 세트 (V01~V12, SPEC.md 25.3)

촬영 공통 조건: 카메라 고정, 초기 5초 빈 장면, 일정 FPS, 실제 크기·FPS와 움직임 구간 사람
라벨링. 정답 라벨은 알고리즘 출력 후에 유리하게 변경하지 않는다.

| ID | 장면 | 주요 기대 또는 측정 |
| - | - | - |
| V01 | 60초 빈 고정 장면 | warmup 이후 후보와 캡처가 없음 |
| V02 | 사람 진입과 보통 속도 걷기 | 사람 내부 전경 조건을 만족하고 원본 저장 |
| V03 | 사람이 천천히 이동 | 미탐 여부와 threshold 민감도 기록 |
| V04 | 사람이 멈춘 후 30초 유지 | 잔여 캡처, 배경 흡수와 안정화 시간 기록 |
| V05 | 사람 없는 장면에서 물체 이동 | person 검출이 없으면 후보 없음 |
| V06 | 사람 없는 장면에서 조명 변화 | person 검출이 없으면 후보 없음 |
| V07 | 정지한 사람과 멀리 떨어진 커튼 이동 | 사람 바깥 전경만으로 후보가 되지 않음 |
| V08 | 정지한 사람 뒤 커튼 또는 조명 변화 | 박스 내부 배경의 오탐 위험 측정 |
| V09 | 사람 두 명 중 한 명 이동 | 사람별 근거와 프레임당 한 번 저장 |
| V10 | 카메라 흔들림이나 장면 전환 | 지원 전제 위반에서 나타나는 오탐 기록 |
| V11 | 멀리 있는 작은 사람 | 최소 픽셀 기준으로 발생하는 미탐 확인 |
| V12 | 한글 경로와 영상 끝, ESC 종료 | 파일·종료 계약 확인 |

V10은 고정 카메라 전제에 대한 제한 테스트다. 이 오탐을 카메라 흔들림 제거 기능으로 해결하는
요구는 이번 범위에 포함하지 않는다.

### 7.2 권장 공개 데이터셋 (팀 평가용 참고 자료)

아래는 팀 내 평가에 활용할 수 있는 공개 데이터셋 추천이며 **프로젝트 결과물에 포함되지
않는다**. 다운로드와 라이선스 확인은 테스트 담당자가 별도로 수행한다.

| 데이터셋 | 활용 목적 |
| - | - |
| CDnet 2014 (Changedetection.net, Pedestrian 서브셋) | 고정 카메라 보행 장면에서 MOG2 오탐과 그림자 제거 검증 |
| CAVIAR | 걷기 → 정지 → 재이동 시나리오로 잔여 전경·안정화 시간과 후보 재발생 평가 |

성공 영상뿐 아니라 정지 직후 잔여 전경, 사람이 있는 조명 변화, 작은 사람 미탐 결과를 함께
제시해야 모듈 범위 안의 설정 조정인지 추가 알고리즘이 필요한 요구인지 판단할 수 있다
(SPEC.md 32.2).

## 8. 문제 해석과 알려진 제약

### 8.1 결과 해석 표 (SPEC.md 28.3)

| 관찰 | 먼저 확인할 항목 |
| - | - |
| 사람이 보이는데 박스 없음 | confidence, 가림, 조도, 사람 크기와 모델 입력 크기 |
| 박스는 있지만 내부 픽셀 0 | 좌표계, 그림자 제거와 배경 모델 흡수 |
| 픽셀은 있는데 후보 없음 | 실제 보정 면적과 비율, warmup |
| 후보인데 저장 안 됨 | 쿨다운과 저장 상태 |
| 멈춰도 계속 저장됨 | 잔여 전경, 배경 갱신과 박스 내부 배경 변화 |
| 조명 변경 때 사람 캡처 | 사람 박스 안 전체 변화와 frame_foreground_ratio |
| 같은 영상을 빠르게 돌리면 캡처 수 다름 | 실제 시각 쿨다운 사용 또는 시간 계산 오류 |

설정 조정은 한 번에 한 가지만 바꾼다. 우선 YOLO 사람 검출과 좌표 일치를 확인한 뒤
`min-person-motion-pixels`(1000/2000/3000)과 `min-person-motion-ratio`(0.03/0.05/0.10)을
비교하고, 선택한 설정을 별도 검증 영상에 다시 적용한다. 단일 장면에 맞춘 threshold로 모든
환경의 안정성을 주장하지 않는다.

### 8.2 알려진 제약 (SPEC.md 30)

| 상황 | 현재 모듈의 한계 |
| - | - |
| 정지 직후 사람 | 전경이 남아 주기적 캡처 가능 |
| 매우 느린 이동 | 배경 갱신으로 전경이 약해질 수 있음 |
| 먼 사람 | 검출 실패 또는 최소 픽셀 조건 미달 |
| 어두운 장면과 가림 | YOLO 사람 누락 가능 |
| 사람 뒤 커튼 | 사람 박스 내부 변화로 오탐 가능 |
| 사람이 있는 장면의 조명 변화 | person와 큰 전경이 함께 있어 오탐 가능 |
| 카메라 이동과 화면 전환 | 고정 배경 전제가 깨져 큰 전경 발생 |
| 사람 박스끼리 겹침 | 같은 픽셀의 근거가 여러 사람에 포함될 수 있음 |
| 긴 영상 | 연속 후보마다 쿨다운 주기로 JPG 증가 |
| VFR 영상 | frame_index / fps 시간이 실제 PTS를 정확히 표현하지 못함 |

이 한계는 threshold 하나로 모두 해결할 수 없다. tracking, optical flow, 행동 분석, 웹캠과
외부 AI는 추가 요구가 확정되기 전까지 넣지 않는다. 사람이 멈추는 즉시 캡처가 끝나는 것도
보장하지 않는다. 잔여 전경 지속 시간은 V04 테스트로 실측해 기록한다.
