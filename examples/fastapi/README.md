# 같은 PC의 FastAPI에 모션 모듈 연결하기

웹캠은 FastAPI가 실행되는 PC에 연결한다. 서버가 카메라 하나와 모델 하나를 소유하고,
React·Streamlit은 같은 미리보기와 사건 사진을 조회한다. 동영상은 파일을 한 번 업로드한다.
이 예제는 소스 저장소에서 실행하며 Python wheel에는 포함되지 않는다.

## 실행

저장소 루트에서 Python 3.11 이상과 가상환경을 준비한다. YOLO 모델은 `models/yolo11n.pt`에 둔다.
모델이 없으면 `.venv/bin/python -c "from ultralytics import YOLO; YOLO('models/yolo11n.pt')"`로 준비한다.

```bash
.venv/bin/python -m pip install -e ".[yolo,web]"
.venv/bin/python -m uvicorn examples.fastapi.app:app --host 127.0.0.1 --port 8000 --workers 1
```

React 화면은 다른 터미널에서 실행한다. Node.js 20.19+ 또는 22.12+를 사용한다.

```bash
cd examples/react
npm ci
npm run dev
```

`http://127.0.0.1:5173`을 연다. 5173이 사용 중이면 `npm run dev -- --port 5174`로 실행한다.
예제의 CORS는 localhost/127.0.0.1의 5173·5174만 허용한다. 다른 프론트 주소는 호스트의 CORS를 수정한다.
화면에서 ‘카메라 시작’ 또는 ‘동영상 분석’을 선택한다. 업로드 원본 재생은 `<video>`의 로컬 재생이며
모션 분석의 시각을 이동시키지 않는다.

Streamlit을 선택하면 저장소 루트에서 다음 명령을 사용한다.

```bash
.venv/bin/python -m pip install -e ".[streamlit]"
.venv/bin/python -m streamlit run examples/streamlit/app.py
```

Streamlit이 다시 실행돼도 카메라를 직접 열지 않는다. 시작 버튼은 FastAPI에 요청하며 동일 설정의
실행이 이미 있으면 같은 run_id를 받는다. 두 화면을 함께 열어도 같은 FastAPI 실행을 조회한다.
웹캠은 OS에서 Python 실행 환경의 카메라 접근을 허용해야 한다.

## 호스트 API

| 요청 | 용도 |
| --- | --- |
| `POST /motion/videos` | multipart의 file 필드로 동영상 업로드. 불투명 video_id 반환 |
| `POST /motion/runs` | source(webcam/video), camera, objects, video_id, mirror, overlay, fast 지정 |
| `GET /motion/status` | 현재 실행 상태, 프레임/사건/묶음 수, 최종 요약 |
| `POST /motion/runs/{run_id}/stop` | 처리 종료 요청 |
| `GET /motion/runs/{run_id}/preview.mjpg` | React `<img>`에 쓰는 MJPEG |
| `GET /motion/runs/{run_id}/preview.jpg` | Streamlit의 주기적 JPEG 조회 |
| `GET /motion/runs/{run_id}/events` | 최근 묶음 메타데이터와 사진 URL, 외부 분석 상태 |
| `GET /motion/events/{event_id}/parts/{part}/images/{image}` | 완료 묶음에 등록된 JPEG만 제공 |

미리보기 준비 중 JPEG 요청은 503일 수 있다. 미리보기는 no-store이며, 화면 연결 종료는
카메라 실행을 중지시키지 않는다. ‘중지’ 요청이나 FastAPI lifespan 종료가 실행을 닫는다.
사건은 event_id 아래 part_index 순서로 표시한다. MAX_DURATION의 episode_end=False는 후속 묶음이 있다는 뜻이다.

## Gemini를 연결할 위치

기본 예제는 사진 저장까지 실행하고 외부 분석 상태를 ready로 표시한다. API 키나 유료 분석 요청을 생성하지 않는다.
실제 프로젝트의 분석 함수를 `create_app(analyze=...)`에 주입한다. 함수는 모듈 처리 스레드와 분리된
호스트 작업 스레드에서 실행되며 JSON으로 반환할 수 있는 값을 돌려준다.

```python
from examples.fastapi.app import create_app
from my_project.gemini import analyze_motion_photos

def analyze_event(bundle):
    return analyze_motion_photos(
        photos=bundle.image_paths,  # 시간순, 최대 6장
        metadata=bundle.to_dict(),  # 촬영 시각·역할·객체·품질 포함
    )

app = create_app(analyze=analyze_event)
```

`my_project.gemini`는 연결 프로젝트가 구현할 함수다. 여러 사진을 한 요청에 보내는 방법은
[Gemini image understanding](https://ai.google.dev/gemini-api/docs/image-understanding)을 참고한다.
사전/사후 부족과 증거 누락을 프롬프트에 알려 모델이 관측하지 못한 장면을 확정하지 않도록 한다.
키·모델 선택·네트워크 시간 제한·결과 검증·재시도는 그 함수가 소유한다.

모듈의 `on_event`를 직접 쓰는 프로젝트도 콜백에서 네트워크 응답을 기다리지 않고 작업 등록만 수행한다.
중복 등록은 run_id/event_id/part_index로 구분한다. 외부 작업을 정확히 한 번 수행하거나 재시작 후 복구해야
하는 경우 프로젝트의 기존 작업 큐/저장소를 사용한다.

## 예제의 보관 정책

최근 32묶음의 메타데이터만 메모리에 두며 새 실행 때 초기화한다. 분석 대기열은 8개로 제한하고
가득 차면 failed로 표시한다. 이미 저장한 사진과 manifest는 남는다. 업로드는 파일당 최대 256 MiB이며
원본과 사진은 `outputs/web/` 아래 저장한다. 파일 삭제와 재시작 후 이력 복원은 이 예제에 구현하지 않았다.
장기 보관·업로드 전체 용량 제한은 실제 프로젝트의 정책으로 연결한다.

초기 운영은 localhost·카메라 1개·Uvicorn worker 1개다. 원격 접근·인증·여러 카메라·여러 서버가 필요해지면
호스트 계층에서 확장한다. Streamlit/React 실행마다 별도의 MotionEngine을 생성하는 방식은 피한다.
