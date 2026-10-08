# 추적 분리·관측 확인·분석 ROI 개선 시험

영상·캡처·로그·환경 기록은 공개 저장소에 포함하지 않는다. 아래 경로는 로컬 재현 시 생성되는 위치다.

2026년 10월 8일, 현재 YOLO11n·Ultralytics·OpenCV MOG2·NumPy·lap을 유지하며 개선했다.
설치 버전은 바꾸지 않았고 Ultralytics 8.4.174를 requirements.in에 고정했다.
YOLO 가중치와 두 사용자 입력 파일의 SHA256은 기존 시험과 동일하다.
최종 코드에서 11회·4,555프레임·99장 저장을 검사했다.
자동 테스트 117개 및 pip check 통과, 모든 최종 실행은 전체 입력 처리와 저장 계약 검사를 통과했다.
장면 전환은 사용자 요청대로 성능 판단에서 제외하며 각 고정 카메라 장면 안의 결과를 본다.

## 1. 변경 사항

- YOLO는 프레임당 한 번 추론하고 종류별 독립 ByteTrack/BoT-SORT에 결과를 전달한다.
  클래스가 없는 프레임도 해당 추적기를 진행한다. 박스 검출 인덱스와 외형 특징의 인덱스를
  원래 추론 목록에 대응시켜 사람·개가 연결 단계에서 섞이지 않게 했다.
- 기본 stable 프로필: 추적 유지 60 업데이트, 새 번호 신뢰도 0.4, 사용자 신뢰도 하한을
  통과한 관측 3회 이후 캡처. 회색 PENDING 박스는 저장을 발생시키지 않는다.
  누적 관측 확인이므로 확정 ID가 잠깐 놓쳤다가 같은 번호로 복구되면 다시 3회를 요구하지 않는다.
- --track-buffer, --track-new-threshold, --track-min-hits로 값을 조정한다.
  --tracking-profile baseline은 이전 공유 추적 연결과 30/0.25/1 조건으로 비교한다.
- --tracker botsort --reid로 기존 YOLO 자체의 64차원 특징을 사용하는 연결을 시험했다.
  별도 재식별 가중치를 추가하지 않았다. 고정 카메라 보정은 gmc_method=none이다.
  실제 사람·개 특징 추출은 별도 확인 기록 (`test_data/tracking_improvements/native_reid_smoke.json`)에 남겼다.
- --roi는 선택 영역을 YOLO·MOG2에 함께 전달한다. 원본 좌표 로그에는 영역 시작 위치를
  반영하고 저장 JPG는 원본 전체 720×1280이다. 표시 영상은 분석 영역을 보여준다.
- 실제 tracker.yaml을 실행 로그 폴더에 저장하고 confirmed/observation_hits/TRACK_PENDING을 기록한다.

## 2. 실제 비교

Mac Apple M1 CPU, OMP_NUM_THREADS=1/MKL_NUM_THREADS=1, GUI 없이 전체 프레임 처리.
사용자 영상은 confidence 0.4, 기준 전경 픽셀 1000, 전경 비율 0.03, 워밍업 2초,
객체별 쿨다운 0.5초, imgsz 640이며 크기 비교만 960이다. 한 명 통행은 기존 0.6/3000을 유지했다.
ROI만 분석 면적과 모델 입력 모양이 달라지고 실제 최소 전경 픽셀도 분석 면적에 따라 보정된다.

| 시험 | 프레임 | JPG | 개 관측 프레임 | FPS | 확정 번호 수 |
| - | -: | -: | -: | -: | - |
| 한 명 통행 / 기존 탐지 | 369 | 17 | 0 | 25.05 | 추적 없음 |
| 한 명 통행 / stable | 369 | 17 | 0 | 26.17 | person: 1 |
| 3080 / 이전 방식 | 258 | 8 | 83 | 16.88 | person: 3, dog: 3 |
| 3080 / stable 60 | 258 | 8 | 83 | 18.71 | person: 2, dog: 2 |
| 3080 / stable 90 | 258 | 8 | 83 | 18.96 | person: 2, dog: 2 |
| 3080 / BoT-SORT ReID 90 | 258 | 8 | 83 | 18.10 | person: 2, dog: 2 |
| 5077 / 이전 방식 | 557 | 6 | 167 | 19.02 | dog: 4 |
| 5077 / stable 60 | 557 | 5 | 167 | 19.32 | dog: 3 |
| 5077 / ROI + stable 90 | 557 | 8 | 179 | 15.93 | dog: 3 |
| 5077 / ROI + ReID 90 | 557 | 8 | 179 | 18.30 | dog: 3 |
| 5077 / stable 60 + imgsz 960 | 557 | 6 | 179 | 12.15 | dog: 2 |

관측 프레임은 신뢰도 기준을 통과한 개 박스가 하나 이상 있었던 프레임이다.
확정 번호 수와 실제 개체 수는 다르다. 번호가 늘거나 줄었다는 사실만으로 ID 정확도를 판단하지 않는다.
FPS는 단회 측정치이며 워밍업 이후 처리 속도다. 시스템 부하와 실행 순서의 변동이 있어
ReID가 더 빠르다는 결론이나 안정적인 실시간 성능 보장을 도출하지 않는다.

## 3. 사용자 영상별 해석

### 3080 — 야간 아기와 개

stable에서도 개로 8장 저장, 사람 트리거 0회를 유지했다. 사람 원시 번호는 3→2로 줄었지만
아기의 중복 person 박스가 완전히 사라진 것은 아니다. stable의 dog 원시 번호는 3개이고
그중 한 번호는 1회 관측으로 미확정, 나머지 두 번호가 3장·5장 저장을 발생시켰다.
유지 시간을 90으로 늘리거나 외형 연결을 켜도 이 영상에서 재등장하는 개의 번호를 하나로
유지하지 못했다. 파라미터와 ReID만으로 영구적인 개체 식별을 해결했다고 보지 않는다.

### 5077 — 야간 작은 개

이전 방식 6장, stable 기본 5장, ROI + stable 90은 8장이다. stable 기본에서는
5.93초의 한 프레임 관측이 미확정이라 저장되지 않았다. 관측 확인은 짧은 노이즈를 억제하지만
빠르게 스쳐가는 실제 객체의 짧은 관측도 놓칠 수 있는 선택이다. 필요하면 --track-min-hits 1로 비교한다.
ROI에서는 원시 dog 번호가 4→3, 개 관측 프레임이 167→179로 바뀌었다.
먼쪽 개의 첫 관측은 15.60→13.60초로 앞당겨졌고 ROI의 첫 먼쪽 캡처는 13.67초였다.
일부 초기 근접 구간은 관측이 줄고 먼쪽에서도 탐지 공백과 번호 변경이 남아 있다.
ROI + ReID는 ROI + ByteTrack과 같은 관측 수·번호 수·8장 저장이었다.
이 파일에서 ReID 활성화의 실질적인 이득은 확인되지 않아 기본은 ByteTrack으로 유지했다.

분석 ROI는 정규화 (0, 0.285, 1, 0.755), 원본 픽셀 (0, 364, 720, 967)이다.
이 제공 파일의 자막·검정 여백을 제외하면서 CCTV 영상 영역을 포함하도록 사람이 지정했다.
일반 카메라 영상에 이 값을 일괄 적용하거나 장면별로 자동 조정하지 않는다.
imgsz 960 비교는 같은 영상에서 선택지를 확인하는 탐색이며 독립 검증 세트의 결과가 아니다.

## 4. 저장·연결 검증

매 프레임 로그·전체 디코드 프레임 수·EOF·실제 JPG 수·요약 수를 대조했다.
모든 저장 JPG를 원본 해당 프레임의 JPEG quality 95 재인코딩 바이트와 비교해 일치를 확인했다.
객체별 0.5초 간격, 워밍업 무캡처, 미확정 객체 무캡처, 선택 종류 제한,
성공 저장 카운터 및 최종 소스 코드 SHA256 일치를 검사했다.
stable 실행의 원시 추적 번호가 다른 종류로 바뀐 기록은 0이다.
합성 시험에서는 동일 위치의 dog/person 및 낮은 신뢰도 다른 종류가 번호를 빼앗지 못하게
검사했고, 같은 종류의 낮은 신뢰도 연결·빈 프레임의 만료·원래 검출 인덱스·재식별 특징
대응·공식 콜백의 빈 결과 처리·ROI 좌표와 전체 원본 저장을 자동 테스트했다.

독립 객체 박스·동일 개체 ID·정지/이동 구간 정답을 작성하지 않았으므로
precision/recall, IDF1, 실제 물리 개체별 캡처 간격의 합격을 주장하지 않는다.
MOG2 전경은 속도 측정이 아니며, 추적 번호가 바뀌면 새 번호의 쿨다운이 시작된다.
실제 Windows 실행과 변경된 GUI 창 조작은 이번에 검증하지 않았다.
박스·번호 MP4는 검증된 로그와 원본 프레임으로 재구성한 진단 영상이다.

## 5. 결과와 직접 실행

### 3080 / stable 60

- 박스·번호 영상 (`test_data/tracking_improvements/tiktok_7675232379845283080_stable_annotated.mp4`)
- 원본 JPG 폴더 (`captures/tracking_improvements/tiktok_7675232379845283080_stable/20261008T035035110417Z_8663b50c`)
- 실행 로그 (`logs/tracking_improvements/tiktok_7675232379845283080_stable/20261008T035035110417Z_8663b50c`)

### 5077 / ROI + stable 90

- 박스·번호 영상 (`test_data/tracking_improvements/tiktok_7622221281492045077_roi_annotated.mp4`)
- 원본 JPG 폴더 (`captures/tracking_improvements/tiktok_7622221281492045077_roi/20261008T035225929775Z_b74bd8d6`)
- 실행 로그 (`logs/tracking_improvements/tiktok_7622221281492045077_roi/20261008T035225929775Z_b74bd8d6`)


프로젝트 폴더에서 첫 영상:

```bash
.venv/bin/python main.py --video "videos/tiktok-7675232379845283080.mp4" --model models/yolo11n.pt --classes person dog --track --capture-scope object --cooldown-sec .5 --confidence .4 --min-object-motion-pixels 1000 --show-mask --debug-decisions
```

두 번째 영상의 ROI 비교:

```bash
.venv/bin/python main.py --video "videos/tiktok-7622221281492045077.mp4" --model models/yolo11n.pt --classes person dog --track --capture-scope object --cooldown-sec .5 --confidence .4 --min-object-motion-pixels 1000 --roi 0 .285 1 .755 --track-buffer 90 --show-mask --debug-decisions
```

처리 상태는 오른쪽 패널, 미확정 번호는 회색 PENDING, 적격 객체는 노란 MATCH다.
캡처는 captures/<run_id>/, 번호별 상태와 설정은 logs/<run_id>/에서 확인한다.
원본 CCTV 파일에 여백이 없다면 --roi 옵션을 생략한다.

재현 기록: 명령·해시·패키지 버전 (`test_data/tracking_improvements/runs.json`),
모든 계약 검사·객체별 상태 (`test_data/tracking_improvements/evaluation.json`).
tools/run_tracking_improvement_tests.py --videos <파일1> <파일2>로 같은 시험을 실행하고,
tools/evaluate_tracking_tests.py --results-dir test_data/tracking_improvements로 검사·표시 영상을 만든다.
모두 기존 가상환경의 Python으로 실행한다.

## 6. 소스 참고

추적 알고리즘은 기존 설치된 라이브러리 API를 사용하며 프로젝트 어댑터는 연결을 종류별로 분리한다.
[Ultralytics 8.4.174 ByteTrack](https://github.com/ultralytics/ultralytics/blob/v8.4.174/ultralytics/trackers/byte_tracker.py),
[BoT-SORT](https://github.com/ultralytics/ultralytics/blob/v8.4.174/ultralytics/trackers/bot_sort.py),
[추적 콜백](https://github.com/ultralytics/ultralytics/blob/v8.4.174/ultralytics/trackers/track.py),
[SORT의 관측 확인](https://github.com/abewley/sort/blob/master/sort.py)을 검토했다.
동일 위치만으로 새 번호를 과거 번호에 강제로 합치는 재연결은 이번 구현에 넣지 않았다.
