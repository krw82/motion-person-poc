# YOLO와 MOG2 공개 영상 테스트 결과

영상·캡처·로그·환경 기록은 공개 저장소에 포함하지 않는다. 아래 경로는 로컬 재현 시 생성되는 위치다.

2026년 10월 8일, 로컬 CPU에서 실제 YOLO11n 추론과 MOG2를 연결해 9회 실행, 총 6,242프레임을 처리했다. 실행·저장 계약은 모두 통과했지만 기본값의 CAVIAR 검출 성능은 부족하다. 기능 smoke test는 통과했고 알고리즘 성능 인수는 미통과 상태다.

## 1 환경과 보완 사항

- Apple M1, RAM 16 GiB, macOS-26.2-arm64-arm-64bit, Python 3.12.14.
- OpenCV 5.0.0, NumPy 2.5.3, Ultralytics 8.4.174, PyTorch 2.14.1.
- CPU, imgsz 640, confidence 0.60, 최소 전경 3000 기준 픽셀, 비율 0.03, warmup 2초, cooldown 3초. 마지막 비교 실행만 최소 전경을 1000으로 변경했다.
- 화면 없이 fast 모드, 매 프레임 판정 로그 사용. 표시·재생 대기를 제외한 처리 FPS이며 Windows나 GUI 성능을 나타내지 않는다.
- 쿨다운 경계의 절대 오차를 1 ns까지 허용했다. 24/25/30/60 FPS의 정확한 3초 경계와 그 직전 프레임을 회귀 테스트했다.
- 면적 계산에 설정한 기준 해상도를 전달하고 정수 연산으로 올림을 계산한다.
- 메타데이터와 실제 읽은 프레임 수가 일치하는 종료와 완료를 확인할 수 없는 종료를 구분한다. 명시적인 디코더 예외는 오류이며, 판별 불가 종료는 경고와 input_completion_verified=false를 남긴다.
- 문서에 최소 구간 수·검출 지연·오탐 빈도를 포함한 초기 성능 인수 프로필을 추가했다.
- 자동 테스트 89개 통과. pip check 통과. 대상 Windows 실행은 미검증이다.

모델 SHA256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`. 패키지 잠금은 `test_data/public_test_results/requirements.macos-py312.lock.txt`이며 Windows용 잠금 파일로 사용하지 않는다.

## 2 실제 실행 결과

| 시험 | 처리 프레임 | 후보 프레임 | JPG | 움직임 구간 검출 | 처리 FPS |
| --- | ---: | ---: | ---: | --- | ---: |
| CAVIAR Walk1 | 611 | 14 | 1 | 1/2 | 22.99 |
| CAVIAR Meet WalkSplit | 623 | 1 | 1 | 1/2 | 25.75 |
| CAVIAR Stop Resume 기본값 | 725 | 0 | 0 | 0/3 | 24.21 |
| CDnet pedestrians | 1099 | 594 | 9 | 별도 정답 구간 없음 | 25.80 |
| CDnet fountain01 | 1184 | 0 | 0 | 별도 정답 구간 없음 | 25.91 |
| 합성 빈 화면 | 300 | 0 | 0 | 별도 정답 구간 없음 | 20.51 |
| 합성 조명 변화 | 300 | 0 | 0 | 별도 정답 구간 없음 | 22.91 |
| 실제 프레임 정지 후 재생 | 675 | 188 | 4 | 2/2 | 25.85 |
| CAVIAR Stop Resume 면적 1000 | 725 | 161 | 4 | 3/3 | 25.08 |

기본 CAVIAR 3개 영상의 라벨 움직임 구간은 7개이며 2개에서 후보가 발생했다(28.6%). 정상 보행 검증 구간 20개를 충족하지 않고 상향 시점·작은 사람이 포함된 표본이므로 이 수치를 일반적인 정확도나 90% 목표 달성률로 해석하지 않는다.

## 3 실패와 해석

**CAVIAR INRIA 상향 시점에서 YOLO 누락이 크다.** Walk1은 611프레임 중 사람 검출이 24프레임, Meet WalkSplit은 623프레임 중 1프레임이다. 진단 이미지의 청록색은 공식 사람 박스, 노란색은 confidence 0.60 이상 YOLO 박스다. 실제 사람이 보이지만 박스가 반환되지 않는 프레임이 확인된다. 최소 전경 기준만 낮춰서는 이 누락을 해결할 수 없다. 정면 시점 영상, 다른 모델 또는 confidence 비교를 별도 검증 세트에서 평가해야 한다.

**CAVIAR 정면 재이동 영상에는 면적 기준이 지나치게 높다.** 384×288에서 기본 3000은 실제 640픽셀이며 후보가 0이다. 1000으로 변경하면 실제 214픽셀이 되어 후보 161프레임, JPG 4장, 움직임 구간 3/3 검출이다. 이 4장 중 3장은 움직임 구간, 1장은 active로 표기된 ambiguous 구간에 있다. 검출 지연은 1.72, 0.00, 2.28초여서 지연 목표까지 달성한 결과는 아니다. 같은 영상에서 조정한 탐색 결과이며 기본값을 변경하거나 독립 검증 통과로 처리하지 않았다.

**멈춘 즉시 무캡처는 보장되지 않는다.** 실제 CDnet 프레임을 10.0~22.0초에 고정한 실험에서 정지 중 후보 47프레임, 11.0초의 잔여 JPG 1장이 발생했다. 마지막 후보는 정지 후 1.84초다. 22.04초에는 재생된 움직임에서 다시 저장됐다. 모든 픽셀을 고정한 대조 실험이며 사람이 실제로 멈춘 상태에서 배경이 변하는 환경의 안정화 시간을 대표하지 않는다. 12초 정지 중 1장으로 초기 정지 오탐 목표인 분당 2개 이하를 만족한다고 주장할 수 없다.

**분수 배경에는 MOG2 오탐이 있어도 사람 조건이 캡처를 억제했다.** fountain01의 후처리 마스크 F1은 0.088로 낮았으나 최종 후보와 JPG는 모두 0이었다. 빈 화면과 사람 없는 조명 변화에서도 후보와 JPG가 0이었다. 사람이 함께 있는 조명 변화·커튼·그림자 조건은 별도의 실제 영상 검증이 남아 있다.

## 4 MOG2 픽셀 평가

| CDnet 영상 | Precision | Recall | F1 | IoU |
| --- | ---: | ---: | ---: | ---: |
| pedestrians | 0.866 | 0.891 | 0.878 | 0.783 |
| fountain01 | 0.048 | 0.557 | 0.088 | 0.046 |

공식 temporalROI 범위(양끝 포함)와 공간 ROI를 적용했다. GT 255는 양성, 0과 50은 음성으로 사용하고 85·170은 제외했다. 그림자 50은 전경으로 검출하면 오탐이다. 수치는 모든 전경의 후처리 마스크 평가이며 YOLO 사람 정확도 또는 캡처 성공률이 아니다.

## 5 데이터와 평가 범위

- [CAVIAR 공식 데이터](https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/): EC Funded CAVIAR project IST 2001 37540의 공개 JPEG 프레임과 XML. Walk1 611장, Meet WalkSplit 623장, OneStopNoEnter1front 725장. 25 FPS로 변환하고 XML의 프레임 0과 첫 이미지 프레임을 맞췄다. 원본 MPEG은 보고 프레임 수가 달라 평가 입력으로 사용하지 않았다.
- [CDnet 2014 공식 데이터](https://changedetection.net/dataset2014/): baseline/pedestrians 1099장, dynamicBackground/fountain01 1184장. 이미지 묶음을 시험용 25 FPS CFR MP4로 변환했다. **CDnet 원본 촬영 FPS는 확인되지 않았으며, 여기서 쿨다운 초 단위와 캡처/분은 지정한 시험 타임베이스 기준이다.**
- MP4는 원본 크기를 유지한 mp4v 재인코딩이며 손실 압축이다. 원본 JPEG에 직접 추론한 공식 벤치마크 점수와 동일하다고 주장하지 않는다.
- CAVIAR의 objectlist만 사용한다. 최상위 hypothesis의 walking/movement는 양성, inactive는 음성, active·상충·정보 부족은 ambiguous다. 화면에 여러 사람이 있으면 한 명이라도 움직이는 프레임을 양성으로 한다. warmup과 ambiguous를 제외하며 제외 수를 JSON에 남긴다. CAVIAR의 정지 사람 라벨은 완전하지 않으므로 전체 화면 사람 Precision은 계산하지 않았다.
- 정답 구간 검출과 JPG 저장은 각각 계산한다. 실패 구간은 검출 지연 평균에 0으로 넣지 않는다. 결과 JSON에 성공 구간 지연만 있으며 구간 성공 수를 함께 읽어야 한다.
- 전체 캡처 시간과 판정 근거는 각 실행의 events.jsonl 및 `runs.json`에 있는 경로에서 확인한다.

## 6 저장 계약 검증과 재현

9회 실행 모두 정상 종료와 전 프레임 처리, 매 프레임 판정, warmup 무저장, 성공 저장 간 3초 쿨다운, 프레임당 JPG 1개, 로그·파일 수 일치, 원본 해상도를 확인했다. 저장 JPG를 같은 원본 프레임에서 같은 품질로 재인코딩한 바이트와 대조해 overlay가 들어가지 않았음을 확인했다. 전체 검사 통과: True.

프로젝트 루트에서 기존 가상환경과 `models/yolo11n.pt`를 사용한다.

```bash
.venv/bin/python tools/download_public_samples.py
.venv/bin/python tools/prepare_public_samples.py
.venv/bin/python tools/run_public_samples.py
.venv/bin/python tools/evaluate_public_samples.py
.venv/bin/python tools/build_public_test_report.py
.venv/bin/python -m pytest tests -q
```

Windows에서는 `.venv/bin/python` 대신 `.\.venv\Scripts\python.exe`를 사용한다. 데이터 다운로드가 필요한 최초 재현에는 인터넷이 필요하고 분석은 로컬에서 수행한다.

원본 URL·SHA256은 `test_data/public_sources/manifest.json`, 변환 영상 정보는 `test_data/public_test_manifest.json`, 실행 경로는 `test_data/public_test_results/runs.json`, 지표는 `evaluation.json`, 환경과 최종 소스 해시는 `environment.json`에 기록했다. 최종 소스 해시는 실행 후 스냅샷이며 실행 당시 해시를 소급 추정한 것이 아니다.

실제 Windows 설치·한글 경로·화면 표시와 ESC/창 닫기, 사람과 조명·그림자가 함께 있는 실제 영상, 독립 검증 정상 구간 20개 이상은 별도 검증이 필요하다. MOT17과 VIRAT는 이번 소규모 시험에 포함하지 않았다.
