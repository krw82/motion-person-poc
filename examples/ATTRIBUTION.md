# 예제 영상의 출처와 이용 조건

이 문서는 `examples/videos/`의 입력 MP4와 `examples/results/`의 처리 MP4·미리보기 PNG에 적용한다.
각 영상은 아래 원본의 Creative Commons BY-SA 조건에 따라 제공한다.
이 저장소에서 추가한 영상 변환·박스·번호·비교 화면도 해당 원본과 같은 조건으로 제공한다.
이 영상들의 이용 조건은 저장소의 Python 코드 전체에 적용하는 라이선스 선언이 아니다.
원저작자가 이 모듈이나 처리 결과를 보증한다는 뜻도 아니다.

## CAVIAR: OneStopNoEnter1front

- 저작자·출처: EC 지원 CAVIAR 프로젝트, IST 2001 37540.
- [공식 데이터셋·이용 조건](https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/).
- [사용한 JPEG 프레임 압축파일](https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA2/OneStopNoEnter1front/OneStopNoEnter1front.tar.gz).
- 공식 페이지의 이용 조건은 **Creative Commons BY-SA**다. 페이지가 버전을 명시하지 않으므로 임의로 3.0이나 4.0을 지정하지 않았다.
- 입력: `videos/caviar_stop_resume.mp4`. 공식 JPEG 시퀀스의 725프레임 전체를 384×288, 공식 25 FPS의 무음 MP4로 변환했다. OpenCV MP4V 변환 후 H.264로 압축했으므로 손실 압축이 적용됐다.
- 처리: `results/caviar_overlay_comparison.mp4`. 탐지·추적 번호·전경 강조를 추가하고 같은 프레임의 `full`, `objects`, `none` 표시를 나란히 배치했다. 비교 화면에서는 오른쪽 상태 패널을 잘라내고 제목과 캡처 상태를 별도로 추가했다.

CAVIAR 예제 데이터는 EC 지원 CAVIAR 프로젝트(IST 2001 37540)에서 제공한 자료다.

## Wikimedia Commons: Pro dog walker - Tokyo - 2024 Nov 1.webm

- 저작자: [Nesnad](https://commons.wikimedia.org/wiki/User:Nesnad).
- [원본 영상·저작자·이용 조건](https://commons.wikimedia.org/wiki/File:Pro_dog_walker_-_Tokyo_-_2024_Nov_1.webm).
- 라이선스: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).
- 입력: `videos/dog_walker_tokyo.mp4`. 원본 WebM에서 실제 디코딩된 271프레임 전체를 1920×1080, 30 FPS의 H.264 MP4로 변환하고 오디오를 제거했다. 손실 압축과 일정 FPS 변환이 적용됐다.
- 처리: `results/dog_walker_annotated.mp4`. 분석용으로 960×540으로 줄이고 `objects` 모드의 사람·개 박스, 추적 번호 및 오른쪽 상태 패널을 추가했다.

두 미리보기 PNG는 처리 MP4에서 각각 추출한 한 프레임이다.

원본 다운로드 URL, SHA-256, 입력 영상의 SHA-256·프레임 수·크기는 [sources.json](sources.json),
처리 결과의 SHA-256과 검증 내용은 [results/summary.json](results/summary.json)에 기록했다.
이 영상들을 다른 곳에 배포할 때도 저작자·출처·이용 조건·변경 사항을 함께 표시하고 같은 조건을 유지한다.
