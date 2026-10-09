"""Small Korean console helpers; no model or camera imports."""
from __future__ import annotations


def error_message(exc) -> str:
    messages = {
        "VIDEO_NOT_FOUND": ("영상 파일을 찾지 못했어요.", "파일 경로를 확인하고 공백이 있으면 따옴표로 감싸주세요."),
        "VIDEO_OPEN_ERROR": ("영상 입력을 열지 못했어요.", "웹캠이면 실행 앱의 카메라 권한을 허용하고, 카메라를 쓰는 다른 앱을 닫아주세요."),
        "VIDEO_DECODE_ERROR": ("영상에서 새 화면을 읽지 못했어요.", "카메라 연결 또는 영상 파일을 확인한 뒤 다시 실행해주세요."),
        "MODEL_NOT_FOUND": ("탐지 모델 파일이 없어요.", "./motion doctor로 모델 경로와 준비 방법을 확인해주세요."),
        "MODEL_LOAD_ERROR": ("탐지 모델을 불러오지 못했어요.", "./motion doctor로 설치 상태와 모델 파일을 확인해주세요."),
        "MODEL_CLASS_ERROR": ("모델에서 요청한 대상을 찾지 못했어요.", "사람·개는 --objects 사람 개로 선택할 수 있어요. 다른 대상은 모델의 영문 이름을 사용해주세요."),
        "CONFIG_ERROR": ("설정값을 확인해주세요.", "./motion webcam --help 또는 ./motion video --help를 확인해주세요."),
        "DISPLAY_ERROR": ("미리보기 창을 표시하지 못했어요.", "화면이 있는 로컬 터미널에서 실행해주세요. 영상 파일은 --no-display로 검사할 수 있어요."),
        "CAPTURE_SAVE_ERROR": ("캡처 파일을 저장하지 못했어요.", "저장 폴더 권한과 남은 디스크 공간을 확인해주세요."),
        "LOG_WRITE_ERROR": ("실행 로그를 저장하지 못했어요.", "로그 폴더 권한과 남은 디스크 공간을 확인해주세요."),
    }
    title, hint = messages.get(exc.code, ("처리 중 오류가 발생했어요.", "아래 상세 내용을 확인해주세요."))
    return f"{title}\n해결 방법: {hint}\n상세 [{exc.code}]: {exc.message}"
