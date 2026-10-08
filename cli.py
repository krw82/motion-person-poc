"""Task-oriented CLI. Existing main.py/webcam_preview.py options remain available."""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models/yolo11n.pt"
COMMANDS = {"webcam", "preview", "video", "run", "doctor"}
TARGETS = {"사람": "person", "아기": "person", "개": "dog", "강아지": "dog", "고양이": "cat"}
TARGET_LABELS = {"person": "사람", "dog": "개", "cat": "고양이"}


class FriendlyHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        super().add_usage(usage, actions, groups, "사용법: " if prefix is None else prefix)


class FriendlyParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", FriendlyHelpFormatter)
        super().__init__(*args, **kwargs)
        self._positionals.title = "명령 / 입력"
        self._optionals.title = "옵션"
        for action in self._actions:
            if action.dest == "help":
                action.help = "도움말 보기"

    def error(self, message):
        message = message.replace("unrecognized arguments:", "알 수 없는 옵션:")
        message = message.replace("invalid choice:", "지원하지 않는 선택:")
        message = message.replace("the following arguments are required:", "필수 입력:")
        message = message.replace("(choose from", "(선택:")
        self.print_usage(sys.stderr)
        self.exit(2, f"입력을 확인해주세요: {message}\n도움말: {self.prog} --help\n")


def positive_int(value):
    try:
        number = int(value)
        if number <= 0:
            raise ValueError
        return number
    except ValueError:
        raise argparse.ArgumentTypeError("1 이상의 정수를 입력해주세요.") from None


def camera_index(value):
    try:
        number = int(value)
        if number < 0:
            raise ValueError
        return number
    except ValueError:
        raise argparse.ArgumentTypeError("카메라 번호는 0 이상의 정수예요.") from None


def nonnegative_float(value):
    try:
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError
        return number
    except ValueError:
        raise argparse.ArgumentTypeError("0 이상의 유한한 숫자를 입력해주세요.") from None


def confidence(value):
    number = nonnegative_float(value)
    if number > 1:
        raise argparse.ArgumentTypeError("신뢰도는 0~1 사이로 입력해주세요.")
    return number


def normalize_targets(values):
    targets = []
    for value in values:
        for part in value.split(","):
            name = part.strip().lower()
            if not name:
                raise ValueError("대상 이름이 비어 있어요. 예: --objects 사람 개")
            targets.append(TARGETS.get(name, name))
    return list(dict.fromkeys(targets))


def input_path(value: str) -> Path:
    """Accept a literal path, a quoted path, or a Finder-dragged escaped path."""
    raw = value.strip()
    literal = Path(raw).expanduser()
    if literal.is_file():
        return literal
    # Windows paths use literal backslashes, not POSIX shell escape sequences.
    if os.name == "nt" and len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
        return Path(raw[1:-1]).expanduser()
    if raw.startswith(('"', "'")) or "\\ " in raw:
        try:
            words = shlex.split(raw)
            if len(words) == 1:
                return Path(words[0]).expanduser()
        except ValueError:
            pass
    return literal


def build_parser():
    parser = FriendlyParser(
        prog="./motion",
        description="영상 이벤트 감지 — 웹캠 확인 또는 영상 파일 분석",
        epilog='바로 시작:\n  ./motion webcam                 웹캠 화면 확인 · 저장 없음\n'
               '  ./motion video "videos/demo.mp4" 영상 분석 · 움직임 캡처\n'
               '  ./motion doctor                 설치·모델 확인\n'
               '  ./motion                        메뉴에서 선택\n'
               'Windows에서는 ./motion 대신 .\\motion.cmd를 사용하세요.',
    )
    commands = parser.add_subparsers(dest="command", title="할 일", metavar="{webcam,video,doctor}")
    webcam = commands.add_parser("webcam", aliases=["preview"], help="웹캠 화면 확인 · 저장 없음",
                                 description="탐지 박스와 움직임을 화면에서 확인해요. 사진·영상·로그는 저장하지 않아요.")
    video = commands.add_parser("video", aliases=["run"], help="영상 파일 분석 · 움직임 캡처",
                                description="영상 경로를 넣으면 사람 탐지·추적과 움직임 캡처를 시작해요.")
    video.add_argument("path", nargs="?", metavar="영상경로", help="영상 파일 경로. 생략하면 입력을 안내해요.")
    for mode in (webcam, video):
        mode.add_argument("--objects", "--classes", nargs="+", default=["person"], metavar="대상",
                          help="분석할 대상 (기본 사람). 예: --objects 사람 개")
        mode.add_argument("--mask", "--show-mask", action="store_true", help="움직임 마스크도 표시")
        mode.add_argument("--overlay", choices=["full", "objects", "none"], default="full",
                          help="화면 표시: full 전체, objects 객체 박스만, none 박스·패널 숨김 (기본 full)")
        tuning = mode.add_argument_group("탐지 조정 (필요할 때만)")
        tuning.add_argument("--model", type=Path, default=DEFAULT_MODEL, metavar="파일", help="모델 경로 (기본 models/yolo11n.pt)")
        tuning.add_argument("--confidence", type=confidence, default=.4, metavar="0~1", help="탐지 신뢰도 하한 (기본 0.4)")
        tuning.add_argument("--imgsz", type=positive_int, default=640, metavar="크기", help="추론 크기 (기본 640, 세부 비교 960)")
        tuning.add_argument("--warmup-sec", type=nonnegative_float, default=2., metavar="초", help="배경 초기 학습 시간 (기본 2초)")
        tuning.add_argument("--min-object-motion-pixels", type=positive_int, default=1000, metavar="픽셀",
                          help="기준 해상도의 최소 전경 면적 (기본 1000)")
    webcam.add_argument("--camera", type=camera_index, default=0, metavar="번호", help="카메라 선택 (기본 0)")
    webcam.add_argument("--seconds", "--duration-sec", type=nonnegative_float, metavar="초", help="지정한 시간 뒤 자동 종료")
    webcam.add_argument("--no-mirror", action="store_true", help="좌우 반전 끄기")
    video.add_argument("--every", "--cooldown-sec", type=nonnegative_float, default=.5, metavar="초",
                       help="같은 추적 번호의 최소 캡처 간격 (기본 0.5초)")
    video.add_argument("--output", "--capture-dir", type=Path, default=ROOT/"captures", metavar="폴더", help="캡처 저장 폴더")
    video.add_argument("--log-dir", type=Path, default=ROOT/"logs", metavar="폴더", help="로그 저장 폴더")
    video.add_argument("--no-display", action="store_true", help="창 없이 분석")
    video.add_argument("--fast", action="store_true", help="영상 재생 대기 없이 처리")
    diagnostics = commands.add_parser("doctor", help="카메라를 열지 않고 설치·모델 상태 확인")
    diagnostics.add_argument("--model", type=Path, default=DEFAULT_MODEL, metavar="파일", help="확인할 모델 경로")
    return parser


def doctor(model_path=None):
    model_path = (DEFAULT_MODEL if model_path is None else model_path).expanduser()
    print("설치 상태를 확인할게요. 모델 추론과 카메라 연결은 실행 명령에서 확인해요.")
    good = sys.version_info >= (3, 11)
    print(f"{'[확인]' if good else '[필요]'} Python {sys.version.split()[0]} (3.11 이상)")
    for package in ("opencv-python", "numpy", "ultralytics", "lap"):
        try:
            print(f"[설치됨] {package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            good = False
            print(f"[필요] {package}")
    model_ok = model_path.is_file()
    good &= model_ok
    print(f"{'[확인]' if model_ok else '[필요]'} 모델: {model_path}")
    if good:
        print("웹캠: ./motion webcam\n영상: ./motion video \"영상 경로\"")
    else:
        print("준비 방법: README.md의 빠른 시작을 확인해주세요.")
    return 0 if good else 2


def menu():
    print("무엇을 확인할까요?\n  1. 웹캠 화면 보기 — 저장 없음\n  2. 영상 파일 분석 — 움직임 캡처 저장\n  3. 설치 상태 확인\n  0. 종료")
    while True:
        selected = input("번호 입력 [1]: ").strip()
        if selected in ("", "1"):
            return ["webcam"]
        if selected == "2":
            return ["video"]
        if selected == "3":
            return ["doctor"]
        if selected.lower() in ("0", "q", "quit"):
            return None
        print("0, 1, 2, 3 중 하나를 골라주세요.")


def launch_webcam(args):
    from webcam_preview import main as preview_main
    return preview_main(args)


def launch_video(args):
    from main import main as video_main
    return video_main(args, friendly=True)


def main(argv=None):
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if not arguments:
            if not sys.stdin.isatty():
                parser.print_help()
                return 0
            arguments = menu()
            if arguments is None:
                print("종료했어요.")
                return 0
        # A direct file path is also enough: ./motion /path/to/video.mp4.
        first = arguments[0]
        if first not in COMMANDS and not first.startswith('-') and (
                input_path(first).is_file() or '/' in first or '\\' in first or
                Path(first).suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.mpg'}):
            arguments.insert(0, 'video')
        options = parser.parse_args(arguments)
        if options.command == "doctor":
            return doctor(options.model)
        if options.command is None:
            parser.print_help()
            return 0
        try:
            targets = normalize_targets(options.objects)
        except ValueError as exc:
            parser.error(str(exc))
        if options.command in ("video", "run"):
            value = options.path
            if value is None:
                if not sys.stdin.isatty():
                    parser.error('영상 경로를 입력해주세요. 예: ./motion video "videos/demo.mp4"')
                value = input("영상 경로 (파일을 끌어다 놓아도 돼요): ")
            path = input_path(value)
            if not value.strip() or not path.is_file():
                parser.error(f"영상 파일을 찾지 못했어요: {path}\n파일 경로를 확인해주세요.")
        if not options.model.expanduser().is_file():
            parser.error(f"모델 파일이 없어요: {options.model}\n./motion doctor로 준비 상태를 확인해주세요.")
        common = ['--model', str(options.model.expanduser()), '--classes', *targets,
                  '--confidence', str(options.confidence), '--imgsz', str(options.imgsz),
                  '--warmup-sec', str(options.warmup_sec),
                  '--min-object-motion-pixels', str(options.min_object_motion_pixels),
                  '--overlay', options.overlay]
        if options.mask:
            common.append('--show-mask')
        print(f"분석 대상: {', '.join(TARGET_LABELS.get(target, target) for target in targets)}", flush=True)
        if options.overlay != "full":
            label = {"objects": "객체 박스만 · 노란 강조 숨김", "none": "박스·상태 패널 숨김"}[options.overlay]
            print(f"화면 표시: {label}", flush=True)
        if options.command in ("webcam", "preview"):
            if options.seconds is not None and options.seconds <= 0:
                parser.error('--seconds는 0보다 큰 시간이어야 해요.')
            extra = ['--camera', str(options.camera)]
            if options.seconds is not None:
                extra.extend(['--duration-sec', str(options.seconds)])
            if options.no_mirror:
                extra.append('--no-mirror')
            return launch_webcam([*common, *extra])
        print(f"영상: {path}\n캡처 간격: 같은 번호별 {options.every:g}초\n"
              f"캡처 폴더: {options.output}\n로그 폴더: {options.log_dir}", flush=True)
        extra = ['--video', str(path), '--track', '--capture-scope', 'object',
                 '--cooldown-sec', str(options.every), '--capture-dir', str(options.output),
                 '--log-dir', str(options.log_dir)]
        if options.no_display:
            extra.append('--no-display')
        if options.fast:
            extra.extend(['--pace', 'fast'])
        return launch_video([*common, *extra])
    except (EOFError, KeyboardInterrupt):
        print("\n종료했어요.")
        return 130
    except ModuleNotFoundError as exc:
        print(f"필요한 라이브러리가 없어요: {exc.name}\n./motion doctor로 설치 상태를 확인하고 "
              "README.md의 빠른 시작에 따라 의존성을 설치해주세요.", file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
