"""User-facing CLI contracts; no camera, model download or display is required."""
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

import cli
from config import build_config
from contracts import ModelNotFoundError
from webcam_preview import parse_args as parse_preview


@pytest.fixture
def files(tmp_path, monkeypatch):
    root = tmp_path / "모듈 폴더"
    root.mkdir()
    model = root / "models" / "yolo11n.pt"
    model.parent.mkdir()
    model.write_bytes(b"fake weights: path validation only")
    video = tmp_path / "걷는 사람's 영상.mp4"
    video.write_bytes(b"fake video: path validation only")
    monkeypatch.setattr(cli, "ROOT", root)
    monkeypatch.setattr(cli, "DEFAULT_MODEL", model)
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    return SimpleNamespace(root=root, model=model, video=video)


def test_path_alone_selects_file_profile_from_another_directory(files, tmp_path, monkeypatch, capsys):
    caller = tmp_path / "실행 위치"
    caller.mkdir()
    monkeypatch.chdir(caller)
    received = []
    monkeypatch.setattr(cli, "launch_video", lambda args: received.append(build_config(args)) or 0)

    assert cli.main([str(files.video)]) == 0
    config, = received
    assert config.video_path == files.video
    assert config.model_path == files.model
    assert config.target_classes == ("person",)
    assert config.tracking and config.capture_scope == "object"
    assert config.overlay_mode == "full"
    assert config.capture_cooldown_sec == .5
    assert config.person_confidence == .4 and config.min_person_motion_pixels_ref == 1000
    assert config.capture_dir == files.root / "captures" and config.log_dir == files.root / "logs"
    assert list(caller.iterdir()) == []
    assert "분석 대상: 사람" in capsys.readouterr().out


def test_video_custom_settings_reach_existing_analyzer(files, tmp_path, monkeypatch):
    received = []
    monkeypatch.setattr(cli, "launch_video", lambda args: received.append(build_config(args)) or 5)
    assert cli.main(["video", str(files.video), "--objects", "사람,개", "person", "강아지",
                     "--every", "1", "--output", str(tmp_path / "사진 저장"),
                     "--log-dir", str(tmp_path / "로그 저장"), "--confidence", ".55",
                     "--imgsz", "960", "--warmup-sec", "1.5", "--mask", "--no-display", "--fast",
                     "--overlay", "none"]) == 5
    config, = received
    assert config.target_classes == ("person", "dog")
    assert config.capture_cooldown_sec == 1
    assert config.capture_dir == tmp_path / "사진 저장" and config.log_dir == tmp_path / "로그 저장"
    assert config.person_confidence == .55 and config.yolo_imgsz == 960
    assert config.warmup_sec == 1.5 and config.show_mask and not config.display and config.pace == "fast"
    assert config.overlay_mode == "none"


def test_webcam_controls_reach_preview_without_output_folders(files, monkeypatch):
    received = []
    monkeypatch.setattr(cli, "launch_webcam", lambda args: received.append(parse_preview(args)) or 3)
    assert cli.main(["webcam", "--camera", "1", "--objects", "사람", "개",
                     "--seconds", "30", "--no-mirror", "--mask", "--overlay", "objects"]) == 3
    options, = received
    assert options.camera == 1 and options.classes == ["person", "dog"]
    assert options.model == files.model and options.duration_sec == 30
    assert options.no_mirror and options.show_mask and not options.no_display
    assert options.overlay == "objects"
    assert not (files.root / "captures").exists() and not (files.root / "logs").exists()


@pytest.mark.parametrize("representation", ["literal", "quoted", "dragged", "home"])
def test_prompt_accepts_spaces_quotes_finder_escape_and_home(files, monkeypatch, representation):
    # Finder escapes apostrophes and spaces instead of adding an outer quote.
    paths = {"literal": str(files.video), "quoted": shlex.quote(str(files.video)),
             "dragged": str(files.video).replace("'", "\\'").replace(" ", "\\ "),
             "home": "~/" + cli.os.path.relpath(files.video, Path.home())}
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: paths[representation])
    received = []
    monkeypatch.setattr(cli, "launch_video", lambda args: received.append(build_config(args)) or 0)
    assert cli.main(["video"]) == 0
    assert received[0].video_path.resolve() == files.video.resolve()


@pytest.mark.parametrize("selection,expected", [("", "webcam"), ("1", "webcam"),
                                               ("2", "video"), ("3", "doctor")])
def test_no_argument_menu_selects_requested_action(files, monkeypatch, selection, expected):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    answers = iter([selection, str(files.video)])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    calls = []
    monkeypatch.setattr(cli, "launch_webcam", lambda args: calls.append("webcam") or 0)
    monkeypatch.setattr(cli, "launch_video", lambda args: calls.append("video") or 0)
    monkeypatch.setattr(cli, "doctor", lambda model: calls.append("doctor") or 0)
    assert cli.main([]) == 0
    assert calls == [expected]


def test_invalid_menu_choice_then_exit_never_starts_analysis(files, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    answers = iter(["9", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr(cli, "launch_webcam", lambda args: pytest.fail("unexpected camera launch"))
    assert cli.main([]) == 0
    output = capsys.readouterr().out
    assert "중 하나를 골라주세요" in output and "종료했어요" in output


def test_noninteractive_no_arguments_prints_help_without_opening_camera(files, monkeypatch, capsys):
    monkeypatch.setattr(cli, "launch_webcam", lambda args: pytest.fail("unexpected camera launch"))
    assert cli.main([]) == 0
    output = capsys.readouterr().out
    assert "사용법:" in output and "./motion webcam" in output


@pytest.mark.parametrize("command", ["webcam", "video"])
def test_command_help_has_one_usage_label_and_clear_options(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([command, "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert output.startswith(f"사용법: ./motion {command}")
    assert output.count("사용법:") == 1 and "탐지 조정 (필요할 때만)" in output


@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
def test_cancelling_path_prompt_is_clean_exit(files, monkeypatch, exception):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    def cancel(prompt):
        raise exception
    monkeypatch.setattr("builtins.input", cancel)
    assert cli.main(["video"]) == 130


@pytest.mark.parametrize("args,detail", [(["video"], "영상 경로"),
                                         (["video", "없는 영상.mp4"], "영상 파일"),
                                         (["webcam", "--camera", "-1"], "카메라 번호"),
                                         (["webcam", "--seconds", "0"], "0보다 큰"),
                                         (["webcam", "--seconds", "nan"], "유한한"),
                                         (["webcam", "--confidence", "1.1"], "신뢰도"),
                                         (["webcam", "--objects", ","], "대상 이름"),
                                         (["webcam", "--overlay", "invalid"], "지원하지 않는 선택"),
                                         (["webcam", "--unknown"], "알 수 없는 옵션")])
def test_invalid_inputs_show_korean_guidance_and_do_not_start(files, monkeypatch, capsys, args, detail):
    monkeypatch.setattr(cli, "launch_webcam", lambda args: pytest.fail("unexpected camera launch"))
    monkeypatch.setattr(cli, "launch_video", lambda args: pytest.fail("unexpected video launch"))
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2
    output = capsys.readouterr().err
    assert detail in output and "도움말:" in output
    assert not (files.root / "captures").exists() and not (files.root / "logs").exists()


def test_missing_model_stops_before_backend_or_camera(files, monkeypatch, capsys):
    files.model.unlink()
    monkeypatch.setattr(cli, "launch_webcam", lambda args: pytest.fail("unexpected camera launch"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["webcam"])
    assert exc.value.code == 2
    assert "모델 파일이 없어요" in capsys.readouterr().err


def test_doctor_is_read_only_and_reports_missing_dependency(files, monkeypatch, capsys):
    before = set(files.root.rglob("*"))
    def version(package):
        if package == "ultralytics":
            raise cli.importlib.metadata.PackageNotFoundError(package)
        return "test-version"
    monkeypatch.setattr(cli.importlib.metadata, "version", version)
    monkeypatch.setattr(cli, "launch_webcam", lambda args: pytest.fail("unexpected camera launch"))
    assert cli.main(["doctor"]) == 2
    output = capsys.readouterr().out
    assert "[필요] ultralytics" in output and "[확인] 모델" in output and "준비 방법:" in output
    assert set(files.root.rglob("*")) == before


def test_doctor_accepts_custom_model_path(files, tmp_path, monkeypatch, capsys):
    files.model.unlink()
    custom = tmp_path / "다른 모델.pt"
    custom.touch()
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda package: "test-version")
    assert cli.main(["doctor", "--model", str(custom)]) == 0
    assert f"[확인] 모델: {custom}" in capsys.readouterr().out


def test_missing_library_is_explained_without_traceback(files, monkeypatch, capsys):
    def missing(args):
        raise ModuleNotFoundError("No module named 'cv2'", name="cv2")
    monkeypatch.setattr(cli, "launch_webcam", missing)
    assert cli.main(["webcam"]) == 2
    output = capsys.readouterr().err
    assert "라이브러리가 없어요: cv2" in output and "./motion doctor" in output
    assert "Traceback" not in output


def test_video_dispatch_enables_readable_messages_and_preserves_exit_code(monkeypatch):
    import main as analyzer
    calls = []
    monkeypatch.setattr(analyzer, "main", lambda args, *, friendly: calls.append((args, friendly)) or 4)
    assert cli.launch_video(["--video", "test.mp4"]) == 4
    assert calls == [(["--video", "test.mp4"], True)]


def test_legacy_error_output_remains_available_beside_friendly_output(capsys):
    from main import _print_error
    error = ModelNotFoundError("model file not found: missing.pt")
    _print_error(error)
    assert capsys.readouterr().err == "[MODEL_NOT_FOUND] model file not found: missing.pt\n"
    _print_error(error, friendly=True)
    output = capsys.readouterr().err
    assert "탐지 모델 파일이 없어요" in output and "./motion doctor" in output
    assert "[MODEL_NOT_FOUND]" in output
