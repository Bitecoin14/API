"""Tests for opt-in feature flag behavior introduced in 2026-04-29 redesign."""
import pytest
from core.config import Config, build_parser


def _parse(*args) -> Config:
    parser = build_parser()
    ns = parser.parse_args(list(args))
    return Config.from_args(ns)


def _parser_rejects(*args) -> int:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(list(args))
    return exc_info.value.code


class TestHandModeDefaults:
    def test_hand_mode_enables_hand_skeleton(self):
        cfg = _parse()
        assert cfg.show_hand_skeleton is True

    def test_hand_mode_pose_off_by_default(self):
        cfg = _parse()
        assert cfg.show_pose is False

    def test_hand_mode_face_off_by_default(self):
        cfg = _parse()
        assert cfg.show_face is False

    def test_hand_mode_blur_off_by_default(self):
        cfg = _parse()
        assert cfg.show_blur is False

    def test_hand_mode_asl_off_by_default(self):
        cfg = _parse()
        assert cfg.show_asl is False


class TestOptInFlags:
    def test_face_flag_enables_face(self):
        cfg = _parse("--face")
        assert cfg.show_face is True

    def test_pose_flag_enables_pose(self):
        cfg = _parse("--pose")
        assert cfg.show_pose is True

    def test_blur_flag_enables_blur(self):
        cfg = _parse("--blur")
        assert cfg.show_blur is True

    def test_asl_flag_enables_asl(self):
        cfg = _parse("--asl")
        assert cfg.show_asl is True

    def test_multiple_flags_together(self):
        cfg = _parse("--face", "--pose", "--blur")
        assert cfg.show_face is True
        assert cfg.show_pose is True
        assert cfg.show_blur is True
        assert cfg.show_hand_skeleton is True  # auto-on in hand mode

    def test_hand_skeleton_still_on_with_other_flags(self):
        cfg = _parse("--face", "--asl")
        assert cfg.show_hand_skeleton is True


class TestFaceMode:
    def test_face_mode_auto_enables_face(self):
        cfg = _parse("--mode", "face")
        assert cfg.show_face is True

    def test_face_mode_disables_hand_skeleton(self):
        cfg = _parse("--mode", "face")
        assert cfg.show_hand_skeleton is False

    def test_face_mode_pose_still_off_by_default(self):
        cfg = _parse("--mode", "face")
        assert cfg.show_pose is False

    def test_face_mode_accepts_pose_opt_in(self):
        cfg = _parse("--mode", "face", "--pose")
        assert cfg.show_pose is True

    def test_face_mode_with_explicit_face_flag_is_idempotent(self):
        cfg = _parse("--mode", "face", "--face")
        assert cfg.show_face is True


class TestRemovedFlags:
    """Ensure the old --no-* flags are gone (argparse exits with error code 2)."""

    def test_no_hand_skeleton_flag_removed(self):
        assert _parser_rejects("--no-hand-skeleton") == 2

    def test_no_pose_flag_removed(self):
        assert _parser_rejects("--no-pose") == 2

    def test_no_face_flag_removed(self):
        assert _parser_rejects("--no-face") == 2

    def test_no_blur_flag_removed(self):
        assert _parser_rejects("--no-blur") == 2
