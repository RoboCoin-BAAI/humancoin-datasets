from review.structured_draft import build, refine
from review.standard import validate_final
from review.base.run import plan, selected_standard
from test_base_review import annotation


def test_draft_and_invalid_model_response():
    draft = build(annotation())
    for response in (None, {}, {"scene": {"category": "invalid", "area": "invalid"}}):
        result, issues = refine(draft, response)
        validate_final(result)
        assert result == draft
        assert issues


def test_valid_scene_survives_other_missing_fields():
    draft = build(annotation())
    result, issues = refine(draft, {
        "scene": {"category": "household", "area": "kitchen"},
    })
    assert result["scene_classification"]["area"] == "kitchen"
    assert issues


def test_regular_video_uses_shared_current_standard(tmp_path):
    video = tmp_path / "left.mp4"
    video.touch()
    item = plan(video, tmp_path / "result.json")
    assert selected_standard(item["command"]) == 3
    assert item["command"][item["command"].index("--model") + 1] == "qwen3.7-plus"
    assert "--annotation-start-s" not in item["command"]


def test_l2_uses_full_video_and_shared_standard(tmp_path):
    from review.L2.run import make_plan
    from review.L3.run import CAMERA
    session = tmp_path / "L2-001-2026-10-27-13-47-37"
    video = session / "videos" / "chunk-000" / CAMERA / "episode_000000.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    item = make_plan(video, [])
    assert selected_standard(item["command"]) == 3
    assert "--annotation-start-s" not in item["command"]
    assert item["output"] == str(session / "annotation/episode_000000_structured.json")
