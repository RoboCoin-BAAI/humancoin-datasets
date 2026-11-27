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


def test_object_raw_name_survives_cleanup_and_repeated_processing():
    source = annotation()
    source["objects"] = [
        dict(name="two small red boxes", attributes=[], location=None),
    ]
    draft = build(source)
    assert draft["objects"][0]["name"] == "two small red boxes"
    assert draft["objects"][0]["raw_name"] == "two small red boxes"
    result, _ = refine(draft, {"objects": [{"index": 0, "name": "box"}]})
    assert result["objects"][0]["name"] == "box"
    repeated = build(result)
    assert repeated["objects"][0]["raw_name"] == "two small red boxes"
    assert source["objects"][0]["name"] == "two small red boxes"


def test_uncleanable_source_object_cannot_break_refinement_fallback():
    source = annotation()
    source["objects"] = [
        dict(name="black wire vase with flowers", attributes=[], location=None),
    ]
    draft = build(source)
    assert draft["objects"][0]["name"] == "black wire vase with flowers"
    for response in ({}, {"objects": []},
                     {"objects": [{"index": 0, "name": "container"}]},
                     {"objects": [{"index": 0, "name": "vase"}]}):
        result, _ = refine(draft, response)
        validate_final(result)
        assert result["objects"][0]["raw_name"] == "black wire vase with flowers"


def test_local_validation_failure_does_not_repeat_model_work(monkeypatch):
    import pytest
    from review.structured_retry import run
    calls = []
    def invalid_result():
        calls.append(1)
        raise ValueError("Local validation bug")
    with pytest.raises(ValueError):
        run(invalid_result)
    assert len(calls) == 1


def test_regular_video_uses_shared_current_standard(tmp_path):
    video = tmp_path / "left.mp4"
    video.touch()
    item = plan(video, tmp_path / "result.json")
    assert selected_standard(item["command"]) == 3
    assert item["command"][item["command"].index("--model") + 1] == "qwen3.7-plus"
    assert "--annotation-start-s" not in item["command"]
    assert item["command"][item["command"].index("--sample-fps") + 1] == "1"


def test_scene_images_are_sent_with_existing_text(monkeypatch):
    import io
    import json
    from review import scene_frames, standard

    images = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,test"}}]
    monkeypatch.setattr(scene_frames, "sample", lambda video, timeline: (images, [2.0]))
    def dispatch(request, timeout):
        body = json.loads(request.data)
        content = body["messages"][1]["content"]
        assert content[0]["type"] == "text"
        assert "timeline" in json.loads(content[0]["text"])
        assert content[1] == images[0]
        payload = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(standard.urllib.request, "urlopen", dispatch)
    receipt = {}
    standard.call_qwen(build(annotation()), "qwen3.7-plus", "https://example.invalid",
                       "key", 10, video="left.mp4", receipt=receipt)
    assert receipt["scene_image_times_s"] == [2.0]


def test_l2_uses_full_video_and_shared_standard(tmp_path):
    from review.L2.run import make_plan
    from review.L3.run import CAMERA
    session = tmp_path / "L2-001-2026-10-27-13-47-37"
    video = session / "videos" / "chunk-000" / CAMERA / "episode_000000.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    item = make_plan(video, [])
    assert selected_standard(item["command"]) == 3
    assert item["command"][item["command"].index("--sample-fps") + 1] == "2"
    assert "--annotation-start-s" not in item["command"]
    assert item["output"] == str(session / "annotation/episode_000000_structured.json")
