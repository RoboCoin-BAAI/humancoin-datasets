import json

import pytest
import numpy as np

from label import episode


def test_only_execution_frames_are_sent_with_original_clock(tmp_path):
    directory = tmp_path / 'episode_000000'
    directory.mkdir()
    np.savez(directory / 'state.npz', state=np.zeros((600, 0)))
    (directory / 'sources.json').write_text(json.dumps({'left': {
        'packed': str(tmp_path / 'video.mp4'), 'base_s': 0, 'n_frames': 600}}))
    (directory / 'context.json').write_text(json.dumps({
        'fps': 30, 'n_state_frames': 600, 'profile': 'ego_head', 'state_kind': 'none',
        'cameras': {'left': {'width': 64, 'height': 36}}}))
    path = directory / 'context.json'
    ctx = json.loads(path.read_text())
    ep = {'context': ctx, 'state': np.zeros((600, 0)), 'action': None,
          'sources': json.loads((directory / 'sources.json').read_text()),
          'kmap': {}, 'times': None, 'signals': {}}
    original = episode.plan(ep)
    start = episode.frame_time(ep, original['ks'][len(original['ks']) // 2]) + 0.001
    ctx['annotation_start_s'] = start
    path.write_text(json.dumps(ctx))
    ep['context'] = ctx
    planned = episode.plan(ep)
    assert all(episode.frame_time(ep, k) >= start for k in planned['ks'])
    assert planned['ks'][0] > 0
    ctx['annotation_start_s'] = 1e9
    path.write_text(json.dumps(ctx))
    with pytest.raises(ValueError, match='No video frames'):
        episode.plan(ep)
