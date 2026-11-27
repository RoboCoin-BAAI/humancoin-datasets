"""Resolve the annotation folder for a local dataset video."""
from pathlib import Path
import re


def default_annotation_output(data: str) -> Path:
    if data.startswith(('http://', 'https://')):
        raise ValueError('a URL requires an explicit --out')
    video = Path(data).resolve()
    camera, chunk, videos, session = video.parent, video.parent.parent, video.parent.parent.parent, video.parent.parent.parent.parent
    if (video.suffix.lower() not in {'.mp4', '.mov', '.m4v', '.mkv', '.webm', '.avi'}
            or not re.fullmatch(r'episode_\d+', video.stem)
            or not camera.name.startswith('observation.images.')
            or not re.fullmatch(r'chunk-\d+', chunk.name)
            or videos.name != 'videos'):
        raise ValueError('provide --out, or use a dataset video in videos/chunk-N/observation.images.CAMERA/episode_N.mp4')
    return session / 'annotation'
