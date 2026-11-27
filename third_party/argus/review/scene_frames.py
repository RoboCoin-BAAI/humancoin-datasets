"""Sample scene evidence from the annotated interval without writing image files."""

import base64
from io import BytesIO


def sample(video, timeline):
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        duration = (float(stream.duration * stream.time_base) if stream.duration is not None
                    else float(container.duration / av.time_base))
        start = min((float(row["start_s"]) for row in timeline), default=0.0)
        end = max((float(row["end_s"]) for row in timeline), default=duration)
        start, end = max(0.0, start), min(duration, end)
        if end <= start:
            raise ValueError("No scene sampling interval")
        content, times = [], []
        for fraction in (0.2, 0.7):
            target = start + (end - start) * fraction
            container.seek(int(target * av.time_base), backward=True)
            selected = None
            for frame in container.decode(stream):
                if frame.time is not None and frame.time >= target:
                    selected = frame
                    break
            if selected is None:
                raise ValueError(f"No video frame near {target}s")
            image = selected.to_image()
            image.thumbnail((960, 960))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            content.extend([
                {"type": "text", "text": f"Scene evidence at {selected.time:.3f} seconds"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")}},
            ])
            times.append(float(selected.time))
        return content, times
