"""HumanCOIN V1: fixed categories, open area vocabulary."""
import json
import re
from pathlib import Path

SCENES = json.loads((Path(__file__).parent / 'schemas/humancoin_scenes_v1.json').read_text(encoding='utf-8'))['categories']


def validate_scene(category, area):
    if not isinstance(category, str) or category not in SCENES:
        raise ValueError('Scene category must be one of the eight HumanCOIN V1 categories')
    if not isinstance(area, str) or not re.fullmatch(r'[a-z][a-z0-9]*(?:_[a-z0-9]+)*', area):
        raise ValueError('Scene area must be a nonempty snake_case string')
    if category in ('other', 'unknown') and area != category:
        raise ValueError('Fallback categories require the matching fallback area')
    for owner, areas in SCENES.items():
        if area in areas and owner != category:
            raise ValueError(f'Typical scene area belongs to a different category: '
                             f'category={category!r}, area={area!r}, area_owner={owner!r}; '
                             f'typical areas for {category}: {SCENES[category]}. '
                             'Preserve any resolved category; choose an evidence-supported compatible area.')
