import pytest
from review.scene_taxonomy import SCENES, validate_scene
from review.base.run import validate


def test_fixed_categories_and_42_typical_areas():
    assert len(SCENES) == 8
    assert sum(len(v) for k,v in SCENES.items() if k not in ('other','unknown')) == 42
    validate_scene('commercial_service', 'warehouse')
    validate_scene('leisure_social', 'gaming_area')
    validate_scene('commercial_service', 'shoe_repair')
    validate_scene('professional_work', 'clinic')


@pytest.mark.parametrize('category,area', [
    ('warehousing_logistics','warehouse'), ('new_category','new_area'),
    ('household','hair_salon'), ('other','kitchen'), ('unknown','other'),
    ('household',None), ('household','Bad Area')])
def test_invalid_pairs(category, area):
    with pytest.raises(ValueError):
        validate_scene(category, area)
