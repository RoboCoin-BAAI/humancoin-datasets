import pytest
from review.object_policy import clean_name, normalize_object, validate_objects


def object_row(name, **values):
    return dict(name=name, attributes=dict(color=values.get('color'),
                material=values.get('material'), shape=values.get('shape'),
                unclassified=[], additional={}), original_attributes=[], location=None)


def test_names_remove_modifiers_without_erasing_category_nouns():
    assert clean_name('two red round plastic cups') == 'cups'
    assert clean_name('3 large blue boxes') == 'boxes'
    for noun in ('orange', 'glass', 'box'):
        assert clean_name(noun) == noun


def test_name_evidence_and_aliases_become_closed_attributes():
    source = object_row('two red round plastic cups', color='grey')
    result = normalize_object(source)
    assert result['name'] == 'cups'
    assert result['attributes'] == dict(color='gray', material='plastic', shape='round',
                                       unclassified=[], additional={'color': ['red']})
    assert source['name'] == 'two red round plastic cups'
    validate_objects([result])


def test_unsupported_properties_are_null_and_preserved_for_audit():
    result = normalize_object(object_row('cup', color='mysterious', material='wooden'))
    assert result['attributes']['color'] is None
    assert result['attributes']['material'] == 'wood'
    assert result['attributes']['unclassified'] == ['mysterious']
    with pytest.raises(ValueError, match='enum'):
        validate_objects([object_row('cup', color='reddish')])


def test_schema_rejects_non_enum_primary_and_additional_values():
    import json
    from pathlib import Path
    from jsonschema import Draft202012Validator, ValidationError
    schema = json.loads((Path(__file__).parents[1] / 'review/schemas/episode_structured.schema.json').read_text(encoding='utf-8'))
    validator = Draft202012Validator({'$ref': '#/$defs/object', '$defs': schema['$defs']})
    row = object_row('cup', color='red')
    validator.validate(row)
    row['attributes']['additional'] = {'color': ['reddish']}
    with pytest.raises(ValidationError):
        validator.validate(row)


def test_shared_extractor_accepts_enum_mapping_and_rejects_free_values():
    from review.extract_dimensions import validate
    prepared = dict(scene=None, objects=[dict(name='two reddish cups', attributes=['reddish'])], subtask=[])
    answer = dict(task_name=None, actions=[], scene=dict(category='household', area='kitchen'),
                  objects=[dict(index=0, attributes=[dict(index=0, kind='color', value='red')])])
    assert validate(answer, prepared)['objects'][0]['attributes']['color'] == 'red'
    answer['objects'][0]['attributes'][0]['value'] = 'reddish'
    with pytest.raises(ValueError, match='enum'):
        validate(answer, prepared)
