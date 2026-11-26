"""Shared closed vocabularies for structured object descriptions."""
import json
import re

VOCABULARY = {
    'color': 'black white gray red orange yellow green blue purple pink brown beige cyan magenta gold silver transparent multicolor'.split(),
    'material': 'plastic metal wood glass ceramic paper cardboard fabric rubber silicone leather stone concrete foam composite'.split(),
    'shape': 'round square rectangular triangular oval cylindrical spherical conical cuboid flat irregular'.split(),
}
ALIASES = {'grey': 'gray', 'wooden': 'wood', 'metallic': 'metal',
           'cylinder': 'cylindrical', 'sphere': 'spherical', 'rectangle': 'rectangular',
           'triangle': 'triangular', 'circle': 'round', 'circular': 'round',
           'box-shaped': 'cuboid', 'multi-colored': 'multicolor', 'colourful': 'multicolor'}
POLICY_PROMPT = '''
Object names must be concise English category nouns, without colors, quantities,
articles, size, material or shape modifiers ("two red round plastic cups" -> "cups").
Preserve category nouns: "orange" (fruit), "box", "glass" (drinking vessel).
Do not merge distinct objects even when their cleaned names coincide.
Object color/material/shape MUST use only the following enumerations, or null
when unknown, unsupported or outside the vocabulary. Never infer unseen properties.
Use additional.color/material/shape for further supported enum values.
Keep original source attributes for provenance, not as normalized values.
''' + json.dumps(VOCABULARY)


def canonical(kind, value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError('Object attribute must be a string or null')
    value = value.strip().lower()
    value = ALIASES.get(value, value)
    return value if value in VOCABULARY[kind] else None


def clean_name(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Object name must be nonempty')
    tokens = name.strip().lower().replace('_', ' ').split()
    modifiers = set(sum(VOCABULARY.values(), [])) | set(ALIASES) | set(
        'a an the one two three four five six seven eight nine ten several some many multiple '
        'single pair small large big little tiny long short wooden'.split())
    # Keep the category noun: "orange", "glass", "square" can themselves be objects.
    return ' '.join(t for i, t in enumerate(tokens) if i == len(tokens)-1 or
                    (t not in modifiers and not re.fullmatch(r'\d+(?:\.\d+)?', t)))


def normalize_object(obj):
    """Normalize without mutating the source; unknown values remain auditable."""
    attrs = obj['attributes']
    unknown = list(attrs.get('unclassified', []))
    normalized, additional = {}, {}
    for kind in VOCABULARY:
        values = [attrs.get(kind)] + attrs.get('additional', {}).get(kind, [])
        values += [t for t in obj['name'].lower().replace('_', ' ').split()[:-1]
                   if canonical(kind, t) is not None]
        accepted = []
        for value in values:
            mapped = canonical(kind, value)
            if mapped is not None and mapped not in accepted:
                accepted.append(mapped)
            elif value is not None and mapped is None and value not in unknown:
                unknown.append(value)
        normalized[kind] = accepted[0] if accepted else None
        if len(accepted) > 1:
            additional[kind] = accepted[1:]
    return {**obj, 'name': clean_name(obj['name']),
            'attributes': {**normalized, 'unclassified': unknown, 'additional': additional}}


def validate_objects(objects):
    for obj in objects:
        if clean_name(obj['name']) != obj['name']:
            raise ValueError('Object name contains forbidden modifiers or is not normalized')
        for kind, vocabulary in VOCABULARY.items():
            value = obj['attributes'][kind]
            if value is not None and value not in vocabulary:
                raise ValueError('Object attribute outside ' + kind + ' enum')
            if any(v not in vocabulary for v in obj['attributes']['additional'].get(kind, [])):
                raise ValueError('Additional object attribute outside ' + kind + ' enum')
