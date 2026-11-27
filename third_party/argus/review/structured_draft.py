"""Build a local draft and accept independently validated model refinements."""

import copy
import re


def build(prepared):
    from review.standard import normalize_verb, preserve_attributes, validate_final
    from review.annotation_input import action_text

    result = copy.deepcopy(prepared)
    result.update(task_name=None, common_task_name="other_task",
                  scene_classification=dict(category="other", area="other",
                                            original_description=prepared.get("scene")))
    objects = []
    for original in prepared.get("objects", []):
        obj = preserve_attributes(original)
        obj.setdefault("raw_name", original.get("name", ""))
        from review.standard import normalize_object_name
        try:
            obj["name"] = normalize_object_name(original.get("name", ""))
        except ValueError:
            obj["name"] = "unclassified"
        obj.setdefault("location", None)
        objects.append(obj)
    result["objects"] = objects
    rows = []
    for original in prepared["subtask"]:
        row = copy.deepcopy(original)
        text = row.get("text") or action_text(row.get("action"), row.get("object"))
        row["text"] = text
        labels = row.get("action")
        if not isinstance(labels, list):
            labels = re.findall(r"[A-Za-z]+", text)[:1]
        normalized = []
        for label in labels:
            try:
                normalized.append(normalize_verb(label))
            except (ValueError, TypeError):
                continue
        row["action"] = list(dict.fromkeys(normalized)) or ["unclassified"]
        for name in ("arm", "object", "destination", "spatial_relation", "contribution",
                     "progress", "hands_visible", "hands_wearing"):
            row.setdefault(name, None)
        if row["arm"] not in ("left", "right", "both", None):
            row["arm"] = None
        if row["contribution"] not in ("advancing", "wasteful", "idle", None):
            row["contribution"] = None
        rows.append(row)
    result["subtask"] = rows
    result["actions"] = dict(vocabulary=list(dict.fromkeys(
        verb for row in rows for verb in row["action"])))
    validate_final(result)
    return result


def refine(draft, response):
    """Apply each valid field independently, retaining the draft for invalid fields."""
    from review.standard import assemble

    baseline = dict(
        scene={key: draft["scene_classification"][key] for key in ("category", "area")},
        objects=[dict(index=i, name=obj["name"]) for i, obj in enumerate(draft["objects"])],
        actions=[dict(index=i, labels=row["action"]) for i, row in enumerate(draft["subtask"])],
        task=None, common_task_name=draft.get("common_task_name", "other_task"))
    issues = []
    if not isinstance(response, dict):
        return copy.deepcopy(draft), ["Model response is not an object"]
    accepted = copy.deepcopy(baseline)
    for key in ("scene", "objects", "actions", "task", "common_task_name"):
        if key not in response:
            issues.append(f"Missing model field: {key}")
            continue
        if key in ("objects", "actions") and isinstance(response[key], list):
            # Validate one supplied index at a time so a malformed row cannot drop other rows.
            used = set()
            for item in response[key]:
                index = item.get("index") if isinstance(item, dict) else None
                if type(index) is not int or not 0 <= index < len(accepted[key]) or index in used:
                    issues.append(f"Invalid or duplicate {key} index")
                    continue
                used.add(index)
                candidate = copy.deepcopy(accepted)
                candidate[key][index] = item
                try:
                    assemble(draft, candidate)
                    accepted = candidate
                except (ValueError, TypeError, KeyError) as error:
                    issues.append(f"{key}[{index}]: {error}")
                except Exception as error:
                    from jsonschema import ValidationError
                    if not isinstance(error, ValidationError):
                        raise
                    issues.append(f"{key}[{index}]: schema validation failed")
            if len(used) != len(accepted[key]):
                issues.append(f"Incomplete model {key}; retained draft entries")
            continue
        candidate = copy.deepcopy(accepted)
        candidate[key] = response[key]
        try:
            assemble(draft, candidate)
            accepted = candidate
        except (ValueError, TypeError, KeyError):
            issues.append(f"Invalid model field: {key}")
        except Exception as error:
            from jsonschema import ValidationError
            if not isinstance(error, ValidationError):
                raise
            issues.append(f"Invalid model field: {key}")
    return assemble(draft, accepted), issues
