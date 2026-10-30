"""A composed method's inputs, each typed by the setting it fills."""

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, get_args, get_origin

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from ...common.error_formatting import error_message
from ...common.hints import Hint, hint_of
from ...common.paths import Loc
from ...common.text import quoted_join
from ..check import CHECK_FIELDS, METHODS, holds_checks
from ..settings import Settings, annotated, item_field, literal
from ..templates import TemplateError, named_in, reference, templated, validate
from .expand import EXPANDED
from .file import TEMPLATE_KEYS, Fault, Input, MethodFile, to_file_path

if TYPE_CHECKING:
    from ..method import Method

_NAMES = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")
"""A reference by name, such as ``inputs.points``."""


class _Typing:
    """Each input's type and field: those of the setting it fills, where the body names it."""

    def __init__(self, file: MethodFile, lookup: Mapping[str, Method[Any]]) -> None:
        self.file = file
        self.lookup = lookup
        self.found: dict[str, tuple[Any, FieldInfo]] = {}
        self.eaches: list[dict[str, tuple[Any, FieldInfo]]] = []
        self.templated = named_in([file.description, file.feedback], "inputs")
        """Inputs that templates and conditions name."""

    def inputs(self) -> dict[str, tuple[Any, FieldInfo]]:
        self._check(self.file.body(), ())
        for key, spec in self.file.inputs.items():
            if key in CHECK_FIELDS:
                raise Fault(
                    f"A check has its own {key}: name the input otherwise.", ("inputs", key)
                )
            if key in self.found:
                continue
            if key not in self.templated:
                raise Fault(f"{spec.title or f'Input {key}'} isn't used.", ("inputs", key))
            self.found[key] = (_type_of(spec.default), FieldInfo())
        return self.found

    def _record(self, expression: str, annotation: Any, info: FieldInfo, path: Loc) -> None:
        if not _NAMES.fullmatch(expression):
            # A template's own refusal says best what's wrong, such as arithmetic.
            try:
                validate(expression, dict.fromkeys(EXPANDED, "any"), expression=True)
            except TemplateError as error:
                raise Fault(str(error), to_file_path(path)) from None
            raise Fault(
                "A setting takes an input as it is, such as {{ inputs.name }}.", to_file_path(path)
            )
        root, *rest = expression.split(".")
        if root == "inputs" and len(rest) == 1 and rest[0] in self.file.inputs:
            existing = self.found.get(rest[0])
            if existing is not None and existing[0] != annotation:
                raise Fault(
                    f"Input {rest[0]} fills settings of different types.", to_file_path(path)
                )
            self.found.setdefault(rest[0], (annotation, info))
        elif root == "each" and self.eaches and len(rest) <= 1:
            self.eaches[-1].setdefault(rest[0] if rest else "", (annotation, info))
        elif root != "index":
            raise Fault(f"There's no {expression}.", to_file_path(path))

    def _check(self, node: Any, path: Loc) -> None:
        if not isinstance(node, Mapping) or "uses" not in node:
            raise Fault("A check needs uses: and a method's name.", to_file_path(path))
        method = self.lookup.get(str(node["uses"]))
        if method is None:
            raise Fault(f"There's no method {node['uses']}.", to_file_path((*path, "uses")))
        for key, value in sorted(node.items(), key=lambda item: item[0] != "with"):
            if key in TEMPLATE_KEYS:
                self.templated |= named_in(value, "inputs")
                continue
            if key in ("uses", "for_each"):
                continue
            if key == "with":
                self._settings(value, method.settings, (*path, "with"))
            elif key not in CHECK_FIELDS:
                raise Fault(f"A check has no {key}:.", to_file_path((*path, key)))
            elif (expression := reference(value)) is not None:
                info = CHECK_FIELDS[key]
                self._record(expression, info.annotation, info, (*path, key))

    def _settings(self, values: Any, model: type[BaseModel], path: Loc) -> None:
        if (expression := reference(values)) is not None:
            raise Fault(
                f"Fill in a method's settings one by one, not with {expression}.",
                to_file_path(path),
            )
        if not isinstance(values, Mapping | None):
            raise Fault("A check's settings aren't a mapping.", to_file_path(path))
        for key, value in (values or {}).items():
            info = model.model_fields.get(key)
            if info is None:
                raise Fault(
                    f"{model.model_config.get('title') or 'This method'} has no setting {key}.",
                    to_file_path((*path, key)),
                )
            if (root := _root(value)) is not None:
                info = _placed(info, root, values or {}, model, (*path, key))
            self._setting(value, info, (*path, key))

    def _setting(self, value: Any, info: FieldInfo, path: Loc) -> None:
        held = holds_checks(info)
        if (expression := reference(value)) is not None:
            self._record(expression, info.annotation, info, path)
        elif held == "one":
            self._check(value, path)
        elif held == "many" and (item := item_field(info)) is not None:
            for index, entry in enumerate(value):
                self._item(entry, item, (*path, index))
        elif templated(value):
            element = get_args(info.annotation)[0] if get_args(info.annotation) else Any
            for index, item in enumerate(value if isinstance(value, list) else ()):
                if (expression := reference(item)) is not None:
                    self._record(expression, element, info, (*path, index))
        else:
            _validate_literal(value, info, path, self.lookup)

    def _item(self, item: Any, info: FieldInfo, path: Loc) -> None:
        if (expression := reference(item)) is not None:
            self._record(expression, annotated(info.annotation, *info.metadata), info, path)
            return
        if isinstance(item, Mapping) and "for_each" in item:
            source = reference(item["for_each"])
            if source is None or not source.startswith("inputs."):
                raise Fault("for_each needs one {{ inputs.… }}.", to_file_path((*path, "for_each")))
            self.eaches.append({})
            self._check(item, path)
            each = self._titled(self.eaches.pop(), source, path)
            if "" in each:
                annotation, field = each[""]
                annotation = annotated(annotation, *field.metadata)
            elif each:
                annotation = create_model(
                    "Item", **{key: (kind, field) for key, (kind, field) in each.items()}
                )  # type: ignore[call-overload]
                field = FieldInfo()
            else:
                raise Fault("A repeated check uses {{ each }}.", to_file_path(path))
            self._record(
                source,
                list[annotation],  # type: ignore[valid-type]
                FieldInfo(title=field.title, json_schema_extra=field.json_schema_extra),
                (*path, "for_each"),
            )
            return
        self._check(item, path)

    def _titled(
        self, each: dict[str, tuple[Any, FieldInfo]], source: str, path: Loc
    ) -> dict[str, tuple[Any, FieldInfo]]:
        """The fields of each item, with the titles the input gives them."""
        spec = self.file.inputs.get(source.removeprefix("inputs."))
        own = (spec.items if spec else None) or {}
        if unused := own.keys() - each.keys():
            raise Fault(
                f"Each item has no {', '.join(sorted(unused))}.", to_file_path((*path, "for_each"))
            )
        return {
            key: (
                kind,
                FieldInfo.merge_field_infos(field, **own[key].model_dump(exclude_none=True)),
            )
            if key in own
            else (kind, field)
            for key, (kind, field) in each.items()
        }


def _root(value: Any) -> str | None:
    """What fills a setting from elsewhere: ``inputs``, or ``each`` of a repeated check."""
    found = reference(value)
    if found is None and isinstance(value, list):
        found = next((expression for item in value if (expression := reference(item))), None)
    return found.split(".")[0] if found else None


def _placed(
    info: FieldInfo, root: str, siblings: Mapping[str, Any], model: type[BaseModel], path: Loc
) -> FieldInfo:
    """``info`` filled from ``root``, applying when what fills the settings it depends on does: a
    condition on a setting filled from the same place names what fills it, and one on a fixed
    setting holds or refuses the file."""
    hint = hint_of(info)
    wanted: dict[str, JsonValue] = {}
    for key, values in (hint.applies_when or {}).items():
        sibling = model.model_fields[key]
        value = siblings.get(key, sibling.get_default(call_default_factory=True))
        options = values if isinstance(values, list) else [values]
        if (expression := reference(value)) is not None:
            scope, _, name = expression.partition(".")
            if scope == root and name:
                wanted[name] = values
        elif not templated(value) and getattr(value, "value", value) not in options:
            labels = hint_of(sibling).labels or {}
            said = quoted_join(
                [labels.get(str(option), option) for option in options], conjunction="or"
            )
            raise Fault(
                f"{info.title} applies only when {sibling.title} is {said}.", to_file_path(path)
            )
    if wanted == (hint.applies_when or {}):
        return info
    placed = hint.model_copy(update={"applies_when": wanted or None})
    extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
    return FieldInfo.merge_field_infos(info, json_schema_extra=extra | placed.schema_extra())


def _type_of(default: Any) -> Any:
    """The type of an input that fills no setting, from its default."""
    for kind in (bool, str, list):
        if isinstance(default, kind):
            return kind
    return float if isinstance(default, int | float) else JsonValue


def _validate_literal(
    value: Any, info: FieldInfo, path: Loc, lookup: Mapping[str, Method[Any]]
) -> None:
    """Refuses a setting written as it is that its method wouldn't accept."""
    try:
        TypeAdapter[Any](annotated(info.annotation, *info.metadata)).validate_python(
            value, context={METHODS: lookup}
        )
    except ValidationError as error:
        first = error.errors()[0]
        raise Fault(error_message(first), to_file_path((*path, *first["loc"]))) from None


def input_field(spec: Input, annotation: Any, info: FieldInfo) -> tuple[Any, FieldInfo]:
    """An input's field: the setting it fills, with the input's own title, default and hints."""
    if spec.choices:
        annotation = literal(spec.choices)
    own = {"advanced": spec.advanced, "labels": spec.labels, "applies_when": spec.applies_when}
    hint = hint_of(info).model_copy(
        update={key: value for key, value in own.items() if value is not None}
    )
    changes: dict[str, Any] = {
        "title": spec.title or info.title,
        "description": spec.description or info.description,
        "examples": spec.examples or info.examples,
        "json_schema_extra": hint.schema_extra() if hint != Hint() else None,
    }
    required = "default" not in spec.model_fields_set
    if required:
        changes["default_factory"] = None
    else:
        changes |= {"default": spec.default, "default_factory": None, "validate_default": True}
    needs_item = get_origin(annotation) is list and spec.default != []
    field = FieldInfo.merge_field_infos(
        info, *([FieldInfo(min_length=1)] if needs_item else []), **changes
    )
    if required:  # merging keeps the setting's own default rather than an undefined one
        field.default = PydanticUndefined
    field.annotation = annotation
    return annotation, field


def settings_model(file: MethodFile, lookup: Mapping[str, Method[Any]]) -> type[Settings]:
    """The settings of the method ``file`` defines: one field for each of its inputs."""
    filled = _Typing(file, lookup).inputs()
    fields = {key: input_field(spec, *filled[key]) for key, spec in file.inputs.items()}
    for key, spec in file.inputs.items():
        if "default" in spec.model_fields_set:
            _validate_literal(spec.default, fields[key][1], ("inputs", key, "default"), lookup)
    model: type[Settings] = create_model(file.label, __base__=Settings, **fields)  # type: ignore[call-overload]
    return model
