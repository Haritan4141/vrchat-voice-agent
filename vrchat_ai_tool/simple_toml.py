from __future__ import annotations

import ast
from typing import Any


def loads(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current: dict[str, Any] = root
    lines = text.splitlines()
    index = 0

    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.strip()
        index += 1

        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            table_path = line[1:-1].strip()
            if not table_path:
                raise ValueError("Empty TOML table path")
            current = _ensure_table(root, table_path.split("."))
            continue

        if "=" not in raw_line:
            raise ValueError(f"Unsupported TOML line: {raw_line}")

        key, value = raw_line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Missing TOML key: {raw_line}")

        if value.startswith('"""'):
            parsed_value, index = _parse_multiline_string(lines, index - 1)
        else:
            parsed_value = _parse_value(value)

        current[key] = parsed_value

    return root


def _ensure_table(root: dict[str, Any], parts: list[str]) -> dict[str, Any]:
    current = root
    for part in parts:
        if part not in current:
            current[part] = {}
        value = current[part]
        if not isinstance(value, dict):
            raise ValueError(f"TOML path is not a table: {'.'.join(parts)}")
        current = value
    return current


def _parse_multiline_string(lines: list[str], start_index: int) -> tuple[str, int]:
    first_line = lines[start_index]
    _, after_equals = first_line.split("=", 1)
    suffix = after_equals.strip()[3:]

    chunks: list[str] = []
    if suffix.endswith('"""'):
        return suffix[:-3], start_index + 1

    if suffix:
        chunks.append(suffix)

    index = start_index + 1
    while index < len(lines):
        line = lines[index]
        if line.endswith('"""'):
            closing_prefix = line[:-3]
            if closing_prefix:
                chunks.append(closing_prefix)
            return "\n".join(chunks), index + 1
        chunks.append(line)
        index += 1

    raise ValueError("Unterminated multiline TOML string")


def _parse_value(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return ast.literal_eval(value)

    lowered = value.casefold()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    raise ValueError(f"Unsupported TOML value: {value}")
