from __future__ import annotations

import re


_TASK_PROMPT_MAP = {
    "lift": "lift the object",
    "can": "pick up the can",
    "square": "insert the square peg into the hole",
    "transport": "transport the object to the target",
    "threading": "thread the object into the target",
    "twoarmcoffee": "make coffee with both arms",
    "twoarmthreading": "perform the threading task with both arms",
    "twoarmthreepieceassembly": "assemble the three pieces with both arms",
    "twoarmtransport": "transport the object with both arms",
    "twoarmlifttray": "lift the tray with both arms",
    "twoarmboxcleanup": "clean up the box with both arms",
    "twoarmdrawercleanup": "clean up the drawer area with both arms",
    "twoarmpouring": "pour the contents into the target container",
    "twoarmcansortrandom": "sort the cans with both arms",
    "twoarmcansort": "sort the cans with both arms",
}


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def infer_task_prompt(
    *,
    task_name: str | None = None,
    dataset_name: str | None = None,
    explicit_prompt: str | None = None,
) -> str:
    """Pick a usable default prompt when a dataset/environment does not expose `task` text."""

    if explicit_prompt:
        return explicit_prompt

    candidates = []
    if task_name:
        candidates.append(task_name)
    if dataset_name:
        candidates.extend(dataset_name.split("/"))

    for candidate in candidates:
        normalized = _normalize_key(candidate)
        if normalized in _TASK_PROMPT_MAP:
            return _TASK_PROMPT_MAP[normalized]

    for candidate in candidates:
        normalized = _normalize_key(candidate)
        for key, prompt in _TASK_PROMPT_MAP.items():
            if key in normalized or normalized in key:
                return prompt

    if task_name:
        cleaned = re.sub(r"[_-]+", " ", task_name).strip()
        if cleaned:
            return cleaned

    if dataset_name:
        cleaned = re.sub(r"[_-]+", " ", dataset_name.split("/")[-1]).strip()
        if cleaned:
            return cleaned

    return "perform the task"
