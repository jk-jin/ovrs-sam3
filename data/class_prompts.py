from __future__ import annotations

from collections.abc import Sequence


def expand_class_prompts(
    class_names: Sequence[str],
) -> tuple[list[str], list[int]]:
    """Expand comma-separated class names into independent prompt channels.

    Each entry in class_names still represents one original semantic class.
    Commas only split that class into multiple text prompts.

    Returns:
        prompt_names:
            Flattened prompt strings in their original order.
        prompt_to_class_id:
            For every prompt channel, the corresponding original forward
            class id.
    """
    if len(class_names) == 0:
        raise ValueError("class_names must not be empty.")

    prompt_names: list[str] = []
    prompt_to_class_id: list[int] = []

    for class_id, raw_class_name in enumerate(class_names):
        raw_class_name = str(raw_class_name)

        aliases = [
            part.strip()
            for part in raw_class_name.split(",")
        ]

        if any(alias == "" for alias in aliases):
            raise ValueError(
                "Class names must not contain empty comma-separated prompts. "
                f"Got class_names[{class_id}]={raw_class_name!r}."
            )

        prompt_names.extend(aliases)
        prompt_to_class_id.extend(
            [class_id] * len(aliases)
        )

    if len(prompt_names) == 0:
        raise ValueError(
            "No prompt names were produced from class_names."
        )

    return prompt_names, prompt_to_class_id
