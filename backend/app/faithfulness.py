from __future__ import annotations

from typing import Callable


def perturbation_curve(
    segments: list[str],
    ranked_indices: list[int],
    probability: Callable[[str], float],
    mode: str = "deletion",
) -> list[dict[str, float | int]]:
    if mode not in {"deletion", "insertion"}:
        raise ValueError("mode must be deletion or insertion")
    original = " ".join(segments)
    reference = probability(original)
    points = []
    for count in range(len(segments) + 1):
        selected = set(ranked_indices[:count])
        if mode == "deletion":
            text = " ".join(segment for index, segment in enumerate(segments) if index not in selected)
        else:
            text = " ".join(segment for index, segment in enumerate(segments) if index in selected)
        points.append({"step": count, "reference_probability": reference, "probability": probability(text)})
    return points


def area_under_curve(points: list[dict[str, float | int]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(
        (float(left["probability"]) + float(right["probability"])) / 2
        for left, right in zip(points, points[1:])
    ) / (len(points) - 1)
