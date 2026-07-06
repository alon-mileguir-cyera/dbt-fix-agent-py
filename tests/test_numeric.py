"""Direct tests for `dbt_fixer._numeric.parse_bounded_number`, the shared
fail-safe numeric-bound parser used by every `DBT_FIXER_*` numeric env var.

These tests specifically cover the non-finite-float regression class
(`nan`, `inf`, `-inf`, and case/whitespace/sign variants of `nan`): a naive
`value < min_value or value > max_value` range check silently treats NaN as
"in range" because every IEEE-754 ordering comparison against NaN is False,
which would let a malformed float value slip through as a live bound with
no warning recorded. That must never happen for any float-typed bound.
"""

from __future__ import annotations

import math

import pytest

from dbt_fixer._numeric import parse_bounded_number


def _parse(raw, *, default=300.0, min_value=1.0, max_value=3600.0, caster=float):
    warnings: list = []
    value = parse_bounded_number(
        {"X": raw},
        "X",
        default=default,
        min_value=min_value,
        max_value=max_value,
        warnings=warnings,
        caster=caster,
    )
    return value, warnings


@pytest.mark.parametrize(
    "raw",
    ["nan", "NaN", "NAN", "+nan", "-nan", " nan ", "nan "],
)
def test_nan_variants_fall_back_to_default_with_warning(raw):
    value, warnings = _parse(raw)
    assert value == 300.0
    assert warnings, f"expected a warning recorded for malformed value {raw!r}"
    assert "X" in warnings[0]


@pytest.mark.parametrize("raw", ["inf", "+inf", "Infinity", "-inf", "-Infinity"])
def test_infinite_variants_fall_back_to_default_with_warning(raw):
    value, warnings = _parse(raw)
    assert value == 300.0
    assert warnings


def test_nan_never_returned_as_a_live_bound():
    # Belt-and-suspenders: whatever the eventual value is, it must never be
    # NaN, and it must never be silently "in range" per an IEEE-754 quirk.
    value, _ = _parse("nan")
    assert not (isinstance(value, float) and math.isnan(value))


def test_huge_but_parseable_float_string_is_treated_as_out_of_range():
    # 1e400 overflows a Python float to +inf via float(), so it must be
    # rejected by the same non-finite guard, not silently accepted.
    value, warnings = _parse("1e400")
    assert value == 300.0
    assert warnings


def test_valid_finite_value_within_range_is_returned_unchanged():
    value, warnings = _parse("42.5")
    assert value == 42.5
    assert warnings == []


def test_int_caster_rejects_nan_text_via_valueerror_not_finiteness():
    # int('nan') already raises ValueError before the finiteness guard is
    # ever reached; confirm the int-typed path is unaffected and still
    # falls back cleanly.
    value, warnings = _parse("nan", default=8, min_value=1, max_value=100, caster=int)
    assert value == 8
    assert warnings
