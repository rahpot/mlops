import numpy as np
import pytest

from tensor_ops import mac

A_VALUES = [1.5, -2.25, 3.0, 0.5, -1.0, 4.5, -0.75, 2.0, -3.5, 1.25]
B_VALUES = [2.0, 4.0, -1.5, 8.0, -2.5, 0.25, -4.0, 1.5, 2.0, -6.0]
C_VALUES = [0.5, -1.0, 2.25, -3.0, 1.5, -0.5, 3.0, -2.0, 0.75, 1.0]


def as_row(values):
    return np.array(values, dtype=np.float64).reshape(1, 1, len(values))


def expected_row(a_values, b_values, c_values):
    return [a * b + c for a, b, c in zip(a_values, b_values, c_values)]


def test_hand_written_values_on_a_long_last_axis():
    out = mac(as_row(A_VALUES), as_row(B_VALUES), as_row(C_VALUES))
    assert out.shape == (1, 1, 10)
    assert out.dtype == np.dtype("float64")
    np.testing.assert_allclose(
        out.reshape(-1), expected_row(A_VALUES, B_VALUES, C_VALUES), rtol=1e-12, atol=1e-12
    )


def test_invalid_input_is_rejected():
    valid = np.ones((2, 3, 4), dtype=np.float64)

    with pytest.raises(ValueError):
        mac(np.ones((2, 3, 4, 5), dtype=np.float64), valid, valid)
    with pytest.raises(ValueError):
        mac(valid, np.ones((24,), dtype=np.float64), valid)

    with pytest.raises(TypeError):
        mac(valid, valid, valid.astype(np.complex128))
    with pytest.raises(TypeError):
        mac(valid.astype(np.float16), valid, valid)

    with pytest.raises(TypeError):
        mac(valid, None, valid)
    with pytest.raises(TypeError):
        mac(valid, valid, "not an array")


def test_transposed_view_is_rejected():
    transposed = np.arange(24.0).reshape(2, 3, 4).transpose(2, 0, 1)
    assert transposed.shape == (4, 2, 3)
    assert not transposed.flags.c_contiguous
    packed = np.ones((4, 2, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        mac(transposed, packed, packed)
