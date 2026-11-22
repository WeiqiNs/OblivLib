import re

import pytest

from oblivlib.dependency import AVLNode, CapacityError, OramBlock
from oblivlib.dependency.layout import (
    Array,
    BitFields,
    Bytes,
    FixedBytes,
    Optional,
    Pair,
    Record,
    Seq,
    UInt,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param(UInt(1), 1, id="uint-1-bit"),
        pytest.param(UInt(64), 2**64 - 1, id="uint-64-bit-max"),
        pytest.param(Bytes(5), b"", id="bytes-empty"),
        pytest.param(Bytes(300), b"\x00" * 299 + b"\x01", id="bytes-two-byte-length"),
        pytest.param(FixedBytes(3), b"abc", id="fixed-bytes"),
        pytest.param(Optional(UInt(8)), None, id="optional-absent"),
        pytest.param(Optional(UInt(8)), 0, id="optional-present-zero"),
        pytest.param(Array(UInt(4), 3), [0, 15, 7], id="array"),
        pytest.param(Seq(Bytes(2), 4), [b"a", b"bc"], id="seq-partial"),
        pytest.param(Pair(UInt(8), Bytes(2)), (5, b"hi"), id="pair"),
        pytest.param(
            Record(
                AVLNode,
                (
                    Bytes(2),
                    UInt(6),
                    Bytes(4),
                    Optional(Bytes(2)),
                    Optional(UInt(6)),
                    UInt(8),
                    Optional(Bytes(2)),
                    Optional(UInt(6)),
                    UInt(8),
                ),
            ),
            AVLNode(
                key=b"k", leaf=5, value=b"v", r_key=b"rk", r_leaf=3, r_height=2, l_key=None, l_leaf=None, l_height=0
            ),
            id="record-avl-node",
        ),
    ],
)
def test_every_value_encodes_to_the_declared_size_and_round_trips(field, value):
    encoded = field.encode(value)
    assert len(encoded) == field.size
    assert field.decode(encoded) == value


def test_a_record_writes_its_fields_in_declaration_order_and_rebuilds_the_block_without_touching_it():
    layout = Record(OramBlock, (UInt(8), UInt(6), Bytes(4)))
    block = OramBlock(key=200, leaf=63, value=b"abcd")
    assert layout.encode(block) == bytes([200, 63, 4]) + b"abcd"
    assert layout.decode(layout.encode(block)) == block
    assert block == OramBlock(key=200, leaf=63, value=b"abcd")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        pytest.param(UInt(8), 256, "8-bit", id="uint-too-large"),
        pytest.param(UInt(8), -1, "8-bit", id="uint-negative"),
        pytest.param(Bytes(3), b"abcd", "3-byte", id="bytes-too-long"),
        pytest.param(FixedBytes(3), b"ab", "exactly 3", id="fixed-bytes-short"),
        pytest.param(Array(UInt(4), 2), [1, 2, 3], "exactly 2", id="array-wrong-count"),
        pytest.param(Seq(UInt(4), 2), [1, 2, 3], "at most 2", id="seq-too-many"),
        pytest.param(Pair(UInt(4), UInt(4)), (1, 16), "4-bit", id="pair-inner-field"),
        pytest.param(Optional(UInt(4)), 16, "4-bit", id="optional-inner-field"),
    ],
)
def test_a_value_that_does_not_fit_its_field_raises_capacity_error(field, value, message):
    with pytest.raises(CapacityError, match=message):
        field.encode(value)


@pytest.mark.parametrize(
    ("field", "name"),
    [
        pytest.param(UInt(8), "UInt(8)", id="uint"),
        pytest.param(Bytes(2), "Bytes(2)", id="bytes"),
        pytest.param(FixedBytes(2), "FixedBytes(2)", id="fixed-bytes"),
        pytest.param(Optional(UInt(4)), "Optional(UInt(4))", id="optional"),
        pytest.param(Array(UInt(4), 2), "Array(UInt(4), 2)", id="array"),
        pytest.param(Seq(Bytes(2), 3), "Seq(Bytes(2), 3)", id="seq"),
        pytest.param(Pair(UInt(8), Bytes(1)), "Pair(UInt(8), Bytes(1))", id="pair"),
        pytest.param(Record(OramBlock, (UInt(8), UInt(6), Bytes(4))), "Record(OramBlock, [", id="record-class"),
    ],
)
def test_a_row_of_the_wrong_length_names_its_layout(field, name):
    with pytest.raises(ValueError, match=re.escape(f"{name}") + ".*expected"):
        field.decode(bytes(field.size + 1))


@pytest.mark.parametrize(
    "build",
    [pytest.param(lambda: UInt(0), id="uint"), pytest.param(lambda: BitFields((2, 0)), id="bit-fields")],
)
def test_a_zero_bit_field_is_refused(build):
    with pytest.raises(ValueError, match="at least 1 bit"):
        build()


def test_bit_fields_set_and_get_each_field_in_place():
    layout = BitFields((3, 2, 1))
    word = layout.set(layout.set(layout.set(0, 0, 7), 1, 2), 2, 1)
    assert word == 0b111101
    assert [layout.get(word, i) for i in range(3)] == [7, 2, 1]
    assert layout.get(layout.set(word, 1, 0), 0) == 7


def test_a_bit_field_value_that_does_not_fit_raises_capacity_error():
    layout = BitFields((3, 2))
    assert layout.set(0, 1, 3) == 0b00011
    with pytest.raises(CapacityError, match="2 bits"):
        layout.set(0, 1, 4)
