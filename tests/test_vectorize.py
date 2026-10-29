"""Merging neighbouring scalar loads into ld.v2 / ld.v4 when the alignment is proven."""
from irkit import lit, load, lowerer, smem, var
from ptxc.vectorize import merge_loads


def test_four_aligned_neighbours_become_one_v4_load_at_the_first_position():
    lines = ["\tld.shared.f32 %f1, [%rd5];", "\tmul.f32 %f9, %f1, %f1;", "\tld.shared.f32 %f2, [%rd5+4];",
             "\tld.shared.f32 %f3, [%rd5+8];", "\tld.shared.f32 %f4, [%rd5+12];"]
    assert merge_loads(lines, {"%rd5": 16}) == [
        "\tld.shared.v4.f32 {%f1, %f2, %f3, %f4}, [%rd5];", "\tmul.f32 %f9, %f1, %f1;"]


def test_an_8_byte_aligned_base_gets_v2_loads_only():
    lines = [f"\tld.shared.f32 %f{i + 1}, [%rd5+{4 * i}];" for i in range(4)]
    assert merge_loads(lines, {"%rd5": 8}) == [
        "\tld.shared.v2.f32 {%f1, %f2}, [%rd5];", "\tld.shared.v2.f32 {%f3, %f4}, [%rd5+8];"]


def test_an_unknown_alignment_leaves_the_loads_alone():
    lines = ["\tld.shared.f32 %f1, [%rd5];", "\tld.shared.f32 %f2, [%rd5+4];"]
    assert merge_loads(lines, {}) == lines


def test_global_loads_are_left_alone():
    # measured slower on a01 / a02; the IR's own grouped loads already give global vectors
    lines = ["\tld.global.f32 %f1, [%rd5];", "\tld.global.f32 %f2, [%rd5+4];"]
    assert merge_loads(lines, {"%rd5": 16}) == lines


def test_a_misaligned_start_shifts_the_pairs():
    lines = [f"\tld.shared.f32 %f{i + 1}, [%rd5+{4 + 4 * i}];" for i in range(3)]
    assert merge_loads(lines, {"%rd5": 8}) == [
        "\tld.shared.f32 %f1, [%rd5+4];", "\tld.shared.v2.f32 {%f2, %f3}, [%rd5+8];"]


def test_loads_are_not_merged_across_a_store_barrier_or_label():
    for boundary in ("\tst.shared.f32 [%rd6], %f9;", "\tbar.sync 0;", "$L_x_1:", "\tcp.async.wait_group 0;"):
        lines = ["\tld.shared.f32 %f1, [%rd5];", boundary, "\tld.shared.f32 %f2, [%rd5+4];"]
        assert merge_loads(lines, {"%rd5": 16}) == lines


def test_shared_tiles_get_vector_loads_when_their_index_is_a_multiple():
    b, lw = lowerer(ints={"t": "%r9"})
    # tile[t, k] is element t*16 + k: the row starts on a 64-byte multiple of a 16-byte aligned array
    stmts = [smem("tile", 64, 16, align=16)] + [load(f"v{k}", "tile", var("t"), lit(k)) for k in range(4)]
    lw.lower_body(stmts)
    merged = merge_loads(list(b.body_lines()), b.alignments())
    assert sum(".v4.f32" in line for line in merged) == 1
