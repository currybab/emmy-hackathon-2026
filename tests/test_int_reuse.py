"""Integer work done once: repeated index math is reused, and indices known to be
non-negative divide unsigned (a shift or a multiply, not a signed-division sequence)."""
from irkit import binop, body_text, cond, lit, load, lowerer, strided_loop, var


def test_a_repeated_index_computation_is_emitted_once():
    b, lw = lowerer(ints={"t": "%r9"})
    index = binop("/", var("t"), lit(64))
    lw.lower_body([load("u", "x", index), load("v", "x", binop("+", index, lit(1)))])
    text = body_text(b)
    assert text.count("div.s32 %r1, %r9, 64;") == 1
    assert sum(line.startswith("div") for line in text) == 1
    assert sum(line.startswith("mul.wide") for line in text) == 1  # the address is shared too
    assert [line for line in text if line.startswith("ld.")] == [
        "ld.global.f32 %f1, [%rd4];", "ld.global.f32 %f2, [%rd4+4];"]


def test_work_done_inside_a_branch_is_not_reused_after_it():
    b, lw = lowerer(ints={"t": "%r9", "lane": "%r8"})
    index = binop("/", var("t"), lit(64))
    lw.lower_body([cond(binop("==", var("lane"), lit(0)), [load("u", "x", index)]), load("v", "x", index)])
    assert sum(line.startswith("div") for line in body_text(b)) == 2


def test_work_done_inside_a_loop_is_not_reused_after_it():
    b, lw = lowerer(ints={"t": "%r9"})
    index = binop("/", var("t"), lit(64))
    lw.lower_body([strided_loop("a0", 64, lit(0), lit(1), [load("u", "x", index)]), load("v", "x", index)])
    assert sum(line.startswith("div") for line in body_text(b)) == 2


def test_a_known_non_negative_index_divides_unsigned():
    b, lw = lowerer(ints={"t": "%r9"})
    b.mark_non_negative("%r9")  # e.g. a Tile axis
    lw.lower_body([load("u", "x", binop("%", binop("/", var("t"), lit(64)), lit(3)))])
    assert body_text(b)[:2] == ["div.u32 %r1, %r9, 64;", "rem.u32 %r2, %r1, 3;"]


def test_a_difference_may_be_negative_so_it_divides_signed():
    b, lw = lowerer(ints={"t": "%r9", "s": "%r8"})
    b.mark_non_negative("%r9")
    b.mark_non_negative("%r8")
    lw.lower_body([load("u", "x", binop("/", binop("-", var("t"), var("s")), lit(64)))])
    assert body_text(b)[:2] == ["sub.s32 %r1, %r9, %r8;", "div.s32 %r2, %r1, 64;"]


def test_loop_counters_are_non_negative():
    b, lw = lowerer()
    lw.lower_body([strided_loop("a0", 1024, lit(0), lit(128), [load("u", "x", binop("/", var("a0"), lit(64)))])])
    assert "div.u32 %r2, %r1, 64;" in body_text(b)
