"""Index constants ride in the address immediate instead of the integer arithmetic.

PTX integer adds wrap and the index is sign-extended to 64 bits, so ptxas may not rewrite
`[base + 4 * (t + 3)]` as `[base + 4*t + 12]` itself; the compiler splits the constant out.
"""
from irkit import binop, body_text, cp_async, lit, load, loop, lowerer, smem, var


def loads(b) -> list[str]:
    return [line for line in body_text(b) if line.startswith("ld.")]


def test_an_added_constant_becomes_the_address_immediate():
    b, lw = lowerer(ints={"t": "%r9"})
    lw.lower_body([load("v", "x", binop("+", var("t"), lit(3)))])
    assert body_text(b) == ["mul.wide.s32 %rd3, %r9, 4;", "add.s64 %rd4, %rd1, %rd3;",
                            "ld.global.f32 %f1, [%rd4+12];"]


def test_a_constant_is_carried_through_a_multiply():
    b, lw = lowerer(ints={"t": "%r9"})
    lw.lower_body([load("v", "x", binop("*", binop("+", var("t"), lit(1)), lit(64)))])
    text = body_text(b)
    assert text[0] == "mul.lo.s32 %r1, %r9, 64;"
    assert text[-1] == "ld.global.f32 %f1, [%rd4+256];"


def test_a_subtracted_constant_is_a_negative_immediate():
    b, lw = lowerer(ints={"t": "%r9"})
    lw.lower_body([load("v", "x", binop("-", var("t"), lit(1)))])
    assert body_text(b)[-1] == "ld.global.f32 %f1, [%rd4+-4];"


def test_division_and_remainder_keep_their_constant_inside():
    # (t + 1) / 2 is not t / 2 + 1/2: the constant cannot leave a division
    b, lw = lowerer(ints={"t": "%r9"})
    lw.lower_body([load("v", "x", binop("/", binop("+", var("t"), lit(1)), lit(2)))])
    assert body_text(b)[:2] == ["add.s32 %r1, %r9, 1;", "div.s32 %r2, %r1, 2;"]
    assert loads(b) == ["ld.global.f32 %f1, [%rd4];"]


def test_unrolled_copies_differ_only_in_the_immediate():
    b, lw = lowerer(ints={"t": "%r9"})
    index = binop("+", binop("*", var("t"), lit(8)), var("k"))
    lw.lower_body([loop("k", 3, [load("v", "x", index)], unroll=True)])
    # the variable part t*8 and its address are computed once; the copies differ in the immediate
    assert [line.split("[")[1] for line in loads(b)] == ["%rd4];", "%rd4+4];", "%rd4+8];"]


def test_cp_async_lane_constants_fold_into_the_shared_address():
    b, lw = lowerer(ints={"t": "%r9"})
    stmt = cp_async("tile", [lit(2), lit(0)], "x", [var("t")], lane=[var("t"), lit(3)])
    lw.lower_body([smem("tile", 64, 4), stmt])
    copy = [line for line in body_text(b) if line.startswith("cp.async")]
    # (2, 0) + lane (t, 3) on 4 columns: 4*t + 11 elements -> variable 4*t, immediate 44 bytes
    assert copy[0].startswith("cp.async.cg.shared.global [%rd") and "+44]" in copy[0]
