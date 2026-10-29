"""Unrolling the compiler decides on itself, as nvcc's front end does: small simple loops are
copied out so their index constants fold into address immediates, and strided loops run
several iterations per trip so each thread keeps several loads in flight."""
from irkit import accum, binop, body_text, cond, init, lit, load, loop, lowerer, strided_loop, sync, var


def loads(b) -> list[str]:
    return [line for line in body_text(b) if line.startswith("ld.")]


def test_a_strided_loop_runs_several_iterations_per_trip_then_the_rest_one_by_one():
    b, lw = lowerer(ints={"t": "%r9"}, auto_unroll=True)
    lw.lower_body([strided_loop("a0", 1024, var("t"), lit(128), [load("v", "x", var("a0")), accum("acc", "v")])])
    text = body_text(b)
    # at most 1024/128 = 8 iterations: eight copies share one address, 128 elements = 512 bytes apart
    assert [line.split("[")[1] for line in loads(b)[:3]] == ["%rd4];", "%rd4+512];", "%rd4+1024];"]
    assert len(loads(b)) == 8 + 1  # plus the remainder loop's one
    assert "add.s32 %r1, %r1, 1024;" in text  # the counter jumps eight strides
    assert "add.s32 %r1, %r1, 128;" in text  # the remainder steps one


def test_the_factor_is_capped_by_the_body_size():
    b, lw = lowerer(ints={"t": "%r9"}, auto_unroll=True)
    body = [load("v", "x", var("a0"))] + [accum(f"acc{i}", "v") for i in range(15)]
    lw.lower_body([strided_loop("a0", 65536, var("t"), lit(128), body)])
    assert len(loads(b)) == 4 + 1  # 64 statements / 16 per copy


def test_the_accumulation_order_is_unchanged():
    b, lw = lowerer(ints={"t": "%r9"}, auto_unroll=True)
    lw.lower_body([strided_loop("a0", 1024, var("t"), lit(128), [load("v", "x", var("a0")), accum("acc", "v")])])
    adds = [line for line in body_text(b) if line.startswith("add.f32")]
    assert adds[:3] == ["add.f32 %f1, %f1, %f2;", "add.f32 %f1, %f1, %f3;", "add.f32 %f1, %f1, %f4;"]


def test_a_thread_with_only_two_iterations_unrolls_by_two():
    b, lw = lowerer(ints={"t": "%r9"}, auto_unroll=True)
    lw.lower_body([strided_loop("a1", 256, var("t"), lit(128), [load("v", "x", var("a1")), accum("acc", "v")])])
    assert "add.s32 %r1, %r1, 256;" in body_text(b)
    assert len(loads(b)) == 3


def test_a_small_loop_is_copied_out_whole():
    b, lw = lowerer(ints={"t": "%r9"}, auto_unroll=True)
    index = binop("+", binop("*", var("t"), lit(32)), var("k"))
    lw.lower_body([init("acc"), loop("k", 8, [load("v", "x", index), accum("acc", "v")])])
    text = body_text(b)
    assert not any(line.startswith(("setp", "bra")) for line in text)
    assert [line.split("[")[1] for line in loads(b)][-1] == "%rd4+28];"


def test_a_large_loop_is_partially_unrolled():
    b, lw = lowerer(auto_unroll=True)
    lw.lower_body([init("acc"), loop("k", 1024, [load("v", "x", var("k")), accum("acc", "v")])])
    assert "add.s32 %r1, %r1, 16;" in body_text(b)


def test_loops_with_barriers_or_branches_stay_loops():
    b, lw = lowerer(ints={"t": "%r9", "lane": "%r8"}, auto_unroll=True)
    lw.lower_body([
        strided_loop("a0", 1024, var("t"), lit(128), [load("v", "x", var("a0")), sync()]),
        strided_loop("a1", 1024, var("t"), lit(128), [cond(binop("==", var("lane"), lit(0)), [load("u", "x", var("a1"))])]),
    ])
    assert len(loads(b)) == 2


def test_loops_inside_unrolled_copies_are_not_unrolled_again():
    b, lw = lowerer(auto_unroll=True)
    inner = loop("k", 8, [load("v", "x", var("k")), accum("acc", "v")])
    lw.lower_body([init("acc"), loop("j", 4, [inner], unroll=True)])
    assert len(loads(b)) == 4  # one rolled inner loop per copy of j


def test_without_auto_unroll_loops_stay_loops():
    b, lw = lowerer(ints={"t": "%r9"})
    lw.lower_body([strided_loop("a0", 1024, var("t"), lit(128), [load("v", "x", var("a0")), accum("acc", "v")])])
    assert len(loads(b)) == 1


def test_an_unroll_the_ir_asks_for_is_skipped_when_the_copies_would_be_huge():
    b, lw = lowerer(auto_unroll=True)
    inner = loop("k", 64, [load("v", "x", var("k")), accum("acc", "v")])
    lw.lower_body([init("acc"), loop("j", 64, [inner, inner], unroll=True)])  # 64 * 256 statements
    text = body_text(b)
    assert any(line.startswith("setp") for line in text)  # j stays a loop
    assert len(loads(b)) == 2 * 64  # its inner loops are copied out instead
