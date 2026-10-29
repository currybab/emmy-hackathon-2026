"""Loop-invariant code motion: work that is the same in every iteration moves before the loop."""
from irkit import accum, assign, binop, body_text, cond, init, lit, load, loop, lowerer, strided_loop, var, write
from ptxc.licm import hoist_invariants


def test_an_instruction_on_outside_values_moves_out():
    body = ["\tmul.lo.s32 %r5, %r1, 32;", "\tadd.s32 %r6, %r5, %r9;"]
    hoisted, rest = hoist_invariants(body, variant={"%r9"}, allow_loads=True)
    assert hoisted == ["\tmul.lo.s32 %r5, %r1, 32;"] and rest == ["\tadd.s32 %r6, %r5, %r9;"]


def test_a_chain_of_invariant_work_moves_out_together():
    body = ["\tmul.wide.s32 %rd5, %r1, 4;", "\tadd.s64 %rd6, %rd1, %rd5;", "\tld.global.f32 %f3, [%rd6];"]
    hoisted, rest = hoist_invariants(body, variant=set(), allow_loads=True)
    assert hoisted == body and rest == []


def test_loads_stay_when_not_allowed_or_when_the_loop_stores():
    body = ["\tld.global.f32 %f3, [%rd6];"]
    assert hoist_invariants(body, variant=set(), allow_loads=False)[0] == []
    stores = body + ["\tst.global.f32 [%rd7], %f3;"]
    assert hoist_invariants(stores, variant=set(), allow_loads=True)[0] == []


def test_an_accumulator_updated_in_place_stays():
    body = ["\tadd.f32 %f1, %f1, %f2;"]
    assert hoist_invariants(body, variant=set(), allow_loads=True)[0] == []


def test_a_register_defined_twice_stays():
    body = ["\tmov.f32 %f4, 0f3F800000;", "\tadd.f32 %f4, %f4, %f4;"]
    assert hoist_invariants(body, variant=set(), allow_loads=True)[0] == []


def test_work_inside_a_branch_or_inner_loop_stays():
    body = ["\t@%p2 bra $L_skip_3;", "\tld.global.f32 %f3, [%rd6];", "$L_skip_3:", "\tld.global.f32 %f4, [%rd6+4];"]
    hoisted, _ = hoist_invariants(body, variant=set(), allow_loads=True)
    assert hoisted == ["\tld.global.f32 %f4, [%rd6+4];"]


def test_a_row_read_in_every_iteration_is_loaded_once_before_the_loop():
    # acc += x[t*32 + k] * x[512 + j]: the first load does not depend on j
    b, lw = lowerer(ints={"t": "%r9"})
    body = [load("a", "x", binop("+", binop("*", var("t"), lit(32)), lit(3))), load("c", "x", binop("+", lit(512), var("j"))),
            assign("p", "multiply", "a", "c"), accum("acc", "p")]
    lw.lower_body([init("acc"), loop("j", 64, body)])
    text = body_text(b)
    top = next(i for i, line in enumerate(text) if line.endswith(":"))
    loads_before = [line for line in text[:top] if line.startswith("ld.")]
    loads_inside = [line for line in text[top:] if line.startswith("ld.")]
    assert len(loads_before) == 1 and len(loads_inside) == 1


def test_a_strided_loop_keeps_its_loads_since_it_may_run_zero_times():
    b, lw = lowerer(ints={"t": "%r9", "u": "%r8"})
    body = [load("a", "x", var("u")), load("c", "x", var("a0")), assign("p", "multiply", "a", "c"), accum("acc", "p")]
    lw.lower_body([strided_loop("a0", 1024, var("t"), lit(128), body)])
    text = body_text(b)
    top = next(i for i, line in enumerate(text) if line.endswith(":"))
    assert not any(line.startswith("ld.") for line in text[:top])
