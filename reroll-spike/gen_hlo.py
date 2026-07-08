#!/usr/bin/env python3
"""Generate HLO benchmark families for the loop re-rolling spike.

Family 1 (loop-carried, scan-like):
  unrolled_N.hlo : static chain of slice -> dot -> tanh -> constant-offset DUS
  rolled_N.hlo   : while loop, trip count N, dynamic-slice/DUS at offset 4*iter

Family 2 (independent iterations):
  indep_unrolled_N.hlo : N static rows, slice -> dot -> tanh, concat
  indep_rolled_N.hlo   : while loop row-at-a-time, DUS into zero-init output
  indep_batched_N.hlo  : single [N,4]x[4,4] dot + tanh

All variants of a family take identical (shape, order) parameters so
bench_hlo --seed=42 feeds them identical inputs.
"""
import os, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hlo")
NS = [8, 32, 128, 512]


def unrolled(n):
    d = 4 * n
    lines = [f"HloModule unrolled_{n}", "", "ENTRY main {"]
    lines.append(f"  state_in = f32[{d}] parameter(0)")
    lines.append("  weight = f32[4,4] parameter(1)")
    prev = "state_in"
    for i in range(n):
        lo, hi = 4 * i, 4 * i + 4
        root = "ROOT " if i == n - 1 else ""
        lines.append(f"  xsl_{i} = f32[4] slice({prev}), slice={{[{lo}:{hi}]}}")
        lines.append(
            f"  dotv_{i} = f32[4] dot(weight, xsl_{i}), "
            "lhs_contracting_dims={1}, rhs_contracting_dims={0}")
        lines.append(f"  yv_{i} = f32[4] tanh(dotv_{i})")
        lines.append(f"  off_{i} = s32[] constant({lo})")
        lines.append(
            f"  {root}state_{i} = f32[{d}] dynamic-update-slice({prev}, yv_{i}, off_{i})")
        prev = f"state_{i}"
    lines.append("}")
    return "\n".join(lines) + "\n"


def rolled(n):
    d = 4 * n
    t = f"(s32[], f32[{d}], f32[4,4])"
    return f"""HloModule rolled_{n}

body {{
  loop_var = {t} parameter(0)
  iter = s32[] get-tuple-element(loop_var), index=0
  state = f32[{d}] get-tuple-element(loop_var), index=1
  weight = f32[4,4] get-tuple-element(loop_var), index=2
  four = s32[] constant(4)
  offset = s32[] multiply(iter, four)
  xv = f32[4] dynamic-slice(state, offset), dynamic_slice_sizes={{4}}
  dotv = f32[4] dot(weight, xv), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}
  yv = f32[4] tanh(dotv)
  newstate = f32[{d}] dynamic-update-slice(state, yv, offset)
  one = s32[] constant(1)
  newiter = s32[] add(iter, one)
  ROOT tup = {t} tuple(newiter, newstate, weight)
}}

cond {{
  loop_var = {t} parameter(0)
  iter = s32[] get-tuple-element(loop_var), index=0
  limit = s32[] constant({n})
  ROOT lt = pred[] compare(iter, limit), direction=LT
}}

ENTRY main {{
  state_in = f32[{d}] parameter(0)
  weight = f32[4,4] parameter(1)
  zero = s32[] constant(0)
  init = {t} tuple(zero, state_in, weight)
  wl = {t} while(init), condition=cond, body=body
  ROOT out = f32[{d}] get-tuple-element(wl), index=1
}}
"""


def indep_unrolled(n):
    lines = [f"HloModule indep_unrolled_{n}", "", "ENTRY main {"]
    lines.append(f"  xs = f32[{n},4] parameter(0)")
    lines.append("  weight = f32[4,4] parameter(1)")
    rows = []
    for i in range(n):
        lines.append(
            f"  row_{i} = f32[1,4] slice(xs), slice={{[{i}:{i+1}], [0:4]}}")
        lines.append(
            f"  dotv_{i} = f32[1,4] dot(row_{i}, weight), "
            "lhs_contracting_dims={1}, rhs_contracting_dims={0}")
        lines.append(f"  yv_{i} = f32[1,4] tanh(dotv_{i})")
        rows.append(f"yv_{i}")
    lines.append(
        f"  ROOT out = f32[{n},4] concatenate({', '.join(rows)}), dimensions={{0}}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def indep_rolled(n):
    t = f"(s32[], f32[{n},4], f32[{n},4], f32[4,4])"
    return f"""HloModule indep_rolled_{n}

body {{
  loop_var = {t} parameter(0)
  iter = s32[] get-tuple-element(loop_var), index=0
  xs = f32[{n},4] get-tuple-element(loop_var), index=1
  acc = f32[{n},4] get-tuple-element(loop_var), index=2
  weight = f32[4,4] get-tuple-element(loop_var), index=3
  zero = s32[] constant(0)
  row = f32[1,4] dynamic-slice(xs, iter, zero), dynamic_slice_sizes={{1,4}}
  dotv = f32[1,4] dot(row, weight), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}
  yv = f32[1,4] tanh(dotv)
  newacc = f32[{n},4] dynamic-update-slice(acc, yv, iter, zero)
  one = s32[] constant(1)
  newiter = s32[] add(iter, one)
  ROOT tup = {t} tuple(newiter, xs, newacc, weight)
}}

cond {{
  loop_var = {t} parameter(0)
  iter = s32[] get-tuple-element(loop_var), index=0
  limit = s32[] constant({n})
  ROOT lt = pred[] compare(iter, limit), direction=LT
}}

ENTRY main {{
  xs = f32[{n},4] parameter(0)
  weight = f32[4,4] parameter(1)
  zero = s32[] constant(0)
  zf = f32[] constant(0)
  acc0 = f32[{n},4] broadcast(zf), dimensions={{}}
  init = {t} tuple(zero, xs, acc0, weight)
  wl = {t} while(init), condition=cond, body=body
  ROOT out = f32[{n},4] get-tuple-element(wl), index=2
}}
"""


def indep_batched(n):
    return f"""HloModule indep_batched_{n}

ENTRY main {{
  xs = f32[{n},4] parameter(0)
  weight = f32[4,4] parameter(1)
  dotv = f32[{n},4] dot(xs, weight), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}
  ROOT out = f32[{n},4] tanh(dotv)
}}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    for n in NS:
        for name, fn in [
            (f"unrolled_{n}", unrolled),
            (f"rolled_{n}", rolled),
            (f"indep_unrolled_{n}", indep_unrolled),
            (f"indep_rolled_{n}", indep_rolled),
            (f"indep_batched_{n}", indep_batched),
        ]:
            path = os.path.join(OUT, name + ".hlo")
            with open(path, "w") as f:
                f.write(fn(n))
            print(path)


if __name__ == "__main__":
    main()
