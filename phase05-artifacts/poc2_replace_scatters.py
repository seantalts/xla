#!/usr/bin/env python3
"""POC-2: replace every scatter with a cheap, hoistable, shape-correct substitute
that keeps all three inputs live:
  result = add(operand, broadcast(add(convert(reduce_sum(updates)),
                                      convert(reduce_sum(indices)))))
Semantics intentionally broken; structural ceiling experiment only.

Usage: poc2_replace_scatters.py IN.txt OUT.txt
"""
import re, sys

def shape_info(shape):
    m = re.match(r'([a-z0-9]+)\[([0-9,]*)\]', shape)
    dtype = m.group(1)
    dims = [d for d in m.group(2).split(',') if d != '']
    return dtype, len(dims)

def main():
    inp, outp = sys.argv[1], sys.argv[2]
    lines = open(inp).read().split('\n')
    # pass 1: name -> shape per computation (names are unique module-wide in dumps,
    # so a single global map is fine)
    shapes = {}
    for l in lines:
        m = re.match(r'\s*(?:ROOT )?(%[\w.\-]+) = (\([^=]*?\)|\S+) [\w\-]+\(', l)
        if m:
            shapes[m.group(1)] = m.group(2)
    helper_dtypes = set()
    out = []
    nrepl = 0
    for l in lines:
        m = re.match(r'(\s*)(ROOT )?(%[\w.\-]+) = (\S+) scatter\(([^)]*)\)', l)
        if not m:
            out.append(l)
            continue
        indent, rootkw, name, shape, opgroup = m.groups()
        ops = re.findall(r'%[\w.\-]+', re.sub(r'/\*.*?\*/', '', opgroup))
        assert len(ops) == 3, f'variadic scatter not handled: {l[:120]}'
        operand, indices, updates = ops
        res_dt, _ = shape_info(shape)
        upd_shape = shapes[updates]; idx_shape = shapes[indices]
        upd_dt, upd_rank = shape_info(upd_shape)
        idx_dt, idx_rank = shape_info(idx_shape)
        helper_dtypes.update([upd_dt, idx_dt])
        tag = 'gfzq' + re.sub(r'[^A-Za-z0-9]', '', name)
        b = []
        # reduce updates to scalar
        if upd_rank > 0:
            b.append(f'{indent}%{tag}zu = {upd_dt}[] constant(0)')
            b.append(f'{indent}%{tag}rsu = {upd_dt}[] reduce({updates}, %{tag}zu), '
                     f'dimensions={{{",".join(str(i) for i in range(upd_rank))}}}, to_apply=%gfzaddred{upd_dt}')
            usum = f'%{tag}rsu'
        else:
            usum = updates
        if upd_dt != res_dt:
            b.append(f'{indent}%{tag}cvu = {res_dt}[] convert({usum})')
            usum = f'%{tag}cvu'
        # reduce indices to scalar
        if idx_rank > 0:
            b.append(f'{indent}%{tag}zi = {idx_dt}[] constant(0)')
            b.append(f'{indent}%{tag}rsi = {idx_dt}[] reduce({indices}, %{tag}zi), '
                     f'dimensions={{{",".join(str(i) for i in range(idx_rank))}}}, to_apply=%gfzaddred{idx_dt}')
            isum = f'%{tag}rsi'
        else:
            isum = indices
        b.append(f'{indent}%{tag}cvi = {res_dt}[] convert({isum})')
        isum = f'%{tag}cvi'
        b.append(f'{indent}%{tag}sum = {res_dt}[] add({usum}, {isum})')
        b.append(f'{indent}%{tag}brd = {shape} broadcast(%{tag}sum), dimensions={{}}')
        b.append(f'{indent}{rootkw or ""}{name} = {shape} add({operand}, %{tag}brd)')
        out.extend(b)
        nrepl += 1
    # insert helper computations after the HloModule line
    helpers = []
    for dt in sorted(helper_dtypes):
        helpers += [f'%gfzaddred{dt} (gfzlhs{dt}: {dt}[], gfzrhs{dt}: {dt}[]) -> {dt}[] {{',
                    f'  %gfzlhs{dt} = {dt}[] parameter(0)',
                    f'  %gfzrhs{dt} = {dt}[] parameter(1)',
                    f'  ROOT %gfzsum{dt} = {dt}[] add(%gfzlhs{dt}, %gfzrhs{dt})',
                    '}', '']
    for i, l in enumerate(out):
        if l.startswith('%'):  # first computation definition
            out = out[:i] + helpers + out[i:]
            break
    open(outp, 'w').write('\n'.join(out))
    print(f'replaced {nrepl} scatters; helper dtypes: {sorted(helper_dtypes)}')

if __name__ == '__main__':
    main()
