#!/usr/bin/env python3
"""POC-1c: merge fusion chains inside hoisted region call bodies into giant kLoop fusions.

For each target *_region computation (the to_apply of a hoisted call):
  - Classify instructions: OUTSIDE = {parameter, get-tuple-element, constant};
    INLINABLE = {fusion, bitcast, copy, concatenate, dynamic-update-slice}.
  - For every ROOT-tuple operand produced by an inlinable chain, build one giant
    fused computation that inlines the whole producing chain (fusions spliced,
    raw ops copied) down to OUTSIDE leaves, which become fusion parameters.
    Shared producers are duplicated into each consumer giant (POC-acceptable).
  - If --dus-leaf is set, DUS-rooted producers are not inlined mid-chain; they
    become materialized giants of their own (DUS stays a fusion root).

Usage: poc1_merge_fusions.py IN.txt OUT.txt [--dus-leaf] REGION1 REGION2 ...
       (region names without leading %; 'ALL' = every *_region to_apply target)
"""
import re, sys

def parse_operand_group(s, start):
    # s[start] == '(' ; return (group_string_without_outer_parens, end_index_after_close)
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return s[start+1:i], i+1
    raise ValueError('unbalanced parens: ' + s[:120])

class Inst:
    __slots__ = ('raw','is_root','name','shape','opcode','opgroup','operands','suffix')
    def __init__(self, line):
        self.raw = line
        m = re.match(r'\s*(ROOT )?(%[\w.\-]+) = ', line)
        if not m:
            raise ValueError('bad inst line: ' + line[:120])
        self.is_root = bool(m.group(1))
        self.name = m.group(2)
        rest = line[m.end():]
        # shape: either balanced-paren tuple or a token up to first space
        if rest[0] == '(':
            g, e = parse_operand_group(rest, 0)
            self.shape = rest[:e]
            rest = rest[e:].lstrip()
        else:
            sp = rest.index(' ')
            self.shape = rest[:sp]
            rest = rest[sp+1:]
        m2 = re.match(r'([\w\-]+)\(', rest)
        self.opcode = m2.group(1)
        g, e = parse_operand_group(rest, m2.end()-1)
        self.opgroup = g
        self.suffix = rest[e:]  # includes leading ', ...' attrs or ''
        clean = re.sub(r'/\*.*?\*/', '', g)
        self.operands = re.findall(r'%[\w.\-]+', clean)

def parse_module(path):
    lines = open(path).read().split('\n')
    segments = []  # ('raw', line) or ('comp', dict)
    i = 0
    while i < len(lines):
        l = lines[i]
        m = re.match(r'(ENTRY )?(%[\w.\-]+) (\(.*\{)\s*$', l)
        if m:
            body = []
            i += 1
            while lines[i].strip() != '}':
                body.append(lines[i]); i += 1
            segments.append(('comp', {'name': m.group(2), 'header': l, 'body': body}))
            i += 1
        else:
            segments.append(('raw', l)); i += 1
    return segments

OUTSIDE = {'parameter', 'get-tuple-element', 'constant'}
DUS = 'dynamic-update-slice'

def strip_layout(shape):
    return re.sub(r'\{[0-9,]*\}', '', shape)

def rename_tokens(text, mapping):
    return re.sub(r'%[\w.\-]+', lambda m: mapping.get(m.group(0), m.group(0)), text)

counter = [0]
def fresh(base):
    counter[0] += 1
    return f'{base}.gfz{counter[0]}'

def transform_region(comp, comps, dus_leaf, single_use=False):
    insts = [Inst(l) for l in comp['body']]
    by_name = {it.name: it for it in insts}
    root = [it for it in insts if it.is_root][0]
    assert root.opcode == 'tuple', comp['name']
    users = {}
    for it in insts:
        for o in (it.operands if it.opcode != 'constant' else []):
            users[o] = users.get(o, 0) + 1

    def inlinable(it):
        return it.opcode in ('fusion','bitcast','copy','concatenate',DUS)
    def dus_rooted(it):
        if it.opcode == DUS: return True
        if it.opcode == 'fusion':
            callee = re.search(r'calls=(%[\w.\-]+)', it.suffix).group(1)
            croot = [Inst(l) for l in comps[callee]['body'] if re.match(r'\s*ROOT ', l)][0]
            return croot.opcode == DUS
        return False

    new_comps = []          # list of (header, body_lines)
    giant_of = {}           # producer name -> giant fusion inst line name
    giant_lines = {}        # producer name -> giant fusion instruction line
    kept_single = set()     # producer names kept as original lines

    def build_giant(pname):
        """Build a giant fusion materializing the value of producer pname."""
        if pname in giant_of: return giant_of[pname]
        P = by_name[pname]
        # collect closure (topological via original order) and leaves
        closure = set(); leaves = []   # leaves ordered by first use
        leaf_seen = set()
        def visit(n):
            it = by_name.get(n)
            is_leaf = (it is None or not inlinable(it)
                       or (dus_leaf and n != pname and dus_rooted(it))
                       or (single_use and n != pname and users.get(n, 0) > 1))
            if is_leaf:
                if n not in leaf_seen:
                    leaf_seen.add(n); leaves.append(n)
                if it is not None and inlinable(it):
                    build_giant(n)  # materialize multi-use/DUS-rooted leaf first
                return
            if n in closure: return
            closure.add(n)
            for o in it.operands: visit(o)
        visit(pname)
        if len(closure) == 1:
            # nothing to merge; keep original instruction (fusion or raw op)
            kept_single.add(pname); giant_of[pname] = pname
            return pname
        cname = fresh('%gfzcomp')
        # map leaf -> param name
        pmap = {}
        plines = []
        for j, leaf in enumerate(leaves):
            leaf_it = by_name.get(leaf)
            if leaf_it is not None and leaf in giant_of and giant_of[leaf] != leaf:
                shape = leaf_it.shape  # value shape unchanged
            else:
                shape = leaf_it.shape if leaf_it else None
            assert shape is not None, f'leaf {leaf} not found in {comp["name"]}'
            par = f'%gfzpar{j}'
            pmap[leaf] = par
            plines.append(f'  {par} = {shape} parameter({j})')
        body = list(plines)
        valmap = dict(pmap)  # region-name -> name inside giant
        ordered = [it for it in insts if it.name in closure]
        for it in ordered:
            if it.opcode == 'fusion':
                callee = re.search(r'calls=(%[\w.\-]+)', it.suffix).group(1)
                cbody = [Inst(l) for l in comps[callee]['body']]
                imap = {}
                for ci in cbody:
                    if ci.opcode == 'parameter':
                        idx = int(ci.opgroup)
                        imap[ci.name] = valmap[it.operands[idx]]
                for ci in cbody:
                    if ci.opcode == 'parameter': continue
                    imap[ci.name] = fresh(ci.name)
                for ci in cbody:
                    if ci.opcode == 'parameter': continue
                    line = ci.raw.replace('ROOT ', '', 1) if ci.is_root else ci.raw
                    body.append(rename_tokens(line, imap))
                    if ci.is_root:
                        valmap[it.name] = imap[ci.name]
            else:
                nm = fresh(it.name)
                line = it.raw.replace('ROOT ', '', 1) if it.is_root else it.raw
                m = {it.name: nm}; m.update({o: valmap[o] for o in it.operands})
                body.append(rename_tokens(line, m))
                valmap[it.name] = nm
        # mark ROOT on the line defining valmap[pname]
        tgt = valmap[pname]
        for k, l in enumerate(body):
            if re.match(rf'\s*{re.escape(tgt)} = ', l):
                body[k] = re.sub(r'^(\s*)', r'\1ROOT ', l, count=1)
                break
        header_params = ', '.join(f'{pmap[leaf][1:]}: {strip_layout(by_name[leaf].shape)}' for leaf in leaves)
        header = f'{cname} ({header_params}) -> {strip_layout(P.shape)} {{'
        new_comps.append((header, body))
        gname = fresh('%gfzfus')
        opstr = ', '.join(valname_for(leaf) for leaf in leaves)
        giant_lines[pname] = f'  {gname} = {P.shape} fusion({opstr}), kind=kLoop, calls={cname}'
        giant_of[pname] = gname
        return gname

    def valname_for(n):
        # name usable in region body for value n (giant name if materialized)
        return giant_of.get(n, n)

    for opnd in root.operands:
        it = by_name.get(opnd)
        if it is not None and inlinable(it):
            build_giant(opnd)

    # rebuild region body in original topological order
    remap = {n: g for n, g in giant_of.items() if g != n}
    newbody = []
    for it in insts:
        if it.is_root: continue
        if it.opcode in OUTSIDE:
            newbody.append(it.raw)
        elif it.name in kept_single:
            newbody.append(rename_tokens(it.raw, remap) if any(o in remap for o in it.operands) else it.raw)
        elif it.name in giant_lines:
            newbody.append(giant_lines[it.name])
        # fully inlined instructions dropped
    tuple_ops = rename_tokens(root.opgroup, remap)
    newbody.append(f'  ROOT {root.name} = {root.shape} tuple({tuple_ops}){root.suffix}')
    comp['body'] = newbody
    return new_comps

def referenced_comps(segments, entry_name):
    comps = {c['name']: c for k, c in segments if k == 'comp'}
    seen = set(); stack = [entry_name]
    while stack:
        n = stack.pop()
        if n in seen or n not in comps: continue
        seen.add(n)
        text = '\n'.join(comps[n]['body'])
        for m in re.finditer(r'(?:calls|to_apply|condition|body)=(%[\w.\-]+)', text):
            stack.append(m.group(1))
        # also handle brace-list form: calls={%a, %b}
        for m in re.finditer(r'(?:calls|branch_computations)=\{([^}]*)\}', text):
            for nm in re.findall(r'%[\w.\-]+', m.group(1)):
                stack.append(nm)
    return seen

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    dus_leaf = '--dus-leaf' in sys.argv
    single_use = '--single-use' in sys.argv
    inp, outp, targets = args[0], args[1], args[2:]
    segments = parse_module(inp)
    comps = {c['name']: c for k, c in segments if k == 'comp'}
    entry = [c['name'] for k, c in segments if k == 'comp' and c['header'].startswith('ENTRY')][0]
    if targets == ['ALL']:
        text = open(inp).read()
        targets = sorted(set(re.findall(r'to_apply=%([\w.\-]+_region)\b', text)))
    inserted = {}
    for t in targets:
        name = '%' + t
        newc = transform_region(comps[name], comps, dus_leaf, single_use)
        inserted[name] = newc
    out = []
    for k, c in segments:
        if k == 'raw':
            out.append(c); continue
        if c['name'] in inserted:
            for hdr, body in inserted[c['name']]:
                out.append(hdr); out.extend(body); out.append('}'); out.append('')
        out.append(c['header']); out.extend(c['body']); out.append('}')
    open(outp, 'w').write('\n'.join(out))
    print(f'transformed {len(inserted)} regions, added {sum(len(v) for v in inserted.values())} giant computations')

if __name__ == '__main__':
    main()
