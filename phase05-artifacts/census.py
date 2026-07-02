import re, sys
from collections import Counter, defaultdict

path="/private/tmp/claude-501/-Users-xitrium-claud/a532ed4d-10b9-40ad-8813-877a77016f7d/scratchpad/dumpA/module_0001.jit_run.cpu_after_optimizations.txt"
lines=open(path).read().split("\n")

# Parse computations: a computation starts with a line beginning "%name (" or "ENTRY %name" and its body is between "{" and matching top-level "}".
# We'll do a simple brace-depth parser at the file level.
comps={}  # name -> list of instruction dicts
order=[]
i=0
N=len(lines)
cur=None
curname=None
depth=0
def comp_name(line):
    m=re.match(r'^(ENTRY\s+)?%?([A-Za-z0-9._-]+)\s*\(', line)
    if m: return m.group(2)
    return None

# Identify computation header lines: top-level (depth 0) lines that end with "{" and match name(
for idx,line in enumerate(lines):
    stripped=line.rstrip()
    if depth==0:
        m=re.match(r'^(ENTRY\s+)?%([A-Za-z0-9._-]+)\s*\(.*\)\s*->.*\{\s*$', stripped)
        if m:
            curname=m.group(2)
            comps[curname]=[]
            order.append(curname)
            depth=1
            continue
    else:
        # count braces to know when computation ends
        # instruction lines are at indent, but nested fusion regions won't appear here (they're separate comps)
        # A top-level instruction inside comp: starts with two spaces then %name = or ROOT
        m=re.match(r'^\s+(ROOT\s+)?%([A-Za-z0-9._-]+)\s*=\s*(.*)$', line)
        if m and depth==1:
            name=m.group(2)
            rest=m.group(3)
            # opcode: find token before first '(' that is not part of a shape tuple.
            # Strategy: remove leading shape. Shape is either scalar/array like f32[..]{..} or tuple (....).
            # The opcode is the alpha-token immediately followed by '(' that starts the operand list.
            # Find all "word(" occurrences; opcode is the first word( where word is a known-ish opcode OR
            # simpler: strip a leading tuple-shape "(...)" if present, then leading array shapes, then take first word.
            r=rest
            # strip leading tuple shape
            if r.startswith('('):
                # find matching close paren
                d=0;j=0
                for j,ch in enumerate(r):
                    if ch=='(':d+=1
                    elif ch==')':
                        d-=1
                        if d==0:break
                r=r[j+1:].strip()
            else:
                # strip leading array shape token(s) up to first space
                # shape like f32[300,1]{1,0} then space then opcode
                sp=r.find(' ')
                if sp>0:
                    r=r[sp+1:].strip()
            mo=re.match(r'([a-z][a-zA-Z0-9-]*)\(', r)
            op=mo.group(1) if mo else 'UNKNOWN:'+r[:20]
            calls=re.search(r'calls=%([A-Za-z0-9._-]+)', rest)
            toapply=re.search(r'to_apply=%([A-Za-z0-9._-]+)', rest)
            called = calls.group(1) if calls else (toapply.group(1) if toapply else None)
            comps[curname].append(dict(name=name,op=op,called=called,root=bool(m.group(1))))
        # track depth via braces on this line for nested structures within the computation
        opens=line.count('{'); closes=line.count('}')
        depth+=opens-closes
        if depth<=0:
            depth=0
            curname=None

print("Parsed %d computations"%len(comps))

# Thunk-generating opcodes (dispatch a kernel or copy). GTE/param/const/tuple/bitcast do not.
DISPATCH={'fusion','copy','scatter','scatter-add','custom-call','convolution','dot','reduce','reduce-window','sort','rng','dynamic-update-slice','dynamic-slice','pad','concatenate','reverse','transpose','select-and-scatter'}
NONDISPATCH={'get-tuple-element','parameter','constant','tuple','bitcast','while','call'}

def classify(comp):
    c=Counter()
    for ins in comps[comp]:
        c[ins['op']]+=1
    return c

def leaf_thunks(comp, seen=None):
    # count dispatched thunks, expanding call/while recursively via 'called'
    if seen is None: seen=set()
    total=0
    detail=Counter()
    for ins in comps.get(comp,[]):
        op=ins['op']
        if op=='call':
            sub=leaf_thunks(ins['called'], seen)
            total+=sub[0]
            detail+=sub[1]
        elif op=='fusion':
            total+=1; detail['fusion']+=1
        elif op=='copy':
            total+=1; detail['copy']+=1
        elif op in DISPATCH:
            total+=1; detail[op]+=1
        # else non-dispatch
    return total, detail

# scatter detection: a fusion whose called computation contains a scatter/scatter-add op,
# OR name contains 'scatter'
def is_scatter_fusion(ins):
    if ins['op']!='fusion': return False
    if 'scatter' in ins['name']: return True
    called=ins.get('called')
    if called and called in comps:
        for x in comps[called]:
            if 'scatter' in x['op']: return True
    return False

# ---- BODY ----
BODY='wide.wide.wide.region_0.133.clone.sunk.clone.clone.sunk.clone'
print("\n===== WHILE BODY (%s) ====="%BODY)
bc=classify(BODY)
print("Top-level instr histogram:", dict(bc))
print("Total top-level instrs:", sum(bc.values()))
body_calls=[ins for ins in comps[BODY] if ins['op']=='call']
body_fusions=[ins for ins in comps[BODY] if ins['op']=='fusion']
body_copies=[ins for ins in comps[BODY] if ins['op']=='copy']
print("call thunks:",len(body_calls),"| top-level fusion thunks:",len(body_fusions),"| copy thunks:",len(body_copies))
scat=[ins for ins in body_fusions if is_scatter_fusion(ins)]
print("top-level fusions that are scatters:",len(scat))
print("top-level non-scatter fusions:",len(body_fusions)-len(scat))

# per region sizes and leaf thunks
print("\n-- per hoisted region (called by body calls) --")
region_leaf_total=0
region_scatter_total=0
region_fusion_total=0
for ins in body_calls:
    rn=ins['called']
    size=len(comps.get(rn,[]))
    lt,detail=leaf_thunks(rn)
    # scatters within region
    sc=sum(1 for x in comps.get(rn,[]) if is_scatter_fusion(x))
    region_leaf_total+=lt
    region_scatter_total+=sc
    print(f"  {ins['name']:>10} -> {rn:45} instrs={size:3d} leaf_thunks={lt:3d} scatter_fusions={sc}  detail={dict(detail)}")

print("\nBODY leaf-thunk rollup (fully expanding calls):")
btotal,bdetail=leaf_thunks(BODY)
print("  total leaf dispatched thunks per body iter:",btotal)
print("  detail:",dict(bdetail))
# count scatter fusions everywhere in body (top-level + regions)
def all_scatter(comp, seen=None):
    if seen is None: seen=set()
    tot=0
    for ins in comps.get(comp,[]):
        if is_scatter_fusion(ins): tot+=1
        if ins['op']=='call':
            tot+=all_scatter(ins['called'],seen)
    return tot
print("  scatter fusions per body iter (all levels):",all_scatter(BODY))

# ---- ENTRY ----
ENTRY='main.135'
print("\n===== ENTRY (%s) ====="%ENTRY)
ec=classify(ENTRY)
print("Top-level instr histogram:", dict(ec))
print("Total top-level instrs:",sum(ec.values()))
etotal,edetail=leaf_thunks(ENTRY)
# entry contains the while; the while's body thunks should NOT be counted per-entry (they run 300x)
# leaf_thunks doesn't expand 'while' (it's non-dispatch and has no 'called' captured), good.
print("Entry leaf thunks (excluding while body):",etotal,"detail:",dict(edetail))
print("  scatter fusions in entry (top-level):",sum(1 for x in comps[ENTRY] if is_scatter_fusion(x)))
ecalls=[ins for ins in comps[ENTRY] if ins['op']=='call']
print("  entry call thunks:",len(ecalls), [i['name'] for i in ecalls])
for ins in ecalls:
    rn=ins['called']; lt,detail=leaf_thunks(rn)
    print("    ",ins['name'],"->",rn,"instrs=",len(comps.get(rn,[])),"leaf=",lt)

print("\n===== THUNKS PER STEP =====")
print("  body leaf thunks:",btotal," x300 =",btotal*300)
print("  entry leaf thunks:",etotal)
print("  TOTAL leaf dispatched thunks/step:",btotal*300+etotal)
print("  (top-level body thunks incl calls-as-1: fusions+copies+calls =",len(body_fusions)+len(body_copies)+len(body_calls),")")
