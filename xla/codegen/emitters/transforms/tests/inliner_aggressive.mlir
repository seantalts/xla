// RUN: emitters_opt %s -split-input-file -xla-inliner="policy=aggressive" \
// RUN: | FileCheck %s

// Chain where consecutive levels share no callees (e.g. rotation/quaternion
// chains emitted for kinematic models): every component of one level calls
// every component of the previous level exactly once. The conservative
// policy never inlines these (no shared third callee), leaving a call per
// use; each surviving call re-evaluates the callee's entire transitive
// computation, so the recomputation compounds exponentially with chain
// depth. The aggressive policy must inline the whole chain.
module {
  func.func private @x0(%a: f32, %b: f32) -> f32 {
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }
  func.func private @y0(%a: f32, %b: f32) -> f32 {
    %ret = arith.mulf %a, %b : f32
    return %ret : f32
  }
  func.func private @x1(%a: f32, %b: f32) -> f32 {
    %x = xla.pure_call @x0(%a, %b) : (f32, f32) -> (f32)
    %y = xla.pure_call @y0(%a, %b) : (f32, f32) -> (f32)
    %ret = arith.subf %x, %y : f32
    return %ret : f32
  }
  func.func private @y1(%a: f32, %b: f32) -> f32 {
    %x = xla.pure_call @x0(%a, %b) : (f32, f32) -> (f32)
    %y = xla.pure_call @y0(%a, %b) : (f32, f32) -> (f32)
    %ret = arith.divf %x, %y : f32
    return %ret : f32
  }
  func.func @caller(%a: f32, %b: f32) -> f32 {
    %x = xla.pure_call @x1(%a, %b) : (f32, f32) -> (f32)
    %y = xla.pure_call @y1(%a, %b) : (f32, f32) -> (f32)
    %ret = arith.addf %x, %y : f32
    return %ret : f32
  }
}

// CHECK-LABEL: func.func @caller
// CHECK-NOT: xla.pure_call

// -----

// A callee with two callers, each with a single call site. The aggressive
// policy inlines both (duplicating the body); the conservative policy would
// keep the calls because the callers share no third callee.
module {
  func.func private @large(%a: f32, %b: f32) -> f32 {
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %a, %mul : f32
    %div = arith.divf %add, %b : f32
    %sub = arith.subf %div, %a : f32
    %atan2 = math.atan2 %b, %sub : f32
    return %atan2 : f32
  }
  func.func private @lhs(%a: f32, %b: f32) -> f32 {
    %ret = xla.pure_call @large(%a, %b) : (f32, f32) -> (f32)
    return %ret : f32
  }
  func.func private @rhs(%a: f32, %b: f32) -> f32 {
    %ret = xla.pure_call @large(%b, %a) : (f32, f32) -> (f32)
    return %ret : f32
  }
  func.func @caller(%a: f32, %b: f32) -> f32 {
    %x = xla.pure_call @lhs(%a, %b) : (f32, f32) -> (f32)
    %y = xla.pure_call @rhs(%a, %b) : (f32, f32) -> (f32)
    %ret = arith.addf %x, %y : f32
    return %ret : f32
  }
}

// CHECK-LABEL: func.func @caller
// CHECK-NOT: xla.pure_call
// CHECK: math.atan2
// CHECK-NOT: xla.pure_call
// CHECK: math.atan2
// CHECK-NOT: xla.pure_call

// -----

// Two calls to the same callee with distinct arguments in one caller are
// not inlined even by the aggressive policy: there is no recomputation to
// eliminate (the arguments differ) and no CSE to collapse the duplicated
// bodies. Identical-argument duplicates are merged by CSE before the
// inliner sees them.
module {
  func.func private @add(%a: f32, %b: f32) -> f32 {
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }
  func.func @caller(%a: f32, %b: f32) -> f32 {
    %x = xla.pure_call @add(%a, %b) : (f32, f32) -> (f32)
    %y = xla.pure_call @add(%b, %a) : (f32, f32) -> (f32)
    %ret = arith.mulf %x, %y : f32
    return %ret : f32
  }
}

// CHECK-LABEL: func.func @caller
// CHECK: xla.pure_call @add
// CHECK: xla.pure_call @add

// -----

// The noinline attribute is respected regardless of policy.
module {
  func.func private @callee(%a: f32) -> f32 {
    %ret = arith.addf %a, %a : f32
    return %ret : f32
  }
  func.func @caller(%a: f32) -> f32 {
    %ret = xla.pure_call @callee(%a) {noinline} : (f32) -> (f32)
    return %ret : f32
  }
}

// CHECK-LABEL: func.func @caller
// CHECK: xla.pure_call @callee
