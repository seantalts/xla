load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/codegen/intrinsic:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "golden_baselines",
    hdrs = ["golden_baselines.h"],
)

cc_library(
    name = "accuracy_budget",
    hdrs = ["accuracy_budget.h"],
)

cc_library(
    name = "accuracy_test_framework",
    testonly = True,
    hdrs = ["accuracy_test_framework.h"],
    deps = [
        ":golden_baselines",
        "//xla:fp_util",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_for_library",
    ],
)

xla_cc_test(
    name = "intrinsic_accuracy_test",
    srcs = ["intrinsic_accuracy_test.cc"],
    deps = [
        ":accuracy_budget",
        ":accuracy_test_framework",
        ":golden_baselines",
        "//xla/codegen/intrinsic:erf",
        "//xla/codegen/intrinsic:exp",
        "//xla/codegen/intrinsic:intrinsic",
        "//xla/codegen/intrinsic:log1p",
        "//xla/codegen/intrinsic:rsqrt",
        "//xla/codegen/intrinsic:simple_jit_runner",
        "//xla/codegen/intrinsic:tanh",
        "//xla:xla_data_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:ir_headers",
    ],
)
