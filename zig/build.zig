const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const usearch_mod = b.addModule("usearch_zig", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const usearch_cpp_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
    });

    usearch_cpp_mod.addCSourceFile(.{
        .file = b.path("usearch/include/lib.cpp"),
        .flags = &.{
            "-std=c++17",
            "-fno-exceptions",
            "-fno-rtti",
        },
    });

    usearch_cpp_mod.addIncludePath(b.path("usearch/include"));
    usearch_cpp_mod.link_libcpp = true;

    const usearch_lib = b.addLibrary(.{
        .name = "usearch",
        .root_module = usearch_cpp_mod,
        .linkage = .static,
    });

    const exe = b.addExecutable(.{
        .name = "usearch_zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "usearch_zig", .module = usearch_mod },
            },
        }),
    });

    exe.root_module.link_libc = true;
    exe.root_module.link_libcpp = true;
    exe.root_module.linkLibrary(usearch_lib);
    exe.root_module.addIncludePath(b.path("usearch/include"));

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the usearch_zig executable");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| run_cmd.addArgs(args);

    const mod_tests = b.addTest(.{
        .root_module = usearch_mod,
    });

    mod_tests.root_module.addIncludePath(b.path("usearch/include"));
    mod_tests.root_module.link_libcpp = true;
    mod_tests.root_module.linkLibrary(usearch_lib);

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const test_step = b.step("test", "Run tests for usearch_zig");
    test_step.dependOn(&run_mod_tests.step);
}
