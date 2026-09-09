const std = @import("std");
const builtin = @import("builtin");

// The manifest's version has to be a literal, so the tracked `VERSION` file is compared against it
// here rather than read into it - a release that bumps one and forgets the other fails the build.
comptime {
    const declared = @import("build.zig.zon").version;
    const tracked = std.mem.trim(u8, @embedFile("VERSION"), &std.ascii.whitespace);
    if (!std.mem.eql(u8, declared, tracked))
        @compileError("`build.zig.zon` declares version " ++ declared ++ ", `VERSION` says " ++ tracked);
}

pub fn build(b: *std.Build) void {
    // Matches `minimum_zig_version` in `build.zig.zon`.
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 16) {
        @panic("USearch requires Zig 0.16.0 or later. Please upgrade your Zig toolchain.");
    }

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // No C or C++ is compiled here: the engine is CMake's to build, and this module consumes it
    // through the stable C headers alone. `-Dlib-dir` names a directory holding the library, and
    // may be repeated; left alone it points where an in-tree CMake build leaves it.
    //
    // The shared `usearch_c` is what gets linked, not the engine's internal archives. It resolves
    // its own dependencies - yyjson, ForkUnion, and the C++ runtime CMake built it against - so
    // one `-l` finishes the link. An archive would leak every libstdc++ symbol into this link,
    // and `linkSystemLibrary("stdc++")` cannot answer them: Zig reads that name as a request for
    // its own libc++ and substitutes it, leaving a GCC-built archive short.
    //
    // The brute-force stand-in publishes the same name in a directory of its own, which is what
    // lets `-Dlib-dir=build/stub` run this binding's tests without building the engine at all.
    const library_directories = b.option([]const []const u8, "lib-dir", "Directory holding the engine library; repeatable") orelse
        &[_][]const u8{"build_release/c"};
    const library_names = b.option([]const []const u8, "lib", "Engine library to link; repeatable") orelse
        &[_][]const u8{"usearch_c"};

    // The binding itself, which a dependent imports. It declares the C ABI with `extern fn`, so it
    // carries the link directives too - importing it should not mean relinking it by hand.
    const module = b.addModule("usearch", .{
        .root_source_file = b.path("zig/usearch.zig"),
        .target = target,
        .optimize = optimize,
    });
    link(b, module, library_directories, library_names);

    // The same sources archived. An archive resolves no symbols, so it carries no link
    // directives and compiles every declaration without the engine on disk - which is what makes
    // a bare `zig build` a compile check.
    const library = b.addLibrary(.{
        .name = "usearch",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/usearch.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(library);

    // The tests link the engine, because a test binary has to resolve every `extern fn` the
    // module declares - `-Dlib-dir` may point them at the brute-force stand-in. The engine-free
    // check is a bare `zig build`, which archives the same sources and resolves nothing.
    const test_step = b.step("test", "Run binding tests");
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/usearch.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    link(b, tests.root_module, library_directories, library_names);
    const run_tests = b.addRunArtifact(tests);
    test_step.dependOn(&run_tests.step);
}

/// Points a module at the directories holding the engine's shared library, and at the library
/// itself. Nothing else is requested: the shared object resolves its own dependencies.
fn link(
    b: *std.Build,
    module: *std.Build.Module,
    library_directories: []const []const u8,
    library_names: []const []const u8,
) void {
    // An absolute `-Dlib-dir` is the common case for an installed engine, and `b.path` refuses
    // anything outside the package, so the two spellings are told apart rather than assumed.
    for (library_directories) |directory| module.addLibraryPath(
        if (std.fs.path.isAbsolute(directory)) .{ .cwd_relative = directory } else b.path(directory),
    );
    for (library_names) |name| module.linkSystemLibrary(name, .{});
    module.link_libc = true;
}
