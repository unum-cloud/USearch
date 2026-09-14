using Cloud.Unum.USearch;

// Run from a restored package, outside the source/build trees. This catches
// missing native dependencies and instructions unsupported by the test CPU.
Console.WriteLine($"USearch: {USearchCapabilities.Version()}");
var compiled = USearchCapabilities.HardwareAccelerationCompiled();
Console.WriteLine($"Compiled: {compiled}");
Console.WriteLine($"Available: {USearchCapabilities.HardwareAccelerationAvailable()}");
if (compiled.Split(',', StringSplitOptions.TrimEntries).All(x => x == "serial"))
{
    throw new Exception("The package must retain NumKong SIMD kernels.");
}

var vector = new float[1536];
vector[0] = 1;
var other = new float[1536];
other[1] = 1;
var path = Path.Combine(Path.GetTempPath(), $"usearch-package-{Guid.NewGuid():N}.usearch");
try
{
    using (var index = new USearchIndex(MetricKind.Cos, ScalarKind.Float32, 1536))
    {
        index.Add(42, vector);
        index.Add(99, other);
        CheckSearch(index);
        index.Save(path);
    }
    using (var restored = new USearchIndex(path))
    {
        CheckSearch(restored);
    }
    Console.WriteLine("Package create/add/search/save/load passed.");
}
finally
{
    File.Delete(path);
}

void CheckSearch(USearchIndex index)
{
    var selected = index.HardwareAcceleration();
    Console.WriteLine($"Selected: {selected}");
    if (args.Length > 0 && selected != args[0])
    {
        throw new Exception($"Expected {args[0]} on the constrained CPU, got {selected}.");
    }
    var count = index.Search(vector, 2, out var keys, out var distances);
    if (count != 2 || keys[0] != 42 || keys[1] != 99 ||
        !float.IsFinite(distances[0]) || Math.Abs(distances[0]) > 0.0001f ||
        !float.IsFinite(distances[1]) || Math.Abs(distances[1] - 1) > 0.0001f)
    {
        throw new Exception("Package search returned incorrect neighbors or distances.");
    }
}
