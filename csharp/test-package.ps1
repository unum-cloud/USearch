param(
    [Parameter(Mandatory)][string]$PackagePath,
    [Parameter(Mandatory)][string]$RuntimeIdentifier,
    [ValidateSet('', 'Haswell', 'Nehalem')][string]$Cpu = ''
)

$ErrorActionPreference = 'Stop'
$PackagePath = (Resolve-Path -LiteralPath $PackagePath).Path
$archive = [System.IO.Compression.ZipFile]::OpenRead($PackagePath)
try {
    $expected = @(
        'runtimes/linux-x64/native/libusearch_c.so',
        'runtimes/linux-x64/native/libnumkong.so',
        'runtimes/osx-arm64/native/libusearch_c.dylib',
        'runtimes/osx-arm64/native/libnumkong.dylib',
        'runtimes/win-x64/native/libusearch_c.dll',
        'runtimes/win-x64/native/numkong.dll'
    )
    foreach ($name in $expected) {
        if (-not $archive.GetEntry($name)) { throw "Package is missing $name" }
    }
    $entry = $archive.Entries | Where-Object { $_.FullName -like '*.nuspec' } | Select-Object -First 1
    $reader = [System.IO.StreamReader]::new($entry.Open())
    try { [xml]$manifest = $reader.ReadToEnd() } finally { $reader.Dispose() }
    $version = $manifest.package.metadata.version
} finally {
    $archive.Dispose()
}

# A fresh project and cache prevent build-tree RPATHs and cached packages from
# hiding a broken archive. Only this package source may supply Cloud.Unum.USearch.
$work = Join-Path ([System.IO.Path]::GetTempPath()) "usearch-package-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Path "$work/source" -Force | Out-Null
Copy-Item -LiteralPath "$PSScriptRoot/PackageSmoke/PackageSmoke.csproj", "$PSScriptRoot/PackageSmoke/Program.cs" -Destination "$work/source"
New-Item -ItemType Directory -Path "$work/feed" | Out-Null
Copy-Item -LiteralPath $PackagePath -Destination "$work/feed"
@'
<configuration>
  <packageSources>
    <clear />
    <add key="package-under-test" value="feed" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
  <packageSourceMapping>
    <packageSource key="package-under-test"><package pattern="Cloud.Unum.USearch" /></packageSource>
    <packageSource key="nuget.org"><package pattern="*" /></packageSource>
  </packageSourceMapping>
</configuration>
'@ | Set-Content -LiteralPath "$work/NuGet.Config"

Write-Host "Testing $PackagePath ($version) in $work"
dotnet restore "$work/source/PackageSmoke.csproj" --configfile "$work/NuGet.Config" --packages "$work/cache" -r $RuntimeIdentifier "-p:USearchPackageVersion=$version"
if ($LASTEXITCODE) { throw "Package restore failed: $LASTEXITCODE" }
dotnet publish "$work/source/PackageSmoke.csproj" --no-restore -c Release -r $RuntimeIdentifier --self-contained false "-p:USearchPackageVersion=$version" "-p:RestorePackagesPath=$work/cache" -o "$work/out"
if ($LASTEXITCODE) { throw "Consumer publish failed: $LASTEXITCODE" }

if ($Cpu) {
    if (-not $IsLinux -or $RuntimeIdentifier -ne 'linux-x64') { throw 'CPU emulation requires Linux x64.' }
    $model = if ($Cpu -eq 'Haswell') { 'Haswell,-hle,-rtm' } else { 'Nehalem' }
    $expectedAcceleration = if ($Cpu -eq 'Haswell') { 'haswell' } else { 'serial' }
    qemu-x86_64 -cpu $model (Get-Command dotnet).Source "$work/out/PackageSmoke.dll" $expectedAcceleration
} else {
    dotnet "$work/out/PackageSmoke.dll"
}
if ($LASTEXITCODE) { throw "Package consumer failed: $LASTEXITCODE" }
