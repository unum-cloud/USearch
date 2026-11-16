#!/usr/bin/env wolframscript
(* ::Package:: *)

(* :Title: Test *)
(* :Author: Ash Vardanian *)
(* :Date: May 4, 2023 *)
(* :Summary: The binding's checks: managed handles, receivers and arenas over a built usearch_wolfram
    library, exiting non-zero at the first failure. *)
(* :Usage: wolframscript -file wolfram/Test.wls [path/to/usearch_wolfram] *)

$library = If[Length[$ScriptCommandLine] > 1, $ScriptCommandLine[[2]], "build_release/usearch_wolfram.so"]

(* The package finds the library by its bare name, so its directory leads the search path. *)
PrependTo[$LibraryPath, DirectoryName[ExpandFileName[$library]]]
Get[FileNameJoin[{DirectoryName[$InputFileName], "USearch.wl"}]]
Print["Package Loaded: ", VersionName[]]

Expect[label_String, actual_, expected_] := If[actual =!= expected,
    Print["FAILED: ", label, " - expected ", InputForm[expected], ", got ", InputForm[actual]]; Exit[1]]

(* Keys are integers below 2^128, so a UUID is the number its hexadecimal digits spell. *)
$uuid = FromDigits[StringDelete["5f9c2b71-8e4d-4a63-b2f1-0c7a9d3e6b58", "-"], 16]

$executor = ExecutorForRam[]
Print["New Executor: ", $executor]
$arena = ArenaInit["Executor" -> $executor]
$local = ArenaInit[]
Expect["a host arena", ArenaDescribe[$arena], <|"Residency" -> "ram", "Device" -> None|>]
Expect["an arena for the calling thread", ArenaDescribe[$local], <|"Residency" -> "ram", "Device" -> None|>]
Expect["an unknown mood", ExecutorDrain[$executor, "chill"],
    Failure["invalid-argument", <|"MessageTemplate" -> "the mood is neither parked nor spinning"|>]]

Expect["an unknown quantization", DenseInitRam["inner", "f24", 5][[1]], "unknown-kind"]
Expect["a quantization the engine never stores", DenseInitRam["inner", "f32", 5][[1]], "unknown-kind"]
$index = DenseInitRam["inner", "bf16", 5]
Print["New Index: ", $index]
Expect["a reservation", DenseReserve[$index, 10], Null]
Expect["the frozen geometry", KeyTake[DenseDescribe[$index], {"Metric", "Quantization", "Dimensions", "Size"}],
    <|"Metric" -> "inner", "Quantization" -> "bf16", "Dimensions" -> 5, "Size" -> 0|>]

(* One identity too wide for a machine integer and one plain integer, so the batch exercises both. *)
$rows = {{1.0, 2.0, 3.0, 4.0, 5.0}, {5.0, 4.0, 3.0, 2.0, 1.0}};
$typed = NumericArray[$rows, "Real32"];
Expect["an insertion", DenseInsert[DenseIndexOn[$index, $executor], {$uuid, 43}, $typed], Null]
Expect["the size after insertion", DenseDescribe[$index]["Size"], 2]
Expect["a narrow row", DenseInsert[$index, {44}, {{1.0, 2.0}}][[1]], "shape-mismatch"]
Expect["one key per vector", DenseInsert[$index, {44}, $rows],
    Failure["shape-mismatch", <|"MessageTemplate" -> "there must be one key per vector"|>]]

(* A search lands in an arena and answers each query's keys and scores, as many as it found. *)
$matches = DenseSearch[DenseIndexOn[$index, $executor], $typed, 10, $arena]
Print["Searched: ", $matches]
Expect["the keys of a batch", $matches["Keys"], {{$uuid, 43}, {43, $uuid}}]
Expect["the scores of a batch", Length /@ $matches["Scores"], {2, 2}]
Expect["the calling thread", DenseSearch[$index, $rows, 10, $local]["Keys"], $matches["Keys"]]
Expect["a result outlives its arena's next call", DenseSearch[$index, {Last[$rows]}, 1, $arena]["Keys"], {{43}}]
Expect["the earlier result", $matches["Keys"], {{$uuid, 43}, {43, $uuid}}]
Expect["no arena", Head[DenseSearch[$index, $rows, 10]], DenseSearch]
Expect["a negative wanted", DenseSearch[$index, $rows, -1, $arena],
    Failure["invalid-argument", <|"MessageTemplate" -> "wanted must not be negative"|>]]
Expect["arguments of the wrong type", Quiet[DenseSearch[$index, {{"one"}}, 1, $arena]],
    Failure["invalid-argument", <|"MessageTemplate" -> "the arguments do not match the verb's types"|>]]
Expect["non-real typed rows", DenseSearch[$index, NumericArray[{{1, 2, 3, 4, 5}}, "Integer32"], 1, $arena][[1]],
    "shape-mismatch"]
Expect["a negative expansion", DenseSearch[$index, $typed, 1, $arena, "ExpansionFactor" -> -1][[1]],
    "invalid-argument"]
Expect["an empty batch", DenseInsert[$index, {}, {}], Null]
Expect["no queries", DenseSearch[$index, {}, 10, $arena], <|"Keys" -> {}, "Scores" -> {}|>]

(* The defaults make an angular bf16 index, and a scan scores every entry exactly, so each vector finds itself first. *)
$scanned = DenseInitRam[5]
Expect["the default geometry", KeyTake[DenseDescribe[$scanned], {"Metric", "Quantization"}],
    <|"Metric" -> "angular", "Quantization" -> "bf16"|>]
Expect["a reservation to scan", DenseReserve[$scanned, 2], Null]
Expect["an insertion to scan", DenseInsert[$scanned, {$uuid, 43}, $rows], Null]
Expect["a scan", First /@ DenseSearch[$scanned, $rows, 2, $local, "Traversal" -> "scan"]["Keys"], {$uuid, 43}]
Expect["an unknown traversal", DenseSearch[$scanned, $rows, 2, $local, "Traversal" -> "sweep"],
    Failure["invalid-argument", <|"MessageTemplate" -> "the traversal is neither walk nor scan"|>]]

Expect["a drain", ExecutorDrain[$executor, "parked"], Null]
Expect["an arena no constructor made", DenseSearch[$index, $rows, 10, Arena[0]],
    Failure["uninitialized", <|"MessageTemplate" -> "the Arena was released"|>]]

(* One arena serves get and multiply in turn, and each result outlives its reuse. *)
Expect["a stored vector", DenseGet[$index, 43, $arena], {Last[$rows]}]
Expect["an absent key", DenseGet[$index, 99, $local], {}]
Expect["the entries under a key", DenseCount[$index, $uuid], 1]
$products = DenseMultiply[DenseIndexOn[$index, $executor], $rows, $arena]
Expect["the product offsets", $products["Offsets"], {0, 2, 4}]
Expect["the product keys", Sort /@ TakeList[$products["Keys"], {2, 2}], {Sort[{$uuid, 43}], Sort[{$uuid, 43}]}]
Expect["the products scored", $products["Required"], 4]
Print["Products: ", $products["Scores"]]
$sparse = DenseMultiply[$index, $typed, {43, $uuid}, $arena];
Expect["sparse columns follow explicit keys", Normal[$sparse], $rows . Transpose[Reverse[$rows]]]
Expect["unknown product columns", DenseMultiply[$index, $rows, {43}, $arena][[1]], "invalid-argument"]

$values = DenseValues[$index, {$uuid, 43}, NumericArray[$rows / 8., "Real64"]];
$attended = DenseAttend[DenseIndexOn[$index, $executor], $typed, $values, $arena,
    "HeadCount" -> 2, "ScoreScale" -> 0.01];
$logits = 0.01 ($rows . Transpose[$rows]);
$weights = Exp[$logits];
$oracle = MapThread[#1 / #2 &, {$weights . ($rows / 8.), Total /@ $weights}];
Expect["folded multi-head attention", Max[Abs[Flatten[$attended["Outputs"] - $oracle]]] < 0.00005, True]
Expect["online softmax maximum", Max[Abs[$attended["Carry"][[All, 1]] - (Max /@ $logits)]] < 0.00005, True]
Expect["online softmax partition", Max[Abs[$attended["Carry"][[All, 2]] -
    (Total /@ Exp[Map[# - Max[#] &, $logits]])]] < 0.00005, True]
Expect["borrowed rows are unchanged", Normal[$typed], $rows]
Expect["duplicate value keys", DenseValues[$index, {43, 43}, $rows][[1]], "invalid-argument"]
Expect["missing attention value", DenseAttend[$index, $typed,
    DenseValues[$index, {43}, {Last[$rows]}], $arena][[1]], "shape-mismatch"]
Expect["single-row queries scored late-interaction style",
    Differences[DenseMaxsim[DenseIndexOn[$index, $executor], $typed, 1, 10, $arena, "ExpansionFactor" -> 0]["Offsets"]],
    {2, 2}]
Expect["a zero wanted lands every scored key", And @@ MapThread[GreaterEqual,
    Differences[DenseMaxsim[$index, $typed, 1, #, $arena]["Offsets"]] & /@ {0, 1}], True]

Expect["a refinement", With[{report = DenseRefine[DenseIndexOn[$index, $executor]]},
    {MemberQ[{"none", "improved", "converged", "stalled", "stopped"}, report["Verdict"]],
        report["Reachability"] === DenseHealth[$index]}],
    {True, True}]
(* A plan keeps its round for the next call naming it, and only a receiver on an executor opens one. *)
$plan = PlanInit[]
Expect["an empty plan", PlanProgress[$plan]["Phase"], "none"]
Expect["a planned round", AssociationQ[DenseRefine[DenseIndexOn[$index, $executor], "Plan" -> $plan]], True]
Expect["a plan that ran its round", PlanProgress[$plan]["Phase"], "done"]
Expect["a deferred finish", DeferredWait[DenseRefinePlanFinish[DenseIndexDeferred[$index, $executor], $plan]][
    "Reachability"]["Audited"], 2]
Expect["a second finish", DenseRefinePlanFinish[DenseIndexOn[$index, $executor], $plan][[1]], "invalid-argument"]
Expect["a plan on the calling thread", DenseRefine[$index, "Plan" -> PlanInit[]][[1]], "invalid-argument"]
Expect["a recut keeps every entry", {AssociationQ[DenseReshard[DenseIndexOn[$index, $executor], {$index}, {$index},
    "MemoryDomains" -> {}, "ShardCapacity" -> 8]], DenseDescribe[$index]["Size"]}, {True, 2}]
$source = DenseInitRam["inner", "bf16", 5]
$gathered = DenseInitRam["inner", "bf16", 5]
Expect["a source insertion", DenseInsert[$source, {44}, {First[$rows]}], Null]
Expect["a gather empties its other sources into the target", {AssociationQ[DenseReshard[$gathered,
    {$gathered, $source}, {$gathered}]], DenseDescribe[$gathered]["Size"], DenseDescribe[$source]["Size"]},
    {True, 1, 0}]
Expect["an empty source list is the library's to refuse", DenseReshard[$index, {}, {$index}][[1]], "invalid-argument"]
Expect["an audit", KeyTake[DenseHealth[$index], {"Audited", "Reachable"}], <|"Audited" -> 2, "Reachable" -> 2|>]

$checkpoint = FileNameJoin[{$TemporaryDirectory, "usearch-test-wolfram.safegraphs"}]
Expect["a save", DenseSave[DenseIndexOn[$index, $executor], $checkpoint], Null]
$loaded = DenseLoadRam[$checkpoint, "Executor" -> $executor]
Expect["a checkpoint round trip", DenseDescribe[$loaded]["Size"], 2]
$nvme = DenseViewNvme[$checkpoint];
Expect["an NVMe view without a key table refuses key lookups", DenseCount[$nvme, $uuid][[1]], "unsupported"]
$drives = ExecutorForNvme["Threads" -> 1]
Expect["a lone monolith has no manifest to load writable",
    DenseLoadNvme[$checkpoint, "Executor" -> $drives][[1]], "unsupported"]
Expect["an NVMe view counts through its key table", DenseCount[DenseIndexOn[
    DenseViewNvme[$checkpoint, "Executor" -> $drives, "KeyTableDomains" -> {}], $drives], $uuid], 1]
Expect["a value cache also serves NVMe attention", Max[Abs[Flatten[
    DenseAttend[$nvme, $typed, $values, $local, "HeadCount" -> 2, "ScoreScale" -> 0.01]["Outputs"] - $oracle]]] < 0.00005,
    True]
Expect["the checkpoint metadata", DenseMetadataRead[$checkpoint],
    <|"Metric" -> "inner", "Quantization" -> "bf16", "Dimensions" -> 5,
        "Connectivity" -> DenseDescribe[$index]["Connectivity"]|>]
$monolith = DenseSaveBuffer[$index, $local]
Expect["a buffer round trip", DenseDescribe[DenseLoadRamBuffer[$monolith]]["Size"], 2]
(* The engine declares views before it serves them, so only the refusal is checked there. *)
$viewed = DenseViewRam[$checkpoint]
If[FailureQ[$viewed],
    Expect["a view the build does not serve yet", $viewed[[1]], "unsupported"],
    Expect["a viewed index refuses writes", DenseInsert[$viewed, {45}, {First[$rows]}][[1]], "read-only"]]
Print["Saved and Loaded"]

Expect["a clear", DenseClear[$loaded], Null]
Expect["the size after clearing", DenseDescribe[$loaded]["Size"], 0]
Expect["empty sparse products", Normal[DenseMultiply[$loaded, $typed, {43, $uuid}, $local]], ConstantArray[0, {2, 2}]]
Expect["no matches in an empty index", DenseSearch[$loaded, $typed, 2, $local],
    <|"Scores" -> {{}, {}}, "Keys" -> {{}, {}}|>]

(* Each basic family prefixes its verbs with its name, so one function runs what test/modalities.py runs per family. *)
verbOf[family_String, verb_String] := Symbol["USearch`" <> family <> verb]

(* An observation at each position, every other field unrecorded. *)
located[latitudes_List, longitudes_List] :=
    Join[Transpose[{latitudes, longitudes}], ConstantArray[Missing[], {Length[latitudes], 9}], 2]

(* Random entries in the form each family's insertion takes, one per key. *)
entriesOf["Sequence", count_Integer, seed_Integer] :=
    (SeedRandom[seed]; Table[RandomInteger[{97, 122}, RandomInteger[{8, 16}]], count])
entriesOf["Shapes", count_Integer, seed_Integer] :=
    (SeedRandom[seed]; Table[RandomReal[1, {RandomInteger[{4, 12}], 3}], count])
entriesOf["Geospatial", count_Integer, seed_Integer] :=
    (SeedRandom[seed]; located[RandomReal[{-60, 60}, count], RandomReal[{-180, 180}, count]])

(* A host index of the family holding count entries under consecutive keys from first. *)
build[family_String, count_Integer, first_Integer : 0, seed_Integer : 0] := With[{index = verbOf[family, "InitRam"][]},
    Expect[family <> ": a build", verbOf[family, "Insert"][verbOf[family, "IndexOn"][index, $executor],
        Range[first, first + count - 1], entriesOf[family, count, seed]], Null];
    index]

sizeOf[family_String, index_] := verbOf[family, "Describe"][index]["Size"]
(* A geospatial count answers the entries alone, and the others add the extent of their payloads. *)
entriesUnder[family_String, index_, key_Integer] :=
    Replace[verbOf[family, "Count"][index, key], tally_Association :> tally["Entries"]]
(* The frozen settings a checkpoint's metadata answers, as a live index describes them; a geospatial checkpoint
   freezes only its connectivity, which answers as a bare integer. *)
metadataOf["Sequence", description_Association] := KeyTake[description, {"Metric", "Connectivity"}]
metadataOf["Shapes", description_Association] :=
    KeyTake[description, {"Metric", "Quantization", "Points", "Connectivity"}]
metadataOf["Geospatial", description_Association] := description["Connectivity"]

(* The cases every basic family shares: rounds, plans, deferred calls, composition and checkpoints. *)
checkFamily[family_String] := Module[{verb = verbOf[family, #] &, say = family <> ": " <> # &, on, deferred, index,
        report, plan, pending, first, donors, second, checkpoint, loaded, vram},
    on = verb["IndexOn"][#, $executor] &;
    deferred = verb["IndexDeferred"][#, $executor] &;

    vram = verb["InitVram"][];
    Expect[say["a VRAM index or the library's refusal"], If[FailureQ[vram],
        !MemberQ[{"success", "deferred"}, vram[[1]]], verb["Describe"][vram]["GPUClass"] =!= "unknown"], True];

    index = build[family, 64];
    report = verb["Refine"][on[index]];
    Expect[say["a refinement's verdict"],
        MemberQ[{"none", "improved", "converged", "stalled", "stopped"}, report["Verdict"]], True];
    Expect[say["a refinement changes only what it reselected"], report["Changed"] <= report["Reselected"], True];
    Expect[say["a refinement audits every entry"], report["Reachability"]["Audited"], 64];
    Expect[say["a refined entry stays findable"], entriesUnder[family, index, 17], 1];

    index = verb["InitRam"][];
    pending = verb["Insert"][deferred[index], Range[0, 63], entriesOf[family, 64, 0]];
    Expect[say["a deferred insertion"], {Head[pending], MemberQ[{"deferred", "success"}, DeferredStatus[pending]]},
        {DeferredCall, True}];
    Expect[say["a deferred insertion waited on"], DeferredWait[pending], Null];
    Expect[say["a deferred call answers once"], DeferredStatus[pending], "uninitialized"];
    Expect[say["the entries a deferred insertion added"], sizeOf[family, index], 64];
    plan = PlanInit[];
    Expect[say["an empty plan"], PlanProgress[plan]["Phase"], "none"];
    Expect[say["a planned round"], AssociationQ[verb["Refine"][on[index], "Plan" -> plan]], True];
    Expect[say["a plan that ran its round"], {PlanProgress[plan]["Phase"],
        PlanProgress[plan]["Completed"] <= PlanProgress[plan]["Total"]}, {"done", True}];
    Expect[say["a deferred finish"],
        DeferredWait[verb["RefinePlanFinish"][deferred[index], plan]]["Reachability"]["Audited"], 64];
    Expect[say["a second finish"], verb["RefinePlanFinish"][on[index], plan][[1]], "invalid-argument"];
    Expect[say["a plan on the calling thread"], verb["Refine"][verb["InitRam"][], "Plan" -> PlanInit[]][[1]],
        "invalid-argument"];

    first = build[family, 32];
    donors = {build[family, 32, 32, 1], build[family, 32, 64, 2]};
    Expect[say["a gather"], AssociationQ[verb["Reshard"][on[first], Join[{first}, donors], {first}]], True];
    Expect[say["a gather takes every entry"], {sizeOf[family, first], entriesUnder[family, first, 40],
        entriesUnder[family, first, 80], sizeOf[family, #] & /@ donors}, {96, 1, 1, {0, 0}}];

    index = build[family, 16];
    Expect[say["a recut of one handle"], verb["Reshard"][on[index], {index}, {index}][[1]], "unsupported"];
    Expect[say["a refused recut keeps every entry"], sizeOf[family, index], 16];
    first = build[family, 8];
    second = build[family, 8, 8, 1];
    Expect[say["an empty source list"], verb["Reshard"][on[first], {}, {first}][[1]], "invalid-argument"];
    Expect[say["a repeated source"], verb["Reshard"][on[first], {first, first}, {first}][[1]], "invalid-argument"];
    Expect[say["a target holding foreign entries"], verb["Reshard"][on[first], {first}, {second}][[1]],
        "config-mismatch"];
    Expect[say["a list naming no index"], Head[verb["Reshard"][on[first], {first, 1}, {first}]], verb["Reshard"]];
    Expect[say["every refusal leaves the entries in place"], {sizeOf[family, first], sizeOf[family, second]}, {8, 8}];

    checkpoint = FileNameJoin[{$TemporaryDirectory, "usearch-test-wolfram-" <> ToLowerCase[family] <> ".safegraphs"}];
    Expect[say["a save"], verb["Save"][on[index], checkpoint], Null];
    loaded = verb["LoadRam"][checkpoint, "Executor" -> $executor];
    Expect[say["a checkpoint round trip"], {sizeOf[family, loaded], entriesUnder[family, loaded, 9]}, {16, 1}];
    Expect[say["the checkpoint metadata"], verb["MetadataRead"][checkpoint],
        metadataOf[family, verb["Describe"][index]]];
    Expect[say["a buffer round trip"], sizeOf[family, verb["LoadRamBuffer"][verb["SaveBuffer"][index, $local]]], 16];
    DeleteFile[checkpoint]]

(* The basic families take their own entries, and each one searched for comes back first under its own key. *)
$sequences = SequenceInitRam[]
Expect["a fresh sequence index", KeyTake[SequenceDescribe[$sequences], {"Metric", "Size", "GPUClass"}],
    <|"Metric" -> "levenshtein", "Size" -> 0, "GPUClass" -> "unknown"|>]
Expect["a sequence index on a CPU class", SequenceDescribe[$sequences]["CPUClass"] =!= "unknown", True]
Expect["the metric a sequence index froze", SequenceDescribe[SequenceInitRam["levenshtein-utf8"]]["Metric"],
    "levenshtein-utf8"]
Expect["text, bytes and byte values in one batch", SequenceInsert[SequenceIndexOn[$sequences, $executor],
    {1, 2, 3}, {"kitten", ByteArray[ToCharacterCode["sitting"]], ToCharacterCode["mitten"]}], Null]
Expect["the size after a sequence insertion", SequenceDescribe[$sequences]["Size"], 3]
$found = SequenceSearch[$sequences, {"sitting"}, 1, $local]
Expect["a sequence finds itself", $found["Keys"], {{2}}]
Expect["a sequence at distance zero", First[First[$found["Scores"]]] == 0, True]
Expect["the sequences under a key", Normal /@ SequenceGet[$sequences, 2, $local], {ToCharacterCode["sitting"]}]
Expect["the sequence tally under a key", SequenceCount[$sequences, 2],
    <|"Entries" -> 1, "TotalBytes" -> 7, "LargestBytes" -> 7|>]
Expect["a verb takes only its own family's receivers", Head[ShapesDescribe[$sequences]], ShapesDescribe]

SeedRandom[42];
$points = RandomReal[1, {#, 3}] & /@ {5, 9, 13};
$shapes = ShapesInitRam[]
Expect["a fresh shapes index", KeyTake[ShapesDescribe[$shapes], {"Metric", "Quantization", "Size", "GPUClass"}],
    <|"Metric" -> "chamfer", "Quantization" -> "f32", "Size" -> 0, "GPUClass" -> "unknown"|>]
Expect["a shapes index on a CPU class", ShapesDescribe[$shapes]["CPUClass"] =!= "unknown", True]
Expect["the metric a shapes index froze", ShapesDescribe[ShapesInitRam["hausdorff"]]["Metric"], "hausdorff"]
Expect["shapes of different point counts in one batch", ShapesInsert[$shapes, {1, 2, 3}, $points], Null]
Expect["the points stored", KeyTake[ShapesDescribe[$shapes], {"Size", "TotalPoints"}],
    <|"Size" -> 3, "TotalPoints" -> 27|>]
Expect["a shape finds itself", ShapesSearch[$shapes, {Last[$points]}, 1, $local]["Keys"], {{3}}]
Expect["a shape in its inserted pose", Max[Abs[Flatten[First[ShapesGet[$shapes, 3, $local]] - Last[$points]]]] < 10^-6,
    True]
Expect["the shapes tally under a key", ShapesCount[$shapes, 3],
    <|"Entries" -> 1, "TotalPoints" -> 13, "LargestPoints" -> 13|>]

$stops = located[{48.8566, 52.52, 40.7128}, {2.3522, 13.405, -74.006}];
$geospatial = GeospatialInitRam[]
Expect["a fresh geospatial index", KeyTake[GeospatialDescribe[$geospatial], {"Size", "GPUClass"}],
    <|"Size" -> 0, "GPUClass" -> "unknown"|>]
Expect["a geospatial index on a CPU class", GeospatialDescribe[$geospatial]["CPUClass"] =!= "unknown", True]
Expect["observations with unrecorded fields", GeospatialInsert[$geospatial, {1, 2, 3}, $stops], Null]
$found = GeospatialSearch[$geospatial, {$stops[[2]]}, 1, $local]
Expect["an observation finds itself", $found["Keys"], {{2}}]
Expect["an observation at zero chord metres", First[First[$found["Scores"]]] == 0, True]
Expect["an observation within its stored rounding", With[{row = First[GeospatialGet[$geospatial, 2, $local]]},
    {Max[Abs[row[[;; 2]] - {52.52, 13.405}]] < 10^-5, row[[3]]}], {True, Missing["NotRecorded"]}]
Expect["the observations under a key", GeospatialCount[$geospatial, 2], 1]
(* Two observations a kilometre apart fall inside a two-kilometre radius, and a far one does not. *)
$nearby = GeospatialInitRam[]
Expect["observations a kilometre apart", GeospatialInsert[$nearby, {1, 2, 3},
    located[{48.8566, 48.8656, 52.52}, {2.3522, 2.3522, 13.405}]], Null]
Expect["only the entries inside the radius", Sort[GeospatialWithin[$nearby, {First[$stops]}, 2000., $local]["Keys"]],
    {1, 2}]

Scan[checkFamily, {"Sequence", "Shapes", "Geospatial"}]
Print["Basic Indexes"]

(* The one-to-many cut runs last, since the RAM engine's cut is known flaky and a crash ends the script. *)
$cut = DenseInitRam["inner", "bf16", 5]
Expect["a cut source insertion", DenseInsert[$cut, {45, 46}, $rows], Null]
$left = DenseInitRam["inner", "bf16", 5]
$right = DenseInitRam["inner", "bf16", 5]
Expect["a cut empties its source into its targets", {AssociationQ[DenseReshard[DenseIndexOn[$cut, $executor],
    {$cut}, {$left, $right}, "Seed" -> 7]], DenseDescribe[$cut]["Size"],
    DenseDescribe[$left]["Size"] + DenseDescribe[$right]["Size"]}, {True, 0, 2}]

ClearAll[$sequences, $shapes, $geospatial, $nearby, $found, $points, $stops, $plan, $loaded, $viewed, $nvme,
    $source, $gathered, $cut, $left, $right, $arena, $local, $scanned, $index, $executor]
DeleteFile[$checkpoint]
Print["Handles Released"]
