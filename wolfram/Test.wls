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
Expect["an unknown mood", ExecutorDrain[$executor, "chill"],
    Failure["invalid-argument", <|"MessageTemplate" -> "the mood is neither parked nor spinning"|>]]

Expect["an unknown quantization", DenseInitRam["inner", "f24", 5],
    Failure["unknown-kind", <|"MessageTemplate" -> "the metric or quantization is not a known name"|>]]
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
    Failure["unknown-kind", <|"MessageTemplate" -> "the traversal is neither walk nor scan"|>]]

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

Expect["a refinement", IntegerQ[DenseRefine[DenseIndexOn[$index, $executor]]], True]
Expect["an audit", KeyTake[DenseHealth[$index], {"Audited", "Reachable"}], <|"Audited" -> 2, "Reachable" -> 2|>]

$checkpoint = FileNameJoin[{$TemporaryDirectory, "usearch-test-wolfram.safegraphs"}]
Expect["a save", DenseSave[DenseIndexOn[$index, $executor], $checkpoint], Null]
$loaded = DenseLoadRam[$checkpoint, "Executor" -> $executor]
Expect["a checkpoint round trip", DenseDescribe[$loaded]["Size"], 2]
$nvme = DenseViewNvme[$checkpoint];
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

ClearAll[$loaded, $viewed, $nvme, $arena, $local, $scanned, $index, $executor]
DeleteFile[$checkpoint]
Print["Handles Released"]
