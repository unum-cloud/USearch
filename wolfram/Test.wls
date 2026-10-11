#!/usr/bin/env wolframscript
(* ::Package:: *)

(* :Title: Test *)
(* :Author: Ash Vardanian *)
(* :Date: 2023-05-04 *)
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
$index = DenseInitRam["inner", "f16", 5]
Print["New Index: ", $index]
Expect["a reservation", DenseReserve[$index, 10], Null]
Expect["the frozen geometry", KeyTake[DenseDescribe[$index], {"Metric", "Quantization", "Dimensions", "Size"}],
    <|"Metric" -> "inner", "Quantization" -> "f16", "Dimensions" -> 5, "Size" -> 0|>]

(* One identity too wide for a machine integer and one plain integer, so the batch exercises both. *)
$rows = {{1.0, 2.0, 3.0, 4.0, 5.0}, {5.0, 4.0, 3.0, 2.0, 1.0}};
Expect["an insertion", DenseInsert[DenseIndexOn[$index, $executor], {$uuid, 43}, $rows], Null]
Expect["the size after insertion", DenseDescribe[$index]["Size"], 2]
Expect["a narrow row", DenseInsert[$index, {44}, {{1.0, 2.0}}][[1]], "shape-mismatch"]
Expect["one key per vector", DenseInsert[$index, {44}, $rows],
    Failure["shape-mismatch", <|"MessageTemplate" -> "there must be one key per vector"|>]]

(* A search lands in an arena and answers each query's keys and scores, as many as it found. *)
$matches = DenseSearch[DenseIndexOn[$index, $executor], $rows, 10, $arena]
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
Expect["an empty batch", DenseInsert[$index, {}, {}], Null]
Expect["no queries", DenseSearch[$index, {}, 10, $arena], <|"Keys" -> {}, "Scores" -> {}|>]

Expect["a drain", ExecutorDrain[$executor, "parked"], Null]
Expect["an arena no constructor made", DenseSearch[$index, $rows, 10, Arena[0]],
    Failure["uninitialized", <|"MessageTemplate" -> "the Arena was released"|>]]

(* One arena serves get and multiply in turn, and each result outlives its reuse. *)
Expect["a stored row", DenseGet[$index, 43, $arena], {Last[$rows]}]
Expect["an absent key", DenseGet[$index, 99, $local], {}]
Expect["the entries under a key", DenseCount[$index, $uuid], 1]
$products = DenseMultiply[DenseIndexOn[$index, $executor], $rows, $arena]
Expect["the product offsets", $products["Offsets"], {0, 2, 4}]
Expect["the product keys", Sort /@ TakeList[$products["Keys"], {2, 2}], {Sort[{$uuid, 43}], Sort[{$uuid, 43}]}]
Expect["the products scored", $products["Required"], 4]
Print["Products: ", $products["Scores"]]
Expect["single-row queries scored late-interaction style",
    Length /@ DenseMaxsim[DenseIndexOn[$index, $executor], $rows, 1, 10, $arena, "ExpansionFactor" -> 0]["Keys"], {2, 2}]

Expect["a refinement", IntegerQ[DenseRefine[DenseIndexOn[$index, $executor]]], True]
Expect["an audit", KeyTake[DenseHealth[$index], {"Audited", "Reachable"}], <|"Audited" -> 2, "Reachable" -> 2|>]

$checkpoint = FileNameJoin[{$TemporaryDirectory, "usearch-test-wolfram.safegraphs"}]
Expect["a save", DenseSave[DenseIndexOn[$index, $executor], $checkpoint], Null]
$loaded = DenseLoadRam[$checkpoint, "Executor" -> $executor]
Expect["a checkpoint round trip", DenseDescribe[$loaded]["Size"], 2]
Expect["the checkpoint metadata", DenseMetadataRead[$checkpoint],
    <|"Metric" -> "inner", "Quantization" -> "f16", "Dimensions" -> 5,
        "Connectivity" -> DenseDescribe[$index]["Connectivity"]|>]
$monolith = DenseSaveBuffer[$index, $local]
Expect["a buffer round trip", DenseDescribe[DenseLoadRamBuffer[$monolith]]["Size"], 2]
$viewed = DenseViewRam[$checkpoint]
Expect["a viewed index refuses writes", DenseInsert[$viewed, {45}, {First[$rows]}][[1]], "read-only"]
Print["Saved and Loaded"]

Expect["a clear", DenseClear[$loaded], Null]
Expect["the size after clearing", DenseDescribe[$loaded]["Size"], 0]

ClearAll[$loaded, $viewed, $arena, $local, $index, $executor]
DeleteFile[$checkpoint]
Print["Handles Released"]
