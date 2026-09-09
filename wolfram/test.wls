#!/usr/bin/env wolframscript

$library = "build_release/usearch_wolfram.so"

ExecutorCreate = LibraryFunctionLoad[$library, "ExecutorCreate", {"UTF8String", Integer}, Integer]
ExecutorDrain = LibraryFunctionLoad[$library, "ExecutorDrain", {Integer}, "Void"]
ExecutorDestroy = LibraryFunctionLoad[$library, "ExecutorDestroy", {Integer}, "Void"]

DenseIndexCreate = LibraryFunctionLoad[$library, "DenseIndexCreate", {"UTF8String", "UTF8String", "UTF8String", Integer, Integer, Integer}, Integer]
DenseIndexDestroy = LibraryFunctionLoad[$library, "DenseIndexDestroy", {Integer}, "Void"]
DenseIndexSave = LibraryFunctionLoad[$library, "DenseIndexSave", {Integer, Integer, "UTF8String"}, "Void"]
DenseIndexLoad = LibraryFunctionLoad[$library, "DenseIndexLoad", {Integer, Integer, "UTF8String"}, "Void"]
DenseIndexSize = LibraryFunctionLoad[$library, "DenseIndexSize", {Integer}, Integer]
DenseIndexCapacity = LibraryFunctionLoad[$library, "DenseIndexCapacity", {Integer}, Integer]
DenseIndexDimensions = LibraryFunctionLoad[$library, "DenseIndexDimensions", {Integer}, Integer]
DenseIndexConnectivity = LibraryFunctionLoad[$library, "DenseIndexConnectivity", {Integer}, Integer]

(* Keys are a rank-two integer tensor of one sixteen-byte identity per row, most significant byte
   first, and vectors a rank-two real one, one row apiece, so a whole batch crosses in one call.
   A single vector is the rank-one case of the vectors argument. *)
DenseIndexAdd = LibraryFunctionLoad[$library, "DenseIndexAdd", {Integer, Integer, {Integer, 2}, {Real, _}}, "Void"]

(* Matches land in three tensors the caller allocates and the verb writes into: identities of
   `queries * wanted * 16`, distances of `queries * wanted`, and one match count per query. *)
DenseIndexSearch = LibraryFunctionLoad[$library, "DenseIndexSearch", {Integer, Integer, {Real, _}, {Integer, 3, "Shared"}, {Real, 2, "Shared"}, {Integer, 1, "Shared"}}, "Void"]
Print["Functions Loaded"]

(* The sixteen identity bytes are the base-256 digits of the key, so an integer identity and a
   full UUID reach the binding through the same encoding, neither of them rounded. *)
KeyBytes[key_Integer] := IntegerDigits[key, 256, 16]
KeyBytes[text_String] := IntegerDigits[FromDigits[StringDelete[text, "-"], 16], 256, 16]
KeyText[bytes_List] := StringInsert[StringJoin[IntegerString[bytes, 16, 2]], "-", {9, 13, 17, 21}]
KeyInteger[bytes_List] := FromDigits[bytes, 256]

$uuid = "5f9c2b71-8e4d-4a63-b2f1-0c7a9d3e6b58"
If[KeyText[KeyBytes[$uuid]] =!= $uuid, Print["FAILED: a UUID does not survive the encoding"]; Exit[1]]
If[KeyInteger[KeyBytes[2^100 + 7]] =!= 2^100 + 7, Print["FAILED: a wide identity was truncated"]; Exit[1]]

$executor = ExecutorCreate["ram", 0]
Print["New Executor Address: ", $executor]

$index = DenseIndexCreate["angular", "f16", "ram", 5, 10, 10]
Print["New Index Address: ", $index]
Print["Index Size: ", DenseIndexSize[$index]]
Print["Index Capacity: ", DenseIndexCapacity[$index]]
Print["Index Dimensions: ", DenseIndexDimensions[$index]]
Print["Index Connectivity: ", DenseIndexConnectivity[$index]]

(* One identity too wide for a machine integer and one plain integer, so the batch exercises both. *)
$identities = {KeyBytes[$uuid], KeyBytes[43]};
$rows = {{1.0, 2.0, 3.0, 4.0, 5.0}, {5.0, 4.0, 3.0, 2.0, 1.0}};
DenseIndexAdd[$executor, $index, $identities, $rows]
Print["Added Rows: ", $rows]
Print["Index Size: ", DenseIndexSize[$index]]
If[DenseIndexSize[$index] =!= 2, Print["FAILED: the batch did not insert both rows"]; Exit[1]]

(* One query, asking for more than the index holds - the trailing rows come back zeroed. *)
$keys = ConstantArray[0, {1, 10, 16}];
$distances = ConstantArray[0., {1, 10}];
$counts = ConstantArray[0, 1];
DenseIndexSearch[$executor, $index, First[$rows], $keys, $distances, $counts]
Print["Searched: ", $counts]
If[First[$counts] =!= 2, Print["FAILED: both entries should have been reachable"]; Exit[1]]
If[KeyText[$keys[[1, 1]]] =!= $uuid, Print["FAILED: a row is not nearest to itself"]; Exit[1]]
If[$keys[[1, 3]] =!= ConstantArray[0, 16], Print["FAILED: a row past the count is not zeroed"]; Exit[1]]
Print["Nearest key and distance: ", KeyText[$keys[[1, 1]]], " ", $distances[[1, 1]]]

(* Both queries at once, which is the shape the interface is built around. *)
$keys = ConstantArray[0, {2, 2, 16}];
$distances = ConstantArray[0., {2, 2}];
$counts = ConstantArray[0, 2];
DenseIndexSearch[$executor, $index, $rows, $keys, $distances, $counts]
If[$counts =!= {2, 2}, Print["FAILED: unexpected batch counts"]; Exit[1]]
If[KeyInteger[$keys[[2, 1]]] =!= 43, Print["FAILED: the integer identity did not round-trip"]; Exit[1]]
Print["Batch searched: ", Map[KeyText, $keys, {2}]]

DenseIndexSave[$executor, $index, "index.safegraphs"]
DenseIndexLoad[$executor, $index, "index.safegraphs"]
Print["Saved and Loaded"]

ExecutorDrain[$executor]
DenseIndexDestroy[$index]
ExecutorDestroy[$executor]
Print["Index and Executor Destroyed"]
