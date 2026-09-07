#!/usr/bin/env wolframscript

$library = "wolfram/USearchWFM.so"

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

(* Keys are a rank-one integer tensor and vectors a rank-two real one, one row apiece, so a whole
   batch crosses in one call. A single vector is the rank-one case of the same argument. *)
DenseIndexAdd = LibraryFunctionLoad[$library, "DenseIndexAdd", {Integer, Integer, {Integer, 1}, {Real, _}}, "Void"]

(* Matches come back as `queries * wanted * 2`, each innermost pair a key and its distance. *)
DenseIndexSearch = LibraryFunctionLoad[$library, "DenseIndexSearch", {Integer, Integer, {Real, _}, Integer}, {Real, 3}]
Print["Functions Loaded"]

$executor = ExecutorCreate["ram", 0]
Print["New Executor Address: ", $executor]

$index = DenseIndexCreate["angular", "f16", "ram", 5, 10, 10]
Print["New Index Address: ", $index]
Print["Index Size: ", DenseIndexSize[$index]]
Print["Index Capacity: ", DenseIndexCapacity[$index]]
Print["Index Dimensions: ", DenseIndexDimensions[$index]]
Print["Index Connectivity: ", DenseIndexConnectivity[$index]]

$rows = {{1.0, 2.0, 3.0, 4.0, 5.0}, {5.0, 4.0, 3.0, 2.0, 1.0}};
DenseIndexAdd[$executor, $index, {42, 43}, $rows]
Print["Added Rows: ", $rows]
Print["Index Size: ", DenseIndexSize[$index]]
If[DenseIndexSize[$index] =!= 2, Print["FAILED: the batch did not insert both rows"]; Exit[1]]

(* One query, asking for more than the index holds - the trailing pairs come back zeroed. *)
$matches = DenseIndexSearch[$executor, $index, First[$rows], 10]
Print["Searched: ", Dimensions[$matches]]
If[Dimensions[$matches] =!= {1, 10, 2}, Print["FAILED: unexpected match shape"]; Exit[1]]
If[First[First[First[$matches]]] =!= 42., Print["FAILED: a row is not nearest to itself"]; Exit[1]]
Print["Nearest key and distance: ", First[First[$matches]]]

(* Both queries at once, which is the shape the interface is built around. *)
$batch = DenseIndexSearch[$executor, $index, $rows, 2]
If[Dimensions[$batch] =!= {2, 2, 2}, Print["FAILED: unexpected batch shape"]; Exit[1]]
Print["Batch searched: ", Dimensions[$batch]]

DenseIndexSave[$executor, $index, "index.safegraphs"]
DenseIndexLoad[$executor, $index, "index.safegraphs"]
Print["Saved and Loaded"]

ExecutorDrain[$executor]
DenseIndexDestroy[$index]
ExecutorDestroy[$executor]
Print["Index and Executor Destroyed"]
