(* ::Package:: *)

(* :Title: USearch *)
(* :File: wolfram/USearch.wl *)
(* :Context: USearch` *)
(* :Author: Ash Vardanian *)
(* :Date: May 4, 2023 *)
(* :Summary: USearch for the Wolfram Language: managed handles and receivers over the LibraryLink
    verbs of `lib.c`. *)

BeginPackage["USearch`"]

DenseIndex::usage = "DenseIndex[id] is a dense vector index, freed once nothing references it."
DenseIndexOn::usage = "DenseIndexOn[index, executor] is a receiver that runs every verb on executor, blocking until the verb is final; a bare DenseIndex runs verbs on the calling thread."
Executor::usage = "Executor[id] owns the threads, streams or submission rings verbs run on, freed once nothing references it."
Arena::usage = "Arena[id] holds the outputs a verb writes, rewound by every call writing into it and freed once nothing references it."

ExecutorForRam::usage = "ExecutorForRam[] spawns host worker threads for RAM-resident indexes. \"MemoryDomains\" lists the NUMA nodes to place shards on, {} deriving every node the process may allocate on, and \"Threads\" is how many threads to spawn, 0 taking 7/8 of the cores those nodes carry."
ExecutorOnPool::usage = "ExecutorOnPool[pool] drives RAM-resident indexes on the ForkUnion thread pool at address pool, which the caller owns. \"MemoryDomains\" lists the NUMA nodes to place shards on, {} deriving every node the process may allocate on."
ExecutorForVram::usage = "ExecutorForVram[] opens one stream per device for VRAM-resident indexes. \"Devices\" lists the device ordinals to place shards on, {} picking device zero; \"Streams\" lends existing streams, one per device, {} creating its own; and \"ScratchLimit\" caps the memory each stream carves scratch and outputs from, past which a call fails with \"out-of-memory\", 0 letting it grow as needed."
ExecutorForNvme::usage = "ExecutorForNvme[] opens one submission ring per worker for NVMe-resident indexes. \"Paths\" lists the devices or directories to serve shards from, {} leaving every shard where its manifest points, and \"Threads\" is how many workers to spawn, 0 deriving it from the devices' queues."
ExecutorDrain::usage = "ExecutorDrain[executor, mood] waits for every call submitted before it, then sets what idle workers do: \"parked\", the default, or \"spinning\"."
ArenaInit::usage = "ArenaInit[] makes an arena for verb outputs, placed where its \"Executor\" computes; None, the default, holds host memory for calls on the calling thread."
ArenaDescribe::usage = "ArenaDescribe[arena] reports where the arena's outputs live: its \"Residency\", \"ram\" for host pages or \"vram\" for one device's memory, and the ordinal of that \"Device\", None for host pages."

DenseInitRam::usage = "DenseInitRam[metric, quantization, dimensions] creates an empty, mutable index in host memory, scoring by metric, \"angular\" by default, and storing quantization scalars, \"bf16\" by default. \"Connectivity\" bounds the connections per node, 0 deriving it; \"ValuesPerKey\" bounds the entries sharing one key, 0 leaving it unbounded; and \"MemoryDomains\" lists the NUMA nodes to place shards on, {} deriving every node the process may allocate on."
DenseInitVram::usage = "DenseInitVram[metric, quantization, dimensions] creates an empty, mutable index in device memory, scoring by metric, \"angular\" by default, and storing quantization scalars, \"bf16\" by default. \"Connectivity\" bounds the connections per node, 0 deriving it; \"ValuesPerKey\" bounds the entries sharing one key, 0 leaving it unbounded; and \"Devices\" lists the device ordinals to place shards on, {} picking device zero."
DenseLoadRam::usage = "DenseLoadRam[path] copies a checkpoint, a manifest or a single monolith, into a new, mutable index in host memory. \"Executor\" runs the call, None meaning the calling thread; \"MemoryDomains\" lists the NUMA nodes to place shards on, {} deriving every node the process may allocate on; and \"ValuesPerKey\" bounds the entries sharing one key, which no checkpoint records, 0 leaving it unbounded."
DenseLoadVram::usage = "DenseLoadVram[path] copies a checkpoint, a manifest or a single monolith, into a new, mutable index in device memory. \"Executor\" runs the call, None meaning the calling thread; \"Devices\" lists the device ordinals to place shards on, {} picking device zero; and \"ValuesPerKey\" bounds the entries sharing one key, which no checkpoint records, 0 leaving it unbounded."
DenseLoadRamBuffer::usage = "DenseLoadRamBuffer[monolith] copies a checkpoint held in a ByteArray or a byte NumericArray into a new, mutable index in host memory, taking the options of DenseLoadRam."
DenseLoadVramBuffer::usage = "DenseLoadVramBuffer[monolith] copies a checkpoint held in a ByteArray or a byte NumericArray into a new, mutable index in device memory, taking the options of DenseLoadVram."
DenseViewRam::usage = "DenseViewRam[path] serves a checkpoint in place through a memory mapping, as a read-only index."
DenseViewNvme::usage = "DenseViewNvme[path] serves a checkpoint in place from NVMe drives as a read-only index, which every later call reads through its executor. \"Executor\" runs the call, None meaning the calling thread, and \"Paths\" lists the devices or directories to serve shards from, {} leaving every shard where its manifest points."
DenseMetadataRead::usage = "DenseMetadataRead[path] reads a checkpoint's geometry and metric without loading it. \"Executor\" runs the call, None meaning the calling thread."
DenseMetadataReadBuffer::usage = "DenseMetadataReadBuffer[monolith] reads the geometry and metric of a checkpoint held in a ByteArray or a byte NumericArray, taking the option of DenseMetadataRead."

DenseReserve::usage = "DenseReserve[receiver, capacity] grows the index ahead of incoming vectors to capacity entries in total, the current ones included."
DenseClear::usage = "DenseClear[receiver] erases every entry, keeping the reserved capacity and the frozen geometry."
DenseInsert::usage = "DenseInsert[receiver, keys, vectors] adds vectors, one row per key, under keys, each an integer below 2^128. \"ExpansionFactor\" sets the breadth of the candidates the walk explores, 0 deriving it."
DenseRefine::usage = "DenseRefine[receiver] re-selects the neighbors of every stored entry against the finished graph, improving those chosen while the graph around them was incomplete. \"ExpansionFactor\" sets the breadth of the candidates the walk explores, 0 deriving it."
DenseHealth::usage = "DenseHealth[receiver] audits how much of the graph a search can reach, leaving the index unchanged."
DenseSearch::usage = "DenseSearch[receiver, queries, wanted, arena] finds up to wanted nearest entries for each query row, answering their \"Keys\" and \"Scores\" per query. \"ExpansionFactor\" sets the breadth of the candidates the walk explores, 0 deriving it, and \"Traversal\" is \"walk\", the default, or \"scan\", an exact pass over every entry."
DenseMaxsim::usage = "DenseMaxsim[receiver, queries, rowsPerQuery, wanted, arena] scores queries of rowsPerQuery rows against multi-row entries: every query row takes its best match within a key, and those maxima sum to the key's score, keeping up to wanted keys per query. \"ExpansionFactor\" sets the breadth of the candidates the walk explores, 0 deriving it."
DenseCount::usage = "DenseCount[receiver, key] counts the entries stored under key."
DenseGet::usage = "DenseGet[receiver, key, arena] retrieves every vector stored under key, one row per entry."
DenseSave::usage = "DenseSave[receiver, path] saves the index under path, as one monolith or as a manifest beside its shards, while readers keep searching."
DenseSaveBuffer::usage = "DenseSaveBuffer[receiver, arena] saves the index into arena, as one monolith for an index of one shard or as a manifest leading one part per shard."
DenseDescribe::usage = "DenseDescribe[receiver] reports the index's frozen metadata and current counters."
DenseMultiply::usage = "DenseMultiply[receiver, left, arena] approximates the sparse product of the rows of left with the indexed vectors, answering every pair the walk scored as \"Offsets\", \"Keys\" and \"Scores\"; DenseMultiply[receiver, left, columnKeys, arena] lays the same products out as a SparseArray with one column per key of columnKeys. \"Wanted\" bounds the pairs kept per row, the first ones its walk scored, 0 keeping every pair, and \"ExpansionFactor\" sets the breadth of the candidates, which widens the pattern, 0 deriving it."
DenseValues::usage = "DenseValues[receiver, keys, values] prepares immutable value rows, one per key, in the index's quantization, for DenseAttend."
DenseAttend::usage = "DenseAttend[receiver, queries, values, arena] folds softmax attention of the query heads over every key their graph walks score, reading each key's row from values, a DenseValues. \"HeadCount\" is how many query heads share one walk per token, dividing the query count; \"ExpansionFactor\" sets the breadth of the candidates the walk explores, 0 deriving it; and \"ScoreScale\" is the logit scale, conventionally the inverse square root of the head dimensionality, 0. deriving it from the index's geometry."

VersionName::usage = "VersionName[] gives the library's semantic version."
CapabilitiesCompiled::usage = "CapabilitiesCompiled[] lists the kernel families this build carries."
CapabilitiesAvailable::usage = "CapabilitiesAvailable[] lists the kernel families this machine can run."

Begin["`Private`"]

$library = FindLibrary["usearch_wolfram"];

verb[name_String, arguments_List] := LibraryFunctionLoad[$library, name, arguments, "DataStore"]

(* Every index verb leads with the IDs of its index and its executor. *)
receiving[name_String, arguments_List] := verb[name, Join[{Integer, Integer}, arguments]]

receivingRows[name_String, before_List, after_List] := Quiet[<|
    List -> receiving[name, Join[before, {{Real, _, "Constant"}, Boolean}, after]],
    NumericArray -> receiving[name, Join[before, {{LibraryDataType[NumericArray], "Constant"}, Boolean}, after]]|>,
    LibraryFunction::overload]

executorForRam = verb["ExecutorForRam", {Integer, "DataStore", Integer}];
executorOnPool = verb["ExecutorOnPool", {Integer, "DataStore", Integer}];
executorForVram = verb["ExecutorForVram", {Integer, "DataStore", "DataStore", Integer}];
executorForNvme = verb["ExecutorForNvme", {Integer, "DataStore", Integer}];
executorDrain = verb["ExecutorDrain", {Integer, "UTF8String"}];
arenaInit = verb["ArenaInit", {Integer, Integer}];
arenaDescribe = verb["ArenaDescribe", {Integer}];
denseInitRam = verb["DenseInitRam", {Integer, "UTF8String", "UTF8String", Integer, Integer, Integer, "DataStore"}];
denseInitVram = verb["DenseInitVram", {Integer, "UTF8String", "UTF8String", Integer, Integer, Integer, "DataStore"}];
denseLoadRam = verb["DenseLoadRam", {Integer, Integer, "UTF8String", "DataStore", Integer}];
denseLoadVram = verb["DenseLoadVram", {Integer, Integer, "UTF8String", "DataStore", Integer}];
denseLoadRamBuffer =
    verb["DenseLoadRamBuffer", {Integer, Integer, LibraryDataType[NumericArray], "DataStore", Integer}];
denseLoadVramBuffer =
    verb["DenseLoadVramBuffer", {Integer, Integer, LibraryDataType[NumericArray], "DataStore", Integer}];
denseViewRam = verb["DenseViewRam", {Integer, "UTF8String"}];
denseViewNvme = verb["DenseViewNvme", {Integer, Integer, "UTF8String", "DataStore"}];
denseMetadataRead = verb["DenseMetadataRead", {Integer, "UTF8String"}];
denseMetadataReadBuffer = verb["DenseMetadataReadBuffer", {Integer, LibraryDataType[NumericArray]}];
denseReserve = receiving["DenseReserve", {Integer}];
denseClear = receiving["DenseClear", {}];
denseInsert = receivingRows["DenseInsert", {{LibraryDataType[NumericArray], "Constant"}}, {Integer}];
denseRefine = receiving["DenseRefine", {Integer}];
denseHealth = receiving["DenseHealth", {}];
denseSearch = receivingRows["DenseSearch", {}, {Integer, Integer, Integer, "UTF8String"}];
denseMaxsim = receivingRows["DenseMaxsim", {}, {Integer, Integer, Integer, Integer}];
denseCount = receiving["DenseCount", {LibraryDataType[NumericArray]}];
denseGet = receiving["DenseGet", {LibraryDataType[NumericArray], Integer}];
denseSave = receiving["DenseSave", {"UTF8String"}];
denseSaveBuffer = receiving["DenseSaveBuffer", {Integer}];
denseDescribe = receiving["DenseDescribe", {}];
denseMultiply = receivingRows["DenseMultiply", {}, {Integer, Integer, Integer}];
denseValues = receivingRows["DenseValues", {{LibraryDataType[NumericArray], "Constant"}}, {}];
denseAttend = receivingRows["DenseAttend", {}, {{LibraryDataType[NumericArray], "Constant"},
    {LibraryDataType[NumericArray], "Constant"}, "UTF8String", Integer, Integer, Integer, Integer, Real}];
versionName = LibraryFunctionLoad[$library, "VersionName", {}, "UTF8String"];
capabilitiesCompiled = LibraryFunctionLoad[$library, "CapabilitiesCompiled", {}, "UTF8String"];
capabilitiesAvailable = LibraryFunctionLoad[$library, "CapabilitiesAvailable", {}, "UTF8String"];

(* A key is an integer below 2^128, crossing as its sixteen base-256 digits, most significant first. *)
keyQ[key_] := IntegerQ[key] && 0 <= key < 2^128
keyBytes[keys_] := NumericArray[IntegerDigits[keys, 256, 16], "UnsignedInteger8"]
keysFromBytes[{}] := {}
keysFromBytes[bytes_NumericArray] := If[MemberQ[Dimensions[bytes], 0], ConstantArray[0, Most[Dimensions[bytes]]],
    Normal[bytes] . (256^Range[15, 0, -1])]

store[list_List] := Developer`DataStore @@ list
rows[vectors_NumericArray] := vectors
rows[vectors_List] := Developer`ToPackedArray[N[vectors]]
monolith[bytes_ByteArray] := NumericArray[Normal[bytes], "UnsignedInteger8"]
monolith[bytes_NumericArray] := bytes

(* Outputs arrive under their C names; found matches split by each query's count, and a lone output stands alone. *)
answer[result_Developer`DataStore] := answer[Association @@ result]
answer[LibraryFunctionError["LIBRARY_MEMORY_ERROR", _]] :=
    Failure["out-of-memory", <|"MessageTemplate" -> "the kernel could not allocate the answer"|>]
answer[<|"Status" -> status_String, "Message" -> message_String|>] := Failure[status, <|"MessageTemplate" -> message|>]
answer[fields_Association] := Module[{result = fields},
    If[KeyExistsQ[result, "Keys"], result["Keys"] = keysFromBytes[result["Keys"]]];
    If[KeyExistsQ[result, "Counts"],
        Scan[(result[#] = TakeList[result[#], fields["Counts"]]) &, {"Keys", "Scores"}];
        KeyDropFrom[result, "Counts"]];
    Switch[Length[result], 0, Null, 1, First[result], _, result]]
(* The kernel leaves a call whose arguments miss the declared types unevaluated, and any other refusal lands here. *)
answer[_] := Failure["invalid-argument", <|"MessageTemplate" -> "the arguments do not match the verb's types"|>]

(* The kernel frees a managed expression once nothing references it, so a call holds every handle it names
   until it returns, and only then do they cross as IDs. *)
call[function_, arguments_List] := Module[{held = arguments},
    answer[function @@ Replace[held, {(DenseIndex | Executor | Arena)[id_Integer] :> id, None -> 0}, {1}]]]

(* A constructor fills a fresh managed expression, which a failure leaves for the kernel to release. *)
construct[head_Symbol, function_, arguments_List] :=
    With[{made = CreateManagedLibraryExpression[SymbolName[head], head]},
        Replace[call[function, Prepend[arguments, made]], Null -> made]]

(* The receiver chooses where the verb runs; the verb stays the same, and blocks either way. *)
submit[index_DenseIndex, function_, arguments_List] := call[function, Join[{index, None}, arguments]]
submit[DenseIndexOn[index_DenseIndex, executor_Executor], function_, arguments_List] :=
    call[function, Join[{index, executor}, arguments]]

submitRows[receiver_, functions_Association, before_List, vectors_, after_List] :=
    submit[receiver, functions[Head[vectors]], Join[before, {rows[vectors], Head[vectors] === NumericArray}, after]]

Options[ExecutorForRam] = {"MemoryDomains" -> {}, "Threads" -> 0};
ExecutorForRam[OptionsPattern[]] := construct[Executor, executorForRam,
    {store[OptionValue["MemoryDomains"]], OptionValue["Threads"]}]
Options[ExecutorOnPool] = {"MemoryDomains" -> {}};
ExecutorOnPool[pool_Integer, OptionsPattern[]] := construct[Executor, executorOnPool,
    {store[OptionValue["MemoryDomains"]], pool}]
Options[ExecutorForVram] = {"Devices" -> {}, "Streams" -> {}, "ScratchLimit" -> 0};
ExecutorForVram[OptionsPattern[]] := construct[Executor, executorForVram,
    {store[OptionValue["Devices"]], store[OptionValue["Streams"]], OptionValue["ScratchLimit"]}]
Options[ExecutorForNvme] = {"Paths" -> {}, "Threads" -> 0};
ExecutorForNvme[OptionsPattern[]] := construct[Executor, executorForNvme,
    {store[OptionValue["Paths"]], OptionValue["Threads"]}]
ExecutorDrain[executor_Executor, mood_String : "parked"] := call[executorDrain, {executor, mood}]
Options[ArenaInit] = {"Executor" -> None};
ArenaInit[OptionsPattern[]] := construct[Arena, arenaInit, {OptionValue["Executor"]}]
ArenaDescribe[arena_Arena] := Replace[call[arenaDescribe, {arena}],
    description_Association :> Append[description, "Device" -> Replace[description["Device"], -1 -> None]]]

Options[DenseInitRam] = {"Connectivity" -> 0, "ValuesPerKey" -> 0, "MemoryDomains" -> {}};
DenseInitRam[metric_String : "angular", quantization_String : "bf16", dimensions_Integer, OptionsPattern[]] :=
    construct[DenseIndex, denseInitRam, {metric, quantization, dimensions, OptionValue["Connectivity"],
        OptionValue["ValuesPerKey"], store[OptionValue["MemoryDomains"]]}]
Options[DenseInitVram] = {"Connectivity" -> 0, "ValuesPerKey" -> 0, "Devices" -> {}};
DenseInitVram[metric_String : "angular", quantization_String : "bf16", dimensions_Integer, OptionsPattern[]] :=
    construct[DenseIndex, denseInitVram, {metric, quantization, dimensions, OptionValue["Connectivity"],
        OptionValue["ValuesPerKey"], store[OptionValue["Devices"]]}]
Options[DenseLoadRam] = {"Executor" -> None, "MemoryDomains" -> {}, "ValuesPerKey" -> 0};
DenseLoadRam[path_String, OptionsPattern[]] := construct[DenseIndex, denseLoadRam,
    {OptionValue["Executor"], path, store[OptionValue["MemoryDomains"]], OptionValue["ValuesPerKey"]}]
Options[DenseLoadVram] = {"Executor" -> None, "Devices" -> {}, "ValuesPerKey" -> 0};
DenseLoadVram[path_String, OptionsPattern[]] := construct[DenseIndex, denseLoadVram,
    {OptionValue["Executor"], path, store[OptionValue["Devices"]], OptionValue["ValuesPerKey"]}]
Options[DenseLoadRamBuffer] = Options[DenseLoadRam];
DenseLoadRamBuffer[bytes : (_ByteArray | _NumericArray), OptionsPattern[]] :=
    construct[DenseIndex, denseLoadRamBuffer, {OptionValue["Executor"], monolith[bytes],
        store[OptionValue["MemoryDomains"]], OptionValue["ValuesPerKey"]}]
Options[DenseLoadVramBuffer] = Options[DenseLoadVram];
DenseLoadVramBuffer[bytes : (_ByteArray | _NumericArray), OptionsPattern[]] :=
    construct[DenseIndex, denseLoadVramBuffer, {OptionValue["Executor"], monolith[bytes],
        store[OptionValue["Devices"]], OptionValue["ValuesPerKey"]}]
DenseViewRam[path_String] := construct[DenseIndex, denseViewRam, {path}]
Options[DenseViewNvme] = {"Executor" -> None, "Paths" -> {}};
DenseViewNvme[path_String, OptionsPattern[]] := construct[DenseIndex, denseViewNvme,
    {OptionValue["Executor"], path, store[OptionValue["Paths"]]}]
Options[DenseMetadataRead] = {"Executor" -> None};
DenseMetadataRead[path_String, OptionsPattern[]] := call[denseMetadataRead, {OptionValue["Executor"], path}]
Options[DenseMetadataReadBuffer] = {"Executor" -> None};
DenseMetadataReadBuffer[bytes : (_ByteArray | _NumericArray), OptionsPattern[]] :=
    call[denseMetadataReadBuffer, {OptionValue["Executor"], monolith[bytes]}]

DenseReserve[receiver_, capacity_Integer] := submit[receiver, denseReserve, {capacity}]
DenseClear[receiver_] := submit[receiver, denseClear, {}]
Options[DenseInsert] = {"ExpansionFactor" -> 0};
(* No NumericArray holds zero keys, and inserting none is the C library's no-op too. *)
DenseInsert[receiver_, {}, {}, OptionsPattern[]] := Null
DenseInsert[receiver_, keys : {__?keyQ}, vectors : (_List | _NumericArray), OptionsPattern[]] :=
    submitRows[receiver, denseInsert, {keyBytes[keys]}, vectors, {OptionValue["ExpansionFactor"]}]
Options[DenseRefine] = {"ExpansionFactor" -> 0};
DenseRefine[receiver_, OptionsPattern[]] := submit[receiver, denseRefine, {OptionValue["ExpansionFactor"]}]
DenseHealth[receiver_] := submit[receiver, denseHealth, {}]
Options[DenseSearch] = {"ExpansionFactor" -> 0, "Traversal" -> "walk"};
(* No tensor holds zero rows, so a call with no queries answers here, as C would, with nothing found. *)
DenseSearch[receiver_, {}, wanted_Integer, arena_Arena, OptionsPattern[]] := <|"Keys" -> {}, "Scores" -> {}|>
DenseSearch[receiver_, queries : ({__List} | _NumericArray), wanted_Integer, arena_Arena, OptionsPattern[]] :=
    submitRows[receiver, denseSearch, {}, queries,
        {wanted, arena, OptionValue["ExpansionFactor"], OptionValue["Traversal"]}]
Options[DenseMaxsim] = {"ExpansionFactor" -> 0};
DenseMaxsim[receiver_, {}, rowsPerQuery_Integer, wanted_Integer, arena_Arena, OptionsPattern[]] :=
    <|"Offsets" -> {0}, "Keys" -> {}, "Scores" -> {}, "Required" -> 0|>
DenseMaxsim[receiver_, queries : ({__List} | _NumericArray), rowsPerQuery_Integer, wanted_Integer, arena_Arena, OptionsPattern[]] :=
    submitRows[receiver, denseMaxsim, {}, queries, {rowsPerQuery, wanted, arena, OptionValue["ExpansionFactor"]}]
DenseCount[receiver_, key_?keyQ] := submit[receiver, denseCount, {keyBytes[key]}]
DenseGet[receiver_, key_?keyQ, arena_Arena] := submit[receiver, denseGet, {keyBytes[key], arena}]
DenseSave[receiver_, path_String] := submit[receiver, denseSave, {path}]
DenseSaveBuffer[receiver_, arena_Arena] := submit[receiver, denseSaveBuffer, {arena}]
DenseDescribe[receiver_] := submit[receiver, denseDescribe, {}]
Options[DenseMultiply] = {"Wanted" -> 0, "ExpansionFactor" -> 0};
DenseMultiply[receiver_, left : (_List | _NumericArray), arena_Arena, OptionsPattern[]] :=
    submitRows[receiver, denseMultiply, {}, left, {arena, OptionValue["Wanted"], OptionValue["ExpansionFactor"]}]
DenseMultiply[receiver_, left : (_List | _NumericArray), columnKeys : {___?keyQ}, arena_Arena,
    options : OptionsPattern[]] :=
    Module[{products, columns, positions},
        If[!DuplicateFreeQ[columnKeys], Return[Failure["invalid-argument",
            <|"MessageTemplate" -> "column keys must be unique"|>]]];
        products = DenseMultiply[receiver, left, arena, options];
        If[FailureQ[products], Return[products]];
        columns = AssociationThread[columnKeys -> Range[Length[columnKeys]]];
        positions = Flatten[MapIndexed[Thread[{First[#2], #1}] &,
            TakeList[Lookup[columns, products["Keys"], Missing["UnknownKey"]], Differences[products["Offsets"]]]], 1];
        If[!FreeQ[positions, _Missing] || !DuplicateFreeQ[positions], Return[Failure["invalid-argument",
            <|"MessageTemplate" -> "each product must identify a unique row and a known column"|>]]];
        SparseArray[Thread[positions -> products["Scores"]], {Length[products["Offsets"]] - 1, Length[columnKeys]}]]
DenseValues[receiver_, keys : {__?keyQ}, values : (_List | _NumericArray)] :=
    Replace[submitRows[receiver, denseValues, {keyBytes[keys]}, values, {}], fields_Association :> DenseValues[fields]]
Options[DenseAttend] = {"HeadCount" -> 1, "ExpansionFactor" -> 0, "ScoreScale" -> 0.};
DenseAttend[receiver_, {}, values_DenseValues, arena_Arena, OptionsPattern[]] := <|"Outputs" -> {}, "Carry" -> {}|>
DenseAttend[receiver_, queries : ({__List} | _NumericArray), DenseValues[values_Association], arena_Arena,
    OptionsPattern[]] := submitRows[receiver, denseAttend, {}, queries,
        {values["ValueKeys"], values["ValueRows"], values["Quantization"], values["Dimensions"], arena,
            OptionValue["HeadCount"], OptionValue["ExpansionFactor"], N[OptionValue["ScoreScale"]]}]

VersionName[] := versionName[]
CapabilitiesCompiled[] := capabilitiesCompiled[]
CapabilitiesAvailable[] := capabilitiesAvailable[]

End[]

EndPackage[]
