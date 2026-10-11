(* ::Package:: *)

(* :Title: USearch *)
(* :Context: USearch` *)
(* :Author: Ash Vardanian *)
(* :Date: 2023-05-04 *)
(* :Summary: USearch for the Wolfram Language: managed handles and receivers over the LibraryLink
    verbs of `lib.c`. *)

BeginPackage["USearch`"]

DenseIndex::usage = "DenseIndex[id] is a dense index, freed through usearch_dense_free once its last reference dies."
DenseIndexOn::usage = "DenseIndexOn[index, executor] runs every verb on the executor, blocking. Binding-only."
Executor::usage = "Executor[id] owns the threads or streams verbs run on, freed through usearch_executor_free."
Arena::usage = "Arena[id] holds the outputs a verb writes, freed through usearch_arena_free."

ExecutorForRam::usage = "ExecutorForRam[] calls usearch_executor_for_ram."
ExecutorOnPool::usage = "ExecutorOnPool[pool] calls usearch_executor_on_pool, borrowing the ForkUnion pool there."
ExecutorForVram::usage = "ExecutorForVram[] calls usearch_executor_for_vram."
ExecutorForNvme::usage = "ExecutorForNvme[] calls usearch_executor_for_nvme."
ExecutorDrain::usage = "ExecutorDrain[executor, mood] calls usearch_executor_drain, \"parked\" or \"spinning\"."
ArenaInit::usage = "ArenaInit[] calls usearch_arena_init, placing the arena where its \"Executor\" computes."

DenseInitRam::usage = "DenseInitRam[metric, quantization, dimensions] calls usearch_dense_init_ram."
DenseInitVram::usage = "DenseInitVram[metric, quantization, dimensions] calls usearch_dense_init_vram."
DenseLoadRam::usage = "DenseLoadRam[path] calls usearch_dense_load_ram."
DenseLoadVram::usage = "DenseLoadVram[path] calls usearch_dense_load_vram."
DenseLoadRamBuffer::usage = "DenseLoadRamBuffer[monolith] calls usearch_dense_load_ram_buffer."
DenseLoadVramBuffer::usage = "DenseLoadVramBuffer[monolith] calls usearch_dense_load_vram_buffer."
DenseViewRam::usage = "DenseViewRam[path] calls usearch_dense_view_ram."
DenseViewNvme::usage = "DenseViewNvme[path] calls usearch_dense_view_nvme."
DenseMetadataRead::usage = "DenseMetadataRead[path] calls usearch_dense_metadata_read."
DenseMetadataReadBuffer::usage = "DenseMetadataReadBuffer[monolith] calls usearch_dense_metadata_read_buffer."

DenseReserve::usage = "DenseReserve[receiver, capacity] calls usearch_dense_reserve."
DenseClear::usage = "DenseClear[receiver] calls usearch_dense_clear."
DenseInsert::usage = "DenseInsert[receiver, keys, vectors] calls usearch_dense_insert."
DenseRefine::usage = "DenseRefine[receiver] calls usearch_dense_refine."
DenseHealth::usage = "DenseHealth[receiver] calls usearch_dense_health."
DenseSearch::usage = "DenseSearch[receiver, queries, wanted, arena] calls usearch_dense_search."
DenseMaxsim::usage = "DenseMaxsim[receiver, queries, rowsPerQuery, wanted, arena] calls usearch_dense_maxsim."
DenseCount::usage = "DenseCount[receiver, key] calls usearch_dense_count."
DenseGet::usage = "DenseGet[receiver, key, arena] calls usearch_dense_get."
DenseSave::usage = "DenseSave[receiver, path] calls usearch_dense_save."
DenseSaveBuffer::usage = "DenseSaveBuffer[receiver, arena] calls usearch_dense_save_buffer."
DenseDescribe::usage = "DenseDescribe[receiver] calls usearch_dense_describe."
DenseMultiply::usage = "DenseMultiply[receiver, left, arena] calls usearch_dense_multiply."

VersionName::usage = "VersionName[] calls usearch_version_name."
CapabilitiesCompiled::usage = "CapabilitiesCompiled[] calls usearch_capabilities_compiled."
CapabilitiesAvailable::usage = "CapabilitiesAvailable[] calls usearch_capabilities_available."

Begin["`Private`"]

$library = FindLibrary["usearch_wolfram"];

verb[name_String, arguments_List] := LibraryFunctionLoad[$library, name, arguments, "DataStore"]

(* Every index verb leads with the IDs of its index and its executor. *)
receiving[name_String, arguments_List] := verb[name, Join[{Integer, Integer}, arguments]]

executorForRam = verb["ExecutorForRam", {Integer, "DataStore", Integer}];
executorOnPool = verb["ExecutorOnPool", {Integer, "DataStore", Integer}];
executorForVram = verb["ExecutorForVram", {Integer, "DataStore", "DataStore", Integer}];
executorForNvme = verb["ExecutorForNvme", {Integer, "DataStore", Integer}];
executorDrain = verb["ExecutorDrain", {Integer, "UTF8String"}];
arenaInit = verb["ArenaInit", {Integer, Integer}];
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
denseInsert = receiving["DenseInsert", {LibraryDataType[NumericArray], {Real, _}, Integer}];
denseRefine = receiving["DenseRefine", {Integer}];
denseHealth = receiving["DenseHealth", {}];
denseSearch = receiving["DenseSearch", {{Real, _}, Integer, Integer, Integer}];
denseMaxsim = receiving["DenseMaxsim", {{Real, _}, Integer, Integer, Integer, Integer}];
denseCount = receiving["DenseCount", {LibraryDataType[NumericArray]}];
denseGet = receiving["DenseGet", {LibraryDataType[NumericArray], Integer}];
denseSave = receiving["DenseSave", {"UTF8String"}];
denseSaveBuffer = receiving["DenseSaveBuffer", {Integer}];
denseDescribe = receiving["DenseDescribe", {}];
denseMultiply = receiving["DenseMultiply", {{Real, _}, Integer, Integer, Integer}];
versionName = LibraryFunctionLoad[$library, "VersionName", {}, "UTF8String"];
capabilitiesCompiled = LibraryFunctionLoad[$library, "CapabilitiesCompiled", {}, "UTF8String"];
capabilitiesAvailable = LibraryFunctionLoad[$library, "CapabilitiesAvailable", {}, "UTF8String"];

(* A key is an integer below 2^128, crossing as its sixteen base-256 digits, most significant first. *)
keyQ[key_] := IntegerQ[key] && 0 <= key < 2^128
keyBytes[keys_] := NumericArray[IntegerDigits[keys, 256, 16], "UnsignedInteger8"]
keysFromBytes[bytes_NumericArray] := If[MemberQ[Dimensions[bytes], 0], ConstantArray[0, Most[Dimensions[bytes]]],
    Normal[bytes] . (256^Range[15, 0, -1])]

store[list_List] := Developer`DataStore @@ list
rows[vectors_] := Developer`ToPackedArray[N[vectors]]
monolith[bytes_ByteArray] := NumericArray[Normal[bytes], "UnsignedInteger8"]
monolith[bytes_NumericArray] := bytes

(* Outputs arrive under their C names; found matches split by each row's count, and a lone output stands alone. *)
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

Options[DenseInitRam] = {"Connectivity" -> 0, "ValuesPerKey" -> 0, "MemoryDomains" -> {}};
DenseInitRam[metric_String, quantization_String, dimensions_Integer, OptionsPattern[]] :=
    construct[DenseIndex, denseInitRam, {metric, quantization, dimensions, OptionValue["Connectivity"],
        OptionValue["ValuesPerKey"], store[OptionValue["MemoryDomains"]]}]
Options[DenseInitVram] = {"Connectivity" -> 0, "ValuesPerKey" -> 0, "Devices" -> {}};
DenseInitVram[metric_String, quantization_String, dimensions_Integer, OptionsPattern[]] :=
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
DenseInsert[receiver_, keys : {__?keyQ}, vectors_List, OptionsPattern[]] :=
    submit[receiver, denseInsert, {keyBytes[keys], rows[vectors], OptionValue["ExpansionFactor"]}]
Options[DenseRefine] = {"ExpansionFactor" -> 0};
DenseRefine[receiver_, OptionsPattern[]] := submit[receiver, denseRefine, {OptionValue["ExpansionFactor"]}]
DenseHealth[receiver_] := submit[receiver, denseHealth, {}]
Options[DenseSearch] = {"ExpansionFactor" -> 0};
(* No tensor holds zero rows, so a call with no queries answers here, as C would, with nothing found. *)
DenseSearch[receiver_, {}, wanted_Integer, arena_Arena, OptionsPattern[]] := <|"Keys" -> {}, "Scores" -> {}|>
DenseSearch[receiver_, queries : {__List}, wanted_Integer, arena_Arena, OptionsPattern[]] :=
    submit[receiver, denseSearch, {rows[queries], wanted, arena, OptionValue["ExpansionFactor"]}]
Options[DenseMaxsim] = {"ExpansionFactor" -> 0};
DenseMaxsim[receiver_, {}, rowsPerQuery_Integer, wanted_Integer, arena_Arena, OptionsPattern[]] :=
    <|"Keys" -> {}, "Scores" -> {}|>
DenseMaxsim[receiver_, queries : {__List}, rowsPerQuery_Integer, wanted_Integer, arena_Arena, OptionsPattern[]] :=
    submit[receiver, denseMaxsim, {rows[queries], rowsPerQuery, wanted, arena, OptionValue["ExpansionFactor"]}]
DenseCount[receiver_, key_?keyQ] := submit[receiver, denseCount, {keyBytes[key]}]
DenseGet[receiver_, key_?keyQ, arena_Arena] := submit[receiver, denseGet, {keyBytes[key], arena}]
DenseSave[receiver_, path_String] := submit[receiver, denseSave, {path}]
DenseSaveBuffer[receiver_, arena_Arena] := submit[receiver, denseSaveBuffer, {arena}]
DenseDescribe[receiver_] := submit[receiver, denseDescribe, {}]
Options[DenseMultiply] = {"Wanted" -> 0, "ExpansionFactor" -> 0};
DenseMultiply[receiver_, left_List, arena_Arena, OptionsPattern[]] := submit[receiver, denseMultiply,
    {rows[left], arena, OptionValue["Wanted"], OptionValue["ExpansionFactor"]}]

VersionName[] := versionName[]
CapabilitiesCompiled[] := capabilitiesCompiled[]
CapabilitiesAvailable[] := capabilitiesAvailable[]

End[]

EndPackage[]
