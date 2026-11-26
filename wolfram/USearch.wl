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
SequenceIndex::usage = "SequenceIndex[id] is an index of byte strings, UTF-8 text, DNA, RNA or proteins, freed once nothing references it."
ShapesIndex::usage = "ShapesIndex[id] is an index of 3D point clouds, freed once nothing references it."
GeospatialIndex::usage = "GeospatialIndex[id] is an index of observations on and above the Earth, freed once nothing references it."
DenseIndexOn::usage = "DenseIndexOn[index, executor] is a receiver that runs every verb on executor, blocking until the verb is final; a bare DenseIndex runs verbs on the calling thread."
SequenceIndexOn::usage = "SequenceIndexOn[index, executor] is a receiver that runs every verb on executor, blocking until the verb is final; a bare SequenceIndex runs verbs on the calling thread."
ShapesIndexOn::usage = "ShapesIndexOn[index, executor] is a receiver that runs every verb on executor, blocking until the verb is final; a bare ShapesIndex runs verbs on the calling thread."
GeospatialIndexOn::usage = "GeospatialIndexOn[index, executor] is a receiver that runs every verb on executor, blocking until the verb is final; a bare GeospatialIndex runs verbs on the calling thread."
DenseIndexDeferred::usage = "DenseIndexDeferred[index, executor] is a receiver that submits every verb to executor and answers a DeferredCall at once."
SequenceIndexDeferred::usage = "SequenceIndexDeferred[index, executor] is a receiver that submits every verb to executor and answers a DeferredCall at once."
ShapesIndexDeferred::usage = "ShapesIndexDeferred[index, executor] is a receiver that submits every verb to executor and answers a DeferredCall at once."
GeospatialIndexDeferred::usage = "GeospatialIndexDeferred[index, executor] is a receiver that submits every verb to executor and answers a DeferredCall at once."
DeferredCall::usage = "DeferredCall[id] is a submitted call, which DeferredStatus polls and DeferredWait finishes; releasing it unwaited waits for its outcome first."
DeferredStatus::usage = "DeferredStatus[call] reads the status of a DeferredCall without waiting: \"deferred\" until the call is final, then its outcome's status, and \"uninitialized\" once it was waited on."
DeferredWait::usage = "DeferredWait[call] waits until a DeferredCall is final and answers what its verb would have answered on a blocking receiver, once."
Plan::usage = "Plan[id] boxes one refine or reshard plan that calls naming it resume, freed once nothing references it; a box released while its plan lives leaks the plan."
PlanInit::usage = "PlanInit[] makes an empty Plan, which the first refine or reshard naming it as \"Plan\" opens."
PlanProgress::usage = "PlanProgress[plan] reports where the last call naming plan stopped: its \"Phase\", that phase's \"Ordinal\", and its \"Completed\" and \"Total\" units."
Executor::usage = "Executor[id] owns the threads, streams or submission rings verbs run on, freed once nothing references it."
Arena::usage = "Arena[id] holds the outputs a verb writes, rewound by every call writing into it and freed once nothing references it."

ExecutorForRam::usage = "ExecutorForRam[] spawns host worker threads for RAM-resident indexes. \"MemoryDomains\" lists the NUMA nodes to place shards on, {} deriving every node the process may allocate on, and \"Threads\" is how many threads to spawn, 0 taking 7/8 of the cores those nodes carry."
ExecutorForVram::usage = "ExecutorForVram[] orders work on one stream per device for VRAM-resident indexes. \"Devices\" lists the device ordinals to place shards on, {} picking device zero; \"Streams\" is declared for your streams, one per device, which answer \"unsupported\", since the executor drives only its own, {} using the one stream per device the executor opens; and \"ScratchLimit\" caps the memory each stream carves scratch and outputs from, past which a call fails with \"out-of-memory\", 0 letting it grow as needed."
ExecutorForNvme::usage = "ExecutorForNvme[] opens one submission ring per worker for NVMe-resident indexes. \"Paths\" lists the devices or directories to serve shards from, {} leaving every shard where its manifest points, and \"Threads\" is how many workers to spawn, 0 deriving it from the devices' queues."
ExecutorDrain::usage = "ExecutorDrain[executor, mood] waits until no call holds the executor, then sets what idle workers do: \"parked\", the default, or \"spinning\"."
ArenaInit::usage = "ArenaInit[] makes an arena for verb outputs, placed where its \"Executor\" computes; None, the default, holds host memory for calls on the calling thread."
ArenaDescribe::usage = "ArenaDescribe[arena] reports where the arena's outputs live: its \"Residency\", \"ram\" for host pages or \"vram\" for one device's memory, the position of that \"Device\" among the library's GPUs, its \"Vendor\" and the \"VendorOrdinal\" its vendor numbers it by, each None for host pages, and \"HostAccess\", \"readable\" when host code may read the outputs or \"refused\" when only the device may."

DenseInitRam::usage = "DenseInitRam[metric, quantization, dimensions] creates an empty, mutable index in host memory, scoring by metric, \"angular\" by default, and storing quantization scalars, \"bf16\" by default. \"Connectivity\" bounds the connections per node, 0 deriving it; \"ValuesPerKey\" bounds the entries sharing one key, 0 leaving it unbounded; and \"MemoryDomains\" lists the NUMA nodes to place shards on, {} deriving every node the process may allocate on."
DenseInitVram::usage = "DenseInitVram[metric, quantization, dimensions] creates an empty, mutable index in device memory, scoring by metric, \"angular\" by default, and storing quantization scalars, \"bf16\" by default. \"Connectivity\" bounds the connections per node, 0 deriving it; \"ValuesPerKey\" bounds the entries sharing one key, 0 leaving it unbounded; and \"Devices\" lists the device ordinals to place shards on, {} picking device zero."
DenseLoadRam::usage = "DenseLoadRam[path] copies a checkpoint, a manifest or a single monolith, into a new, mutable index in host memory. \"Executor\" runs the call, None meaning the calling thread; \"MemoryDomains\" lists the NUMA nodes to place shards on, {} deriving every node the process may allocate on; and \"ValuesPerKey\" bounds the entries sharing one key, which no checkpoint records, 0 leaving it unbounded."
DenseLoadVram::usage = "DenseLoadVram[path] copies a checkpoint, a manifest or a single monolith, into a new, mutable index in device memory. \"Executor\" runs the call, None meaning the calling thread; \"Devices\" lists the device ordinals to place shards on, {} picking device zero; and \"ValuesPerKey\" bounds the entries sharing one key, which no checkpoint records, 0 leaving it unbounded."
DenseLoadRamBuffer::usage = "DenseLoadRamBuffer[monolith] copies a checkpoint held in a ByteArray or a byte NumericArray into a new, mutable index in host memory, taking the options of DenseLoadRam."
DenseLoadVramBuffer::usage = "DenseLoadVramBuffer[monolith] copies a checkpoint held in a ByteArray or a byte NumericArray into a new, mutable index in device memory, taking the options of DenseLoadVram."
DenseViewRam::usage = "DenseViewRam[path] serves a checkpoint in place through a memory mapping, as a read-only index."
DenseViewNvme::usage = "DenseViewNvme[path] serves a checkpoint in place from NVMe drives as a read-only index, which every later call reads through its executor. \"Executor\" runs the call, None meaning the calling thread, \"Paths\" lists the devices or directories to serve shards from, {} leaving every shard where its manifest points; and \"KeyTableDomains\" lists the NUMA nodes of a key table held in RAM, which DenseCount, DenseGet, DenseMaxsim and hinted searches read, {} deriving them and None holding none."
DenseLoadNvme::usage = "DenseLoadNvme[path] opens a checkpoint's manifest from NVMe drives as a writable index: it answers every read a view answers, and DenseRefine and DenseReshard rewrite its parts one at a time. It takes the options of DenseViewNvme, with \"Paths\" naming none, and a lone monolith fails as \"unsupported\"."
DenseMetadataRead::usage = "DenseMetadataRead[path] reads a checkpoint's geometry and metric without loading it. \"Executor\" runs the call, None meaning the calling thread."
DenseMetadataReadBuffer::usage = "DenseMetadataReadBuffer[monolith] reads the geometry and metric of a checkpoint held in a ByteArray or a byte NumericArray, taking the option of DenseMetadataRead."

DenseReserve::usage = "DenseReserve[receiver, capacity] grows the index ahead of incoming vectors to capacity entries in total, the current ones included."
DenseClear::usage = "DenseClear[receiver] erases every entry, keeping the reserved capacity and the frozen geometry."
DenseInsert::usage = "DenseInsert[receiver, keys, vectors] adds vectors, one row per key, under keys, each an integer below 2^128. \"ExpansionFactor\" sets the breadth of the candidates the walk explores, 0 deriving it."
DenseRefine::usage = "DenseRefine[receiver] runs one round over the whole graph: it heals every island a search cannot reach, gathers every entry into a neighborhood of near entries and re-stitches each one. \"ExpansionFactor\" sets the breadth of any walk beside the round, 0 deriving it; \"MemoryLimit\" the bytes a loaded NVMe index holds for the one part it rewrites, 0 deriving it and less streaming the part from the drives; and \"Plan\" names a Plan the call opens or resumes, None running one round on its own. It answers the \"Reselected\", \"Changed\", \"IslandsFound\", \"IslandsReconnected\", \"Neighborhoods\", \"WidestNeighborhood\", \"AddedLinks\" and \"DroppedLinks\" counts, the \"Verdict\" of \"none\", \"improved\", \"converged\", \"stalled\" or \"stopped\", and the \"Opening\" and \"Reachability\" audits, each as DenseHealth answers it."
DenseRefinePlanFinish::usage = "DenseRefinePlanFinish[receiver, plan] closes a refine plan, leaving its Plan empty whatever it answers, and answers the report of every round the plan ran, as DenseRefine does."
DenseReshard::usage = "DenseReshard[receiver, sources, targets] moves the entries of the sources into the targets, keeping their links, then runs a round on every target that took entries and answers its report as DenseRefine does; the receiver chooses where the call runs, and its index only names the family. Several sources into one target among them gather into it, one source into several empty targets cuts it into one graph-local part apiece, and one index as its only source and target recuts its own shards; a source the targets do not name ends empty and live, and every listed index must outlive the call and its plan. \"MemoryDomains\", \"Devices\" and \"Paths\" name a RAM, VRAM or NVMe placement for a recut, None keeping the index's own; \"ShardCapacity\" bounds the entries per RAM or VRAM shard, 0 keeping the index's bound; \"ExpansionFactor\" sets the breadth of any walk beside the rounds, 0 deriving it; \"MemoryLimit\" bounds a loaded NVMe target as in DenseRefine; \"Seed\" picks the part roots of a cut; and \"Plan\" names a Plan as in DenseRefine."
DenseReshardPlanFinish::usage = "DenseReshardPlanFinish[receiver, plan] closes a reshard plan, leaving its Plan empty whatever it answers, and answers the report of every round the plan ran, as DenseRefine does."
DenseHealth::usage = "DenseHealth[receiver] audits how much of the graph a search can reach, leaving the index unchanged: the \"Audited\" entries, the \"Reachable\" ones, the \"DepthFromRoot\" and \"DepthToRoot\" of its two sweeps, the \"Unlinked\", \"Underlinked\" and \"BeyondMostPopulatedDepth\" entries, the \"UnreachedIslands\" and \"StrandedIslands\", and the undirected \"Components\"."
DenseSearch::usage = "DenseSearch[receiver, queries, wanted, arena] finds up to wanted nearest entries for each query row, answering their \"Keys\" and \"Scores\" per query. \"ExpansionFactor\" sets the breadth of the candidates the walk explores, 0 deriving it, and \"Traversal\" is \"walk\", the default, or \"scan\", an exact pass over every entry."
DenseMaxsim::usage = "DenseMaxsim[receiver, queries, rowsPerQuery, wanted, arena] scores queries of rowsPerQuery rows against multi-row entries: every query row takes its best match within a key, and those maxima sum to the key's score, keeping up to wanted keys per query. \"ExpansionFactor\" sets the breadth of the candidates the walk explores, 0 deriving it."
DenseCount::usage = "DenseCount[receiver, key] counts the entries stored under key."
DenseGet::usage = "DenseGet[receiver, key, arena] retrieves every vector stored under key, one row per entry."
DenseSave::usage = "DenseSave[receiver, path] saves the index under path, as one monolith or as a manifest beside its shards, while readers keep searching."
DenseSaveBuffer::usage = "DenseSaveBuffer[receiver, arena] saves the index into arena, as one monolith for an index of one shard or as a manifest leading one part per shard."
DenseDescribe::usage = "DenseDescribe[receiver] reports the index's frozen metadata and current counters."
DenseMultiply::usage = "DenseMultiply[receiver, left, arena] approximates the sparse product of the rows of left with the indexed vectors, answering every pair the walk scored as \"Offsets\", \"Keys\" and \"Scores\"; DenseMultiply[receiver, left, columnKeys, arena] lays the same products out as a SparseArray with one column per key of columnKeys, on a blocking receiver. \"Wanted\" bounds the pairs kept per row, the first ones its walk scored, 0 keeping every pair, and \"ExpansionFactor\" sets the breadth of the candidates, which widens the pattern, 0 deriving it."
DenseValues::usage = "DenseValues[receiver, keys, values] prepares immutable value rows, one per key, in the index's quantization, for DenseAttend, on a blocking receiver."
DenseAttend::usage = "DenseAttend[receiver, queries, values, arena] folds softmax attention of the query heads over every key their graph walks score, reading each key's row from values, a DenseValues. \"HeadCount\" is how many query heads share one walk per token, dividing the query count; \"ExpansionFactor\" sets the breadth of the candidates the walk explores, 0 deriving it; and \"ScoreScale\" is the logit scale, conventionally the inverse square root of the head dimensionality, 0. deriving it from the index's geometry."

SequenceInitRam::usage = "SequenceInitRam[metric] creates an empty, mutable sequence index in host memory, scoring by metric, \"levenshtein\" by default, or \"levenshtein-utf8\", \"needleman-wunsch\" or \"overlap\". \"SubstitutionMatrix\" is the 256 by 256 integer matrix Needleman-Wunsch scores with, None naming none, beside its \"Gap\" cost; \"WindowWidths\" lists the window widths overlap scores with; \"Connectivity\" bounds the connections per node, 0 deriving it; \"ValuesPerKey\" bounds the entries sharing one key, 0 leaving it unbounded; and \"MemoryDomains\" lists the NUMA nodes to place shards on, {} deriving every node the process may allocate on."
SequenceInitVram::usage = "SequenceInitVram[metric] creates an empty, mutable sequence index in device memory, taking the options of SequenceInitRam with \"Devices\", the device ordinals to place shards on, {} picking device zero, in place of \"MemoryDomains\"."
SequenceLoadRam::usage = "SequenceLoadRam[path] copies a checkpoint into a new, mutable sequence index in host memory. \"Executor\" runs the call, None meaning the calling thread; \"MemoryDomains\" lists the NUMA nodes to place shards on; \"ValuesPerKey\" bounds the entries sharing one key; and \"SubstitutionMatrix\" and \"Gap\" name the weights again, since no checkpoint records them."
SequenceLoadVram::usage = "SequenceLoadVram[path] copies a checkpoint into a new, mutable sequence index in device memory, taking the options of SequenceLoadRam with \"Devices\" in place of \"MemoryDomains\"."
SequenceLoadRamBuffer::usage = "SequenceLoadRamBuffer[monolith] copies a checkpoint held in a ByteArray or a byte NumericArray into a new, mutable sequence index in host memory, taking the options of SequenceLoadRam."
SequenceLoadVramBuffer::usage = "SequenceLoadVramBuffer[monolith] copies a checkpoint held in a ByteArray or a byte NumericArray into a new, mutable sequence index in device memory, taking the options of SequenceLoadVram."
SequenceViewRam::usage = "SequenceViewRam[path] serves a checkpoint in place through a memory mapping, as a read-only sequence index, taking \"SubstitutionMatrix\" and \"Gap\" as SequenceLoadRam does."
SequenceViewNvme::usage = "SequenceViewNvme[path] serves a checkpoint in place from NVMe drives as a read-only sequence index. \"Executor\" runs the call, None meaning the calling thread; \"Paths\" lists the devices or directories to serve shards from, {} leaving every shard where its manifest points; and \"SubstitutionMatrix\" and \"Gap\" name the weights as SequenceLoadRam does."
SequenceMetadataRead::usage = "SequenceMetadataRead[path] reads a checkpoint's frozen \"Metric\" and \"Connectivity\" without loading it. \"Executor\" runs the call, None meaning the calling thread."
SequenceMetadataReadBuffer::usage = "SequenceMetadataReadBuffer[monolith] reads the frozen metric and connectivity of a checkpoint held in a ByteArray or a byte NumericArray, taking the option of SequenceMetadataRead."
SequenceReserve::usage = "SequenceReserve[receiver, entries, payloadBytes] grows the index ahead of incoming sequences to entries in total, the current ones included, holding payloadBytes bytes in all, 0 by default."
SequenceClear::usage = "SequenceClear[receiver] erases every entry, keeping the reserved capacity and the frozen settings."
SequenceInsert::usage = "SequenceInsert[receiver, keys, sequences] adds sequences, each a string read as its UTF-8 bytes, a ByteArray or a list of byte values, one per key. \"ExpansionFactor\" sets the breadth of the candidates the walk explores, 0 deriving it."
SequenceRefine::usage = "SequenceRefine[receiver] runs one refinement round over the whole graph, taking the options of DenseRefine and answering its report."
SequenceRefinePlanFinish::usage = "SequenceRefinePlanFinish[receiver, plan] closes a refine plan as DenseRefinePlanFinish does."
SequenceReshard::usage = "SequenceReshard[receiver, sources, targets] moves the entries of the sources into the targets as DenseReshard does, taking its \"ExpansionFactor\", \"MemoryLimit\", \"Seed\" and \"Plan\" options."
SequenceReshardPlanFinish::usage = "SequenceReshardPlanFinish[receiver, plan] closes a reshard plan as DenseReshardPlanFinish does."
SequenceHealth::usage = "SequenceHealth[receiver] audits how much of the graph a search can reach, answering as DenseHealth does."
SequenceSearch::usage = "SequenceSearch[receiver, queries, wanted, arena] finds up to wanted nearest entries for each query sequence, answering their \"Keys\" and \"Scores\" per query, taking the options of DenseSearch."
SequenceCount::usage = "SequenceCount[receiver, key] counts the \"Entries\" stored under key, with their \"TotalBytes\" and \"LargestBytes\"."
SequenceGet::usage = "SequenceGet[receiver, key, arena] retrieves every sequence stored under key, one ByteArray per entry."
SequenceSave::usage = "SequenceSave[receiver, path] saves the index under path, as DenseSave does."
SequenceSaveBuffer::usage = "SequenceSaveBuffer[receiver, arena] saves the index into arena, as DenseSaveBuffer does."
SequenceDescribe::usage = "SequenceDescribe[receiver] reports the index's frozen metric and connectivity and its current counters."

ShapesInitRam::usage = "ShapesInitRam[metric, quantization] creates an empty, mutable shapes index in host memory, scoring by metric, \"chamfer\" by default, or \"hausdorff\", \"gromov-wasserstein\" or \"wasserstein-procrustes\", and storing quantization scalars, \"f32\" by default, or \"f16\". \"Points\" freezes every shape's point count, 0 leaving it free; \"Connectivity\" bounds the connections per node, 0 deriving it; \"ValuesPerKey\" bounds the entries sharing one key, 0 leaving it unbounded; and \"MemoryDomains\" lists the NUMA nodes to place shards on, {} deriving every node the process may allocate on."
ShapesInitVram::usage = "ShapesInitVram[metric, quantization] creates an empty, mutable shapes index in device memory, taking the options of ShapesInitRam with \"Devices\", the device ordinals to place shards on, {} picking device zero, in place of \"MemoryDomains\"."
ShapesLoadRam::usage = "ShapesLoadRam[path] copies a checkpoint into a new, mutable shapes index in host memory, taking the options of DenseLoadRam."
ShapesLoadVram::usage = "ShapesLoadVram[path] copies a checkpoint into a new, mutable shapes index in device memory, taking the options of DenseLoadVram."
ShapesLoadRamBuffer::usage = "ShapesLoadRamBuffer[monolith] copies a checkpoint held in a ByteArray or a byte NumericArray into a new, mutable shapes index in host memory, taking the options of ShapesLoadRam."
ShapesLoadVramBuffer::usage = "ShapesLoadVramBuffer[monolith] copies a checkpoint held in a ByteArray or a byte NumericArray into a new, mutable shapes index in device memory, taking the options of ShapesLoadVram."
ShapesViewRam::usage = "ShapesViewRam[path] serves a checkpoint in place through a memory mapping, as a read-only shapes index."
ShapesViewNvme::usage = "ShapesViewNvme[path] serves a checkpoint in place from NVMe drives as a read-only shapes index. \"Executor\" runs the call, None meaning the calling thread, and \"Paths\" lists the devices or directories to serve shards from, {} leaving every shard where its manifest points."
ShapesMetadataRead::usage = "ShapesMetadataRead[path] reads a checkpoint's frozen \"Metric\", \"Quantization\", \"Points\" and \"Connectivity\" without loading it. \"Executor\" runs the call, None meaning the calling thread."
ShapesMetadataReadBuffer::usage = "ShapesMetadataReadBuffer[monolith] reads the frozen geometry of a checkpoint held in a ByteArray or a byte NumericArray, taking the option of ShapesMetadataRead."
ShapesReserve::usage = "ShapesReserve[receiver, entries, points] grows the index ahead of incoming shapes to entries in total, the current ones included, holding points points in all, 0 by default."
ShapesClear::usage = "ShapesClear[receiver] erases every entry, keeping the reserved capacity and the frozen settings."
ShapesInsert::usage = "ShapesInsert[receiver, keys, shapes] adds shapes, each a real matrix of {x, y, z} rows, one per key. \"ExpansionFactor\" sets the breadth of the candidates the walk explores, 0 deriving it."
ShapesRefine::usage = "ShapesRefine[receiver] runs one refinement round over the whole graph, taking the options of DenseRefine and answering its report."
ShapesRefinePlanFinish::usage = "ShapesRefinePlanFinish[receiver, plan] closes a refine plan as DenseRefinePlanFinish does."
ShapesReshard::usage = "ShapesReshard[receiver, sources, targets] moves the entries of the sources into the targets as DenseReshard does, taking its \"ExpansionFactor\", \"MemoryLimit\", \"Seed\" and \"Plan\" options."
ShapesReshardPlanFinish::usage = "ShapesReshardPlanFinish[receiver, plan] closes a reshard plan as DenseReshardPlanFinish does."
ShapesHealth::usage = "ShapesHealth[receiver] audits how much of the graph a search can reach, answering as DenseHealth does."
ShapesSearch::usage = "ShapesSearch[receiver, queries, wanted, arena] finds up to wanted nearest entries for each query shape, answering their \"Keys\" and \"Scores\" per query, taking the options of DenseSearch."
ShapesCount::usage = "ShapesCount[receiver, key] counts the \"Entries\" stored under key, with their \"TotalPoints\" and \"LargestPoints\"."
ShapesGet::usage = "ShapesGet[receiver, key, arena] retrieves every shape stored under key in its inserted pose, one real matrix of {x, y, z} rows per entry."
ShapesSave::usage = "ShapesSave[receiver, path] saves the index under path, as DenseSave does."
ShapesSaveBuffer::usage = "ShapesSaveBuffer[receiver, arena] saves the index into arena, as DenseSaveBuffer does."
ShapesDescribe::usage = "ShapesDescribe[receiver] reports the index's frozen geometry, its current counters and the \"TotalPoints\" it stores."

GeospatialInitRam::usage = "GeospatialInitRam[] creates an empty, mutable geospatial index in host memory. \"Connectivity\" bounds the connections per node, 0 deriving it; \"ValuesPerKey\" bounds the entries sharing one key, 0 leaving it unbounded; and \"MemoryDomains\" lists the NUMA nodes to place shards on, {} deriving every node the process may allocate on."
GeospatialInitVram::usage = "GeospatialInitVram[] creates an empty, mutable geospatial index in device memory, taking the options of GeospatialInitRam with \"Devices\", the device ordinals to place shards on, {} picking device zero, in place of \"MemoryDomains\"."
GeospatialLoadRam::usage = "GeospatialLoadRam[path] copies a checkpoint into a new, mutable geospatial index in host memory, taking the options of DenseLoadRam."
GeospatialLoadVram::usage = "GeospatialLoadVram[path] copies a checkpoint into a new, mutable geospatial index in device memory, taking the options of DenseLoadVram."
GeospatialLoadRamBuffer::usage = "GeospatialLoadRamBuffer[monolith] copies a checkpoint held in a ByteArray or a byte NumericArray into a new, mutable geospatial index in host memory, taking the options of GeospatialLoadRam."
GeospatialLoadVramBuffer::usage = "GeospatialLoadVramBuffer[monolith] copies a checkpoint held in a ByteArray or a byte NumericArray into a new, mutable geospatial index in device memory, taking the options of GeospatialLoadVram."
GeospatialViewRam::usage = "GeospatialViewRam[path] serves a checkpoint in place through a memory mapping, as a read-only geospatial index."
GeospatialViewNvme::usage = "GeospatialViewNvme[path] serves a checkpoint in place from NVMe drives as a read-only geospatial index, taking the options of ShapesViewNvme."
GeospatialMetadataRead::usage = "GeospatialMetadataRead[path] reads a checkpoint's frozen \"Connectivity\" without loading it. \"Executor\" runs the call, None meaning the calling thread."
GeospatialMetadataReadBuffer::usage = "GeospatialMetadataReadBuffer[monolith] reads the frozen connectivity of a checkpoint held in a ByteArray or a byte NumericArray, taking the option of GeospatialMetadataRead."
GeospatialReserve::usage = "GeospatialReserve[receiver, entries] grows the index ahead of incoming observations to entries in total, the current ones included."
GeospatialClear::usage = "GeospatialClear[receiver] erases every entry, keeping the reserved capacity and the frozen settings."
GeospatialInsert::usage = "GeospatialInsert[receiver, keys, observations] adds observations, one per key, each a list of the eleven fields of usearch_geospatial_observation_t in its order, from latitude and longitude in degrees to the begin and end Unix seconds, a field no device recorded being any non-numeric value such as Missing[]. \"ExpansionFactor\" sets the breadth of the candidates the walk explores, 0 deriving it."
GeospatialRefine::usage = "GeospatialRefine[receiver] runs one refinement round over the whole graph, taking the options of DenseRefine and answering its report."
GeospatialRefinePlanFinish::usage = "GeospatialRefinePlanFinish[receiver, plan] closes a refine plan as DenseRefinePlanFinish does."
GeospatialReshard::usage = "GeospatialReshard[receiver, sources, targets] moves the entries of the sources into the targets as DenseReshard does, taking its \"ExpansionFactor\", \"MemoryLimit\", \"Seed\" and \"Plan\" options."
GeospatialReshardPlanFinish::usage = "GeospatialReshardPlanFinish[receiver, plan] closes a reshard plan as DenseReshardPlanFinish does."
GeospatialHealth::usage = "GeospatialHealth[receiver] audits how much of the graph a search can reach, answering as DenseHealth does."
GeospatialSearch::usage = "GeospatialSearch[receiver, queries, wanted, arena] finds up to wanted nearest entries the relation admits for each query observation, answering their \"Keys\" and their \"Scores\" in chord metres per query. \"Relation\" is the usearch_geospatial_relation_t value: 1 near, the default, 2 visible, 3 sees or 4 seen; \"Metres\" and \"Seconds\" narrow every relation by a radius and a time slack, 0. ranking by distance alone; \"ExpansionFactor\" and \"Traversal\" act as in DenseSearch."
GeospatialWithin::usage = "GeospatialWithin[receiver, queries, metres, arena] reports every entry within metres of each query observation that the relation admits, as \"Offsets\", \"Keys\" and \"Scores\" in chord metres, laid out as DenseMultiply lays its pairs. \"Relation\", \"Seconds\" and \"Traversal\" act as in GeospatialSearch."
GeospatialCount::usage = "GeospatialCount[receiver, key] counts the entries stored under key."
GeospatialGet::usage = "GeospatialGet[receiver, key, arena] retrieves every observation stored under key within the stored rounding, one eleven-field row per entry, a field no device recorded reading as Missing[\"NotRecorded\"]."
GeospatialSave::usage = "GeospatialSave[receiver, path] saves the index under path, as DenseSave does."
GeospatialSaveBuffer::usage = "GeospatialSaveBuffer[receiver, arena] saves the index into arena, as DenseSaveBuffer does."
GeospatialDescribe::usage = "GeospatialDescribe[receiver] reports the index's frozen connectivity and its current counters."

VersionName::usage = "VersionName[] gives the library's semantic version."
CapabilitiesCompiled::usage = "CapabilitiesCompiled[] lists the kernel families this build holds."
CapabilitiesAvailable::usage = "CapabilitiesAvailable[] lists the kernel families this machine can run."

Begin["`Private`"]

$library = FindLibrary["usearch_wolfram"];

verb[name_String, arguments_List] := LibraryFunctionLoad[$library, name, arguments, "DataStore"]

(* Every index verb leads with the IDs of its index, its executor and its DeferredCall. *)
receiving[name_String, arguments_List] := verb[name, Join[{Integer, Integer, Integer}, arguments]]

(* Rows cross "Shared", since a deferred call reads them after the verb returns and disowns them once final. *)
receivingRows[name_String, before_List, after_List] := Quiet[<|
    List -> receiving[name, Join[before, {{Real, _, "Shared"}, Boolean}, after]],
    NumericArray -> receiving[name, Join[before, {{LibraryDataType[NumericArray], "Shared"}, Boolean}, after]]|>,
    LibraryFunction::overload]

passed = {LibraryDataType[NumericArray], "Shared"};
weights = Sequence["Boolean", {LibraryDataType[NumericArray], "Constant"}, Integer];
bytesTape = Sequence[passed, {Integer, 1, "Shared"}];
pointsTape = Sequence[{Real, 2, "Shared"}, {Integer, 1, "Shared"}];
observed = Sequence[{Real, 2, "Constant"}, {Integer, 2, "Constant"}];
reshardLists = Sequence["DataStore", "DataStore", Integer, Integer, Integer, Integer];

executorForRam = verb["ExecutorForRam", {Integer, "DataStore", Integer}];
executorForVram = verb["ExecutorForVram", {Integer, "DataStore", "DataStore", Integer}];
executorForNvme = verb["ExecutorForNvme", {Integer, "DataStore", Integer}];
executorDrain = verb["ExecutorDrain", {Integer, "UTF8String"}];
arenaInit = verb["ArenaInit", {Integer, Integer}];
arenaDescribe = verb["ArenaDescribe", {Integer}];
deferredStatus = LibraryFunctionLoad[$library, "DeferredStatus", {Integer}, "UTF8String"];
deferredWait = verb["DeferredWait", {Integer}];
planProgress = verb["PlanProgress", {Integer}];

denseInitRam = verb["DenseInitRam", {Integer, "UTF8String", "UTF8String", Integer, Integer, Integer, "DataStore"}];
denseInitVram = verb["DenseInitVram", {Integer, "UTF8String", "UTF8String", Integer, Integer, Integer, "DataStore"}];
denseLoadRam = verb["DenseLoadRam", {Integer, Integer, "UTF8String", "DataStore", Integer}];
denseLoadVram = verb["DenseLoadVram", {Integer, Integer, "UTF8String", "DataStore", Integer}];
denseLoadRamBuffer =
    verb["DenseLoadRamBuffer", {Integer, Integer, LibraryDataType[NumericArray], "DataStore", Integer}];
denseLoadVramBuffer =
    verb["DenseLoadVramBuffer", {Integer, Integer, LibraryDataType[NumericArray], "DataStore", Integer}];
denseViewRam = verb["DenseViewRam", {Integer, "UTF8String"}];
denseViewNvme = verb["DenseViewNvme", {Integer, Integer, "UTF8String", "DataStore", "Boolean", "DataStore"}];
denseLoadNvme = verb["DenseLoadNvme", {Integer, Integer, "UTF8String", "DataStore", "Boolean", "DataStore"}];
denseMetadataRead = verb["DenseMetadataRead", {Integer, "UTF8String"}];
denseMetadataReadBuffer = verb["DenseMetadataReadBuffer", {Integer, LibraryDataType[NumericArray]}];
denseReserve = receiving["DenseReserve", {Integer}];
denseClear = receiving["DenseClear", {}];
denseInsert = receivingRows["DenseInsert", {passed}, {Integer}];
denseRefine = receiving["DenseRefine", {Integer, Integer, Integer}];
denseRefinePlanFinish = receiving["DenseRefinePlanFinish", {Integer}];
denseReshard = receiving["DenseReshard", {"DataStore", "DataStore", "Boolean", "DataStore", "Boolean", "DataStore",
    "Boolean", "DataStore", Integer, Integer, Integer, Integer, Integer}];
denseReshardPlanFinish = receiving["DenseReshardPlanFinish", {Integer}];
denseHealth = receiving["DenseHealth", {}];
denseSearch = receivingRows["DenseSearch", {}, {Integer, Integer, Integer, "UTF8String"}];
denseMaxsim = receivingRows["DenseMaxsim", {}, {Integer, Integer, Integer, Integer}];
denseCount = receiving["DenseCount", {LibraryDataType[NumericArray]}];
denseGet = receiving["DenseGet", {LibraryDataType[NumericArray], Integer}];
denseSave = receiving["DenseSave", {"UTF8String"}];
denseSaveBuffer = receiving["DenseSaveBuffer", {Integer}];
denseDescribe = receiving["DenseDescribe", {}];
denseMultiply = receivingRows["DenseMultiply", {}, {Integer, Integer, Integer}];
denseValues = receivingRows["DenseValues", {passed}, {}];
denseAttend = receivingRows["DenseAttend", {}, {passed, passed, "UTF8String", Integer, Integer, Integer, Integer, Real}];

sequenceInitRam = verb["SequenceInitRam", {Integer, "UTF8String", weights, "DataStore", Integer, Integer, "DataStore"}];
sequenceInitVram = verb["SequenceInitVram", {Integer, "UTF8String", weights, "DataStore", Integer, Integer, "DataStore"}];
sequenceLoadRam = verb["SequenceLoadRam", {Integer, Integer, "UTF8String", "DataStore", Integer, weights}];
sequenceLoadVram = verb["SequenceLoadVram", {Integer, Integer, "UTF8String", "DataStore", Integer, weights}];
sequenceLoadRamBuffer =
    verb["SequenceLoadRamBuffer", {Integer, Integer, LibraryDataType[NumericArray], "DataStore", Integer, weights}];
sequenceLoadVramBuffer =
    verb["SequenceLoadVramBuffer", {Integer, Integer, LibraryDataType[NumericArray], "DataStore", Integer, weights}];
sequenceViewRam = verb["SequenceViewRam", {Integer, "UTF8String", weights}];
sequenceViewNvme = verb["SequenceViewNvme", {Integer, Integer, "UTF8String", "DataStore", weights}];
sequenceMetadataRead = verb["SequenceMetadataRead", {Integer, "UTF8String"}];
sequenceMetadataReadBuffer = verb["SequenceMetadataReadBuffer", {Integer, LibraryDataType[NumericArray]}];
sequenceReserve = receiving["SequenceReserve", {Integer, Integer}];
sequenceClear = receiving["SequenceClear", {}];
sequenceInsert = receiving["SequenceInsert", {passed, bytesTape, Integer}];
sequenceRefine = receiving["SequenceRefine", {Integer, Integer, Integer}];
sequenceRefinePlanFinish = receiving["SequenceRefinePlanFinish", {Integer}];
sequenceReshard = receiving["SequenceReshard", {reshardLists}];
sequenceReshardPlanFinish = receiving["SequenceReshardPlanFinish", {Integer}];
sequenceHealth = receiving["SequenceHealth", {}];
sequenceSearch = receiving["SequenceSearch", {bytesTape, Integer, Integer, Integer, "UTF8String"}];
sequenceCount = receiving["SequenceCount", {LibraryDataType[NumericArray]}];
sequenceGet = receiving["SequenceGet", {LibraryDataType[NumericArray], Integer}];
sequenceSave = receiving["SequenceSave", {"UTF8String"}];
sequenceSaveBuffer = receiving["SequenceSaveBuffer", {Integer}];
sequenceDescribe = receiving["SequenceDescribe", {}];

shapesInitRam = verb["ShapesInitRam", {Integer, "UTF8String", "UTF8String", Integer, Integer, Integer, "DataStore"}];
shapesInitVram = verb["ShapesInitVram", {Integer, "UTF8String", "UTF8String", Integer, Integer, Integer, "DataStore"}];
shapesLoadRam = verb["ShapesLoadRam", {Integer, Integer, "UTF8String", "DataStore", Integer}];
shapesLoadVram = verb["ShapesLoadVram", {Integer, Integer, "UTF8String", "DataStore", Integer}];
shapesLoadRamBuffer =
    verb["ShapesLoadRamBuffer", {Integer, Integer, LibraryDataType[NumericArray], "DataStore", Integer}];
shapesLoadVramBuffer =
    verb["ShapesLoadVramBuffer", {Integer, Integer, LibraryDataType[NumericArray], "DataStore", Integer}];
shapesViewRam = verb["ShapesViewRam", {Integer, "UTF8String"}];
shapesViewNvme = verb["ShapesViewNvme", {Integer, Integer, "UTF8String", "DataStore"}];
shapesMetadataRead = verb["ShapesMetadataRead", {Integer, "UTF8String"}];
shapesMetadataReadBuffer = verb["ShapesMetadataReadBuffer", {Integer, LibraryDataType[NumericArray]}];
shapesReserve = receiving["ShapesReserve", {Integer, Integer}];
shapesClear = receiving["ShapesClear", {}];
shapesInsert = receiving["ShapesInsert", {passed, pointsTape, Integer}];
shapesRefine = receiving["ShapesRefine", {Integer, Integer, Integer}];
shapesRefinePlanFinish = receiving["ShapesRefinePlanFinish", {Integer}];
shapesReshard = receiving["ShapesReshard", {reshardLists}];
shapesReshardPlanFinish = receiving["ShapesReshardPlanFinish", {Integer}];
shapesHealth = receiving["ShapesHealth", {}];
shapesSearch = receiving["ShapesSearch", {pointsTape, Integer, Integer, Integer, "UTF8String"}];
shapesCount = receiving["ShapesCount", {LibraryDataType[NumericArray]}];
shapesGet = receiving["ShapesGet", {LibraryDataType[NumericArray], Integer}];
shapesSave = receiving["ShapesSave", {"UTF8String"}];
shapesSaveBuffer = receiving["ShapesSaveBuffer", {Integer}];
shapesDescribe = receiving["ShapesDescribe", {}];

geospatialInitRam = verb["GeospatialInitRam", {Integer, Integer, Integer, "DataStore"}];
geospatialInitVram = verb["GeospatialInitVram", {Integer, Integer, Integer, "DataStore"}];
geospatialLoadRam = verb["GeospatialLoadRam", {Integer, Integer, "UTF8String", "DataStore", Integer}];
geospatialLoadVram = verb["GeospatialLoadVram", {Integer, Integer, "UTF8String", "DataStore", Integer}];
geospatialLoadRamBuffer =
    verb["GeospatialLoadRamBuffer", {Integer, Integer, LibraryDataType[NumericArray], "DataStore", Integer}];
geospatialLoadVramBuffer =
    verb["GeospatialLoadVramBuffer", {Integer, Integer, LibraryDataType[NumericArray], "DataStore", Integer}];
geospatialViewRam = verb["GeospatialViewRam", {Integer, "UTF8String"}];
geospatialViewNvme = verb["GeospatialViewNvme", {Integer, Integer, "UTF8String", "DataStore"}];
geospatialMetadataRead = verb["GeospatialMetadataRead", {Integer, "UTF8String"}];
geospatialMetadataReadBuffer = verb["GeospatialMetadataReadBuffer", {Integer, LibraryDataType[NumericArray]}];
geospatialReserve = receiving["GeospatialReserve", {Integer}];
geospatialClear = receiving["GeospatialClear", {}];
geospatialInsert = receiving["GeospatialInsert", {passed, observed, Integer}];
geospatialRefine = receiving["GeospatialRefine", {Integer, Integer, Integer}];
geospatialRefinePlanFinish = receiving["GeospatialRefinePlanFinish", {Integer}];
geospatialReshard = receiving["GeospatialReshard", {reshardLists}];
geospatialReshardPlanFinish = receiving["GeospatialReshardPlanFinish", {Integer}];
geospatialHealth = receiving["GeospatialHealth", {}];
geospatialSearch = receiving["GeospatialSearch",
    {observed, Integer, Real, Real, Integer, Integer, Integer, "UTF8String"}];
geospatialWithin = receiving["GeospatialWithin", {observed, Integer, Real, Real, Integer, "UTF8String"}];
geospatialCount = receiving["GeospatialCount", {LibraryDataType[NumericArray]}];
geospatialGet = receiving["GeospatialGet", {LibraryDataType[NumericArray], Integer}];
geospatialSave = receiving["GeospatialSave", {"UTF8String"}];
geospatialSaveBuffer = receiving["GeospatialSaveBuffer", {Integer}];
geospatialDescribe = receiving["GeospatialDescribe", {}];

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

(* A tape lays its entries end to end beside one more offset than entries, as Apache Arrow does. *)
offsets[lengths_List] := Developer`ToPackedArray[Prepend[Accumulate[lengths], 0]]
sequenceBytes[text_String] := ToCharacterCode[text, "UTF-8"]
sequenceBytes[bytes_ByteArray] := Normal[bytes]
sequenceBytes[bytes : {___Integer}] := bytes
(* No NumericArray or packed matrix holds zero cells, so a tape of empty entries passes one cell its offsets never
   reach. *)
sequencesTape[sequences_List] := With[{bytes = sequenceBytes /@ sequences},
    {NumericArray[Replace[Join @@ bytes, {} -> {0}], "UnsignedInteger8"], offsets[Length /@ bytes]}]
shapesTape[shapes_List] := {Developer`ToPackedArray[N[Replace[Join @@ shapes, {} -> {{0, 0, 0}}]]],
    offsets[Length /@ shapes]}
(* A tensor holds no NaN, so a field no device recorded crosses as zero beside a mask marking it unrecorded. *)
observations[fields_List] := {Developer`ToPackedArray[N[Replace[fields, field_ /; !NumericQ[field] -> 0, {2}]]],
    Developer`ToPackedArray[Boole[Map[NumericQ, fields, {2}]]]}
(* A weightless kind still fills the matrix slot, which the library reads only when weights are named. *)
weighting[None, gap_Integer] := Sequence[False, NumericArray[{0}, "Integer8"], gap]
weighting[matrix_, gap_Integer] := Sequence[True, NumericArray[matrix, "Integer8"], gap]

(* Outputs arrive under their C names, a nested audit as its own store; found matches split by each query's count,
   tapes by each entry's length, observations by their recorded mask, and a lone output stands alone. *)
answer[result_Developer`DataStore] := answer[Replace[result, fields_Developer`DataStore :> Association @@ fields, {0, 2}]]
answer[LibraryFunctionError["LIBRARY_MEMORY_ERROR", _]] :=
    Failure["out-of-memory", <|"MessageTemplate" -> "the kernel could not allocate the answer"|>]
answer[<|"Status" -> status_String, "Message" -> message_String|>] := Failure[status, <|"MessageTemplate" -> message|>]
answer[fields_Association] := Module[{result = fields},
    If[KeyExistsQ[result, "Keys"], result["Keys"] = keysFromBytes[result["Keys"]]];
    If[KeyExistsQ[result, "Counts"],
        Scan[(result[#] = TakeList[result[#], fields["Counts"]]) &, {"Keys", "Scores"}];
        KeyDropFrom[result, "Counts"]];
    If[KeyExistsQ[result, "Sequences"],
        result["Sequences"] = ByteArray /@ TakeList[Normal[result["Sequences"]], fields["Lengths"]]];
    If[KeyExistsQ[result, "Points"] && KeyExistsQ[result, "Lengths"],
        result["Points"] = TakeList[result["Points"], fields["Lengths"]]];
    KeyDropFrom[result, "Lengths"];
    If[KeyExistsQ[result, "Recorded"],
        result["Observations"] = MapThread[If[#2 == 1, #1, Missing["NotRecorded"]] &,
            {result["Observations"], fields["Recorded"]}, 2];
        KeyDropFrom[result, "Recorded"]];
    Switch[Length[result], 0, Null, 1, First[result], _, result]]
(* The kernel leaves a call whose arguments miss the declared types unevaluated, and any other refusal lands here. *)
answer[_] := Failure["invalid-argument", <|"MessageTemplate" -> "the arguments do not match the verb's types"|>]

indexQ[_DenseIndex | _SequenceIndex | _ShapesIndex | _GeospatialIndex] := True
indexQ[_] := False

(* The kernel frees a managed expression once nothing references it, so a call holds every handle it names
   until it returns, and only then do they cross as IDs, a list of indexes as a store of them. A handle the
   library still reads after that is the library's to wait for, as each free waits out the calls naming it. *)
call[function_, arguments_List] := Module[{held = arguments},
    answer[function @@ Replace[held, {
        (DenseIndex | SequenceIndex | ShapesIndex | GeospatialIndex | Executor | Arena | Plan | DeferredCall)[
            id_Integer] :> id,
        None -> 0, indexes : {__?indexQ} :> store[First /@ indexes]}, {1}]]]

(* A constructor fills a fresh managed expression, which a failure leaves for the kernel to release. *)
construct[head_Symbol, function_, arguments_List] :=
    With[{made = CreateManagedLibraryExpression[SymbolName[head], head]},
        Replace[call[function, Prepend[arguments, made]], Null -> made]]

(* The receiver chooses where the verb runs and how it completes; the verb and its arguments stay the same. *)
denseBlocking = _DenseIndex | DenseIndexOn[_DenseIndex, _Executor];
dense = denseBlocking | DenseIndexDeferred[_DenseIndex, _Executor];
sequence = _SequenceIndex | SequenceIndexOn[_SequenceIndex, _Executor] |
    SequenceIndexDeferred[_SequenceIndex, _Executor];
shapes = _ShapesIndex | ShapesIndexOn[_ShapesIndex, _Executor] | ShapesIndexDeferred[_ShapesIndex, _Executor];
geospatial = _GeospatialIndex | GeospatialIndexOn[_GeospatialIndex, _Executor] |
    GeospatialIndexDeferred[_GeospatialIndex, _Executor];

submit[index_?indexQ, function_, arguments_List] := call[function, Join[{index, None, None}, arguments]]
submit[(DenseIndexOn | SequenceIndexOn | ShapesIndexOn | GeospatialIndexOn)[index_, executor_], function_,
    arguments_List] := call[function, Join[{index, executor, None}, arguments]]
(* A deferred call parks in a fresh DeferredCall, which alone polls it and lands its outputs. *)
submit[(DenseIndexDeferred | SequenceIndexDeferred | ShapesIndexDeferred | GeospatialIndexDeferred)[index_,
    executor_], function_, arguments_List] :=
    With[{deferred = CreateManagedLibraryExpression["DeferredCall", DeferredCall]},
        Replace[call[function, Join[{index, executor, deferred}, arguments]], Null -> deferred]]

submitRows[receiver_, functions_Association, before_List, vectors_, after_List] :=
    submit[receiver, functions[Head[vectors]], Join[before, {rows[vectors], Head[vectors] === NumericArray}, after]]

DeferredStatus[deferred_DeferredCall] := deferredStatus[First[deferred]]
DeferredWait[deferred_DeferredCall] := call[deferredWait, {deferred}]
PlanInit[] := CreateManagedLibraryExpression["Plan", Plan]
PlanProgress[plan_Plan] := call[planProgress, {plan}]

Options[ExecutorForRam] = {"MemoryDomains" -> {}, "Threads" -> 0};
ExecutorForRam[OptionsPattern[]] := construct[Executor, executorForRam,
    {store[OptionValue["MemoryDomains"]], OptionValue["Threads"]}]
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
    description_Association :> Join[description, Replace[description[[{"Device", "VendorOrdinal"}]], -1 -> None, {1}]]]

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
Options[DenseViewNvme] = {"Executor" -> None, "Paths" -> {}, "KeyTableDomains" -> None};
DenseViewNvme[path_String, OptionsPattern[]] := construct[DenseIndex, denseViewNvme,
    {OptionValue["Executor"], path, store[OptionValue["Paths"]], ListQ[OptionValue["KeyTableDomains"]],
        store[Replace[OptionValue["KeyTableDomains"], None -> {}]]}]
Options[DenseLoadNvme] = Options[DenseViewNvme];
DenseLoadNvme[path_String, OptionsPattern[]] := construct[DenseIndex, denseLoadNvme,
    {OptionValue["Executor"], path, store[OptionValue["Paths"]], ListQ[OptionValue["KeyTableDomains"]],
        store[Replace[OptionValue["KeyTableDomains"], None -> {}]]}]
Options[DenseMetadataRead] = {"Executor" -> None};
DenseMetadataRead[path_String, OptionsPattern[]] := call[denseMetadataRead, {OptionValue["Executor"], path}]
Options[DenseMetadataReadBuffer] = {"Executor" -> None};
DenseMetadataReadBuffer[bytes : (_ByteArray | _NumericArray), OptionsPattern[]] :=
    call[denseMetadataReadBuffer, {OptionValue["Executor"], monolith[bytes]}]

DenseReserve[receiver : dense, capacity_Integer] := submit[receiver, denseReserve, {capacity}]
DenseClear[receiver : dense] := submit[receiver, denseClear, {}]
Options[DenseInsert] = {"ExpansionFactor" -> 0};
(* No NumericArray holds zero keys, and inserting none is the C library's no-op too. *)
DenseInsert[receiver : dense, {}, {}, OptionsPattern[]] := Null
DenseInsert[receiver : dense, keys : {__?keyQ}, vectors : (_List | _NumericArray), OptionsPattern[]] :=
    submitRows[receiver, denseInsert, {keyBytes[keys]}, vectors, {OptionValue["ExpansionFactor"]}]
Options[DenseRefine] = {"ExpansionFactor" -> 0, "MemoryLimit" -> 0, "Plan" -> None};
DenseRefine[receiver : dense, OptionsPattern[]] := submit[receiver, denseRefine,
    {OptionValue["ExpansionFactor"], OptionValue["MemoryLimit"], OptionValue["Plan"]}]
DenseRefinePlanFinish[receiver : dense, plan_Plan] := submit[receiver, denseRefinePlanFinish, {plan}]
Options[DenseReshard] = {"MemoryDomains" -> None, "Devices" -> None, "Paths" -> None, "ShardCapacity" -> 0,
    "ExpansionFactor" -> 0, "MemoryLimit" -> 0, "Seed" -> 0, "Plan" -> None};
(* An empty list crosses as an empty store, so the C library answers it as it answers any malformed list. *)
DenseReshard[receiver : dense, sources : {___DenseIndex}, targets : {___DenseIndex}, OptionsPattern[]] :=
    submit[receiver, denseReshard, {Replace[sources, {} -> store[{}]], Replace[targets, {} -> store[{}]],
        ListQ[OptionValue["MemoryDomains"]], store[Replace[OptionValue["MemoryDomains"], None -> {}]],
        ListQ[OptionValue["Devices"]], store[Replace[OptionValue["Devices"], None -> {}]],
        ListQ[OptionValue["Paths"]], store[Replace[OptionValue["Paths"], None -> {}]],
        OptionValue["ShardCapacity"], OptionValue["ExpansionFactor"], OptionValue["MemoryLimit"],
        OptionValue["Seed"], OptionValue["Plan"]}]
DenseReshardPlanFinish[receiver : dense, plan_Plan] := submit[receiver, denseReshardPlanFinish, {plan}]
DenseHealth[receiver : dense] := submit[receiver, denseHealth, {}]
Options[DenseSearch] = {"ExpansionFactor" -> 0, "Traversal" -> "walk"};
(* No tensor holds zero rows, so a call with no queries answers here, as C would, with nothing found. *)
DenseSearch[receiver : dense, {}, wanted_Integer, arena_Arena, OptionsPattern[]] := <|"Keys" -> {}, "Scores" -> {}|>
DenseSearch[receiver : dense, queries : ({__List} | _NumericArray), wanted_Integer, arena_Arena, OptionsPattern[]] :=
    submitRows[receiver, denseSearch, {}, queries,
        {wanted, arena, OptionValue["ExpansionFactor"], OptionValue["Traversal"]}]
Options[DenseMaxsim] = {"ExpansionFactor" -> 0};
DenseMaxsim[receiver : dense, {}, rowsPerQuery_Integer, wanted_Integer, arena_Arena, OptionsPattern[]] :=
    <|"Offsets" -> {0}, "Keys" -> {}, "Scores" -> {}, "Required" -> 0|>
DenseMaxsim[receiver : dense, queries : ({__List} | _NumericArray), rowsPerQuery_Integer, wanted_Integer,
    arena_Arena, OptionsPattern[]] :=
    submitRows[receiver, denseMaxsim, {}, queries, {rowsPerQuery, wanted, arena, OptionValue["ExpansionFactor"]}]
DenseCount[receiver : dense, key_?keyQ] := submit[receiver, denseCount, {keyBytes[key]}]
DenseGet[receiver : dense, key_?keyQ, arena_Arena] := submit[receiver, denseGet, {keyBytes[key], arena}]
DenseSave[receiver : dense, path_String] := submit[receiver, denseSave, {path}]
DenseSaveBuffer[receiver : dense, arena_Arena] := submit[receiver, denseSaveBuffer, {arena}]
DenseDescribe[receiver : dense] := submit[receiver, denseDescribe, {}]
Options[DenseMultiply] = {"Wanted" -> 0, "ExpansionFactor" -> 0};
DenseMultiply[receiver : dense, left : (_List | _NumericArray), arena_Arena, OptionsPattern[]] :=
    submitRows[receiver, denseMultiply, {}, left, {arena, OptionValue["Wanted"], OptionValue["ExpansionFactor"]}]
(* The layout reads the products the call answers, so only a blocking receiver answers them in time. *)
DenseMultiply[receiver : denseBlocking, left : (_List | _NumericArray), columnKeys : {___?keyQ}, arena_Arena,
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
(* The library prepares no value rows, so the binding fills them in one blocking call. *)
DenseValues[receiver : denseBlocking, keys : {__?keyQ}, values : (_List | _NumericArray)] :=
    Replace[submitRows[receiver, denseValues, {keyBytes[keys]}, values, {}], fields_Association :> DenseValues[fields]]
Options[DenseAttend] = {"HeadCount" -> 1, "ExpansionFactor" -> 0, "ScoreScale" -> 0.};
DenseAttend[receiver : dense, {}, values_DenseValues, arena_Arena, OptionsPattern[]] := <|"Outputs" -> {}, "Carry" -> {}|>
DenseAttend[receiver : dense, queries : ({__List} | _NumericArray), DenseValues[values_Association], arena_Arena,
    OptionsPattern[]] := submitRows[receiver, denseAttend, {}, queries,
        {values["ValueKeys"], values["ValueRows"], values["Quantization"], values["Dimensions"], arena,
            OptionValue["HeadCount"], OptionValue["ExpansionFactor"], N[OptionValue["ScoreScale"]]}]

Options[SequenceInitRam] = {"SubstitutionMatrix" -> None, "Gap" -> 0, "WindowWidths" -> {}, "Connectivity" -> 0,
    "ValuesPerKey" -> 0, "MemoryDomains" -> {}};
SequenceInitRam[metric_String : "levenshtein", OptionsPattern[]] := construct[SequenceIndex, sequenceInitRam,
    {metric, weighting[OptionValue["SubstitutionMatrix"], OptionValue["Gap"]], store[OptionValue["WindowWidths"]],
        OptionValue["Connectivity"], OptionValue["ValuesPerKey"], store[OptionValue["MemoryDomains"]]}]
Options[SequenceInitVram] = {"SubstitutionMatrix" -> None, "Gap" -> 0, "WindowWidths" -> {}, "Connectivity" -> 0,
    "ValuesPerKey" -> 0, "Devices" -> {}};
SequenceInitVram[metric_String : "levenshtein", OptionsPattern[]] := construct[SequenceIndex, sequenceInitVram,
    {metric, weighting[OptionValue["SubstitutionMatrix"], OptionValue["Gap"]], store[OptionValue["WindowWidths"]],
        OptionValue["Connectivity"], OptionValue["ValuesPerKey"], store[OptionValue["Devices"]]}]
Options[SequenceLoadRam] = {"Executor" -> None, "MemoryDomains" -> {}, "ValuesPerKey" -> 0,
    "SubstitutionMatrix" -> None, "Gap" -> 0};
SequenceLoadRam[path_String, OptionsPattern[]] := construct[SequenceIndex, sequenceLoadRam,
    {OptionValue["Executor"], path, store[OptionValue["MemoryDomains"]], OptionValue["ValuesPerKey"],
        weighting[OptionValue["SubstitutionMatrix"], OptionValue["Gap"]]}]
Options[SequenceLoadVram] = {"Executor" -> None, "Devices" -> {}, "ValuesPerKey" -> 0,
    "SubstitutionMatrix" -> None, "Gap" -> 0};
SequenceLoadVram[path_String, OptionsPattern[]] := construct[SequenceIndex, sequenceLoadVram,
    {OptionValue["Executor"], path, store[OptionValue["Devices"]], OptionValue["ValuesPerKey"],
        weighting[OptionValue["SubstitutionMatrix"], OptionValue["Gap"]]}]
Options[SequenceLoadRamBuffer] = Options[SequenceLoadRam];
SequenceLoadRamBuffer[bytes : (_ByteArray | _NumericArray), OptionsPattern[]] :=
    construct[SequenceIndex, sequenceLoadRamBuffer, {OptionValue["Executor"], monolith[bytes],
        store[OptionValue["MemoryDomains"]], OptionValue["ValuesPerKey"],
        weighting[OptionValue["SubstitutionMatrix"], OptionValue["Gap"]]}]
Options[SequenceLoadVramBuffer] = Options[SequenceLoadVram];
SequenceLoadVramBuffer[bytes : (_ByteArray | _NumericArray), OptionsPattern[]] :=
    construct[SequenceIndex, sequenceLoadVramBuffer, {OptionValue["Executor"], monolith[bytes],
        store[OptionValue["Devices"]], OptionValue["ValuesPerKey"],
        weighting[OptionValue["SubstitutionMatrix"], OptionValue["Gap"]]}]
Options[SequenceViewRam] = {"SubstitutionMatrix" -> None, "Gap" -> 0};
SequenceViewRam[path_String, OptionsPattern[]] := construct[SequenceIndex, sequenceViewRam,
    {path, weighting[OptionValue["SubstitutionMatrix"], OptionValue["Gap"]]}]
Options[SequenceViewNvme] = {"Executor" -> None, "Paths" -> {}, "SubstitutionMatrix" -> None, "Gap" -> 0};
SequenceViewNvme[path_String, OptionsPattern[]] := construct[SequenceIndex, sequenceViewNvme,
    {OptionValue["Executor"], path, store[OptionValue["Paths"]],
        weighting[OptionValue["SubstitutionMatrix"], OptionValue["Gap"]]}]
Options[SequenceMetadataRead] = {"Executor" -> None};
SequenceMetadataRead[path_String, OptionsPattern[]] := call[sequenceMetadataRead, {OptionValue["Executor"], path}]
Options[SequenceMetadataReadBuffer] = {"Executor" -> None};
SequenceMetadataReadBuffer[bytes : (_ByteArray | _NumericArray), OptionsPattern[]] :=
    call[sequenceMetadataReadBuffer, {OptionValue["Executor"], monolith[bytes]}]

SequenceReserve[receiver : sequence, entries_Integer, payloadBytes_Integer : 0] :=
    submit[receiver, sequenceReserve, {entries, payloadBytes}]
SequenceClear[receiver : sequence] := submit[receiver, sequenceClear, {}]
Options[SequenceInsert] = {"ExpansionFactor" -> 0};
SequenceInsert[receiver : sequence, {}, {}, OptionsPattern[]] := Null
SequenceInsert[receiver : sequence, keys : {__?keyQ}, sequences_List, OptionsPattern[]] :=
    submit[receiver, sequenceInsert, Join[{keyBytes[keys]}, sequencesTape[sequences], {OptionValue["ExpansionFactor"]}]]
Options[SequenceRefine] = Options[DenseRefine];
SequenceRefine[receiver : sequence, OptionsPattern[]] := submit[receiver, sequenceRefine,
    {OptionValue["ExpansionFactor"], OptionValue["MemoryLimit"], OptionValue["Plan"]}]
SequenceRefinePlanFinish[receiver : sequence, plan_Plan] := submit[receiver, sequenceRefinePlanFinish, {plan}]
Options[SequenceReshard] = {"ExpansionFactor" -> 0, "MemoryLimit" -> 0, "Seed" -> 0, "Plan" -> None};
SequenceReshard[receiver : sequence, sources : {___SequenceIndex}, targets : {___SequenceIndex}, OptionsPattern[]] :=
    submit[receiver, sequenceReshard, {Replace[sources, {} -> store[{}]], Replace[targets, {} -> store[{}]],
        OptionValue["ExpansionFactor"], OptionValue["MemoryLimit"], OptionValue["Seed"], OptionValue["Plan"]}]
SequenceReshardPlanFinish[receiver : sequence, plan_Plan] := submit[receiver, sequenceReshardPlanFinish, {plan}]
SequenceHealth[receiver : sequence] := submit[receiver, sequenceHealth, {}]
Options[SequenceSearch] = Options[DenseSearch];
SequenceSearch[receiver : sequence, {}, wanted_Integer, arena_Arena, OptionsPattern[]] := <|"Keys" -> {}, "Scores" -> {}|>
SequenceSearch[receiver : sequence, queries_List, wanted_Integer, arena_Arena, OptionsPattern[]] :=
    submit[receiver, sequenceSearch, Join[sequencesTape[queries],
        {wanted, arena, OptionValue["ExpansionFactor"], OptionValue["Traversal"]}]]
SequenceCount[receiver : sequence, key_?keyQ] := submit[receiver, sequenceCount, {keyBytes[key]}]
SequenceGet[receiver : sequence, key_?keyQ, arena_Arena] := submit[receiver, sequenceGet, {keyBytes[key], arena}]
SequenceSave[receiver : sequence, path_String] := submit[receiver, sequenceSave, {path}]
SequenceSaveBuffer[receiver : sequence, arena_Arena] := submit[receiver, sequenceSaveBuffer, {arena}]
SequenceDescribe[receiver : sequence] := submit[receiver, sequenceDescribe, {}]

Options[ShapesInitRam] = {"Points" -> 0, "Connectivity" -> 0, "ValuesPerKey" -> 0, "MemoryDomains" -> {}};
ShapesInitRam[metric_String : "chamfer", quantization_String : "f32", OptionsPattern[]] :=
    construct[ShapesIndex, shapesInitRam, {metric, quantization, OptionValue["Points"], OptionValue["Connectivity"],
        OptionValue["ValuesPerKey"], store[OptionValue["MemoryDomains"]]}]
Options[ShapesInitVram] = {"Points" -> 0, "Connectivity" -> 0, "ValuesPerKey" -> 0, "Devices" -> {}};
ShapesInitVram[metric_String : "chamfer", quantization_String : "f32", OptionsPattern[]] :=
    construct[ShapesIndex, shapesInitVram, {metric, quantization, OptionValue["Points"], OptionValue["Connectivity"],
        OptionValue["ValuesPerKey"], store[OptionValue["Devices"]]}]
Options[ShapesLoadRam] = Options[DenseLoadRam];
ShapesLoadRam[path_String, OptionsPattern[]] := construct[ShapesIndex, shapesLoadRam,
    {OptionValue["Executor"], path, store[OptionValue["MemoryDomains"]], OptionValue["ValuesPerKey"]}]
Options[ShapesLoadVram] = Options[DenseLoadVram];
ShapesLoadVram[path_String, OptionsPattern[]] := construct[ShapesIndex, shapesLoadVram,
    {OptionValue["Executor"], path, store[OptionValue["Devices"]], OptionValue["ValuesPerKey"]}]
Options[ShapesLoadRamBuffer] = Options[ShapesLoadRam];
ShapesLoadRamBuffer[bytes : (_ByteArray | _NumericArray), OptionsPattern[]] :=
    construct[ShapesIndex, shapesLoadRamBuffer, {OptionValue["Executor"], monolith[bytes],
        store[OptionValue["MemoryDomains"]], OptionValue["ValuesPerKey"]}]
Options[ShapesLoadVramBuffer] = Options[ShapesLoadVram];
ShapesLoadVramBuffer[bytes : (_ByteArray | _NumericArray), OptionsPattern[]] :=
    construct[ShapesIndex, shapesLoadVramBuffer, {OptionValue["Executor"], monolith[bytes],
        store[OptionValue["Devices"]], OptionValue["ValuesPerKey"]}]
ShapesViewRam[path_String] := construct[ShapesIndex, shapesViewRam, {path}]
Options[ShapesViewNvme] = {"Executor" -> None, "Paths" -> {}};
ShapesViewNvme[path_String, OptionsPattern[]] := construct[ShapesIndex, shapesViewNvme,
    {OptionValue["Executor"], path, store[OptionValue["Paths"]]}]
Options[ShapesMetadataRead] = {"Executor" -> None};
ShapesMetadataRead[path_String, OptionsPattern[]] := call[shapesMetadataRead, {OptionValue["Executor"], path}]
Options[ShapesMetadataReadBuffer] = {"Executor" -> None};
ShapesMetadataReadBuffer[bytes : (_ByteArray | _NumericArray), OptionsPattern[]] :=
    call[shapesMetadataReadBuffer, {OptionValue["Executor"], monolith[bytes]}]

ShapesReserve[receiver : shapes, entries_Integer, points_Integer : 0] :=
    submit[receiver, shapesReserve, {entries, points}]
ShapesClear[receiver : shapes] := submit[receiver, shapesClear, {}]
Options[ShapesInsert] = {"ExpansionFactor" -> 0};
ShapesInsert[receiver : shapes, {}, {}, OptionsPattern[]] := Null
ShapesInsert[receiver : shapes, keys : {__?keyQ}, entries_List, OptionsPattern[]] :=
    submit[receiver, shapesInsert, Join[{keyBytes[keys]}, shapesTape[entries], {OptionValue["ExpansionFactor"]}]]
Options[ShapesRefine] = Options[DenseRefine];
ShapesRefine[receiver : shapes, OptionsPattern[]] := submit[receiver, shapesRefine,
    {OptionValue["ExpansionFactor"], OptionValue["MemoryLimit"], OptionValue["Plan"]}]
ShapesRefinePlanFinish[receiver : shapes, plan_Plan] := submit[receiver, shapesRefinePlanFinish, {plan}]
Options[ShapesReshard] = Options[SequenceReshard];
ShapesReshard[receiver : shapes, sources : {___ShapesIndex}, targets : {___ShapesIndex}, OptionsPattern[]] :=
    submit[receiver, shapesReshard, {Replace[sources, {} -> store[{}]], Replace[targets, {} -> store[{}]],
        OptionValue["ExpansionFactor"], OptionValue["MemoryLimit"], OptionValue["Seed"], OptionValue["Plan"]}]
ShapesReshardPlanFinish[receiver : shapes, plan_Plan] := submit[receiver, shapesReshardPlanFinish, {plan}]
ShapesHealth[receiver : shapes] := submit[receiver, shapesHealth, {}]
Options[ShapesSearch] = Options[DenseSearch];
ShapesSearch[receiver : shapes, {}, wanted_Integer, arena_Arena, OptionsPattern[]] := <|"Keys" -> {}, "Scores" -> {}|>
ShapesSearch[receiver : shapes, queries_List, wanted_Integer, arena_Arena, OptionsPattern[]] :=
    submit[receiver, shapesSearch, Join[shapesTape[queries],
        {wanted, arena, OptionValue["ExpansionFactor"], OptionValue["Traversal"]}]]
ShapesCount[receiver : shapes, key_?keyQ] := submit[receiver, shapesCount, {keyBytes[key]}]
ShapesGet[receiver : shapes, key_?keyQ, arena_Arena] := submit[receiver, shapesGet, {keyBytes[key], arena}]
ShapesSave[receiver : shapes, path_String] := submit[receiver, shapesSave, {path}]
ShapesSaveBuffer[receiver : shapes, arena_Arena] := submit[receiver, shapesSaveBuffer, {arena}]
ShapesDescribe[receiver : shapes] := submit[receiver, shapesDescribe, {}]

Options[GeospatialInitRam] = {"Connectivity" -> 0, "ValuesPerKey" -> 0, "MemoryDomains" -> {}};
GeospatialInitRam[OptionsPattern[]] := construct[GeospatialIndex, geospatialInitRam,
    {OptionValue["Connectivity"], OptionValue["ValuesPerKey"], store[OptionValue["MemoryDomains"]]}]
Options[GeospatialInitVram] = {"Connectivity" -> 0, "ValuesPerKey" -> 0, "Devices" -> {}};
GeospatialInitVram[OptionsPattern[]] := construct[GeospatialIndex, geospatialInitVram,
    {OptionValue["Connectivity"], OptionValue["ValuesPerKey"], store[OptionValue["Devices"]]}]
Options[GeospatialLoadRam] = Options[DenseLoadRam];
GeospatialLoadRam[path_String, OptionsPattern[]] := construct[GeospatialIndex, geospatialLoadRam,
    {OptionValue["Executor"], path, store[OptionValue["MemoryDomains"]], OptionValue["ValuesPerKey"]}]
Options[GeospatialLoadVram] = Options[DenseLoadVram];
GeospatialLoadVram[path_String, OptionsPattern[]] := construct[GeospatialIndex, geospatialLoadVram,
    {OptionValue["Executor"], path, store[OptionValue["Devices"]], OptionValue["ValuesPerKey"]}]
Options[GeospatialLoadRamBuffer] = Options[GeospatialLoadRam];
GeospatialLoadRamBuffer[bytes : (_ByteArray | _NumericArray), OptionsPattern[]] :=
    construct[GeospatialIndex, geospatialLoadRamBuffer, {OptionValue["Executor"], monolith[bytes],
        store[OptionValue["MemoryDomains"]], OptionValue["ValuesPerKey"]}]
Options[GeospatialLoadVramBuffer] = Options[GeospatialLoadVram];
GeospatialLoadVramBuffer[bytes : (_ByteArray | _NumericArray), OptionsPattern[]] :=
    construct[GeospatialIndex, geospatialLoadVramBuffer, {OptionValue["Executor"], monolith[bytes],
        store[OptionValue["Devices"]], OptionValue["ValuesPerKey"]}]
GeospatialViewRam[path_String] := construct[GeospatialIndex, geospatialViewRam, {path}]
Options[GeospatialViewNvme] = Options[ShapesViewNvme];
GeospatialViewNvme[path_String, OptionsPattern[]] := construct[GeospatialIndex, geospatialViewNvme,
    {OptionValue["Executor"], path, store[OptionValue["Paths"]]}]
Options[GeospatialMetadataRead] = {"Executor" -> None};
GeospatialMetadataRead[path_String, OptionsPattern[]] := call[geospatialMetadataRead, {OptionValue["Executor"], path}]
Options[GeospatialMetadataReadBuffer] = {"Executor" -> None};
GeospatialMetadataReadBuffer[bytes : (_ByteArray | _NumericArray), OptionsPattern[]] :=
    call[geospatialMetadataReadBuffer, {OptionValue["Executor"], monolith[bytes]}]

GeospatialReserve[receiver : geospatial, entries_Integer] := submit[receiver, geospatialReserve, {entries}]
GeospatialClear[receiver : geospatial] := submit[receiver, geospatialClear, {}]
Options[GeospatialInsert] = {"ExpansionFactor" -> 0};
GeospatialInsert[receiver : geospatial, {}, {}, OptionsPattern[]] := Null
GeospatialInsert[receiver : geospatial, keys : {__?keyQ}, entries : {__List}, OptionsPattern[]] :=
    submit[receiver, geospatialInsert, Join[{keyBytes[keys]}, observations[entries], {OptionValue["ExpansionFactor"]}]]
Options[GeospatialRefine] = Options[DenseRefine];
GeospatialRefine[receiver : geospatial, OptionsPattern[]] := submit[receiver, geospatialRefine,
    {OptionValue["ExpansionFactor"], OptionValue["MemoryLimit"], OptionValue["Plan"]}]
GeospatialRefinePlanFinish[receiver : geospatial, plan_Plan] := submit[receiver, geospatialRefinePlanFinish, {plan}]
Options[GeospatialReshard] = Options[SequenceReshard];
GeospatialReshard[receiver : geospatial, sources : {___GeospatialIndex}, targets : {___GeospatialIndex},
    OptionsPattern[]] :=
    submit[receiver, geospatialReshard, {Replace[sources, {} -> store[{}]], Replace[targets, {} -> store[{}]],
        OptionValue["ExpansionFactor"], OptionValue["MemoryLimit"], OptionValue["Seed"], OptionValue["Plan"]}]
GeospatialReshardPlanFinish[receiver : geospatial, plan_Plan] := submit[receiver, geospatialReshardPlanFinish, {plan}]
GeospatialHealth[receiver : geospatial] := submit[receiver, geospatialHealth, {}]
Options[GeospatialSearch] = {"Relation" -> 1, "Metres" -> 0., "Seconds" -> 0., "ExpansionFactor" -> 0,
    "Traversal" -> "walk"};
GeospatialSearch[receiver : geospatial, {}, wanted_Integer, arena_Arena, OptionsPattern[]] :=
    <|"Keys" -> {}, "Scores" -> {}|>
GeospatialSearch[receiver : geospatial, queries : {__List}, wanted_Integer, arena_Arena, OptionsPattern[]] :=
    submit[receiver, geospatialSearch, Join[observations[queries], {OptionValue["Relation"],
        N[OptionValue["Metres"]], N[OptionValue["Seconds"]], wanted, arena, OptionValue["ExpansionFactor"],
        OptionValue["Traversal"]}]]
Options[GeospatialWithin] = {"Relation" -> 1, "Seconds" -> 0., "Traversal" -> "walk"};
GeospatialWithin[receiver : geospatial, {}, metres_?NumericQ, arena_Arena, OptionsPattern[]] :=
    <|"Offsets" -> {0}, "Keys" -> {}, "Scores" -> {}|>
GeospatialWithin[receiver : geospatial, queries : {__List}, metres_?NumericQ, arena_Arena, OptionsPattern[]] :=
    submit[receiver, geospatialWithin, Join[observations[queries], {OptionValue["Relation"], N[metres],
        N[OptionValue["Seconds"]], arena, OptionValue["Traversal"]}]]
GeospatialCount[receiver : geospatial, key_?keyQ] := submit[receiver, geospatialCount, {keyBytes[key]}]
GeospatialGet[receiver : geospatial, key_?keyQ, arena_Arena] := submit[receiver, geospatialGet, {keyBytes[key], arena}]
GeospatialSave[receiver : geospatial, path_String] := submit[receiver, geospatialSave, {path}]
GeospatialSaveBuffer[receiver : geospatial, arena_Arena] := submit[receiver, geospatialSaveBuffer, {arena}]
GeospatialDescribe[receiver : geospatial] := submit[receiver, geospatialDescribe, {}]

VersionName[] := versionName[]
CapabilitiesCompiled[] := capabilitiesCompiled[]
CapabilitiesAvailable[] := capabilitiesAvailable[]

End[]

EndPackage[]
