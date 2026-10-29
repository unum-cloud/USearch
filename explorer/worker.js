/**
 *  @file explorer/worker.js
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The one place that touches the C ABI. It loads the Emscripten factory at `moduleUrl`, lays out every argument
 *      structure from the headers that module was built from, calls the verbs blocking, and copies their outputs out of
 *      the arena before the next call rewinds it.
 *
 *  Behind the factory may be the engine, the `src_stub` stand-in, or a forwarder to a server, so a call may block on a
 *  synchronous fetch. A NULL executor runs every call on this thread, and one arena serves them all in turn.
 *
 *  Nothing here is written per modality. A modality's entries travel as whatever its `insert` takes after `keys` -
 *  rows, a tape, or an array of records - and every other verb carries them in the member of that same type.
 */

/** @typedef {'unsigned' | 'signed' | 'float' | 'structure'} Kind */
/** @typedef {{name: string, type: string, pointers: number, length: number, offset: number}} Member */
/** @typedef {{kind: Kind, size: number, alignment: number, members?: Map<string, Member>}} Type */

/** @typedef {import('./modalities.js').Payload} Payload */
/** @typedef {import('./modalities.js').Rows} Rows */
/** @typedef {import('./modalities.js').Tape} Tape */
/** @typedef {import('./modalities.js').Records} Records */

/**
 *  @typedef {{
 *    HEAPU8: Uint8Array,
 *    _malloc: (size: number) => number,
 *    _free: (pointer: number) => void,
 *    stackAlloc: (size: number) => number,
 *    stackSave: () => number,
 *    stackRestore: (pointer: number) => void,
 *    UTF8ToString: (pointer: number) => string,
 *    FS: {mkdirTree: (path: string) => void, writeFile: (path: string, data: Uint8Array) => void},
 *    [symbol: string]: any
 *  }} UsearchModule
 */

/** @type {Record<string, [Kind, number]>} */
const primitives = {
  char: ["signed", 1],
  bool: ["unsigned", 1],
  int8_t: ["signed", 1],
  uint8_t: ["unsigned", 1],
  int16_t: ["signed", 2],
  uint16_t: ["unsigned", 2],
  int32_t: ["signed", 4],
  uint32_t: ["unsigned", 4],
  int64_t: ["signed", 8],
  uint64_t: ["unsigned", 8],
  size_t: ["unsigned", 4],
  float: ["float", 4],
  double: ["float", 8],
};

/** @type {Type} */
const pointer = { kind: "unsigned", size: 4, alignment: 4 };
const keyBytes = 16;
const cursorBatch = 1024;

/** @type {Map<string, Type>} */
const types = new Map(
  Object.entries(primitives).map(([name, [kind, size]]) => [name, { kind, size, alignment: size }]),
);
/** @type {Map<string, number>} */
const enumerators = new Map();

/** @type {UsearchModule} */
let library;
let modality = "";
let tapeColumns = 1;
/** The member `insert` carries entries in, whose type every other verb's entries share. @type {Member} */
let entries;
let arena = 0;
let outcome = 0;
let index = 0;
/** Heap copies the current request borrows, freed once it is answered. @type {number[]} */
let borrowed = [];

/**
 *  Reads every typedef of C `source` in order - enumerations, handles, function pointers, aliases, and structures laid
 *  out as wasm32 lays them out, where pointers and `size_t` take 4 bytes and 64-bit scalars align to 8.
 *  @param {string} source
 */
function readHeaders(source) {
  const text = source
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\/\/.*$/gm, "")
    .replace(/^\s*#.*$/gm, "");
  const keyword = /\btypedef\b/g;
  for (let match; (match = keyword.exec(text));) {
    let depth = 0;
    let end = keyword.lastIndex;
    for (; end < text.length && (text[end] !== ";" || depth > 0); end++)
      depth += text[end] === "{" ? 1 : text[end] === "}" ? -1 : 0;
    const statement = text.slice(keyword.lastIndex, end).trim();
    keyword.lastIndex = end;
    let parsed;
    if ((parsed = /^enum\b[^{]*\{([\s\S]*)\}\s*(\w+)$/.exec(statement))) {
      let next = 0;
      for (const item of parsed[1]
        .split(",")
        .map((part) => part.trim())
        .filter(Boolean)) {
        const [name, value] = item.split("=").map((part) => part.trim());
        if (value !== undefined) next = Number(value);
        if (Number.isNaN(next)) throw new Error(`Unreadable enumerator ${item}`);
        enumerators.set(name, next++);
      }
      types.set(parsed[2], { kind: "signed", size: 4, alignment: 4 });
    } else if ((parsed = /^struct\b[^{]*\{([\s\S]*)\}\s*(\w+)$/.exec(statement))) {
      types.set(parsed[2], structure(parsed[1]));
    } else if (
      (parsed = /\(\s*\*\s*(\w+)\s*\)\s*\(/.exec(statement) ?? /^struct\s+\w+\s*\*\s*(\w+)$/.exec(statement))
    ) {
      types.set(parsed[1], pointer);
    } else if ((parsed = /^(\w+)\s+(\w+)$/.exec(statement)) && types.has(parsed[1])) {
      types.set(parsed[2], /** @type {Type} */ (types.get(parsed[1])));
    }
  }
}

/** @param {string} body The declarations between a structure's braces. @returns {Type} */
function structure(body) {
  /** @type {Map<string, Member>} */
  const members = new Map();
  let offset = 0;
  let alignment = 1;
  for (const declaration of body.split(";")) {
    const text = declaration
      .replace(/\b(const|struct)\b/g, " ")
      .replace(/\s*\*\s*/g, "*")
      .replace(/\s+/g, " ")
      .trim();
    if (!text) continue;
    const parsed = /^(\w+)(\**) ?(\w+)(?: ?\[(\d+)\])?$/.exec(text);
    if (!parsed) throw new Error(`Unreadable member ${text}`);
    const [, type, stars, name, length = "1"] = parsed;
    const member = { name, type, pointers: stars.length, length: Number(length), offset: 0 };
    const placed = typeOf(member);
    member.offset = offset = Math.ceil(offset / placed.alignment) * placed.alignment;
    offset += placed.size * member.length;
    alignment = Math.max(alignment, placed.alignment);
    members.set(name, member);
  }
  return { kind: "structure", size: Math.ceil(offset / alignment) * alignment, alignment, members };
}

/** @param {Member} member */
function typeOf(member) {
  const type = member.pointers ? pointer : types.get(member.type);
  if (!type) throw new Error(`Unknown type ${member.type} of ${member.name}`);
  return type;
}

/** @param {string} name @returns {Type} */
function typeNamed(name) {
  const type = types.get(name);
  if (!type) throw new Error(`The headers declare no ${name}`);
  return type;
}

/** @param {string} name @param {string} member @returns {Member} */
function memberOf(name, member) {
  const found = typeNamed(name).members?.get(member);
  if (!found) throw new Error(`${name} has no member ${member}`);
  return found;
}

/** What a pointer member of `name` points to. @param {string} name @param {string} member */
function pointee(name, member) {
  const found = memberOf(name, member);
  return typeOf({ ...found, pointers: found.pointers - 1 });
}

const memory = () => new DataView(library.HEAPU8.buffer);

/** @param {number} address @param {Type} type @param {number} value */
function store(address, type, value) {
  const view = memory();
  if (type.kind === "float")
    type.size === 8 ? view.setFloat64(address, value, true) : view.setFloat32(address, value, true);
  else if (type.size === 8) view.setBigUint64(address, BigInt.asUintN(64, BigInt(value)), true);
  else if (type.size === 4) view.setInt32(address, value, true);
  else if (type.size === 2) view.setInt16(address, value, true);
  else view.setInt8(address, value);
}

/** @param {number} address @param {Type} type @returns {number} */
function load(address, type) {
  const view = memory();
  const signed = type.kind === "signed";
  if (type.kind === "float") return type.size === 8 ? view.getFloat64(address, true) : view.getFloat32(address, true);
  if (type.size === 8) return Number(signed ? view.getBigInt64(address, true) : view.getBigUint64(address, true));
  if (type.size === 4) return signed ? view.getInt32(address, true) : view.getUint32(address, true);
  if (type.size === 2) return signed ? view.getInt16(address, true) : view.getUint16(address, true);
  return signed ? view.getInt8(address) : view.getUint8(address);
}

/**
 *  Writes `values` by member name into the structure of `type` at `address`: nested objects fill nested structures,
 *  byte arrays land as they are, and strings name enumerators.
 *  @param {number} address @param {Type} type @param {Record<string, any>} values
 */
function write(address, type, values) {
  for (const [name, value] of Object.entries(values)) {
    if (value === undefined) continue;
    const member = type.members?.get(name);
    if (!member) throw new Error(`No member ${name} to write`);
    const at = address + member.offset;
    const placed = typeOf(member);
    if (value instanceof Uint8Array) library.HEAPU8.set(value, at);
    else if (placed.kind === "structure") write(at, placed, value);
    else store(at, placed, typeof value === "string" ? enumerator(value) : value);
  }
}

/**
 *  The structure of `type` at `address` by member name, nested structures nested and arrays copied out as bytes.
 *  @param {number} address @param {Type} type @returns {Record<string, any>}
 */
function read(address, type) {
  /** @type {Record<string, any>} */
  const values = {};
  for (const member of type.members?.values() ?? []) {
    const at = address + member.offset;
    const placed = typeOf(member);
    if (member.length > 1) values[member.name] = library.HEAPU8.slice(at, at + placed.size * member.length);
    else values[member.name] = placed.kind === "structure" ? read(at, placed) : load(at, placed);
  }
  return values;
}

/** @param {string} name */
function enumerator(name) {
  const value = enumerators.get(name);
  if (value === undefined) throw new Error(`The headers declare no ${name}`);
  return value;
}

/** @param {number} value */
function scalarName(value) {
  for (const [name, known] of enumerators) if (known === value && name.startsWith("usearch_scalar_")) return name;
  throw new Error(`No scalar numbered ${value}`);
}

/** Bytes per scalar of the plain kinds a payload comes out in. @param {string} name */
function scalarBytes(name) {
  const bits = /^usearch_scalar_(?:b?f|i|u)(\d+)_k$/.exec(name);
  if (!bits) throw new Error(`No byte width for ${name}`);
  return Number(bits[1]) / 8;
}

/** Zeroed room on the stack for one `type`. @param {Type} type */
function room(type) {
  const address = library.stackAlloc(type.size);
  library.HEAPU8.fill(0, address, address + type.size);
  return address;
}

/** A heap copy of `bytes` the current request borrows. @param {Uint8Array} bytes */
function place(bytes) {
  const address = library._malloc(Math.max(1, bytes.byteLength));
  library.HEAPU8.set(bytes, address);
  borrowed.push(address);
  return address;
}

/** @param {number} address @param {number} bytes */
function copy(address, bytes) {
  return library.HEAPU8.slice(address, address + bytes);
}

/** `count` keys at `stride` bytes apart, packed. @param {number} address @param {number} stride @param {number} count */
function copyKeys(address, stride, count) {
  const keys = new Uint8Array(count * keyBytes);
  for (let key = 0; key < count; key++)
    keys.set(library.HEAPU8.subarray(address + key * stride, address + key * stride + keyBytes), key * keyBytes);
  return keys;
}

/** `count` values of `size_t`. @param {number} address @param {number} count */
function copySizes(address, count) {
  return new Uint32Array(copy(address, count * 4).buffer);
}

/** @param {Uint8Array[]} parts */
function concatenated(parts) {
  const joined = new Uint8Array(parts.reduce((sum, part) => sum + part.byteLength, 0));
  let position = 0;
  for (const part of parts) {
    joined.set(part, position);
    position += part.byteLength;
  }
  return joined;
}

/** The verb `action` of this modality. @param {string} action */
const named = (action) => `usearch_${modality}_${action}`;

/** Calls `name` with `values` in its argument structure. @param {string} name @param {Record<string, any>} values */
function call(name, values) {
  const type = typeNamed(`${name}_t`);
  const address = room(type);
  write(address, type, values);
  library[`_${name}`](address);
}

/** Zeroed room for what each named pointer member of `name`'s arguments points to. @param {string} name @param {string[]} members */
function outputs(name, ...members) {
  return Object.fromEntries(members.map((member) => [member, room(pointee(`${name}_t`, member))]));
}

/** What a call left in the room of its pointer member. @param {string} name @param {string} member @param {number} address @returns {any} */
function landed(name, member, address) {
  const type = pointee(`${name}_t`, member);
  return type.kind === "structure" ? read(address, type) : load(address, type);
}

/** The member of `name`'s arguments that carries entries - typed as `insert`'s - `depth` pointers deeper. @param {string} name @param {number} depth */
function entriesMember(name, depth) {
  for (const member of typeNamed(`${name}_t`).members?.values() ?? [])
    if (member.type === entries.type && member.pointers === entries.pointers + depth) return member.name;
  throw new Error(`${name} carries no ${entries.type}`);
}

/**
 *  Copies `payloads` into the heap, joined end to end, and answers the member value that borrows them with their count.
 *  @param {Payload[]} payloads
 */
function borrow(payloads) {
  const count = payloads.reduce((sum, payload) => sum + payload.count, 0);
  const [first] = payloads;
  if ("records" in first) {
    const record = typeNamed(entries.type);
    const records = payloads.flatMap((payload) => /** @type {Records} */ (payload).records);
    const address = place(new Uint8Array(count * record.size));
    records.forEach((values, entry) => write(address + entry * record.size, record, values));
    return { count, value: address };
  }
  const data = place(concatenated(payloads.map((payload) => /** @type {Rows | Tape} */ (payload).bytes)));
  if ("offsets" in first) {
    const offsets = new Uint32Array(count + 1);
    let entry = 0;
    for (const payload of /** @type {Tape[]} */ (payloads)) {
      const base = offsets[entry];
      for (let local = 1; local <= payload.count; local++) offsets[++entry] = base + payload.offsets[local];
    }
    return { count, value: { data, offsets: place(new Uint8Array(offsets.buffer)), scalar: first.scalar } };
  }
  return { count, value: { data, stride: first.stride, columns: first.columns, scalar: first.scalar } };
}

/**
 *  Copies `count` entries a call wrote into the arena out of it, as `view` - a view, or a pointer to records - names them.
 *  @param {Record<string, number> | number} view @param {number} count @returns {Payload}
 */
function copied(view, count) {
  if (typeof view === "number") {
    const record = typeNamed(entries.type);
    return { count, records: Array.from({ length: count }, (_, entry) => read(view + entry * record.size, record)) };
  }
  const scalar = scalarName(view.scalar);
  if ("offsets" in view) {
    const offsets = count ? copySizes(view.offsets, count + 1) : new Uint32Array(1);
    const first = offsets[0];
    const element = count ? tapeColumns * scalarBytes(scalar) : 0;
    const bytes = copy(view.data + first * element, (offsets[count] - first) * element);
    return { count, bytes, scalar, offsets: offsets.map((offset) => offset - first), columns: tapeColumns };
  }
  return { count, bytes: copy(view.data, count * view.stride), scalar, stride: view.stride, columns: view.columns };
}

/** The outcome as the library reported it, with the status' canonical name. */
function reported() {
  const { status, message } = read(outcome, typeNamed("usearch_outcome_t"));
  return {
    status,
    statusName: library.UTF8ToString(library._usearch_status_name(status)),
    message: message ? library.UTF8ToString(message) : "",
  };
}

const succeeded = () => reported().status === enumerator("usearch_status_success_k");

/**
 *  Each handler runs one request to completion inside the stack frame and heap copies the dispatcher releases.
 *  @type {Record<string, (request: any) => Promise<Record<string, any>> | Record<string, any>>}
 */
const handlers = {
  init_ram({ options }) {
    const name = named("init_ram");
    const created = outputs(name, "index");
    call(name, { ...options, index: created.index, outcome });
    index = landed(name, "index", created.index);
    return {};
  },

  async view_ram({ path, options }) {
    const file = `/indexes/${new URL(path).pathname.split("/").pop()}`;
    library.FS.mkdirTree("/indexes");
    library.FS.writeFile(file, new Uint8Array(await (await fetch(path)).arrayBuffer()));
    const text = new TextEncoder().encode(file);
    const name = named("view_ram");
    const viewed = outputs(name, "index");
    call(name, { ...options, index: viewed.index, outcome, path: { data: place(text), length: text.length } });
    index = landed(name, "index", viewed.index);
    return {};
  },

  insert({ keys, payload }) {
    const name = named("insert");
    const entries = borrow([payload]);
    const out = outputs(name, "stats");
    call(name, {
      index,
      outcome,
      count: entries.count,
      keys: { data: place(keys), stride: keyBytes },
      [entriesMember(name, 0)]: entries.value,
      ...out,
    });
    return { stats: landed(name, "stats", out.stats) };
  },

  sample({ anchors: payloads, options }) {
    const name = named("sample");
    const region = payloads?.length ? borrow(payloads) : null;
    const out = outputs(name, "keys", "links", "written", "stats");
    call(name, {
      ...options,
      index,
      outcome,
      arena,
      anchors_count: region?.count ?? 0,
      anchors: region?.value,
      ...out,
    });
    const stats = landed(name, "stats", out.stats);
    const written = succeeded() ? landed(name, "written", out.written) : 0;
    if (written === 0) return { written, stats };
    const keys = landed(name, "keys", out.keys);
    const links = landed(name, "links", out.links);
    const linkOffsets = copySizes(links.offsets, written + 1);
    const linked = linkOffsets[written];
    return {
      written,
      stats,
      keys: copyKeys(keys.data, keys.stride, written),
      linkOffsets,
      linkKeys: copyKeys(links.keys, keyBytes, linked),
      linkDistances: new Float64Array(copy(links.scores, linked * 8).buffer),
    };
  },

  get({ keys }) {
    const name = named("get");
    const member = entriesMember(name, 1);
    const scalar = typeNamed(`${name}_t`).members?.has("outputs_scalar")
      ? { outputs_scalar: "usearch_scalar_f32_k" }
      : {};
    /** @type {Payload[]} */
    const payloads = [];
    for (let key = 0; key < keys.byteLength / keyBytes; key++) {
      const out = outputs(name, member, "written");
      call(name, {
        ...scalar,
        index,
        outcome,
        arena,
        key: keys.subarray(key * keyBytes, (key + 1) * keyBytes),
        ...out,
      });
      const written = succeeded() ? landed(name, "written", out.written) : 0;
      payloads.push(copied(landed(name, member, out[member]), written));
    }
    return { payloads };
  },

  search({ queries, options }) {
    const name = named("search");
    const entries = borrow([queries]);
    const out = outputs(name, "matches", "stats");
    call(name, {
      ...options,
      index,
      outcome,
      arena,
      count: entries.count,
      [entriesMember(name, 0)]: entries.value,
      ...out,
    });
    const stats = landed(name, "stats", out.stats);
    if (!succeeded()) return { stats };
    const matches = landed(name, "matches", out.matches);
    const cells = entries.count * matches.columns;
    return {
      stats,
      columns: matches.columns,
      counts: copySizes(matches.counts, entries.count),
      keys: copyKeys(matches.keys, keyBytes, cells),
      distances: new Float64Array(copy(matches.scores, cells * 8).buffer),
    };
  },

  /**
   *  The entries each start key's entries link to, as one key tape: one cursor over every start yields the starts first,
   *  in order and each with its links, so their entries are the first ones it writes.
   */
  cursor({ starts }) {
    const init = named("cursor_init");
    const next = named("cursor_next");
    const count = starts.byteLength / keyBytes;
    const offsets = new Uint32Array(count + 1);
    /** @type {Uint8Array[]} */
    const pulled = [];
    const opened = outputs(init, "cursor");
    call(init, { index, outcome, starts_count: count, starts: place(starts), ...opened });
    const cursor = succeeded() ? landed(init, "cursor", opened.cursor) : 0;
    for (let entry = 0; cursor && entry < count;) {
      const wanted = Math.min(cursorBatch, count - entry);
      const out = outputs(next, "keys", "links", "written");
      call(next, { index, outcome, arena, cursor, wanted, ...out });
      const written = succeeded() ? landed(next, "written", out.written) : 0;
      if (written === 0) break;
      const links = landed(next, "links", out.links);
      const linkOffsets = copySizes(links.offsets, written + 1);
      pulled.push(copyKeys(links.keys, keyBytes, linkOffsets[written] - linkOffsets[0]).subarray(0));
      for (let local = 0; local < written; local++)
        offsets[entry + local + 1] = offsets[entry] + linkOffsets[local + 1] - linkOffsets[0];
      entry += written;
    }
    if (cursor) library[`_${named("cursor_free")}`](cursor);
    return { offsets, keys: concatenated(pulled) };
  },
};

/**
 *  Loads the module and the headers it was built from, and opens the arena every call lands its outputs in.
 *  @param {{moduleUrl: string, headersUrl: string, modality: string, tapeColumns?: number}} request
 */
async function loadModule(request) {
  const headers = await Promise.all(
    ["types.h", `${request.modality}_index.h`].map(async (file) =>
      (await fetch(new URL(file, request.headersUrl))).text(),
    ),
  );
  readHeaders(headers.join("\n"));
  library = await (await import(request.moduleUrl)).default();
  modality = request.modality;
  tapeColumns = request.tapeColumns ?? 1;
  const inserted = [...(typeNamed(`${named("insert")}_t`).members?.values() ?? [])];
  entries = inserted[inserted.findIndex((member) => member.name === "keys") + 1];
  const outcomeType = typeNamed("usearch_outcome_t");
  outcome = library._malloc(outcomeType.size);
  library.HEAPU8.fill(0, outcome, outcome + outcomeType.size);
  const saved = library.stackSave();
  const opened = outputs("usearch_arena_init", "arena");
  call("usearch_arena_init", { ...opened, outcome });
  arena = landed("usearch_arena_init", "arena", opened.arena);
  library.stackRestore(saved);
  return { version: library.UTF8ToString(library._usearch_version_name()) };
}

/** @param {Record<string, any>} response */
function transferables(response) {
  /** @type {Set<ArrayBuffer>} */
  const buffers = new Set();
  const visit = (/** @type {any} */ value) => {
    if (ArrayBuffer.isView(value)) buffers.add(/** @type {ArrayBuffer} */ (value.buffer));
    else if (value && typeof value === "object") Object.values(value).forEach(visit);
  };
  visit(response);
  return [...buffers];
}

let queue = Promise.resolve();

self.onmessage = (/** @type {MessageEvent} */ event) => {
  const request = event.data;
  queue = queue.then(async () => {
    const started = performance.now();
    /** @type {Record<string, any>} */
    let response;
    try {
      if (request.verb === "load") response = { ...(await loadModule(request)), ...reported() };
      else {
        const handler = handlers[request.verb];
        if (!handler) throw new Error(`No handler for ${request.verb}`);
        library.HEAPU8.fill(0, outcome, outcome + typeNamed("usearch_outcome_t").size);
        const saved = library.stackSave();
        try {
          response = { ...(await handler(request)), ...reported() };
        } finally {
          library.stackRestore(saved);
          for (const address of borrowed) library._free(address);
          borrowed = [];
        }
      }
    } catch (error) {
      response = { status: -1, statusName: "exception", message: String(error) };
    }
    response = {
      ...response,
      id: request.id,
      verb: request.verb === "load" ? "load" : named(request.verb),
      milliseconds: performance.now() - started,
    };
    self.postMessage(response, { transfer: transferables(response) });
  });
};
