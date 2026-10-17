/**
 *  @file explorer/explorer.js
 *  @brief The scaffolding around the projection core: the key-to-ordinal map, what to draw next, gestures, hover arcs,
 *      selection, hubs, landmarks, glyphs and the readout strip, for every modality alike.
 *
 *  Layout edges always carry distances, from `sample` links or from `search`. Graph links are topology, so they appear
 *  only as hover arcs, and a long link never drags two remote clusters together. What differs per modality lives in
 *  `modalities.js`, and only `worker.js` touches the C ABI.
 */

import { NeighborhoodProjection, alongView, emptySlot, project, rotationOf } from "./projection.js";
import { modalities } from "./modalities.js";

/** @typedef {import('./projection.js').Camera} Camera */
/** @typedef {import('./projection.js').Vector} Vector */
/** @typedef {import('./modalities.js').Payload} Payload */
/** @typedef {import('./modalities.js').Palette} Palette */
/** @typedef {import('./modalities.js').GlyphItem} GlyphItem */
/** @typedef {{seed: number, offset: number, anchors: Payload[] | null, radius: number, seen: Map<string, number>, beside: number, remaining: number, exhausted: boolean}} SampleSequence */

const nodesBudget = 2 << 20;
const edgesPerNode = 12;
// Zooms and selections keep adding edges among nodes already drawn, so a small index still gets room for them.
const edgesFloor = 1 << 20;
const linksPerDraw = 8;
const drawsPerCall = 8192;
// Sample calls are sized to finish near this, so a hover never queues behind a long one.
const sampleCallMilliseconds = 40;
const overviewDraws = 2048;
const zoomDraws = 1024;
const zoomAnchors = 16;
const zoomViewNodes = 256;
const zoomStep = 1.6;
const searchWanted = 16;
const dwellMilliseconds = 150;
const settleMilliseconds = 250;
const pickRadiusPixels = 14;
const dragThresholdPixels = 4;
const turnRadiansPerPixel = 0.008;
// A node this close to the pointer lends a zoom its depth.
const anchorPixels = 60;
const layoutBudgetMilliseconds = 8;
const layoutEpochsAfterChange = 900;
const defaultEdges = 0.5;
const glyphCandidates = 48;
const glyphSparsePixels = 160;
const fitSampleNodes = 4096;
const followedNodes = 32;
const overviewSeed = 0x5eed;

const integerFormat = new Intl.NumberFormat("en-US");
const textDecoder = new TextDecoder();

/** Requests to the worker, answered in order, with the last call kept for the readout. */
class Channel {
  /** @type {Map<number, (response: any) => void>} */
  #pending = new Map();
  #next = 1;
  /** @type {any} */
  last = null;

  /** @param {URL} url */
  constructor(url) {
    this.worker = new Worker(url, { type: "module" });
    this.worker.onmessage = (event) => {
      const resolve = this.#pending.get(event.data.id);
      this.#pending.delete(event.data.id);
      if (event.data.verb !== "load") this.last = event.data;
      resolve?.(event.data);
    };
  }

  /** @param {Record<string, any>} message @param {Transferable[]} transfer @returns {Promise<any>} */
  request(message, transfer = []) {
    const id = this.#next++;
    return new Promise((resolve) => {
      this.#pending.set(id, resolve);
      this.worker.postMessage({ ...message, id }, transfer);
    });
  }

  get busy() {
    return this.#pending.size > 0;
  }
}

/** Keys travel as 16 bytes; as Map keys they are 8 UTF-16 units. @param {Uint8Array} bytes @param {number} offset */
function keyName(bytes, offset) {
  let name = "";
  for (let index = 0; index < 16; index += 2)
    name += String.fromCharCode(bytes[offset + index] | (bytes[offset + index + 1] << 8));
  return name;
}

// The C ABI keeps an integer identity in a key's last bytes, big-endian, with the rest zeroed.

/** Integer identities read as numbers, the rest as dashed UUIDs. @param {Uint8Array} bytes @param {number} offset */
function keyText(bytes, offset) {
  let high = 0;
  for (let index = 0; index < 10; index++) high |= bytes[offset + index];
  if (high === 0) {
    let value = 0;
    for (let index = 10; index < 16; index++) value = value * 256 + bytes[offset + index];
    return `#${value}`;
  }
  const hex = Array.from(bytes.subarray(offset, offset + 16), (byte) => byte.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

/** @param {number} value */
function integerKey(value) {
  const bytes = new Uint8Array(16);
  for (let index = 15; index >= 10; index--, value = Math.floor(value / 256)) bytes[index] = value % 256;
  return bytes;
}

/** Answers a worker's `response`, or throws its outcome when the call failed. @param {any} response */
function checked(response) {
  if (response.status !== 0)
    throw new Error(response.message ? `${response.statusName}: ${response.message}` : response.statusName);
  return response;
}

/**
 *  The titles tape `prepare.py` writes, indexed by integer identity: a `u32` count, that many plus one `u32` offsets,
 *  then the UTF-8 bytes they span.
 *  @param {string | null} path @returns {Promise<string[]>}
 */
async function readTitles(path) {
  if (!path) return [];
  const tape = new Uint8Array(await (await fetch(new URL(path, location.href))).arrayBuffer());
  const words = new DataView(tape.buffer);
  const count = words.getUint32(0, true);
  const base = 4 + (count + 1) * 4;
  return Array.from({ length: count }, (_, title) =>
    textDecoder.decode(
      tape.subarray(base + words.getUint32(4 + title * 4, true), base + words.getUint32(8 + title * 4, true)),
    ),
  );
}

/** @param {HTMLElement} element @param {string} property */
function cssValue(element, property) {
  return getComputedStyle(element).getPropertyValue(property).trim();
}

/** A token resolved to `rgb(...)`, so `light-dark()` and `var()` read as the canvas needs. @param {HTMLElement} element @param {string} property */
function resolvedColor(element, property) {
  const probe = document.createElement("span");
  probe.style.color = `var(${property})`;
  element.append(probe);
  const color = getComputedStyle(probe).color;
  probe.remove();
  return color;
}

/** @param {HTMLElement} element @param {string} property @returns {number[]} */
function colorChannels(element, property) {
  const channels = (resolvedColor(element, property).match(/[\d.]+/g) ?? ["0", "0", "0"]).map(Number);
  return [channels[0] / 255, channels[1] / 255, channels[2] / 255];
}

/**
 *  @typedef {{
 *    frame: HTMLElement,
 *    canvas: HTMLCanvasElement,
 *    overlay: HTMLCanvasElement,
 *    status: HTMLElement,
 *    readouts: Record<'nodes' | 'edges' | 'layout' | 'draw' | 'call' | 'traversal' | 'outcome', HTMLElement>,
 *    toggles: Record<'edges' | 'hubs' | 'surface', HTMLButtonElement>
 *  }} Elements
 */

/**
 *  Opens the index the URL names - or, on the dev path, builds one inside the worker - and explores it in `elements`.
 *  @param {Elements} elements
 *  @param {URLSearchParams} parameters
 */
export async function explore(elements, parameters) {
  const { frame, canvas, overlay, status } = elements;
  const showStatus = (/** @type {string} */ message) => {
    status.textContent = message;
    status.hidden = !message;
  };

  const development = parameters.get("dev");
  const modalityName = development ?? parameters.get("modality") ?? "dense";
  const modality = modalities[modalityName];
  if (!modality) throw new Error(`Unknown modality ${modalityName}`);
  const scenery = modality.scenery?.() ?? [];

  const channel = new Channel(new URL("./worker.js", import.meta.url));
  showStatus("Loading the module…");
  checked(
    await channel.request({
      verb: "load",
      moduleUrl: new URL(parameters.get("module") ?? "./usearch.mjs", location.href).href,
      headersUrl: new URL(parameters.get("headers") ?? "../c/usearch/", location.href).href,
      modality: modality.name,
      tapeColumns: modality.tapeColumns,
    }),
  );

  let capacity = Number(parameters.get("budget") ?? nodesBudget);
  if (development) {
    const dataset = await modality.develop(parameters, showStatus);
    checked(await channel.request({ verb: "init_ram", options: dataset.options }));
    showStatus(`Inserting ${integerFormat.format(dataset.payload.count)} entries through the C ABI…`);
    const keys = new Uint8Array(dataset.identities.length * 16);
    dataset.identities.forEach((identity, entry) => keys.set(integerKey(identity), entry * 16));
    checked(await channel.request({ verb: "insert", keys, payload: dataset.payload }));
    capacity = Math.min(capacity, dataset.payload.count);
  } else {
    const path = parameters.get("index");
    if (!path)
      throw new Error("Name an index with ?index= and its ?modality=, or a development one with ?dev=<modality>.");
    showStatus("Opening the index…");
    checked(await channel.request({ verb: "view_ram", path: new URL(path, location.href).href }));
  }
  const titles = await readTitles(parameters.get("titles"));

  showStatus("Preparing the GPU…");
  const edgesCapacity = Math.max(capacity * edgesPerNode, edgesFloor);
  const projection = await NeighborhoodProjection.create(canvas, capacity, edgesCapacity);
  projection.device.lost.then((information) => showStatus(`The GPU device was lost: ${information.message}`));

  const ordinalsByKey = /** @type {Map<string, number[]>} */ (new Map());
  const keyBytes = new Uint8Array(capacity * 16);
  const linkReach = new Float32Array(capacity);
  const payloadCache = /** @type {Map<string, Payload>} */ (new Map());
  const hubCounts = /** @type {Map<number, number>} */ (new Map());
  const pinned = /** @type {Set<number>} */ (new Set());
  let nodes = 0;
  let edges = 0;

  /** @param {number} ordinal */
  const nameOf = (ordinal) => keyName(keyBytes, ordinal * 16);
  /** @param {number} ordinal */
  const bytesOf = (ordinal) => keyBytes.slice(ordinal * 16, ordinal * 16 + 16);
  /** @param {string} name */
  const firstOrdinal = (name) => ordinalsByKey.get(name)?.[0];
  /** @param {number} ordinal */
  const labelOf = (ordinal) => {
    const text = keyText(keyBytes, ordinal * 16);
    const title = titles[Number(text.slice(1))];
    return text.startsWith("#") && title !== undefined ? title : text;
  };

  /** @param {string} name @param {Uint8Array} bytes @param {number} offset */
  function register(name, bytes, offset) {
    const ordinal = nodes++;
    keyBytes.set(bytes.subarray(offset, offset + 16), ordinal * 16);
    const known = ordinalsByKey.get(name);
    if (known) known.push(ordinal);
    else ordinalsByKey.set(name, [ordinal]);
    return ordinal;
  }

  /** Appends edges and keeps each node's longest link, the scale a zoom samples around. */
  function appendEdges(
    /** @type {number[]} */ sources,
    /** @type {number[]} */ targets,
    /** @type {number[]} */ distances,
  ) {
    const room = Math.max(0, edgesCapacity - edges);
    const count = Math.min(room, sources.length);
    if (count === 0) return;
    projection.appendEdges(
      Uint32Array.from(sources.slice(0, count)),
      Uint32Array.from(targets.slice(0, count)),
      Float32Array.from(distances.slice(0, count)),
    );
    for (let edge = 0; edge < count; edge++) {
      linkReach[sources[edge]] = Math.max(linkReach[sources[edge]], distances[edge]);
      linkReach[targets[edge]] = Math.max(linkReach[targets[edge]], distances[edge]);
    }
    edges += count;
  }

  /** @param {number[]} beside */
  function appendNodes(beside) {
    if (beside.length === 0) return;
    projection.appendNodes(Uint32Array.from(beside), [0, 0, 0]);
    layoutEpochsLeft = layoutEpochsAfterChange;
    if (modality.place) pinToPlane(Array.from(beside, (_, index) => nodes - beside.length + index));
  }

  /** Holds `ordinals` where the modality's plane puts their entries, once `get` brings them. @param {number[]} ordinals */
  async function pinToPlane(ordinals) {
    const place = modality.place;
    if (!place) return;
    const payloads = await payloadsOf(ordinals.map(bytesOf));
    /** @type {number[]} */ const kept = [];
    /** @type {number[]} */ const coordinates = [];
    ordinals.forEach((ordinal, index) => {
      const payload = payloads[index];
      if (!payload?.count) return;
      // Nodes under one key take its entries in turn, which `get` may list in another order than they were drawn.
      const entry = Math.min(
        payload.count - 1,
        Math.max(0, (ordinalsByKey.get(nameOf(ordinal)) ?? []).indexOf(ordinal)),
      );
      kept.push(ordinal);
      coordinates.push(...place(payload, entry));
    });
    if (kept.length > 0) projection.pin(Uint32Array.from(kept), Float32Array.from(coordinates));
    // A placement is no layout motion for the view to follow.
    followedCentre = null;
    if (!userMoved) fit();
  }

  const style = {
    ink: [1, 1, 1],
    paper: [0, 0, 0],
    accent: [0.4, 0.565, 1],
    edges: Number(parameters.get("edges") ?? defaultEdges),
  };
  /** @type {Palette} */
  let palette = {
    ink: "#fff",
    muted: "#8a90a2",
    paper: "#000",
    accent: "#6690ff",
    danger: "#ff6369",
    partial: "#d29922",
  };
  function readTokens() {
    style.ink = colorChannels(frame, "--ink");
    style.paper = colorChannels(frame, "--paper");
    style.accent = colorChannels(frame, "--accent");
    palette = {
      ink: resolvedColor(frame, "--ink"),
      muted: resolvedColor(frame, "--ink-muted"),
      paper: resolvedColor(frame, "--paper"),
      accent: resolvedColor(frame, "--accent"),
      danger: resolvedColor(frame, "--danger"),
      partial: resolvedColor(frame, "--status-partial"),
    };
    dirty = true;
  }

  /** @type {Camera} */
  const camera = { center: [0, 0, 0], scale: 40, yaw: 0, pitch: 0, radius: 1 };
  /** @type {Camera} */
  let home = { ...camera };
  let userMoved = false;
  let dirty = true;
  let layoutEpochsLeft = 0;
  let lastCameraChange = 0;
  let settledHandled = true;
  let lastSampledScale = 0;

  const cssWidth = () => canvas.clientWidth || 1;
  const cssHeight = () => canvas.clientHeight || 1;
  /** Where a layout position lands on screen, with its depth toward the viewer. @param {ArrayLike<number>} position */
  const screenOf = (position) => project(camera, cssWidth(), cssHeight(), position);

  function cameraChanged() {
    dirty = true;
    lastCameraChange = performance.now();
    settledHandled = false;
  }

  /**
   *  Fits the camera around the middle 98% of a strided subset of the nodes, at any rotation. A modality that places its
   *  nodes frames the whole surface around the placement's origin instead, turned so the nodes face the viewer.
   */
  async function fit() {
    if (nodes === 0) return home;
    const stride = Math.max(1, Math.floor(nodes / fitSampleNodes));
    const ordinals = new Uint32Array(Math.ceil(nodes / stride)).map((_, index) => index * stride);
    const positions = await projection.positionsOf(ordinals);
    const axes = [0, 1, 2].map((axis) =>
      Array.from(ordinals, (_, index) => positions[index * 3 + axis]).sort((left, right) => left - right),
    );
    const pick = (/** @type {number[]} */ values, /** @type {number} */ fraction) =>
      values[Math.floor(fraction * (values.length - 1))];
    let center = /** @type {Vector} */ (axes.map((values) => (pick(values, 0.99) + pick(values, 0.01)) / 2));
    let radius = Math.max(1e-3, ...axes.map((values) => (pick(values, 0.99) - pick(values, 0.01)) / 2));
    let { yaw, pitch } = camera;
    if (modality.place) {
      [yaw, pitch] = [Math.atan2(-center[0], center[2]), Math.atan2(center[1], Math.hypot(center[0], center[2]))];
      radius = Math.max(
        ...Array.from(ordinals, (_, index) => Math.hypot(...positions.subarray(index * 3, index * 3 + 3))),
      );
      center = [0, 0, 0];
    }
    home = { center, scale: (0.86 * Math.min(cssWidth(), cssHeight())) / (2 * radius), yaw, pitch, radius };
    if (!userMoved) {
      Object.assign(camera, home, { center: [...home.center] });
      lastSampledScale = home.scale * zoomStep;
      dirty = true;
    }
    return home;
  }

  // Sampling: the overview sequence, then one sequence per zoom, each extended a call at a time when the worker idles.
  /** @type {SampleSequence} */
  const overview = {
    seed: overviewSeed,
    offset: 0,
    anchors: null,
    radius: 0,
    seen: new Map(),
    beside: emptySlot,
    remaining: Number(parameters.get("overview") ?? overviewDraws),
    exhausted: false,
  };
  /** @type {SampleSequence[]} */
  const zooms = [];
  let drawsWanted = 512;
  let sampling = false;
  const samplingTotals = { calls: 0, draws: 0, milliseconds: 0 };
  let interactive = 0;

  /** @param {SampleSequence} sequence @param {any} response */
  function integrate(sequence, response) {
    const count = response.written;
    /** @type {number[]} */
    const beside = [];
    const ordinals = new Uint32Array(count);
    let landed = 0;
    for (; landed < count; landed++) {
      const name = keyName(response.keys, landed * 16);
      const occurrence = sequence.seen.get(name) ?? 0;
      const known = ordinalsByKey.get(name);
      if (known && occurrence < known.length) {
        ordinals[landed] = known[occurrence];
      } else {
        if (nodes >= capacity) break;
        const linkStart = response.linkOffsets[landed];
        const first =
          linkStart < response.linkOffsets[landed + 1]
            ? firstOrdinal(keyName(response.linkKeys, linkStart * 16))
            : undefined;
        beside.push(first ?? sequence.beside);
        ordinals[landed] = register(name, response.keys, landed * 16);
      }
      sequence.seen.set(name, occurrence + 1);
    }
    appendNodes(beside);
    /** @type {number[]} */ const sources = [];
    /** @type {number[]} */ const targets = [];
    /** @type {number[]} */ const distances = [];
    for (let draw = 0; draw < landed; draw++) {
      for (let link = response.linkOffsets[draw]; link < response.linkOffsets[draw + 1]; link++) {
        const target = firstOrdinal(keyName(response.linkKeys, link * 16));
        if (target === undefined || target === ordinals[draw]) continue;
        sources.push(ordinals[draw]);
        targets.push(target);
        distances.push(response.linkDistances[link]);
      }
    }
    appendEdges(sources, targets, distances);
    sequence.offset += landed;
    sequence.remaining -= landed;
    if (landed < count || response.written < response.wanted || nodes >= capacity) sequence.exhausted = true;
  }

  async function sampleTick() {
    if (sampling || interactive > 0 || nodes >= capacity) return;
    const sequence = [...zooms, overview].find((candidate) => !candidate.exhausted && candidate.remaining > 0);
    if (!sequence) return;
    sampling = true;
    const wanted = Math.max(16, Math.min(drawsPerCall, drawsWanted, sequence.remaining, capacity - nodes));
    const response = await channel.request({
      verb: "sample",
      anchors: sequence.anchors,
      options: {
        seed: sequence.seed,
        radius: sequence.radius,
        wanted,
        offset: sequence.offset,
        links_per_draw: linksPerDraw,
      },
    });
    sampling = false;
    if (response.status !== 0) {
      sequence.exhausted = true;
      return;
    }
    response.wanted = wanted;
    samplingTotals.calls++;
    samplingTotals.draws += response.written;
    samplingTotals.milliseconds += response.milliseconds;
    integrate(sequence, response);
    drawsWanted = Math.round(
      Math.max(16, Math.min(drawsPerCall, (wanted * sampleCallMilliseconds) / Math.max(1, response.milliseconds))),
    );
    if (!userMoved && (sequence === overview || nodes < 4096)) fit();
  }

  /** Every entry under each of `keys`, through `get`, cached. @param {Uint8Array[]} keys @returns {Promise<(Payload | null)[]>} */
  async function payloadsOf(keys) {
    const missing = keys.filter((bytes) => !payloadCache.has(keyName(bytes, 0)));
    if (missing.length > 0) {
      const joinedKeys = new Uint8Array(missing.length * 16);
      missing.forEach((bytes, index) => joinedKeys.set(bytes, index * 16));
      interactive++;
      const response = await channel.request({ verb: "get", keys: joinedKeys });
      interactive--;
      (response.payloads ?? []).forEach((/** @type {Payload} */ payload, /** @type {number} */ index) =>
        payloadCache.set(keyName(missing[index], 0), payload),
      );
    }
    return keys.map((bytes) => payloadCache.get(keyName(bytes, 0)) ?? null);
  }

  /** The keys each start key's entries link to in the graph, one cursor apiece. @param {Uint8Array[]} starts */
  async function linkedKeys(starts) {
    const joinedKeys = new Uint8Array(starts.length * 16);
    starts.forEach((bytes, index) => joinedKeys.set(bytes, index * 16));
    interactive++;
    const response = await channel.request({ verb: "cursor", starts: joinedKeys });
    interactive--;
    return starts.map((_, index) => {
      const keys = [];
      for (let entry = response.offsets?.[index] ?? 0; entry < (response.offsets?.[index + 1] ?? 0); entry++)
        keys.push(response.keys.slice(entry * 16, entry * 16 + 16));
      return keys;
    });
  }

  async function zoomTick() {
    if (camera.scale < lastSampledScale || nodes >= capacity || nodes === 0) return;
    lastSampledScale = camera.scale * zoomStep;
    const nearby = await projection.nearest(camera, cssWidth() / 2, cssHeight() / 2, zoomViewNodes);
    if (nearby.nodes.length === 0) return;
    const anchors = Array.from(nearby.nodes.subarray(0, zoomAnchors));
    const uniqueAnchors = [...new Map(anchors.map((ordinal) => [nameOf(ordinal), ordinal])).values()];
    const payloads = (await payloadsOf(uniqueAnchors.map(bytesOf))).filter((payload) => payload?.count);
    if (payloads.length === 0) return;
    let radius = 0;
    for (const ordinal of nearby.nodes) radius = Math.max(radius, linkReach[ordinal]);
    zooms.push({
      seed: overviewSeed + zooms.length + 1,
      offset: 0,
      anchors: /** @type {Payload[]} */ (payloads),
      radius,
      seen: new Map(),
      beside: anchors[0],
      remaining: zoomDraws,
      exhausted: false,
    });
  }

  // Hover: the newest pointer wins, one request in flight; the first hop at once, the second after a dwell.
  /** @typedef {{bytes: Uint8Array, ordinal: number | undefined, reached: number[] | null}} Hop */
  /** @type {{node: number, hops: Hop[], started: number} | null} */
  let hover = null;
  /** @type {{x: number, y: number, time: number} | null} */
  let pointer = null;
  let hoverRunning = false;
  /** @type {ReturnType<typeof setTimeout> | undefined} */
  let dwellTimer;
  const latencies = { first: /** @type {number[]} */ ([]), second: /** @type {number[]} */ ([]), dwellStart: 0 };

  async function runHover() {
    if (hoverRunning || !pointer) return;
    hoverRunning = true;
    while (pointer) {
      const current = pointer;
      const found = await projection.nearest({ ...camera, center: [...camera.center] }, current.x, current.y, 1);
      if (pointer !== current) continue;
      pointer = null;
      const node = found.nodes[0];
      const [x, y] = found.nodes.length ? screenOf(found.positions) : [Infinity, Infinity];
      if (found.nodes.length && Math.hypot(x - current.x, y - current.y) < anchorPixels)
        pointerAnchor = Array.from(found.positions.subarray(0, 3));
      if (node === undefined || Math.hypot(x - current.x, y - current.y) > pickRadiusPixels) {
        if (hover) {
          hover = null;
          clearTimeout(dwellTimer);
          dirty = true;
        }
        continue;
      }
      if (hover?.node === node) continue;
      clearTimeout(dwellTimer);
      const [linked] = await linkedKeys([bytesOf(node)]);
      if (pointer) continue;
      hover = {
        node,
        started: current.time,
        hops: linked.map((bytes) => ({ bytes, ordinal: firstOrdinal(keyName(bytes, 0)), reached: null })),
      };
      for (const hop of hover.hops)
        if (hop.ordinal !== undefined) hubCounts.set(hop.ordinal, (hubCounts.get(hop.ordinal) ?? 0) + 1);
      if (hubsShown) showHubs();
      await refreshPositions();
      drawOverlay();
      latencies.first.push(performance.now() - current.time);
      payloadsOf([bytesOf(node)]).then(() => (dirty = true));
      const hovered = hover;
      dwellTimer = setTimeout(() => secondHop(hovered), dwellMilliseconds);
    }
    hoverRunning = false;
  }

  /** @param {{node: number, hops: Hop[], started: number}} hovered */
  async function secondHop(hovered) {
    const started = performance.now();
    const undrawn = hovered.hops.filter((hop) => hop.ordinal === undefined);
    const reached = await linkedKeys(undrawn.map((hop) => hop.bytes));
    if (hover !== hovered) return;
    undrawn.forEach((hop, index) => {
      hop.reached = reached[index]
        .map((bytes) => firstOrdinal(keyName(bytes, 0)))
        .filter((ordinal) => ordinal !== undefined);
    });
    await refreshPositions();
    drawOverlay();
    latencies.second.push(performance.now() - started);
  }

  // Selection: every entry under the key, their nearest entries as nodes and layout edges, and two highlighted hops.
  /** @type {{name: string, entries: number[], first: number[], second: number[]} | null} */
  let selection = null;
  let hubsShown = false;

  /** @param {number} node */
  async function select(node) {
    follow([node]);
    const bytes = bytesOf(node);
    const name = nameOf(node);
    const [payload] = await payloadsOf([bytes]);
    const count = payload?.count ?? 0;
    const known = ordinalsByKey.get(name) ?? [];
    /** @type {number[]} */
    const beside = [];
    for (let entry = known.length; entry < count && nodes < capacity; entry++) {
      beside.push(node);
      register(name, bytes, 0);
    }
    appendNodes(beside);
    const entries = (ordinalsByKey.get(name) ?? [node]).slice(0, Math.max(1, count));
    if (payload && count > 0) {
      interactive++;
      const found = await channel.request({
        verb: "search",
        queries: payload,
        options: { wanted: searchWanted, ...modality.search },
      });
      interactive--;
      if (found.status === 0) {
        /** @type {number[]} */ const newcomers = [];
        /** @type {number[]} */ const sources = [];
        /** @type {number[]} */ const targets = [];
        /** @type {number[]} */ const distances = [];
        for (let row = 0; row < count; row++) {
          for (let column = 0; column < found.counts[row]; column++) {
            const match = row * found.columns + column;
            const matchName = keyName(found.keys, match * 16);
            if (matchName === name) continue;
            let target = firstOrdinal(matchName);
            if (target === undefined) {
              if (nodes >= capacity) continue;
              newcomers.push(entries[row] ?? node);
              target = register(matchName, found.keys, match * 16);
            }
            sources.push(entries[row] ?? node);
            targets.push(target);
            distances.push(found.distances[match]);
          }
        }
        appendNodes(newcomers);
        appendEdges(sources, targets, distances);
      }
    }
    const [firstHop] = await linkedKeys([bytes]);
    const secondHop = await linkedKeys(firstHop);
    const drawn = (/** @type {Uint8Array[]} */ keys) =>
      keys.map((key) => firstOrdinal(keyName(key, 0))).filter((ordinal) => ordinal !== undefined);
    const first = drawn(firstHop);
    const second = drawn(secondHop.flat()).filter((ordinal) => !first.includes(ordinal) && !entries.includes(ordinal));
    selection = { name, entries, first, second };
    if (!hubsShown) showSelection();
    await refreshPositions();
    dirty = true;
  }

  function showSelection() {
    if (!selection) return projection.highlight(new Uint32Array(0), new Float32Array(0));
    const ordinals = [...selection.second, ...selection.first, ...selection.entries];
    const strengths = [
      ...selection.second.map(() => 0.3),
      ...selection.first.map(() => 0.6),
      ...selection.entries.map(() => 1),
    ];
    projection.highlight(Uint32Array.from(ordinals), Float32Array.from(strengths));
    dirty = true;
  }

  function showHubs() {
    let most = 1;
    for (const count of hubCounts.values()) most = Math.max(most, count);
    const ordinals = Uint32Array.from(hubCounts.keys());
    projection.highlight(
      ordinals,
      Float32Array.from(ordinals, (ordinal) => Math.sqrt((hubCounts.get(ordinal) ?? 0) / most)),
    );
    dirty = true;
  }

  // Overlay: arcs, rings, landmarks and glyphs in CSS pixels, from positions refreshed while the layout moves.
  const positions = /** @type {Map<number, number[]>} */ (new Map());
  /** @type {{ordinal: number, item: GlyphItem}[]} */
  let glyphs = [];
  let positionsRunning = false;

  // The view keeps what it shows in place while the layout moves: the selection, or the nodes nearest its centre.
  /** @type {number[]} */
  let followed = [];
  /** @type {number[] | null} */
  let followedCentre = null;

  /** @param {number[]} ordinals */
  function follow(ordinals) {
    followed = ordinals;
    followedCentre = null;
  }

  /** Shifts the camera by how far the followed nodes moved since the last refresh, unless a node is being dragged. */
  function keepFollowed() {
    const known = followed.map((ordinal) => positions.get(ordinal)).filter((position) => position !== undefined);
    if (known.length === 0) return;
    const centre = [0, 1, 2].map((axis) => known.reduce((sum, position) => sum + position[axis], 0) / known.length);
    if (followedCentre && userMoved && drag?.node == null) {
      const moved = /** @type {number[]} */ (followedCentre);
      camera.center = /** @type {Vector} */ (camera.center.map((value, axis) => value + centre[axis] - moved[axis]));
      dirty = true;
    }
    followedCentre = centre;
  }

  function overlayOrdinals() {
    const ordinals = new Set([...pinned, ...followed]);
    if (hover) {
      ordinals.add(hover.node);
      for (const hop of hover.hops) {
        if (hop.ordinal !== undefined) ordinals.add(hop.ordinal);
        for (const ordinal of hop.reached ?? []) ordinals.add(ordinal);
      }
    }
    if (selection) for (const ordinal of selection.entries) ordinals.add(ordinal);
    for (const glyph of glyphs) ordinals.add(glyph.ordinal);
    return ordinals;
  }

  async function refreshPositions() {
    if (positionsRunning) return;
    const ordinals = Uint32Array.from(overlayOrdinals());
    if (ordinals.length === 0) return;
    positionsRunning = true;
    const found = await projection.positionsOf(ordinals);
    ordinals.forEach((ordinal, index) => positions.set(ordinal, Array.from(found.subarray(index * 3, index * 3 + 3))));
    keepFollowed();
    positionsRunning = false;
  }

  const overlayContext = /** @type {CanvasRenderingContext2D} */ (overlay.getContext("2d"));

  function drawOverlay() {
    const ratio = devicePixelRatio || 1;
    const width = Math.round(cssWidth() * ratio);
    const height = Math.round(cssHeight() * ratio);
    if (overlay.width !== width || overlay.height !== height) {
      overlay.width = width;
      overlay.height = height;
    }
    const context = overlayContext;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, cssWidth(), cssHeight());
    const at = (/** @type {number} */ ordinal) => {
      const position = positions.get(ordinal);
      return position ? /** @type {[number, number]} */ (screenOf(position).slice(0, 2)) : null;
    };
    // Scenery lies under everything else, fading with depth, and drawn only within the depth the core draws.
    const slab = Math.max(cssWidth(), cssHeight()) / (2 * camera.scale);
    const depthRange = Math.max(1e-9, Math.min(camera.radius, slab));
    context.strokeStyle = palette.ink;
    for (const polyline of scenery) {
      const points = polyline.map(screenOf);
      for (let point = 1; point < points.length; point++) {
        const [from, to] = [points[point - 1], points[point]];
        if (Math.abs(from[2]) > slab || Math.abs(to[2]) > slab) continue;
        // The far half fades out entirely, so a surface reads as solid rather than as a cage.
        context.globalAlpha = 0.25 * Math.min(1, Math.max(0, (0.5 * (from[2] + to[2])) / depthRange / 0.3 + 0.3));
        context.beginPath();
        context.moveTo(from[0], from[1]);
        context.lineTo(to[0], to[1]);
        context.stroke();
      }
    }
    context.globalAlpha = 1;

    const ring = (
      /** @type {[number, number] | null} */ point,
      /** @type {number} */ radius,
      /** @type {string} */ color,
    ) => {
      if (!point) return;
      context.strokeStyle = color;
      context.beginPath();
      context.arc(point[0], point[1], radius, 0, Math.PI * 2);
      context.stroke();
    };
    context.lineWidth = 1;

    if (hover) {
      const source = at(hover.node);
      if (source) {
        context.strokeStyle = palette.accent;
        for (const hop of hover.hops) {
          let target = hop.ordinal !== undefined ? at(hop.ordinal) : null;
          const remote = hop.ordinal === undefined;
          if (remote) {
            const reached = (hop.reached ?? []).map(at).filter((point) => point !== null);
            if (reached.length === 0) continue;
            target = [
              reached.reduce((sum, point) => sum + point[0], 0) / reached.length,
              reached.reduce((sum, point) => sum + point[1], 0) / reached.length,
            ];
          }
          if (!target) continue;
          const [x, y] = target;
          const bend = 0.18;
          context.globalAlpha = remote ? 0.55 : 0.85;
          context.setLineDash(remote ? [3, 3] : []);
          context.beginPath();
          context.moveTo(source[0], source[1]);
          context.quadraticCurveTo(
            (source[0] + x) / 2 - (y - source[1]) * bend,
            (source[1] + y) / 2 + (x - source[0]) * bend,
            x,
            y,
          );
          context.stroke();
          if (!remote) ring(target, 2.5, palette.accent);
        }
        context.setLineDash([]);
        context.globalAlpha = 1;
        ring(source, 6, palette.ink);
      }
    }

    if (selection) for (const ordinal of selection.entries) ring(at(ordinal), 8, palette.accent);

    for (const ordinal of pinned) {
      const point = at(ordinal);
      if (!point) continue;
      context.strokeStyle = palette.ink;
      context.strokeRect(point[0] - 5, point[1] - 5, 10, 10);
    }

    // Glyphs are placed again at every draw, since the layout keeps moving them, the hovered one first and any that
    // would overlap one already placed skipped.
    context.font = `11px ${cssValue(frame, "--font-mono") || "monospace"}`;
    const [glyphWidth, glyphHeight] = modality.glyphSize;
    /** @type {[number, number, number, number][]} */
    const placed = [];
    /** @param {[number, number]} point @param {GlyphItem} item */
    const place = (point, item) => {
      const box = /** @type {[number, number, number, number]} */ ([
        point[0],
        point[1] - glyphHeight / 2,
        point[0] + glyphWidth,
        point[1] + glyphHeight / 2,
      ]);
      if (placed.some((other) => box[0] < other[2] && other[0] < box[2] && box[1] < other[3] && other[1] < box[3]))
        return;
      placed.push(box);
      modality.glyph(context, item, point[0], point[1]);
    };
    const hovered = hover ? at(hover.node) : null;
    if (hover && hovered)
      place([hovered[0] + 8, hovered[1]], {
        label: labelOf(hover.node),
        payload: payloadCache.get(nameOf(hover.node)) ?? null,
        palette,
      });
    for (const glyph of glyphs) {
      const point = at(glyph.ordinal);
      if (!point || glyph.ordinal === hover?.node) continue;
      context.fillStyle = palette.muted;
      context.fillRect(point[0] - 1, point[1] - 1, 2, 2);
      place([point[0], point[1]], glyph.item);
    }

    // The rotation legend: the layout's axes as the camera turns them, nearer ones brighter, and the angles beside them.
    const [legendX, legendY] = [30, cssHeight() - 30];
    const turned = rotationOf(camera);
    context.font = `10px ${cssValue(frame, "--font-mono") || "monospace"}`;
    ["x", "y", "z"].forEach((name, axis) => {
      const [tipX, tipY] = [legendX + turned[0][axis] * 18, legendY - turned[1][axis] * 18];
      context.globalAlpha = 0.35 + 0.65 * (0.5 + 0.5 * turned[2][axis]);
      context.strokeStyle = context.fillStyle = palette.ink;
      context.beginPath();
      context.moveTo(legendX, legendY);
      context.lineTo(tipX, tipY);
      context.stroke();
      context.fillText(name, tipX + (tipX - legendX) * 0.25 - 3, tipY + (tipY - legendY) * 0.25 + 3);
    });
    context.globalAlpha = 1;
    context.fillStyle = palette.muted;
    const degrees = (/** @type {number} */ radians) => Math.round((((radians * 180) / Math.PI + 540) % 360) - 180);
    context.fillText(`yaw ${degrees(camera.yaw)}°  pitch ${degrees(camera.pitch)}°`, legendX + 30, legendY + 4);
  }

  async function refreshGlyphs() {
    const nearby = await projection.nearest(
      { ...camera, center: [...camera.center] },
      cssWidth() / 2,
      cssHeight() / 2,
      glyphCandidates,
    );
    const count = nearby.nodes.length;
    if (count === 0) return void (glyphs = []);
    if (!selection) follow(Array.from(nearby.nodes.subarray(0, followedNodes)));
    const [lastX, lastY] = screenOf(nearby.positions.subarray((count - 1) * 3, count * 3));
    const sparse =
      count < glyphCandidates || Math.hypot(lastX - cssWidth() / 2, lastY - cssHeight() / 2) > glyphSparsePixels;
    if (!sparse) return void (glyphs = []);
    const kept = Array.from(nearby.nodes);
    const payloads = await payloadsOf(kept.map(bytesOf));
    glyphs = kept.map((ordinal, index) => ({
      ordinal,
      item: { label: labelOf(ordinal), payload: payloads[index], palette },
    }));
    nearby.nodes.forEach((ordinal, index) =>
      positions.set(ordinal, Array.from(nearby.positions.subarray(index * 3, index * 3 + 3))),
    );
    dirty = true;
  }

  // Gestures: a drag turns the layout about the view's centre, Shift or the right button pans, Option drags a node to pin
  // it as a landmark, the wheel or a trackpad pinch zooms at the cursor, and a double-click resets.
  /** The last node found near the pointer. @type {number[] | null} */
  let pointerAnchor = null;
  /** @param {number} x @param {number} y @param {number} factor */
  function zoomAround(x, y, factor) {
    const right = x - cssWidth() / 2;
    const down = y - cssHeight() / 2;
    // The point under the cursor stays there, at the depth of the last node found near the pointer, so zooming into a
    // surface lands on it rather than inside it.
    const depth = pointerAnchor ? screenOf(pointerAnchor)[2] : 0;
    const sight = rotationOf(camera)[2];
    const before = alongView(camera, right, down);
    const target = camera.center.map((value, axis) => value + before[axis] + depth * sight[axis]);
    camera.scale = Math.max(1e-3, Math.min(1e7, camera.scale * factor));
    const after = alongView(camera, right, down);
    camera.center = /** @type {Vector} */ (target.map((value, axis) => value - after[axis]));
    userMoved = true;
    cameraChanged();
  }

  /** @param {MouseEvent} event @returns {[number, number]} */
  const local = (event) => {
    const bounding = canvas.getBoundingClientRect();
    return [event.clientX - bounding.left, event.clientY - bounding.top];
  };

  frame.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const lineScale = event.deltaMode === 1 ? 16 : 1;
      const sensitivity = event.ctrlKey ? 0.01 : 0.0015;
      const [x, y] = local(event);
      zoomAround(x, y, Math.exp(-event.deltaY * lineScale * sensitivity));
    },
    { passive: false },
  );

  /** @type {{x: number, y: number, lastX: number, lastY: number, moved: boolean, node: number | null, panning: boolean} | null} */
  let drag = null;
  frame.addEventListener("contextmenu", (event) => event.preventDefault());
  frame.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 && event.button !== 2) return;
    frame.setPointerCapture(event.pointerId);
    const [x, y] = local(event);
    const hovered = hover ? positions.get(hover.node) : null;
    const onNode =
      hover && hovered && Math.hypot(screenOf(hovered)[0] - x, screenOf(hovered)[1] - y) <= pickRadiusPixels;
    const node = event.altKey && onNode && hover ? hover.node : null;
    drag = { x, y, lastX: x, lastY: y, moved: false, node, panning: event.shiftKey || event.button === 2 };
  });
  frame.addEventListener("pointermove", (event) => {
    const [x, y] = local(event);
    if (!drag) {
      pointer = { x, y, time: performance.now() };
      runHover();
      return;
    }
    if (Math.hypot(x - drag.x, y - drag.y) > dragThresholdPixels) {
      drag.moved = true;
      frame.classList.add("dragging");
    }
    if (!drag.moved) return;
    const offset = alongView(camera, x - drag.lastX, y - drag.lastY);
    if (drag.node !== null) {
      const moved = (positions.get(drag.node) ?? camera.center).map((value, axis) => value + offset[axis]);
      pinned.add(drag.node);
      projection.pin(Uint32Array.of(drag.node), Float32Array.from(moved));
      positions.set(drag.node, moved);
      layoutEpochsLeft = layoutEpochsAfterChange;
      dirty = true;
    } else {
      if (drag.panning)
        camera.center = /** @type {Vector} */ (camera.center.map((value, axis) => value - offset[axis]));
      else {
        camera.yaw += (x - drag.lastX) * turnRadiansPerPixel;
        camera.pitch = Math.max(
          -Math.PI / 2,
          Math.min(Math.PI / 2, camera.pitch + (y - drag.lastY) * turnRadiansPerPixel),
        );
      }
      userMoved = true;
      cameraChanged();
    }
    drag.lastX = x;
    drag.lastY = y;
  });
  frame.addEventListener("pointerup", async (event) => {
    if (!drag) return;
    const { moved } = drag;
    drag = null;
    frame.classList.remove("dragging");
    if (moved) return;
    const [x, y] = local(event);
    const found = await projection.nearest({ ...camera, center: [...camera.center] }, x, y, 1);
    const [nodeX, nodeY] = found.nodes.length > 0 ? screenOf(found.positions) : [Infinity, Infinity];
    const near = Math.hypot(nodeX - x, nodeY - y) <= pickRadiusPixels;
    if (near) await select(found.nodes[0]);
    else {
      selection = null;
      if (!hubsShown) showSelection();
    }
  });
  frame.addEventListener("pointercancel", () => {
    drag = null;
    frame.classList.remove("dragging");
  });
  frame.addEventListener("dblclick", async (event) => {
    const [x, y] = local(event);
    const found = await projection.nearest({ ...camera, center: [...camera.center] }, x, y, 1);
    const ordinal = found.nodes[0];
    if (ordinal !== undefined && pinned.has(ordinal)) {
      pinned.delete(ordinal);
      projection.release(Uint32Array.of(ordinal));
      pinToPlane([ordinal]);
      layoutEpochsLeft = layoutEpochsAfterChange;
      dirty = true;
      return;
    }
    userMoved = false;
    Object.assign(camera, { yaw: 0, pitch: 0 });
    await fit();
    cameraChanged();
  });

  elements.toggles.edges.addEventListener("click", () => {
    style.edges = style.edges > 0 ? 0 : defaultEdges;
    elements.toggles.edges.textContent = style.edges > 0 ? "on" : "off";
    dirty = true;
  });
  elements.toggles.hubs.addEventListener("click", () => {
    hubsShown = !hubsShown;
    elements.toggles.hubs.textContent = hubsShown ? "on" : "off";
    if (hubsShown) showHubs();
    else showSelection();
  });
  elements.toggles.surface.addEventListener("click", () => {
    const onPaper = frame.classList.toggle("ink-on-paper");
    frame.classList.toggle("dark-surface", !onPaper);
    elements.toggles.surface.textContent = onPaper ? "ink on paper" : "poster";
    readTokens();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", readTokens);
  new ResizeObserver(() => (dirty = true)).observe(canvas);
  elements.toggles.edges.textContent = style.edges > 0 ? "on" : "off";

  function updateReadouts() {
    const { readouts } = elements;
    readouts.nodes.textContent = integerFormat.format(nodes);
    readouts.edges.textContent = integerFormat.format(edges);
    readouts.layout.textContent = `${projection.timings.layoutMilliseconds.toFixed(2)} ms`;
    readouts.draw.textContent = `${projection.timings.drawMilliseconds.toFixed(2)} ms`;
    const last = channel.last;
    if (!last) return;
    readouts.call.textContent = `${last.verb.replace(/^usearch_/, "")} ${last.milliseconds.toFixed(1)} ms`;
    readouts.traversal.textContent = last.stats
      ? `${integerFormat.format(last.stats.visited_members)} / ${integerFormat.format(last.stats.computed_distances)}`
      : "–";
    readouts.outcome.textContent = last.message ? `${last.statusName}: ${last.message}` : last.statusName;
  }

  let frameCount = 0;
  function tick() {
    requestAnimationFrame(tick);
    frameCount++;
    const now = performance.now();
    if (!settledHandled && now - lastCameraChange > settleMilliseconds) {
      settledHandled = true;
      zoomTick();
      refreshGlyphs();
    }
    sampleTick();
    const layouting = layoutEpochsLeft > 0;
    if (layouting && frameCount % 4 === 0) refreshPositions().then(() => (dirty = true));
    if (frameCount % 15 === 0) updateReadouts();
    if (!dirty && !layouting) return;
    projection.frame(camera, style, layouting ? layoutBudgetMilliseconds : 0);
    if (layouting) layoutEpochsLeft -= Math.max(1, projection.timings.epochsPerFrame);
    drawOverlay();
    dirty = false;
  }

  readTokens();
  showStatus("");
  requestAnimationFrame(tick);

  return {
    projection,
    camera,
    style,
    latencies,
    get nodes() {
      return nodes;
    },
    get edges() {
      return edges;
    },
    get sampling() {
      return sampling || [...zooms, overview].some((sequence) => !sequence.exhausted && sequence.remaining > 0);
    },
    samplingTotals,
    get zoomSequences() {
      return zooms.length;
    },
    get hover() {
      return hover
        ? {
            node: hover.node,
            hops: hover.hops.length,
            drawn: hover.hops.filter((hop) => hop.ordinal !== undefined).length,
          }
        : null;
    },
    get selection() {
      return selection;
    },
    get glyphs() {
      return glyphs.length;
    },
    pinned,
    fit,
    zoomAround,
    screenOf,
    labelOf,
    /** Screen points of the given nodes, for drivers. @param {Uint32Array<ArrayBuffer>} ordinals */
    async screenPoints(ordinals) {
      const found = await projection.positionsOf(ordinals);
      return Array.from(ordinals, (_, index) => screenOf(found.subarray(index * 3, index * 3 + 3)));
    },
  };
}
