/**
 *  @file explorer/modalities.js
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief What differs between modalities, and nothing else: how a close-up draws one entry, how a development index
 *      fills from local data, and the arguments a search needs beyond the shared ones. Every flow - samples, zooms,
 *      hover arcs, hubs, selection and landmarks - stays shared in `explorer.js`, and every structure in `worker.js`.
 */

/** Rows of equal width, as `usearch_rows_view_t` borrows them. @typedef {{count: number, bytes: Uint8Array, scalar: string, stride: number, columns: number}} Rows */
/** Entries end to end around `count + 1` element offsets starting at zero, each element `columns` scalars wide, as `usearch_tape_view_t` borrows them. @typedef {{count: number, bytes: Uint8Array, scalar: string, offsets: Uint32Array, columns: number}} Tape */
/** One C structure per entry, by member name. @typedef {{count: number, records: Record<string, number>[]}} Records */
/** @typedef {Rows | Tape | Records} Payload */
/** @typedef {{ink: string, muted: string, paper: string, accent: string, danger: string, partial: string}} Palette */
/** @typedef {{label: string, payload: Payload | null, palette: Palette}} GlyphItem */
/** A development index: its `init_ram` arguments, and entries under the integer identities of their keys. @typedef {{options: Record<string, any>, identities: Uint32Array, payload: Payload}} Dataset */
/**
 *  @typedef {{
 *    name: string,
 *    tapeColumns?: number,
 *    search?: Record<string, string>,
 *    develop: (parameters: URLSearchParams, report: (message: string) => void) => Promise<Dataset>,
 *    place?: (payload: Payload, entry: number) => [number, number, number],
 *    scenery?: () => number[][][],
 *    glyphSize: [number, number],
 *    glyph: (context: CanvasRenderingContext2D, item: GlyphItem, x: number, y: number) => void
 *  }} Modality
 */

const posterSide = 64;
const posterGap = 8;
const lineHeight = 15;
// Heavy-atom bonds run 1.2 to 1.9 Å, while atoms two bonds apart sit at least 2.3 Å away.
const bondAngstroms = 1.95;

/** @param {string} text @param {number} limit */
function shortened(text, limit) {
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}

/** `text` broken at spaces into lines of up to `width` characters, at most `limit` of them, the last shortened. */
function wrapped(/** @type {string} */ text, /** @type {number} */ width, /** @type {number} */ limit) {
  /** @type {string[]} */
  const lines = [];
  for (const word of text.split(/\s+/).filter(Boolean)) {
    const last = lines.length - 1;
    if (last >= 0 && lines[last].length + 1 + word.length <= width) lines[last] += ` ${word}`;
    else lines.push(word);
  }
  if (lines.length <= limit) return lines.map((line) => shortened(line, width));
  return [...lines.slice(0, limit - 1), shortened(lines.slice(limit - 1).join(" "), width)];
}

/** @param {string} path @param {URLSearchParams} parameters */
function sourceOf(path, parameters) {
  return new URL(parameters.get("source") ?? path, location.href);
}

/** @param {URL} url @param {number} first @param {number} last */
async function bytesOf(url, first, last) {
  return new Uint8Array(await (await fetch(url, { headers: { Range: `bytes=${first}-${last}` } })).arrayBuffer());
}

/** @param {number} count */
const identitiesUpTo = (count) => Uint32Array.from({ length: count }, (_, identity) => identity);

/** Mono text left-aligned at `x` and centred on `y`, over a translucent paper backdrop so it reads over dense ink. */
function caption(
  /** @type {CanvasRenderingContext2D} */ context,
  /** @type {GlyphItem} */ item,
  /** @type {string} */ text,
  /** @type {number} */ x,
  /** @type {number} */ y,
) {
  context.globalAlpha = 0.8;
  context.fillStyle = item.palette.paper;
  context.fillRect(x - 3, y - 8, context.measureText(text).width + 6, 15);
  context.globalAlpha = 1;
  context.fillStyle = item.palette.ink;
  context.fillText(text, x, y + 4);
}

/** A black poster square right of `x`, centred on `y`, with `draw` given its centre and half side, then `text` beside it. */
function poster(
  /** @type {CanvasRenderingContext2D} */ context,
  /** @type {GlyphItem} */ item,
  /** @type {string} */ text,
  /** @type {number} */ x,
  /** @type {number} */ y,
  /** @type {(centerX: number, centerY: number, half: number) => void} */ draw,
) {
  context.fillStyle = "#000";
  context.fillRect(x + posterGap, y - posterSide / 2, posterSide, posterSide);
  draw(x + posterGap + posterSide / 2, y, posterSide / 2 - 6);
  caption(context, item, text, x + 2 * posterGap + posterSide, y);
}

/** @type {Modality} */
export const dense = {
  name: "dense",

  /** The first `entries` vectors of a `.u8bin` file - SIFT by default - under Euclidean distance. */
  async develop(parameters, report) {
    const source = sourceOf("../data/sift-10m/base.10M.u8bin", parameters);
    const entries = Number(parameters.get("entries") ?? 20000);
    const [, dimensions] = new Uint32Array((await bytesOf(source, 0, 7)).buffer);
    report(`Reading ${entries.toLocaleString("en-US")} vectors of ${dimensions} bytes…`);
    return {
      options: {
        metric: "usearch_dense_metric_euclidean_k",
        quantization: "usearch_scalar_u8_k",
        dimensions,
        connectivity: 16,
      },
      identities: identitiesUpTo(entries),
      payload: {
        count: entries,
        bytes: await bytesOf(source, 8, 7 + entries * dimensions),
        scalar: "usearch_scalar_u8_k",
        stride: dimensions,
        columns: dimensions,
      },
    };
  },

  glyphSize: [280, 16],

  /** The title the dataset's sidecar names, or the key. */
  glyph(context, item, x, y) {
    caption(context, item, shortened(item.label, 40), x + posterGap, y);
  },
};

const proteinPattern = /^[ACDEFGHIKLMNPQRSTVWY]{8,}$/;
const residueClasses = /** @type {Record<string, keyof Palette>} */ ({
  K: "accent",
  R: "accent",
  H: "accent",
  D: "danger",
  E: "danger",
  G: "partial",
  P: "partial",
  C: "partial",
  S: "muted",
  T: "muted",
  N: "muted",
  Q: "muted",
  Y: "muted",
});

/** @type {Modality} */
export const sequence = {
  name: "sequence",

  /** `entries` sentences of 24 to 80 bytes from a text file sorted alphabetically - Leipzig by default - drawn from chunks
   *  spread over the whole file, under byte-level Levenshtein distance. */
  async develop(parameters, report) {
    const source = sourceOf("../data/leipzig-1m/leipzig1M_en.txt", parameters);
    const entries = Number(parameters.get("entries") ?? 5000);
    const size = Number(
      ((await fetch(source, { headers: { Range: "bytes=0-0" } })).headers.get("Content-Range") ?? "/0").split("/")[1],
    );
    report(`Reading ${entries.toLocaleString("en-US")} sentences…`);
    const chunks = 64;
    const chunkBytes = 96 << 10;
    /** @type {string[]} */
    const candidates = [];
    for (let chunk = 0; chunk < chunks; chunk++) {
      const first = Math.floor((chunk * (size - chunkBytes)) / chunks);
      const text = new TextDecoder().decode(await bytesOf(source, first, first + chunkBytes - 1));
      candidates.push(
        ...text
          .split("\n")
          .slice(1, -1)
          .filter((line) => line.length >= 24 && line.length <= 80),
      );
    }
    const stride = Math.max(1, candidates.length / entries);
    const encoder = new TextEncoder();
    const encoded = Array.from({ length: Math.min(entries, Math.floor(candidates.length / stride)) }, (_, entry) =>
      encoder.encode(candidates[Math.floor(entry * stride)]),
    );
    const offsets = new Uint32Array(encoded.length + 1);
    encoded.forEach((bytes, entry) => (offsets[entry + 1] = offsets[entry] + bytes.length));
    const bytes = new Uint8Array(offsets[encoded.length]);
    encoded.forEach((sentence, entry) => bytes.set(sentence, offsets[entry]));
    return {
      options: { metric: "usearch_sequence_metric_levenshtein_bytes_k", connectivity: 16 },
      identities: identitiesUpTo(encoded.length),
      payload: { count: encoded.length, bytes, scalar: "usearch_scalar_u8_k", offsets, columns: 1 },
    };
  },

  glyphSize: [300, 3 * lineHeight],

  /** The string itself wrapped over up to three lines, or one coloured cell per residue when the index holds proteins. */
  glyph(context, item, x, y) {
    const tape = /** @type {Tape | null} */ (item.payload);
    const text = tape?.count
      ? new TextDecoder().decode(tape.bytes.subarray(tape.offsets[0], tape.offsets[1]))
      : item.label;
    if (!proteinPattern.test(text)) {
      const lines = wrapped(text, 40, 3);
      lines.forEach((line, index) =>
        caption(context, item, line, x + posterGap, y + (index - (lines.length - 1) / 2) * lineHeight),
      );
      return;
    }
    const cell = 4;
    const shown = text.slice(0, 64);
    context.fillStyle = item.palette.paper;
    context.fillRect(x + posterGap - 3, y - 5, shown.length * cell + 6, 10);
    for (let residue = 0; residue < shown.length; residue++) {
      context.fillStyle = item.palette[residueClasses[shown[residue]] ?? "ink"];
      context.fillRect(x + posterGap + residue * cell, y - cell / 2, cell - 1, cell);
    }
  },
};

/** The unit axis of widest spread among centred `points`, orthogonal to `away` when given, by power iteration. */
function widestAxis(/** @type {Float32Array} */ points, /** @type {number[] | null} */ away) {
  let axis = away ? [away[1], -away[0], 0.5] : [1, 0.5, 0.25];
  for (let iteration = 0; iteration < 24; iteration++) {
    const next = [0, 0, 0];
    for (let point = 0; point < points.length; point += 3) {
      const along = points[point] * axis[0] + points[point + 1] * axis[1] + points[point + 2] * axis[2];
      for (let coordinate = 0; coordinate < 3; coordinate++) next[coordinate] += along * points[point + coordinate];
    }
    if (away) {
      const shared = next[0] * away[0] + next[1] * away[1] + next[2] * away[2];
      for (let coordinate = 0; coordinate < 3; coordinate++) next[coordinate] -= shared * away[coordinate];
    }
    const norm = Math.hypot(...next) || 1;
    axis = next.map((value) => value / norm);
  }
  return axis;
}

/** @type {Modality} */
export const shapes = {
  name: "shapes",
  tapeColumns: 3,

  /** The first `entries` conformers `prepare.py molecules` wrote, each molecule's under one key, under Chamfer distance. */
  async develop(parameters, report) {
    const source = sourceOf("../data/usearch-molecules-explorer/conformers.bin", parameters);
    const file = new Uint8Array(await (await fetch(source)).arrayBuffer());
    const words = new Uint32Array(file.buffer, 0, 1);
    const stored = words[0];
    const entries = Math.min(stored, Number(parameters.get("entries") ?? 600));
    report(`Read ${entries.toLocaleString("en-US")} conformers…`);
    const identities = new Uint32Array(file.buffer, 4, entries).slice();
    const offsets = new Uint32Array(file.buffer, 4 + stored * 4, entries + 1).slice();
    const points = 4 + stored * 4 + (stored + 1) * 4;
    return {
      options: {
        metric: "usearch_shapes_metric_chamfer_k",
        quantization: "usearch_scalar_f32_k",
        connectivity: 16,
        values_per_key: 3,
      },
      identities,
      payload: {
        count: entries,
        bytes: file.slice(points, points + offsets[entries] * 12),
        scalar: "usearch_scalar_f32_k",
        offsets,
        columns: 3,
      },
    };
  },

  glyphSize: [300, 64],

  /** The shape projected orthographically onto its two widest axes in a poster square - points closer than a bond joined
   *  into a skeleton - and its title beside it. */
  glyph(context, item, x, y) {
    const tape = /** @type {Tape | null} */ (item.payload);
    if (!tape?.count) return dense.glyph(context, item, x, y);
    const floats = new Float32Array(tape.bytes.buffer, tape.bytes.byteOffset, tape.bytes.byteLength / 4);
    const points = floats.slice(tape.offsets[0] * 3, tape.offsets[1] * 3);
    const centre = [0, 0, 0];
    for (let position = 0; position < points.length; position++)
      centre[position % 3] += (points[position] * 3) / points.length;
    for (let position = 0; position < points.length; position++) points[position] -= centre[position % 3];
    const first = widestAxis(points, null);
    const second = widestAxis(points, first);
    const projected = [];
    let extent = 1e-6;
    for (let point = 0; point < points.length; point += 3) {
      const u = points[point] * first[0] + points[point + 1] * first[1] + points[point + 2] * first[2];
      const v = points[point] * second[0] + points[point + 1] * second[1] + points[point + 2] * second[2];
      projected.push([u, v]);
      extent = Math.max(extent, Math.abs(u), Math.abs(v));
    }
    poster(context, item, shortened(item.label, 26), x, y, (centerX, centerY, half) => {
      const screen = projected.map(([u, v]) => [centerX + (u / extent) * half, centerY - (v / extent) * half]);
      context.strokeStyle = "#fff";
      context.lineWidth = 1.25;
      context.beginPath();
      for (let first = 0; first < screen.length; first++)
        for (let second = first + 1; second < screen.length; second++) {
          const offsets = [0, 1, 2].map(
            (coordinate) => points[first * 3 + coordinate] - points[second * 3 + coordinate],
          );
          if (offsets[0] ** 2 + offsets[1] ** 2 + offsets[2] ** 2 > bondAngstroms ** 2) continue;
          context.moveTo(screen[first][0], screen[first][1]);
          context.lineTo(screen[second][0], screen[second][1]);
        }
      context.stroke();
      context.lineWidth = 1;
      context.fillStyle = "#fff";
      for (const [left, top] of screen) context.fillRect(left - 1.5, top - 1.5, 3, 3);
    });
  },
};

/** Mulberry32, so the synthetic fixes repeat exactly. @param {number} seed */
function randomFrom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

const metresPerDegree = 111320;
const earthKilometres = 6378.137;

/** @param {number} latitude @param {number} longitude in degrees @returns {[number, number, number]} */
function onGlobe(latitude, longitude) {
  const [north, east] = [(latitude * Math.PI) / 180, (longitude * Math.PI) / 180];
  return [
    earthKilometres * Math.cos(north) * Math.sin(east),
    earthKilometres * Math.sin(north),
    earthKilometres * Math.cos(north) * Math.cos(east),
  ];
}
const fixesPerVehicle = 8;
const hubs = [
  [51.5074, -0.1278],
  [48.8566, 2.3522],
  [52.52, 13.405],
  [40.4168, -3.7038],
  [45.4642, 9.19],
  [50.0755, 14.4378],
];

/** @type {Modality} */
export const geospatial = {
  name: "geospatial",
  search: { relation: "usearch_geospatial_relation_near_k" },
  glyphSize: [300, 64],

  /** On a sphere of the Earth's radius in kilometres, x east and y north at the meridian, so the globe reads north-up. */
  place(payload, entry) {
    const record = /** @type {Records} */ (payload).records[entry];
    return onGlobe(record.latitude_degrees, record.longitude_degrees);
  },

  /** A graticule every 15 degrees. */
  scenery() {
    const steps = Array.from({ length: 121 }, (_, step) => step * 3 - 180);
    const parallels = [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75].map((latitude) =>
      steps.map((longitude) => onGlobe(latitude, longitude)),
    );
    const meridians = steps
      .filter((longitude) => longitude % 15 === 0 && longitude > -180)
      .map((longitude) =>
        steps.filter((latitude) => Math.abs(latitude) <= 90).map((latitude) => onGlobe(latitude, longitude)),
      );
    return [...parallels, ...meridians];
  },

  /** `entries` synthetic fixes, since no geotagged corpus ships with the repository: vehicles leave six European city
   *  centres on winding tracks, one fix every 50 m looking along the way, each vehicle's fixes under its own key. */
  async develop(parameters) {
    const entries = Number(parameters.get("entries") ?? 8000);
    const random = randomFrom(7);
    /** @type {Records['records']} */
    const records = [];
    const identities = new Uint32Array(entries);
    for (let vehicle = 0; records.length < entries; vehicle++) {
      const [latitude, longitude] = hubs[vehicle % hubs.length];
      let north = latitude + (random() - 0.5) * 0.2;
      let east = longitude + (random() - 0.5) * 0.3;
      let heading = random() * 360;
      for (let fix = 0; fix < fixesPerVehicle && records.length < entries; fix++) {
        heading = (heading + (random() - 0.5) * 40 + 360) % 360;
        north += (50 * Math.cos((heading * Math.PI) / 180)) / metresPerDegree;
        east += (50 * Math.sin((heading * Math.PI) / 180)) / (metresPerDegree * Math.cos((north * Math.PI) / 180));
        const time = 1.7e9 + vehicle * 600 + fix * 4;
        identities[records.length] = vehicle;
        records.push({
          latitude_degrees: north,
          longitude_degrees: east,
          altitude_metres: NaN,
          accuracy_metres: 5,
          heading_degrees: heading,
          pitch_degrees: NaN,
          heading_accuracy_degrees: 5,
          field_of_view_degrees: 70,
          range_metres: 120,
          begin_seconds: time,
          end_seconds: time + 1,
        });
      }
    }
    return { options: { values_per_key: fixesPerVehicle }, identities, payload: { count: entries, records } };
  },

  /** The view cone, north up, in a poster square, and the position beside it. */
  glyph(context, item, x, y) {
    const record = /** @type {Records | null} */ (item.payload)?.records[0];
    if (!record) return dense.glyph(context, item, x, y);
    const field = Number.isNaN(record.field_of_view_degrees) ? 360 : record.field_of_view_degrees;
    const heading = Number.isNaN(record.heading_degrees) ? 0 : record.heading_degrees;
    const north = record.latitude_degrees;
    const east = record.longitude_degrees;
    const position = `${Math.abs(north).toFixed(4)}° ${north < 0 ? "S" : "N"} ${Math.abs(east).toFixed(4)}° ${east < 0 ? "W" : "E"}`;
    poster(context, item, position, x, y, (centerX, centerY, half) => {
      const radians = (/** @type {number} */ degrees) => ((degrees - 90) * Math.PI) / 180;
      context.fillStyle = item.palette.accent;
      context.globalAlpha = 0.6;
      context.beginPath();
      context.moveTo(centerX, centerY);
      context.arc(centerX, centerY, half, radians(heading - field / 2), radians(heading + field / 2));
      context.closePath();
      context.fill();
      context.globalAlpha = 1;
      context.fillStyle = "#fff";
      context.fillRect(centerX - 1, centerY - half, 2, 4);
      context.fillRect(centerX - 1.5, centerY - 1.5, 3, 3);
    });
  },
};

/** @type {Record<string, Modality>} */
export const modalities = { dense, sequence, shapes, geospatial };
