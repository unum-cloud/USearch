/**
 *  @file explorer/projection.js
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The GPU projection core: a spring-electrical layout of a growing graph, relaxed and drawn every frame.
 *
 *  It knows ordinals, distances, positions, the camera and the style, and nothing about keys, payloads or modalities.
 *  The layout is the spring-electrical family of Eades and of Fruchterman and Reingold, with the charges estimated by
 *  negative sampling as in LargeVis, and `balance` placing it on the attraction-repulsion spectrum of Böhm, Berens and
 *  Kobak (JMLR 2022). Both forces use the Cauchy kernel `q = 1 / (1 + d²)`.
 *
 *  The layout lives in three dimensions and the camera looks at it orthographically. A camera is `{center, scale, yaw,
 *  pitch, radius}`: `center` in layout units, `scale` in CSS pixels per unit, `yaw` turning about the vertical axis and
 *  `pitch` about the horizontal one, in radians, and `radius` the layout's extent around `center`, over which far ink
 *  fades. Only nodes within half the view's larger side of `center` along the line of sight are drawn, so zooming in
 *  narrows the depth seen as well as the width.
 */

export const emptySlot = 0xffffffff;

/** @typedef {[number, number, number]} Vector */
/** @typedef {{center: Vector, scale: number, yaw: number, pitch: number, radius: number}} Camera */
/** @typedef {{ink: number[], paper: number[], accent: number[], edges: number}} Style */
/** @typedef {{layoutMilliseconds: number, drawMilliseconds: number, epochsPerFrame: number}} Timings */

const workgroupSize = 256;
const workgroupsLimit = 65535;
// Few enough that a settling layout glides across frames rather than jumping within one.
const epochsPerFrameLimit = 4;
const uniformStride = 256;
// Forces accumulate as i32 at 1/65536 resolution: one edge adds at most half a unit, so 65536 edges fit one node.
const fixedPointScale = 65536;
const candidatesLimit = 4096;
const histogramBuckets = 256;
const statisticCellPixels = 8;
const layoutSeed = 0x2962ff;

// Strong enough repulsion to resolve hubs in SIFT and Leipzig samples, weak enough to keep a manifold whole.
const negatives = 6;
const balance = 40;
const epsilon = 0.001;
// New nodes spawn beside their nearest earlier draw, so a small first step suffices, and a settled node's step is small
// enough that the noise of its sampled charges stays below a pixel.
const stepInitial = 0.25;
const stepFinal = 0.001;
const settleEpochs = 400;
const forceClip = 2;
// A pin or release restarts every node's step from this age, so the rest re-sorts gently rather than anew.
const reheatAge = 300;
const spawnRadius = 0.4;
// Optical depth at the ink-weighted mean density, so the typical point renders at about a third of full ink.
const typicalInkDepth = 0.4;
// Highlights are discs this many CSS pixels across, since a lone tinted pixel vanishes at any density.
const highlightPixels = 7;
const highlightsLimit = 65536;
// Edges longer than this many device pixels spread a fixed amount of ink along their length instead of more.
const edgeInkPixels = 40;
// Shorter edges lie inside the ink of their own two points, so drawing them only costs blending.
const edgeShortestPixels = 3;
// Ink kept at the far side of the depth drawn, so the near side reads in front.
const farInk = 0.3;

/** The rows of the rotation taking layout offsets from `camera.center` into view space. @param {Camera} camera */
export function rotationOf(camera) {
  const [yawCosine, yawSine] = [Math.cos(camera.yaw), Math.sin(camera.yaw)];
  const [pitchCosine, pitchSine] = [Math.cos(camera.pitch), Math.sin(camera.pitch)];
  return [
    [yawCosine, 0, yawSine],
    [pitchSine * yawSine, pitchCosine, -pitchSine * yawCosine],
    [-pitchCosine * yawSine, pitchSine, pitchCosine * yawCosine],
  ];
}

/**
 *  Where `position` lands for `camera` on a canvas `width` × `height` CSS pixels: `[x, y, depth]`, the depth in layout
 *  units toward the viewer.
 *  @param {Camera} camera @param {number} width @param {number} height @param {ArrayLike<number>} position @returns {Vector}
 */
export function project(camera, width, height, position) {
  const offset = [0, 1, 2].map((axis) => position[axis] - camera.center[axis]);
  const [x, y, depth] = rotationOf(camera).map((row) => row[0] * offset[0] + row[1] * offset[1] + row[2] * offset[2]);
  return [width / 2 + x * camera.scale, height / 2 - y * camera.scale, depth];
}

/** The layout offset a screen offset of `right` and `down` CSS pixels spans, in the view's plane. @param {Camera} camera @param {number} right @param {number} down @returns {Vector} */
export function alongView(camera, right, down) {
  const [first, second] = rotationOf(camera);
  const [x, y] = [right / camera.scale, -down / camera.scale];
  return [first[0] * x + second[0] * y, first[1] * x + second[1] * y, first[2] * x + second[2] * y];
}

const sharedShader = /* wgsl */ `
  const pinned_bit = 0x80000000u;

  fn mix_bits(value: u32) -> u32 {
    var bits = value;
    bits ^= bits >> 16u;
    bits *= 0x7feb352du;
    bits ^= bits >> 15u;
    bits *= 0x846ca68bu;
    bits ^= bits >> 16u;
    return bits;
  }
`;

const layoutShader = /* wgsl */ `
  ${sharedShader}

  struct Settings {
    nodes: u32,
    edges: u32,
    epoch: u32,
    seed: u32,
    negatives: u32,
    reheat_epoch: u32,
    settle_epochs: f32,
    reheat_age: f32,
    balance: f32,
    epsilon: f32,
    step_initial: f32,
    step_final: f32,
    clip: f32,
    fixed_scale: f32,
  }

  @group(0) @binding(0) var<uniform> settings: Settings;
  @group(0) @binding(1) var<storage, read> source_positions: array<vec4f>;
  @group(0) @binding(2) var<storage, read_write> target_positions: array<vec4f>;
  @group(0) @binding(3) var<storage, read_write> forces: array<atomic<i32>>;
  @group(0) @binding(4) var<storage, read> states: array<u32>;
  @group(0) @binding(5) var<storage, read> edges: array<vec2u>;
  @group(0) @binding(6) var<storage, read> weights: array<f32>;

  fn random_node(epoch: u32, node: u32, sample: u32) -> u32 {
    return mix_bits(settings.seed ^ mix_bits(epoch ^ mix_bits(node ^ mix_bits(sample)))) % settings.nodes;
  }

  fn step_size(state: u32) -> f32 {
    let age = f32(settings.epoch - (state & ~pinned_bit));
    let reheated = f32(settings.epoch - settings.reheat_epoch) + settings.reheat_age;
    let progress = min(min(age, reheated) / settings.settle_epochs, 1.0);
    return mix(settings.step_initial, settings.step_final, progress);
  }

  @compute @workgroup_size(${workgroupSize})
  fn springs(@builtin(global_invocation_id) invocation: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let stride = groups.x * ${workgroupSize}u;
    for (var edge = invocation.x; edge < settings.edges; edge += stride) {
      let ends = edges[edge];
      let delta = source_positions[ends.y].xyz - source_positions[ends.x].xyz;
      let q = 1.0 / (1.0 + dot(delta, delta));
      let pull = vec3i(round(weights[edge] * q * delta * settings.fixed_scale));
      for (var axis = 0u; axis < 3u; axis++) {
        atomicAdd(&forces[4u * ends.x + axis], pull[axis]);
        atomicSub(&forces[4u * ends.y + axis], pull[axis]);
      }
    }
  }

  @compute @workgroup_size(${workgroupSize})
  fn charges(@builtin(global_invocation_id) invocation: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let stride = groups.x * ${workgroupSize}u;
    for (var node = invocation.x; node < settings.nodes; node += stride) {
      var force = vec3f(0.0);
      for (var axis = 0u; axis < 3u; axis++) { force[axis] = f32(atomicExchange(&forces[4u * node + axis], 0)); }
      force /= settings.fixed_scale;
      let state = states[node];
      let here = source_positions[node].xyz;
      if ((state & pinned_bit) != 0u) {
        target_positions[node] = vec4f(here, 0.0);
        continue;
      }
      for (var sample = 0u; sample < settings.negatives; sample++) {
        let delta = here - source_positions[random_node(settings.epoch, node, sample)].xyz;
        let distance_squared = dot(delta, delta);
        force += settings.balance * delta / ((settings.epsilon + distance_squared) * (1.0 + distance_squared));
      }
      target_positions[node] = vec4f(here + step_size(state) * clamp(force, vec3f(-settings.clip), vec3f(settings.clip)), 0.0);
    }
  }
`;

const spawnShader = /* wgsl */ `
  struct Spawn {
    first: u32,
    count: u32,
    radius: f32,
    origin: vec4f,
  }

  @group(0) @binding(0) var<uniform> spawn: Spawn;
  @group(0) @binding(1) var<storage, read> roots: array<vec2u>;
  @group(0) @binding(2) var<storage, read_write> positions: array<vec4f>;

  @compute @workgroup_size(${workgroupSize})
  fn spawn_nodes(@builtin(global_invocation_id) invocation: vec3u) {
    if (invocation.x >= spawn.count) { return; }
    let root = roots[invocation.x];
    var base = spawn.origin.xyz;
    if (root.x != ${emptySlot}u) { base = positions[root.x].xyz; }
    let node = spawn.first + invocation.x;
    // Golden-angle turns over golden-ratio heights spread siblings of one parent over a sphere without random numbers.
    let angle = f32(node) * 2.39996323;
    let height = fract(f32(node) * 0.61803399) * 2.0 - 1.0;
    let around = sqrt(1.0 - height * height);
    positions[node] = vec4f(base + spawn.radius * sqrt(f32(root.y)) * vec3f(around * cos(angle), height, around * sin(angle)), 0.0);
  }
`;

const densityShader = /* wgsl */ `
  struct View {
    rotation: array<vec4f, 3>,
    center: vec4f,
    clip_scale: vec2f,
    pixels: vec2f,
    edge_ink: f32,
    edge_ink_pixels: f32,
    highlight_pixels: f32,
    depth: f32,
  }

  struct Highlight {
    node: u32,
    strength: f32,
  }

  @group(0) @binding(0) var<uniform> view: View;
  @group(0) @binding(1) var<storage, read> positions: array<vec4f>;
  @group(0) @binding(2) var<storage, read> highlights: array<Highlight>;
  @group(0) @binding(3) var<storage, read> edges: array<vec2u>;
  @group(0) @binding(4) var<storage, read> weights: array<f32>;
  @group(0) @binding(5) var<storage, read> statistic: Statistic;

  struct Statistic {
    contribution: f32,
    cells: u32,
    spread: f32,
  }

  // A node in view space: x and y in clip units, z its depth toward the viewer in layout units.
  fn viewed(node: u32) -> vec3f {
    let offset = positions[node].xyz - view.center.xyz;
    let rotated = vec3f(dot(view.rotation[0].xyz, offset), dot(view.rotation[1].xyz, offset), dot(view.rotation[2].xyz, offset));
    return vec3f(rotated.xy * view.clip_scale, rotated.z);
  }

  // How much ink a node keeps at a depth: all of it at the near edge of the depth drawn, the far share at the far one.
  fn fog(depth: f32) -> f32 {
    return mix(${farInk}, 1.0, clamp(0.5 + 0.5 * depth / view.depth, 0.0, 1.0));
  }

  // Outside the viewport or past the depth drawn, so the rasterizer drops it before any blending.
  fn hidden(at: vec3f) -> bool {
    return any(abs(at.xy) > vec2f(1.0)) || abs(at.z) > view.center.w;
  }

  struct Point {
    @builtin(position) position: vec4f,
    @location(0) ink: f32,
  }

  @vertex fn point_vertex(@builtin(vertex_index) node: u32) -> Point {
    var point: Point;
    let at = viewed(node);
    point.position = select(vec4f(at.xy, 0.5, 1.0), vec4f(0.0, 0.0, 2.0, 1.0), hidden(at));
    point.ink = fog(at.z);
    return point;
  }

  @fragment fn point_fragment(point: Point) -> @location(0) vec4f {
    return vec4f(point.ink, 0.0, 0.0, 0.0);
  }

  struct Disc {
    @builtin(position) position: vec4f,
    @location(0) corner: vec2f,
    @location(1) strength: f32,
  }

  @vertex fn disc_vertex(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> Disc {
    var disc: Disc;
    let highlight = highlights[instance];
    let corners = array(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
    disc.corner = corners[index];
    disc.strength = highlight.strength;
    let at = viewed(highlight.node);
    disc.position = select(vec4f(at.xy + disc.corner * view.highlight_pixels / view.pixels, 0.5, 1.0), vec4f(0.0, 0.0, 2.0, 1.0), hidden(at));
    return disc;
  }

  @fragment fn disc_fragment(disc: Disc) -> @location(0) vec4f {
    let coverage = 1.0 - smoothstep(0.6, 1.0, length(disc.corner));
    return vec4f(0.0, 0.0, disc.strength * coverage, 0.0);
  }

  struct Line {
    @builtin(position) position: vec4f,
    @location(0) ink: f32,
  }

  @vertex fn line_vertex(@builtin(vertex_index) index: u32) -> Line {
    var line: Line;
    let edge = index >> 1u;
    let ends = edges[edge];
    let from_at = viewed(ends.x);
    let to_at = viewed(ends.y);
    let from_clip = from_at.xy;
    let to_clip = to_at.xy;
    let length_pixels = length((to_clip - from_clip) * view.pixels * 0.5);
    let outside = any(min(from_clip, to_clip) > vec2f(1.0)) || any(max(from_clip, to_clip) < vec2f(-1.0)) ||
      abs(from_at.z) > view.center.w || abs(to_at.z) > view.center.w;
    line.ink = view.edge_ink * weights[edge] * min(1.0, view.edge_ink_pixels / max(length_pixels, 1.0)) * fog(0.5 * (from_at.z + to_at.z));
    // The previous frame's contribution tells which lines would tone below one 8-bit step; those only cost blending.
    let faint = line.ink * statistic.contribution < ${1 / 256};
    if (outside || faint || length_pixels < ${edgeShortestPixels}.0) {
      line.position = vec4f(0.0, 0.0, 2.0, 1.0);
      return line;
    }
    let clip = select(from_clip, to_clip, (index & 1u) == 1u);
    line.position = vec4f(clip, 0.5, 1.0);
    return line;
  }

  @fragment fn line_fragment(line: Line) -> @location(0) vec4f {
    return vec4f(0.0, line.ink, 0.0, 0.0);
  }
`;

const statisticShader = /* wgsl */ `
  struct Statistic {
    contribution: f32,
    cells: u32,
    spread: f32,
  }

  @group(0) @binding(0) var density: texture_2d<f32>;
  @group(0) @binding(1) var<storage, read_write> partials: array<vec2f>;
  @group(0) @binding(2) var<storage, read_write> statistic: Statistic;

  var<workgroup> sums: array<vec2f, 64>;

  @compute @workgroup_size(8, 8)
  fn cells(
    @builtin(global_invocation_id) invocation: vec3u,
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u
  ) {
    let size = textureDimensions(density);
    var count = 0.0;
    for (var row = 0u; row < ${statisticCellPixels}u; row++) {
      for (var column = 0u; column < ${statisticCellPixels}u; column++) {
        let pixel = invocation.xy * ${statisticCellPixels}u + vec2u(column, row);
        if (all(pixel < size)) { count += textureLoad(density, pixel, 0).r; }
      }
    }
    sums[lane] = vec2f(count, count * count);
    workgroupBarrier();
    for (var half = 32u; half > 0u; half >>= 1u) {
      if (lane < half) { sums[lane] += sums[lane + half]; }
      workgroupBarrier();
    }
    if (lane == 0u) { partials[group.y * groups.x + group.x] = sums[0]; }
  }

  var<workgroup> totals: array<vec2f, ${workgroupSize}>;

  @compute @workgroup_size(${workgroupSize})
  fn reduce(@builtin(local_invocation_index) lane: u32) {
    var total = vec2f(0.0);
    for (var cell = lane; cell < statistic.cells; cell += ${workgroupSize}u) { total += partials[cell]; }
    totals[lane] = total;
    workgroupBarrier();
    for (var half = ${workgroupSize / 2}u; half > 0u; half >>= 1u) {
      if (lane < half) { totals[lane] += totals[lane + half]; }
      workgroupBarrier();
    }
    if (lane == 0u) {
      let sum = totals[0];
      let typical_density = select(1.0, sum.y / sum.x / ${statisticCellPixels * statisticCellPixels}.0, sum.x > 0.0);
      statistic.contribution = min(1.0, ${typicalInkDepth} / typical_density);
      // Views sparser than the typical depth widen each point into a soft bead, fully so at an eighth of it.
      statistic.spread = clamp(log2(${typicalInkDepth} / typical_density) / 3.0, 0.0, 1.0);
    }
  }
`;

const toneShader = /* wgsl */ `
  struct Tone {
    paper: vec4f,
    ink: vec4f,
    accent: vec4f,
  }

  struct Statistic {
    contribution: f32,
    cells: u32,
    spread: f32,
  }

  @group(0) @binding(0) var density: texture_2d<f32>;
  @group(0) @binding(1) var<uniform> tone: Tone;
  @group(0) @binding(2) var<storage, read> statistic: Statistic;

  @vertex fn tone_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
  }

  @fragment fn tone_fragment(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
    let here = vec2i(pixel.xy);
    let sample = textureLoad(density, here, 0);
    let contribution = statistic.contribution;
    var points = sample.r;
    if (statistic.spread > 0.0) {
      let last = vec2i(textureDimensions(density)) - 1;
      var beads = 0.0;
      for (var row = -2; row <= 2; row++) {
        for (var column = -2; column <= 2; column++) {
          let weight = exp(-f32(row * row + column * column) / 2.88);
          beads += weight * textureLoad(density, clamp(here + vec2i(column, row), vec2i(0), last), 0).r;
        }
      }
      points = mix(points, beads, statistic.spread);
    }
    let depth = (points + sample.g) * contribution;
    let inked = mix(tone.paper, tone.ink, 1.0 - exp(-depth));
    let tint = 1.0 - exp(-2.0 * sample.b);
    return mix(inked, tone.accent, tint);
  }
`;

const nearestShader = /* wgsl */ `
  struct Query {
    rotation: array<vec4f, 3>,
    center: vec4f,
    pixels: vec2f,
    point: vec2f,
    scale: f32,
    nodes: u32,
    wanted: u32,
  }

  struct Control {
    threshold: f32,
    count: atomic<u32>,
  }

  struct Candidate {
    position: vec4f,
    node: u32,
    distance: f32,
  }

  @group(0) @binding(0) var<uniform> query: Query;
  @group(0) @binding(1) var<storage, read> positions: array<vec4f>;
  @group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>, ${histogramBuckets}>;
  @group(0) @binding(3) var<storage, read_write> control: Control;
  @group(0) @binding(4) var<storage, read_write> candidates: array<Candidate, ${candidatesLimit}>;

  // Screen distance from the query point, or -1 for a node off screen or past the depth drawn.
  fn screen_distance(node: u32) -> f32 {
    let offset = positions[node].xyz - query.center.xyz;
    let rotated = vec3f(dot(query.rotation[0].xyz, offset), dot(query.rotation[1].xyz, offset), dot(query.rotation[2].xyz, offset));
    let screen = vec2f(query.pixels.x * 0.5 + rotated.x * query.scale, query.pixels.y * 0.5 - rotated.y * query.scale);
    if (any(screen < vec2f(0.0)) || any(screen > query.pixels) || abs(rotated.z) > query.center.w) { return -1.0; }
    return distance(screen, query.point);
  }

  // Sixteen buckets per octave of distance, so a bucket holds few nodes even where they crowd.
  fn bucket_of(separation: f32) -> u32 {
    return min(u32(log2(1.0 + separation) * 16.0), ${histogramBuckets - 1}u);
  }

  @compute @workgroup_size(${workgroupSize})
  fn histogram_nodes(@builtin(global_invocation_id) invocation: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let stride = groups.x * ${workgroupSize}u;
    for (var node = invocation.x; node < query.nodes; node += stride) {
      let separation = screen_distance(node);
      if (separation >= 0.0) { atomicAdd(&histogram[bucket_of(separation)], 1u); }
    }
  }

  @compute @workgroup_size(1)
  fn pick_threshold() {
    var total = 0u;
    var bucket = 0u;
    for (; bucket < ${histogramBuckets - 1}u; bucket++) {
      total += atomicLoad(&histogram[bucket]);
      if (total >= query.wanted) { break; }
    }
    control.threshold = exp2(f32(bucket + 1u) / 16.0) - 1.0;
  }

  @compute @workgroup_size(${workgroupSize})
  fn collect_nodes(@builtin(global_invocation_id) invocation: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let stride = groups.x * ${workgroupSize}u;
    for (var node = invocation.x; node < query.nodes; node += stride) {
      let separation = screen_distance(node);
      if (separation < 0.0 || separation > control.threshold) { continue; }
      let slot = atomicAdd(&control.count, 1u);
      if (slot < ${candidatesLimit}u) { candidates[slot] = Candidate(positions[node], node, separation); }
    }
  }
`;

const gatherShader = /* wgsl */ `
  struct Gather {
    count: u32,
    nodes: u32,
  }

  @group(0) @binding(0) var<uniform> gather: Gather;
  @group(0) @binding(1) var<storage, read> positions: array<vec4f>;
  @group(0) @binding(2) var<storage, read> ordinals: array<u32>;
  @group(0) @binding(3) var<storage, read_write> gathered: array<vec4f>;

  @compute @workgroup_size(${workgroupSize})
  fn gather_positions(@builtin(global_invocation_id) invocation: vec3u) {
    if (invocation.x >= gather.count) { return; }
    let node = ordinals[invocation.x];
    if (node < gather.nodes) { gathered[invocation.x] = positions[node]; }
  }
`;

/** @param {number} count */
function workgroupsFor(count) {
  return Math.max(1, Math.min(workgroupsLimit, Math.ceil(count / workgroupSize)));
}

/** @param {number[]} color */
function opaque(color) {
  return [color[0], color[1], color[2], 1];
}

/** A spring-electrical layout of a growing graph, relaxed and drawn on the GPU every frame. */
export class NeighborhoodProjection {
  /** @type {Timings} */
  timings = { layoutMilliseconds: 0, drawMilliseconds: 0, epochsPerFrame: 1 };

  #nodes = 0;
  #edges = 0;
  #epoch = 0;
  #reheatEpoch = 0;
  #current = 0;
  #epochMilliseconds = 0;
  #highlights = 0;
  /** @type {GPUTexture | null} */ #densityTexture = null;
  /** @type {GPUBindGroup | null} */ #statisticBindGroup = null;
  /** @type {GPUBindGroup | null} */ #toneBindGroup = null;
  #cellColumns = 0;
  #cellRows = 0;
  /** @type {GPUBuffer[]} */ #timestampReadbacks = [];

  /**
   *  @param {HTMLCanvasElement} canvas
   *  @param {GPUDevice} device
   *  @param {number} nodesCapacity
   *  @param {number} edgesCapacity
   */
  constructor(canvas, device, nodesCapacity, edgesCapacity) {
    this.canvas = canvas;
    this.device = device;
    this.nodesCapacity = nodesCapacity;
    this.edgesCapacity = edgesCapacity;
    this.states = new Uint32Array(nodesCapacity);

    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("The canvas has no WebGPU context.");
    this.context = context;
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: this.canvasFormat, alphaMode: "opaque" });

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const buffer = (/** @type {number} */ size, /** @type {number} */ usage) =>
      device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
    this.positionBuffers = [buffer(nodesCapacity * 16, storage), buffer(nodesCapacity * 16, storage)];
    this.forceBuffer = buffer(nodesCapacity * 16, storage);
    this.stateBuffer = buffer(nodesCapacity * 4, storage);
    this.highlightBuffer = buffer(highlightsLimit * 8, storage);
    this.edgeBuffer = buffer(edgesCapacity * 8, storage);
    this.weightBuffer = buffer(edgesCapacity * 4, storage);
    this.settingsBuffer = buffer(uniformStride * epochsPerFrameLimit, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.viewBuffer = buffer(96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.toneBuffer = buffer(48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.statisticBuffer = buffer(16, storage);
    this.partialBuffer = buffer(16, storage);
    this.spawnBuffer = buffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.queryBuffer = buffer(96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.histogramBuffer = buffer(histogramBuckets * 4, storage);
    this.controlBuffer = buffer(16, storage);
    this.candidateBuffer = buffer(candidatesLimit * 32, storage);
    this.gatherBuffer = buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    const layoutModule = device.createShaderModule({ code: layoutShader });
    const binding = (/** @type {number} */ index, /** @type {GPUBufferBindingType} */ type, dynamic = false) => ({
      binding: index,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type, hasDynamicOffset: dynamic },
    });
    const layoutBindGroupLayout = device.createBindGroupLayout({
      entries: [
        binding(0, "uniform", true),
        binding(1, "read-only-storage"),
        binding(2, "storage"),
        binding(3, "storage"),
        binding(4, "read-only-storage"),
        binding(5, "read-only-storage"),
        binding(6, "read-only-storage"),
      ],
    });
    const layoutPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layoutBindGroupLayout] });
    this.springsPipeline = device.createComputePipeline({
      layout: layoutPipelineLayout,
      compute: { module: layoutModule, entryPoint: "springs" },
    });
    this.chargesPipeline = device.createComputePipeline({
      layout: layoutPipelineLayout,
      compute: { module: layoutModule, entryPoint: "charges" },
    });
    this.layoutBindGroups = [0, 1].map((current) =>
      device.createBindGroup({
        layout: layoutBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.settingsBuffer, size: uniformStride } },
          { binding: 1, resource: { buffer: this.positionBuffers[current] } },
          { binding: 2, resource: { buffer: this.positionBuffers[1 - current] } },
          { binding: 3, resource: { buffer: this.forceBuffer } },
          { binding: 4, resource: { buffer: this.stateBuffer } },
          { binding: 5, resource: { buffer: this.edgeBuffer } },
          { binding: 6, resource: { buffer: this.weightBuffer } },
        ],
      }),
    );

    this.spawnPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: spawnShader }), entryPoint: "spawn_nodes" },
    });

    const densityModule = device.createShaderModule({ code: densityShader });
    const densityFormat = /** @type {GPUTextureFormat} */ ("rgba16float");
    const additive = /** @type {GPUBlendState} */ ({
      color: { srcFactor: "one", dstFactor: "one", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
    });
    const densityBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        ...[1, 2, 3, 4, 5].map((index) => ({
          binding: index,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: /** @type {GPUBufferBindingType} */ ("read-only-storage") },
        })),
      ],
    });
    const densityPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [densityBindGroupLayout] });
    const densityPipeline = (/** @type {string} */ stage, /** @type {GPUPrimitiveTopology} */ topology) =>
      device.createRenderPipeline({
        layout: densityPipelineLayout,
        vertex: { module: densityModule, entryPoint: `${stage}_vertex` },
        fragment: {
          module: densityModule,
          entryPoint: `${stage}_fragment`,
          targets: [{ format: densityFormat, blend: additive }],
        },
        primitive: { topology },
      });
    this.densityFormat = densityFormat;
    this.pointPipeline = densityPipeline("point", "point-list");
    this.linePipeline = densityPipeline("line", "line-list");
    this.discPipeline = densityPipeline("disc", "triangle-list");
    this.densityBindGroups = [0, 1].map((current) =>
      device.createBindGroup({
        layout: densityBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.viewBuffer } },
          { binding: 1, resource: { buffer: this.positionBuffers[current] } },
          { binding: 2, resource: { buffer: this.highlightBuffer } },
          { binding: 3, resource: { buffer: this.edgeBuffer } },
          { binding: 4, resource: { buffer: this.weightBuffer } },
          { binding: 5, resource: { buffer: this.statisticBuffer } },
        ],
      }),
    );

    const statisticModule = device.createShaderModule({ code: statisticShader });
    this.cellsPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: statisticModule, entryPoint: "cells" },
    });
    this.reducePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: statisticModule, entryPoint: "reduce" },
    });

    const toneModule = device.createShaderModule({ code: toneShader });
    this.tonePipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: toneModule, entryPoint: "tone_vertex" },
      fragment: { module: toneModule, entryPoint: "tone_fragment", targets: [{ format: this.canvasFormat }] },
      primitive: { topology: "triangle-list" },
    });

    const nearestModule = device.createShaderModule({ code: nearestShader });
    const nearestBindGroupLayout = device.createBindGroupLayout({
      entries: [
        binding(0, "uniform"),
        binding(1, "read-only-storage"),
        binding(2, "storage"),
        binding(3, "storage"),
        binding(4, "storage"),
      ],
    });
    const nearestPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [nearestBindGroupLayout] });
    const nearestPipeline = (/** @type {string} */ entryPoint) =>
      device.createComputePipeline({ layout: nearestPipelineLayout, compute: { module: nearestModule, entryPoint } });
    this.histogramPipeline = nearestPipeline("histogram_nodes");
    this.thresholdPipeline = nearestPipeline("pick_threshold");
    this.collectPipeline = nearestPipeline("collect_nodes");
    this.nearestBindGroups = [0, 1].map((current) =>
      device.createBindGroup({
        layout: nearestBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.queryBuffer } },
          { binding: 1, resource: { buffer: this.positionBuffers[current] } },
          { binding: 2, resource: { buffer: this.histogramBuffer } },
          { binding: 3, resource: { buffer: this.controlBuffer } },
          { binding: 4, resource: { buffer: this.candidateBuffer } },
        ],
      }),
    );

    this.gatherPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: gatherShader }), entryPoint: "gather_positions" },
    });

    this.timestamps = device.features.has("timestamp-query")
      ? device.createQuerySet({ type: "timestamp", count: 4 })
      : null;
    this.timestampResolve = buffer(32, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
    for (let index = 0; index < 3; index++)
      this.#timestampReadbacks.push(buffer(32, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
  }

  /**
   *  Reserves room for `nodes` nodes and `edges` edges, on the GPU behind `canvas`.
   *  @param {HTMLCanvasElement} canvas
   *  @param {number} nodes
   *  @param {number} edges
   *  @returns {Promise<NeighborhoodProjection>}
   */
  static async create(canvas, nodes, edges) {
    if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("WebGPU is unavailable in this browser.");
    const largest = Math.max(nodes * 16, edges * 8);
    if (largest > adapter.limits.maxStorageBufferBindingSize)
      throw new RangeError(`${edges} edges need ${largest} bytes in one binding, past this GPU's limit.`);
    const device = await adapter.requestDevice({
      requiredFeatures: adapter.features.has("timestamp-query") ? ["timestamp-query"] : [],
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
    return new NeighborhoodProjection(canvas, device, nodes, edges);
  }

  /**
   *  Appends one node per entry of `beside`, each starting next to that ordinal, or at `origin` for `emptySlot`.
   *  Answers the first new ordinal.
   *  @param {Uint32Array} beside
   *  @param {Vector} origin
   */
  appendNodes(beside, origin) {
    const first = this.#nodes;
    const count = beside.length;
    if (first + count > this.nodesCapacity)
      throw new RangeError(`Appending ${count} nodes passes the ${this.nodesCapacity} reserved.`);
    if (count === 0) return first;
    // A node beside an earlier node of the same batch starts beside that node's own root, a little further out.
    const roots = new Uint32Array(count * 2);
    for (let index = 0; index < count; index++) {
      const neighbor = beside[index];
      const local = neighbor - first;
      if (neighbor < first) roots.set([neighbor, 1], index * 2);
      else if (neighbor !== emptySlot && local < index)
        roots.set([roots[local * 2], roots[local * 2 + 1] + 1], index * 2);
      else roots.set([emptySlot, 1], index * 2);
    }
    this.states.fill(this.#epoch, first, first + count);
    this.device.queue.writeBuffer(this.stateBuffer, first * 4, this.states, first, count);

    const rootBuffer = this.device.createBuffer({
      size: roots.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(rootBuffer, 0, roots);
    const spawn = new ArrayBuffer(32);
    new Uint32Array(spawn, 0, 2).set([first, count]);
    new Float32Array(spawn, 8, 1).set([spawnRadius]);
    new Float32Array(spawn, 16, 3).set(origin);
    this.device.queue.writeBuffer(this.spawnBuffer, 0, spawn);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.spawnPipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: this.spawnPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.spawnBuffer } },
          { binding: 1, resource: { buffer: rootBuffer } },
          { binding: 2, resource: { buffer: this.positionBuffers[this.#current] } },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(count / workgroupSize));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    rootBuffer.destroy();
    this.#nodes += count;
    return first;
  }

  /**
   *  Appends edges grouped in runs of equal source. Each run is calibrated alone, so any metric's distances draw alike.
   *  @param {Uint32Array} sources
   *  @param {Uint32Array} targets
   *  @param {Float32Array} distances
   */
  appendEdges(sources, targets, distances) {
    const count = sources.length;
    if (this.#edges + count > this.edgesCapacity)
      throw new RangeError(`Appending ${count} edges passes the ${this.edgesCapacity} reserved.`);
    if (count === 0) return;
    const ends = new Uint32Array(count * 2);
    const weights = new Float32Array(count);
    for (let begin = 0, end = 0; begin < count; begin = end) {
      let nearest = Infinity;
      let sum = 0;
      for (end = begin; end < count && sources[end] === sources[begin]; end++) {
        nearest = Math.min(nearest, distances[end]);
        sum += distances[end];
      }
      const mean = sum / (end - begin);
      for (let edge = begin; edge < end; edge++) {
        ends[edge * 2] = sources[edge];
        ends[edge * 2 + 1] = targets[edge];
        weights[edge] = mean > 0 && Number.isFinite(mean) ? Math.exp(-(distances[edge] - nearest) / mean) : 1;
      }
    }
    this.device.queue.writeBuffer(this.edgeBuffer, this.#edges * 8, ends);
    this.device.queue.writeBuffer(this.weightBuffer, this.#edges * 4, weights);
    this.#edges += count;
  }

  /**
   *  Holds `nodes` at `positions` - three layout coordinates apiece - until released; everything else settles around them.
   *  @param {Uint32Array} nodes
   *  @param {Float32Array<ArrayBuffer>} positions
   */
  pin(nodes, positions) {
    for (let index = 0; index < nodes.length; index++) {
      const node = nodes[index];
      this.states[node] |= 0x80000000;
      this.device.queue.writeBuffer(this.stateBuffer, node * 4, this.states, node, 1);
      this.device.queue.writeBuffer(this.positionBuffers[this.#current], node * 16, positions, index * 3, 3);
    }
    this.#reheatEpoch = this.#epoch;
  }

  /** @param {Uint32Array} nodes */
  release(nodes) {
    for (const node of nodes) {
      this.states[node] &= 0x7fffffff;
      this.device.queue.writeBuffer(this.stateBuffer, node * 4, this.states, node, 1);
    }
    this.#reheatEpoch = this.#epoch;
  }

  /**
   *  Tints `nodes` with `style.accent` at `strengths` in [0, 1], replacing the previous highlight.
   *  @param {Uint32Array} nodes
   *  @param {Float32Array} strengths
   */
  highlight(nodes, strengths) {
    const count = Math.min(nodes.length, highlightsLimit);
    const list = new ArrayBuffer(Math.max(8, count * 8));
    const ordinals = new Uint32Array(list);
    const values = new Float32Array(list);
    for (let index = 0; index < count; index++) {
      ordinals[index * 2] = nodes[index];
      values[index * 2 + 1] = strengths[index];
    }
    this.device.queue.writeBuffer(this.highlightBuffer, 0, list);
    this.#highlights = count;
  }

  #resize() {
    const ratio = devicePixelRatio || 1;
    const width = Math.max(1, Math.round(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * ratio));
    if (this.#densityTexture && width === this.canvas.width && height === this.canvas.height) return;
    this.canvas.width = width;
    this.canvas.height = height;
    this.#densityTexture?.destroy();
    this.#densityTexture = this.device.createTexture({
      size: [width, height],
      format: this.densityFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.#cellColumns = Math.ceil(width / statisticCellPixels / 8);
    this.#cellRows = Math.ceil(height / statisticCellPixels / 8);
    this.partialBuffer.destroy();
    this.partialBuffer = this.device.createBuffer({
      size: this.#cellColumns * this.#cellRows * 8,
      usage: GPUBufferUsage.STORAGE,
    });
    this.device.queue.writeBuffer(this.statisticBuffer, 4, new Uint32Array([this.#cellColumns * this.#cellRows]));
    const densityView = this.#densityTexture.createView();
    this.#statisticBindGroup = this.device.createBindGroup({
      layout: this.cellsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: densityView },
        { binding: 1, resource: { buffer: this.partialBuffer } },
      ],
    });
    this.reduceBindGroup = this.device.createBindGroup({
      layout: this.reducePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.partialBuffer } },
        { binding: 2, resource: { buffer: this.statisticBuffer } },
      ],
    });
    this.#toneBindGroup = this.device.createBindGroup({
      layout: this.tonePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: densityView },
        { binding: 1, resource: { buffer: this.toneBuffer } },
        { binding: 2, resource: { buffer: this.statisticBuffer } },
      ],
    });
  }

  /** @param {number} budgetMilliseconds */
  #epochsWithin(budgetMilliseconds) {
    if (budgetMilliseconds <= 0 || this.#nodes === 0) return 0;
    if (this.#epochMilliseconds <= 0) return 1;
    return Math.max(1, Math.min(epochsPerFrameLimit, Math.floor(budgetMilliseconds / this.#epochMilliseconds)));
  }

  /**
   *  Runs layout epochs for up to `budgetMilliseconds`, then draws the frame `camera` sees in `style`.
   *  @param {Camera} camera
   *  @param {Style} style
   *  @param {number} budgetMilliseconds
   */
  frame(camera, style, budgetMilliseconds) {
    this.#resize();
    const width = this.canvas.width;
    const height = this.canvas.height;
    const cssWidth = this.canvas.clientWidth || 1;
    const cssHeight = this.canvas.clientHeight || 1;
    const epochs = this.#epochsWithin(budgetMilliseconds);

    const settings = new ArrayBuffer(uniformStride * Math.max(1, epochs));
    for (let epoch = 0; epoch < epochs; epoch++) {
      const offset = epoch * uniformStride;
      new Uint32Array(settings, offset, 6).set([
        this.#nodes,
        this.#edges,
        this.#epoch + epoch,
        layoutSeed,
        negatives,
        this.#reheatEpoch,
      ]);
      new Float32Array(settings, offset + 24, 8).set([
        settleEpochs,
        reheatAge,
        balance,
        epsilon,
        stepInitial,
        stepFinal,
        forceClip,
        fixedPointScale,
      ]);
    }
    if (epochs > 0) this.device.queue.writeBuffer(this.settingsBuffer, 0, settings);
    const slab = Math.max(cssWidth, cssHeight) / (2 * camera.scale);
    const view = new Float32Array(24);
    rotationOf(camera).forEach((row, index) => view.set(row, index * 4));
    view.set([...camera.center, slab], 12);
    view.set(
      [
        (2 * camera.scale) / cssWidth,
        (2 * camera.scale) / cssHeight,
        width,
        height,
        style.edges,
        edgeInkPixels,
        highlightPixels * (width / cssWidth),
        Math.max(1e-9, Math.min(camera.radius, slab)),
      ],
      16,
    );
    this.device.queue.writeBuffer(this.viewBuffer, 0, view);
    this.device.queue.writeBuffer(
      this.toneBuffer,
      0,
      new Float32Array([...opaque(style.paper), ...opaque(style.ink), ...opaque(style.accent)]),
    );

    const readback = this.timestamps
      ? this.#timestampReadbacks.find((candidate) => candidate.mapState === "unmapped")
      : undefined;
    const timed = this.timestamps && readback ? this.timestamps : null;
    const encoder = this.device.createCommandEncoder();
    if (epochs > 0) {
      const layoutPass = encoder.beginComputePass(
        timed ? { timestampWrites: { querySet: timed, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {},
      );
      for (let epoch = 0; epoch < epochs; epoch++) {
        const bindGroup = this.layoutBindGroups[(this.#current + epoch) % 2];
        layoutPass.setBindGroup(0, bindGroup, [epoch * uniformStride]);
        layoutPass.setPipeline(this.springsPipeline);
        layoutPass.dispatchWorkgroups(workgroupsFor(this.#edges));
        layoutPass.setPipeline(this.chargesPipeline);
        layoutPass.dispatchWorkgroups(workgroupsFor(this.#nodes));
      }
      layoutPass.end();
      this.#current = (this.#current + epochs) % 2;
      this.#epoch += epochs;
    }

    const densityTexture = /** @type {GPUTexture} */ (this.#densityTexture);
    const densityPass = encoder.beginRenderPass({
      colorAttachments: [
        { view: densityTexture.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" },
      ],
      ...(timed ? { timestampWrites: { querySet: timed, beginningOfPassWriteIndex: 2 } } : {}),
    });
    densityPass.setBindGroup(0, this.densityBindGroups[this.#current]);
    if (style.edges > 0 && this.#edges > 0) {
      densityPass.setPipeline(this.linePipeline);
      densityPass.draw(this.#edges * 2);
    }
    if (this.#nodes > 0) {
      densityPass.setPipeline(this.pointPipeline);
      densityPass.draw(this.#nodes);
    }
    if (this.#highlights > 0) {
      densityPass.setPipeline(this.discPipeline);
      densityPass.draw(6, this.#highlights);
    }
    densityPass.end();

    const statisticPass = encoder.beginComputePass();
    statisticPass.setPipeline(this.cellsPipeline);
    statisticPass.setBindGroup(0, this.#statisticBindGroup);
    statisticPass.dispatchWorkgroups(this.#cellColumns, this.#cellRows);
    statisticPass.setPipeline(this.reducePipeline);
    statisticPass.setBindGroup(0, this.reduceBindGroup);
    statisticPass.dispatchWorkgroups(1);
    statisticPass.end();

    const tonePass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: [0, 0, 0, 1],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      ...(timed ? { timestampWrites: { querySet: timed, endOfPassWriteIndex: 3 } } : {}),
    });
    tonePass.setPipeline(this.tonePipeline);
    tonePass.setBindGroup(0, this.#toneBindGroup);
    tonePass.draw(3);
    tonePass.end();

    if (timed && readback) {
      encoder.resolveQuerySet(timed, 0, 4, this.timestampResolve, 0);
      encoder.copyBufferToBuffer(this.timestampResolve, 0, readback, 0, 32);
    }
    const submitted = performance.now();
    this.device.queue.submit([encoder.finish()]);
    if (timed && readback) {
      readback.mapAsync(GPUMapMode.READ).then(
        () => {
          const stamps = new BigInt64Array(readback.getMappedRange());
          const layoutMilliseconds = epochs > 0 ? Number(stamps[1] - stamps[0]) / 1e6 : 0;
          this.#record(layoutMilliseconds, Number(stamps[3] - stamps[2]) / 1e6, epochs);
          readback.unmap();
        },
        () => {},
      );
    } else if (!this.timestamps) {
      this.device.queue.onSubmittedWorkDone().then(() => this.#record(0, performance.now() - submitted, epochs));
    }
  }

  /**
   *  @param {number} layoutMilliseconds
   *  @param {number} drawMilliseconds
   *  @param {number} epochs
   */
  #record(layoutMilliseconds, drawMilliseconds, epochs) {
    this.timings = { layoutMilliseconds, drawMilliseconds, epochsPerFrame: epochs };
    if (epochs > 0 && layoutMilliseconds > 0) {
      const perEpoch = layoutMilliseconds / epochs;
      this.#epochMilliseconds = this.#epochMilliseconds > 0 ? 0.8 * this.#epochMilliseconds + 0.2 * perEpoch : perEpoch;
    }
  }

  /**
   *  Up to `wanted` drawn nodes nearest a screen point, nearest first, with their layout positions, three apiece.
   *  @param {Camera} camera
   *  @param {number} x CSS pixels from the canvas' left edge.
   *  @param {number} y CSS pixels from the canvas' top edge.
   *  @param {number} wanted
   *  @returns {Promise<{nodes: Uint32Array, positions: Float32Array}>}
   */
  async nearest(camera, x, y, wanted) {
    const cssWidth = this.canvas.clientWidth || 1;
    const cssHeight = this.canvas.clientHeight || 1;
    const query = new ArrayBuffer(96);
    const floats = new Float32Array(query);
    rotationOf(camera).forEach((row, index) => floats.set(row, index * 4));
    floats.set(
      [...camera.center, Math.max(cssWidth, cssHeight) / (2 * camera.scale), cssWidth, cssHeight, x, y, camera.scale],
      12,
    );
    new Uint32Array(query, 84, 2).set([this.#nodes, Math.min(wanted, candidatesLimit)]);
    this.device.queue.writeBuffer(this.queryBuffer, 0, query);
    const readback = this.device.createBuffer({
      size: 16 + candidatesLimit * 32,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.clearBuffer(this.histogramBuffer);
    encoder.clearBuffer(this.controlBuffer);
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, this.nearestBindGroups[this.#current]);
    pass.setPipeline(this.histogramPipeline);
    pass.dispatchWorkgroups(workgroupsFor(this.#nodes));
    pass.setPipeline(this.thresholdPipeline);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.collectPipeline);
    pass.dispatchWorkgroups(workgroupsFor(this.#nodes));
    pass.end();
    encoder.copyBufferToBuffer(this.controlBuffer, 0, readback, 0, 16);
    encoder.copyBufferToBuffer(this.candidateBuffer, 0, readback, 16, candidatesLimit * 32);
    this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = readback.getMappedRange();
    const found = Math.min(new Uint32Array(mapped, 4, 1)[0], candidatesLimit);
    const words = new Uint32Array(mapped, 16, found * 8);
    const values = new Float32Array(mapped, 16, found * 8);
    const order = Array.from({ length: found }, (_, index) => index).sort(
      (left, right) => values[left * 8 + 5] - values[right * 8 + 5],
    );
    const kept = order.slice(0, wanted);
    const nodes = new Uint32Array(kept.length);
    const positions = new Float32Array(kept.length * 3);
    kept.forEach((candidate, index) => {
      nodes[index] = words[candidate * 8 + 4];
      positions.set(values.subarray(candidate * 8, candidate * 8 + 3), index * 3);
    });
    readback.destroy();
    return { nodes, positions };
  }

  /**
   *  Layout positions of `nodes`, three apiece, for overlays of selections that may be off screen.
   *  @param {Uint32Array<ArrayBuffer>} nodes
   *  @returns {Promise<Float32Array>}
   */
  async positionsOf(nodes) {
    const count = nodes.length;
    if (count === 0) return new Float32Array(0);
    const ordinals = this.device.createBuffer({
      size: count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const gathered = this.device.createBuffer({
      size: count * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readback = this.device.createBuffer({
      size: count * 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(ordinals, 0, nodes);
    this.device.queue.writeBuffer(this.gatherBuffer, 0, new Uint32Array([count, this.#nodes]));
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.gatherPipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: this.gatherPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.gatherBuffer } },
          { binding: 1, resource: { buffer: this.positionBuffers[this.#current] } },
          { binding: 2, resource: { buffer: ordinals } },
          { binding: 3, resource: { buffer: gathered } },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(count / workgroupSize));
    pass.end();
    encoder.copyBufferToBuffer(gathered, 0, readback, 0, count * 16);
    this.device.queue.submit([encoder.finish()]);
    ordinals.destroy();
    gathered.destroy();
    await readback.mapAsync(GPUMapMode.READ);
    const padded = new Float32Array(readback.getMappedRange());
    const positions = new Float32Array(count * 3);
    for (let node = 0; node < count; node++) positions.set(padded.subarray(node * 4, node * 4 + 3), node * 3);
    readback.destroy();
    return positions;
  }
}
