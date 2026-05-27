import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const palette = [
  0x0f7f8a,
  0xe0604f,
  0xd6a12f,
  0x3f6fb5,
  0x5c9f55,
  0xa05f9f,
  0x2f9fb3,
  0xc65f2f,
  0x4f8f79,
  0x7d6ccf,
];

const state = {
  bins: [],
  pieces: [],
  result: null,
  currentStep: 0,
  playing: false,
  playTimer: null,
  speed: 3,
};

const els = {
  binRows: document.querySelector("#binRows"),
  pieceRows: document.querySelector("#pieceRows"),
  addBin: document.querySelector("#addBin"),
  addPiece: document.querySelector("#addPiece"),
  loadDemo: document.querySelector("#loadDemo"),
  optimize: document.querySelector("#optimize"),
  allowRotation: document.querySelector("#allowRotation"),
  solverMode: document.querySelector("#solverMode"),
  message: document.querySelector("#message"),
  metricCost: document.querySelector("#metricCost"),
  metricBins: document.querySelector("#metricBins"),
  metricUtilization: document.querySelector("#metricUtilization"),
  metricPlaced: document.querySelector("#metricPlaced"),
  objectiveText: document.querySelector("#objectiveText"),
  variantText: document.querySelector("#variantText"),
  wasteText: document.querySelector("#wasteText"),
  placementRows: document.querySelector("#placementRows"),
  unplacedList: document.querySelector("#unplacedList"),
  playToggle: document.querySelector("#playToggle"),
  prevStep: document.querySelector("#prevStep"),
  nextStep: document.querySelector("#nextStep"),
  stepSlider: document.querySelector("#stepSlider"),
  speedSlider: document.querySelector("#speedSlider"),
  stepLabel: document.querySelector("#stepLabel"),
  canvas: document.querySelector("#scene"),
  viewport: document.querySelector(".viewport-panel"),
  pieceTip: document.querySelector("#pieceTip"),
};

const sceneState = {
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  root: null,
  raycaster: new THREE.Raycaster(),
  pointer: new THREE.Vector2(),
  hoverables: [],
  lastHover: null,
};

function uid(prefix) {
  return `${prefix}-${Math.random().toString(16).slice(2, 9)}`;
}

function loadDemoData() {
  state.bins = [
    { id: uid("bin"), name: "S", width: 100, height: 100, cost: 100 },
    { id: uid("bin"), name: "M", width: 150, height: 100, cost: 136 },
    { id: uid("bin"), name: "L", width: 200, height: 120, cost: 190 },
  ];
  state.pieces = [
    { id: uid("piece"), name: "A", width: 48, height: 42, qty: 4 },
    { id: uid("piece"), name: "B", width: 60, height: 28, qty: 3 },
    { id: uid("piece"), name: "C", width: 35, height: 30, qty: 5 },
    { id: uid("piece"), name: "D", width: 24, height: 18, qty: 8 },
    { id: uid("piece"), name: "E", width: 70, height: 22, qty: 2 },
  ];
  renderInputs();
}

function renderInputs() {
  els.binRows.replaceChildren(
    ...state.bins.map((bin, index) => {
      const row = document.createElement("div");
      row.className = "data-row";
      row.innerHTML = `
        <label><span>Name</span><input data-field="name" value="${escapeAttr(bin.name)}" /></label>
        <label><span>Width</span><input data-field="width" type="number" min="1" step="1" value="${bin.width}" /></label>
        <label><span>Height</span><input data-field="height" type="number" min="1" step="1" value="${bin.height}" /></label>
        <label><span>Cost</span><input data-field="cost" type="number" min="0" step="1" value="${bin.cost}" /></label>
        <button class="row-delete" type="button" aria-label="Remove bin">x</button>
      `;
      bindRow(row, state.bins, index);
      return row;
    }),
  );

  els.pieceRows.replaceChildren(
    ...state.pieces.map((piece, index) => {
      const row = document.createElement("div");
      row.className = "data-row piece-row";
      row.innerHTML = `
        <label><span>Name</span><input data-field="name" value="${escapeAttr(piece.name)}" /></label>
        <label><span>Width</span><input data-field="width" type="number" min="1" step="1" value="${piece.width}" /></label>
        <label><span>Height</span><input data-field="height" type="number" min="1" step="1" value="${piece.height}" /></label>
        <label><span>Qty</span><input data-field="qty" type="number" min="1" step="1" value="${piece.qty}" /></label>
        <button class="row-delete" type="button" aria-label="Remove piece">x</button>
      `;
      bindRow(row, state.pieces, index);
      return row;
    }),
  );
}

function escapeAttr(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function bindRow(row, collection, index) {
  row.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", () => {
      const field = input.dataset.field;
      collection[index][field] = field === "name" ? input.value : Number(input.value);
    });
  });
  row.querySelector(".row-delete").addEventListener("click", () => {
    collection.splice(index, 1);
    renderInputs();
  });
}

function readModel() {
  const bins = state.bins
    .map((bin, index) => ({
      index,
      name: String(bin.name || `B${index + 1}`).trim() || `B${index + 1}`,
      width: Math.round(Number(bin.width)),
      height: Math.round(Number(bin.height)),
      cost: Number(bin.cost),
    }))
    .filter((bin) => bin.width > 0 && bin.height > 0 && Number.isFinite(bin.cost));

  const pieces = [];
  state.pieces.forEach((group, groupIndex) => {
    const width = Math.round(Number(group.width));
    const height = Math.round(Number(group.height));
    const qty = Math.round(Number(group.qty));
    if (width <= 0 || height <= 0 || qty <= 0) return;
    for (let copy = 0; copy < qty; copy += 1) {
      pieces.push({
        id: `${group.name || `P${groupIndex + 1}`}-${copy + 1}`,
        group: group.name || `P${groupIndex + 1}`,
        groupIndex,
        copy: copy + 1,
        width,
        height,
        area: width * height,
        perimeter: 2 * (width + height),
      });
    }
  });

  return {
    bins,
    pieces,
    allowRotation: els.allowRotation.checked,
    solver: els.solverMode.value,
  };
}

function validateModel(model) {
  if (!model.bins.length) return "Add at least one bin type.";
  if (!model.pieces.length) return "Add at least one piece.";
  if (model.bins.some((bin) => bin.cost < 0)) return "Bin cost cannot be negative.";
  return "";
}

function solvePacking(model) {
  const orderModes =
    model.searchLevel === "fast"
      ? ["area"]
      : ["area", "maxSide", "perimeter", "height", "width", "original"];
  const scoreModes = model.searchLevel === "fast" ? ["area"] : ["area", "short", "contact", "bottomLeft"];
  const openModes =
    model.searchLevel === "deep"
      ? ["cheapest", "smallest", "density", "balanced"]
      : ["cheapest", "smallest", "balanced"];

  const variants = [];
  for (const orderMode of orderModes) {
    for (const scoreMode of scoreModes) {
      for (const openMode of openModes) {
        variants.push(packVariant(model, orderMode, scoreMode, openMode));
      }
    }
  }

  if (model.searchLevel === "deep" && model.pieces.length <= 180) {
    for (let seed = 1; seed <= 12; seed += 1) {
      variants.push(packVariant(model, "seeded", "area", "balanced", seed));
      variants.push(packVariant(model, "seeded", "contact", "cheapest", seed + 31));
    }
  }

  variants.sort(compareResults);
  variants.push(...searchFixedBinPlans(model, variants[0], orderModes, scoreModes));
  variants.sort(compareResults);
  const best = variants[0];
  best.variantCount = variants.length;
  best.objective = "Min opened-bin cost, then opened bins, then waste";
  return best;
}

function searchFixedBinPlans(model, incumbent, orderModes, scoreModes) {
  const totalPieceArea = model.pieces.reduce((sum, piece) => sum + piece.area, 0);
  const maxCost =
    incumbent && incumbent.unplaced.length === 0
      ? incumbent.totalCost
      : Math.min(...model.bins.map((bin) => bin.cost || 1)) * Math.min(model.pieces.length, 12);
  const maxCombos = model.searchLevel === "deep" ? 900 : model.searchLevel === "fast" ? 120 : 360;
  const combos = enumerateBinCombos(model.bins, totalPieceArea, maxCost, maxCombos);
  const binOrderModes =
    model.searchLevel === "fast"
      ? ["areaAsc", "costAsc"]
      : ["areaAsc", "areaDesc", "costAsc", "densityAsc", "input"];
  const results = [];

  for (const combo of combos) {
    for (const orderMode of orderModes) {
      for (const scoreMode of scoreModes) {
        for (const binOrderMode of binOrderModes) {
          results.push(packFixedVariant(model, combo, orderMode, scoreMode, binOrderMode));
        }
      }
    }
  }

  return results;
}

function enumerateBinCombos(binTypes, totalPieceArea, maxCost, maxCombos) {
  const minPositiveCost = Math.min(...binTypes.map((bin) => bin.cost).filter((cost) => cost > 0), 1);
  const maxArea = Math.max(...binTypes.map((bin) => bin.width * bin.height), 1);
  const costCap = Math.max(1, Math.floor(maxCost / minPositiveCost));
  const areaCap = Math.ceil(totalPieceArea / maxArea) + binTypes.length + 2;
  const maxBins = Math.min(12, Math.max(1, Math.min(costCap, areaCap)));
  const combos = [];

  function walk(typeIndex, counts) {
    if (combos.length >= maxCombos) return;
    if (typeIndex === binTypes.length) {
      const count = counts.reduce((sum, value) => sum + value, 0);
      if (!count || count > maxBins) return;
      const cost = counts.reduce((sum, value, index) => sum + value * binTypes[index].cost, 0);
      const area = counts.reduce((sum, value, index) => sum + value * binTypes[index].width * binTypes[index].height, 0);
      if (cost <= maxCost + 1e-9 && area >= totalPieceArea) {
        combos.push(materializeCombo(binTypes, counts));
      }
      return;
    }

    for (let count = 0; count <= maxBins; count += 1) {
      counts[typeIndex] = count;
      const partialCount = counts.reduce((sum, value) => sum + value, 0);
      const partialCost = counts.reduce((sum, value, index) => sum + value * binTypes[index].cost, 0);
      if (partialCount <= maxBins && partialCost <= maxCost + 1e-9) walk(typeIndex + 1, counts);
    }
    counts[typeIndex] = 0;
  }

  walk(0, new Array(binTypes.length).fill(0));
  return combos.sort((left, right) => {
    const costDiff = left.totalCost - right.totalCost;
    if (Math.abs(costDiff) > 1e-9) return costDiff;
    return left.templates.length - right.templates.length || left.totalArea - right.totalArea;
  });
}

function materializeCombo(binTypes, counts) {
  const templates = [];
  counts.forEach((count, typeIndex) => {
    for (let copy = 1; copy <= count; copy += 1) {
      const binType = binTypes[typeIndex];
      templates.push({
        ...binType,
        typeIndex,
        typeName: binType.name,
        instanceName: `${binType.name}${copy}`,
      });
    }
  });
  return {
    templates,
    totalArea: templates.reduce((sum, bin) => sum + bin.width * bin.height, 0),
    totalCost: templates.reduce((sum, bin) => sum + bin.cost, 0),
  };
}

function orderedPieces(pieces, mode, seed) {
  const copy = pieces.map((piece, index) => ({ ...piece, originalIndex: index }));
  const byNumber = (fn) => (a, b) => fn(b) - fn(a) || b.area - a.area || a.originalIndex - b.originalIndex;

  if (mode === "area") return copy.sort(byNumber((piece) => piece.area));
  if (mode === "maxSide") return copy.sort(byNumber((piece) => Math.max(piece.width, piece.height)));
  if (mode === "perimeter") return copy.sort(byNumber((piece) => piece.perimeter));
  if (mode === "height") return copy.sort(byNumber((piece) => piece.height));
  if (mode === "width") return copy.sort(byNumber((piece) => piece.width));
  if (mode === "seeded") return seededShuffle(copy, seed).sort((a, b) => b.area - a.area + jitter(seed, a.id, b.id));
  return copy;
}

function seededShuffle(items, seed) {
  const output = [...items];
  let value = seed || 1;
  for (let i = output.length - 1; i > 0; i -= 1) {
    value = (value * 1664525 + 1013904223) % 4294967296;
    const j = value % (i + 1);
    [output[i], output[j]] = [output[j], output[i]];
  }
  return output;
}

function jitter(seed, left, right) {
  const text = `${seed}:${left}:${right}`;
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) hash = (hash * 31 + text.charCodeAt(i)) | 0;
  return (hash % 17) * 0.0001;
}

function packVariant(model, orderMode, scoreMode, openMode, seed = 0) {
  const bins = [];
  const steps = [];
  const unplaced = [];
  const pieces = orderedPieces(model.pieces, orderMode, seed);

  for (const piece of pieces) {
    const existingCandidates = [];
    for (const bin of bins) {
      const placement = findBestPlacement(bin, piece, model.allowRotation, scoreMode);
      if (placement) {
        existingCandidates.push({ kind: "existing", bin, placement, rank: placement.rank });
      }
    }

    let chosen = chooseExisting(existingCandidates);
    if (!chosen) {
      const newCandidates = [];
      for (const binType of model.bins) {
        const bin = createBin(binType, bins.length + newCandidates.length + 1);
        const placement = findBestPlacement(bin, piece, model.allowRotation, scoreMode);
        if (placement) {
          newCandidates.push({ kind: "new", bin, placement, rank: rankNewBin(binType, placement, piece, openMode) });
        }
      }
      chosen = chooseNew(newCandidates);
      if (chosen) bins.push(chosen.bin);
    }

    if (!chosen) {
      unplaced.push(piece);
      continue;
    }

    const placed = commitPlacement(chosen.bin, piece, chosen.placement);
    steps.push({
      ...placed,
      step: steps.length + 1,
      binName: chosen.bin.name,
      binTypeIndex: chosen.bin.typeIndex,
    });
  }

  const placedArea = steps.reduce((sum, item) => sum + item.width * item.height, 0);
  const totalArea = bins.reduce((sum, bin) => sum + bin.width * bin.height, 0);
  const totalCost = bins.reduce((sum, bin) => sum + bin.cost, 0);
  const waste = totalArea - placedArea;

  return {
    bins,
    steps,
    placements: steps,
    unplaced,
    placedArea,
    totalArea,
    totalCost,
    waste,
    utilization: totalArea ? (placedArea / totalArea) * 100 : 0,
    variant: `${orderMode}/${scoreMode}/${openMode}${seed ? `/${seed}` : ""}`,
  };
}

function packFixedVariant(model, combo, orderMode, scoreMode, binOrderMode) {
  const bins = combo.templates.map((template, index) => ({
    id: index + 1,
    name: template.instanceName,
    typeName: template.typeName,
    typeIndex: template.typeIndex,
    width: template.width,
    height: template.height,
    cost: template.cost,
    freeRects: [{ x: 0, y: 0, width: template.width, height: template.height }],
    placements: [],
  }));
  const steps = [];
  const unplaced = [];
  const pieces = orderedPieces(model.pieces, orderMode, 0);

  for (const piece of pieces) {
    const candidates = [];
    for (const bin of orderedBins(bins, binOrderMode, piece)) {
      const placement = findBestPlacement(bin, piece, model.allowRotation, scoreMode);
      if (placement) {
        candidates.push({
          bin,
          placement,
          rank: fixedBinRank(bin, placement, piece, binOrderMode),
        });
      }
    }

    const chosen = chooseExisting(candidates);
    if (!chosen) {
      unplaced.push(piece);
      continue;
    }

    const placed = commitPlacement(chosen.bin, piece, chosen.placement);
    steps.push({
      ...placed,
      step: steps.length + 1,
      binName: chosen.bin.name,
      binTypeIndex: chosen.bin.typeIndex,
    });
  }

  const usedIds = new Set(steps.map((step) => step.binId));
  const usedBins = bins.filter((bin) => usedIds.has(bin.id));
  const placedArea = steps.reduce((sum, item) => sum + item.width * item.height, 0);
  const totalArea = usedBins.reduce((sum, bin) => sum + bin.width * bin.height, 0);
  const totalCost = usedBins.reduce((sum, bin) => sum + bin.cost, 0);
  const waste = totalArea - placedArea;

  return {
    bins: usedBins,
    steps,
    placements: steps,
    unplaced,
    placedArea,
    totalArea,
    totalCost,
    waste,
    utilization: totalArea ? (placedArea / totalArea) * 100 : 0,
    variant: `fixed/${orderMode}/${scoreMode}/${binOrderMode}`,
  };
}

function orderedBins(bins, mode, piece) {
  const sorted = [...bins];
  if (mode === "areaAsc") {
    return sorted.sort((a, b) => a.width * a.height - b.width * b.height || a.cost - b.cost);
  }
  if (mode === "areaDesc") {
    return sorted.sort((a, b) => b.width * b.height - a.width * a.height || a.cost - b.cost);
  }
  if (mode === "costAsc") {
    return sorted.sort((a, b) => a.cost - b.cost || a.width * a.height - b.width * b.height);
  }
  if (mode === "densityAsc") {
    return sorted.sort((a, b) => a.cost / (a.width * a.height) - b.cost / (b.width * b.height));
  }
  return sorted.sort((a, b) => a.id - b.id || piece.area - piece.area);
}

function fixedBinRank(bin, placement, piece, mode) {
  const usedArea = bin.placements.reduce((sum, item) => sum + item.width * item.height, 0);
  const remainingAfter = bin.width * bin.height - usedArea - piece.area;
  const area = bin.width * bin.height;
  if (mode === "costAsc") return [bin.cost, remainingAfter, ...placement.rank];
  if (mode === "areaDesc") return [remainingAfter, -area, ...placement.rank];
  if (mode === "densityAsc") return [bin.cost / area, remainingAfter, ...placement.rank];
  if (mode === "input") return [bin.id, ...placement.rank];
  return [remainingAfter, area, ...placement.rank];
}

function createBin(binType, id) {
  return {
    id,
    name: `${binType.name}${id}`,
    typeName: binType.name,
    typeIndex: binType.index,
    width: binType.width,
    height: binType.height,
    cost: binType.cost,
    freeRects: [{ x: 0, y: 0, width: binType.width, height: binType.height }],
    placements: [],
  };
}

function findBestPlacement(bin, piece, allowRotation, scoreMode) {
  let best = null;
  const orientations = [{ width: piece.width, height: piece.height, rotated: false }];
  if (allowRotation && piece.width !== piece.height) {
    orientations.push({ width: piece.height, height: piece.width, rotated: true });
  }

  for (const free of bin.freeRects) {
    for (const orientation of orientations) {
      if (orientation.width > free.width || orientation.height > free.height) continue;
      const candidate = {
        x: free.x,
        y: free.y,
        width: orientation.width,
        height: orientation.height,
        rotated: orientation.rotated,
      };
      candidate.rank = rankPlacement(bin, free, candidate, scoreMode);
      if (!best || compareRank(candidate.rank, best.rank) < 0) best = candidate;
    }
  }
  return best;
}

function rankPlacement(bin, free, candidate, scoreMode) {
  const leftoverArea = free.width * free.height - candidate.width * candidate.height;
  const shortSide = Math.min(free.width - candidate.width, free.height - candidate.height);
  const longSide = Math.max(free.width - candidate.width, free.height - candidate.height);
  const contact = contactScore(bin, candidate);

  if (scoreMode === "short") return [shortSide, leftoverArea, longSide, candidate.y, candidate.x];
  if (scoreMode === "contact") return [-contact, leftoverArea, shortSide, candidate.y, candidate.x];
  if (scoreMode === "bottomLeft") return [candidate.y, candidate.x, leftoverArea, shortSide];
  return [leftoverArea, shortSide, longSide, candidate.y, candidate.x];
}

function contactScore(bin, rect) {
  let score = 0;
  if (rect.x === 0) score += rect.height;
  if (rect.y === 0) score += rect.width;
  if (rect.x + rect.width === bin.width) score += rect.height;
  if (rect.y + rect.height === bin.height) score += rect.width;

  for (const placed of bin.placements) {
    const xOverlap = overlapLength(rect.x, rect.x + rect.width, placed.x, placed.x + placed.width);
    const yOverlap = overlapLength(rect.y, rect.y + rect.height, placed.y, placed.y + placed.height);
    if (rect.x === placed.x + placed.width || rect.x + rect.width === placed.x) score += yOverlap;
    if (rect.y === placed.y + placed.height || rect.y + rect.height === placed.y) score += xOverlap;
  }
  return score;
}

function overlapLength(a0, a1, b0, b1) {
  return Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));
}

function rankNewBin(binType, placement, piece, openMode) {
  const area = binType.width * binType.height;
  const waste = area - piece.area;
  const densityCost = binType.cost / area;
  if (openMode === "smallest") return [area, binType.cost, waste, ...placement.rank];
  if (openMode === "density") return [densityCost, binType.cost, waste, ...placement.rank];
  if (openMode === "balanced") return [binType.cost + waste * 0.015, binType.cost, waste, ...placement.rank];
  return [binType.cost, waste, area, ...placement.rank];
}

function chooseExisting(candidates) {
  if (!candidates.length) return null;
  return candidates.sort((a, b) => compareRank(a.rank, b.rank))[0];
}

function chooseNew(candidates) {
  if (!candidates.length) return null;
  return candidates.sort((a, b) => compareRank(a.rank, b.rank))[0];
}

function compareRank(left, right) {
  const max = Math.max(left.length, right.length);
  for (let i = 0; i < max; i += 1) {
    const diff = (left[i] ?? 0) - (right[i] ?? 0);
    if (Math.abs(diff) > 1e-9) return diff;
  }
  return 0;
}

function commitPlacement(bin, piece, placement) {
  const used = {
    pieceId: piece.id,
    group: piece.group,
    groupIndex: piece.groupIndex,
    copy: piece.copy,
    sourceWidth: piece.width,
    sourceHeight: piece.height,
    x: placement.x,
    y: placement.y,
    width: placement.width,
    height: placement.height,
    rotated: placement.rotated,
    binId: bin.id,
    binName: bin.name,
    binType: bin.typeName,
  };

  const nextFree = [];
  for (const free of bin.freeRects) {
    if (!intersects(free, used)) {
      nextFree.push(free);
      continue;
    }
    nextFree.push(...splitFreeRect(free, used));
  }
  bin.freeRects = pruneFreeRects(nextFree);
  bin.placements.push(used);
  return used;
}

function intersects(a, b) {
  return a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
}

function splitFreeRect(free, used) {
  const rects = [];
  const freeRight = free.x + free.width;
  const freeTop = free.y + free.height;
  const usedRight = used.x + used.width;
  const usedTop = used.y + used.height;

  if (used.y > free.y && used.y < freeTop) {
    rects.push({ x: free.x, y: free.y, width: free.width, height: used.y - free.y });
  }
  if (usedTop < freeTop) {
    rects.push({ x: free.x, y: usedTop, width: free.width, height: freeTop - usedTop });
  }
  if (used.x > free.x && used.x < freeRight) {
    rects.push({ x: free.x, y: free.y, width: used.x - free.x, height: free.height });
  }
  if (usedRight < freeRight) {
    rects.push({ x: usedRight, y: free.y, width: freeRight - usedRight, height: free.height });
  }

  return rects.filter((rect) => rect.width > 0 && rect.height > 0);
}

function pruneFreeRects(rects) {
  const pruned = [];
  rects.forEach((rect, index) => {
    const contained = rects.some((other, otherIndex) => otherIndex !== index && containsRect(other, rect));
    if (!contained && rect.width > 0 && rect.height > 0) pruned.push(rect);
  });
  return pruned;
}

function containsRect(outer, inner) {
  return (
    inner.x >= outer.x &&
    inner.y >= outer.y &&
    inner.x + inner.width <= outer.x + outer.width &&
    inner.y + inner.height <= outer.y + outer.height
  );
}

function compareResults(left, right) {
  if (left.unplaced.length !== right.unplaced.length) return left.unplaced.length - right.unplaced.length;
  if (Math.abs(left.totalCost - right.totalCost) > 1e-9) return left.totalCost - right.totalCost;
  if (left.bins.length !== right.bins.length) return left.bins.length - right.bins.length;
  if (Math.abs(left.waste - right.waste) > 1e-9) return left.waste - right.waste;
  return right.utilization - left.utilization;
}

function updateResults(result) {
  els.metricCost.textContent = formatNumber(result.totalCost);
  els.metricBins.textContent = String(result.bins.length);
  els.metricUtilization.textContent = `${result.utilization.toFixed(1)}%`;
  els.metricPlaced.textContent = `${result.placements.length}/${result.placements.length + result.unplaced.length}`;
  els.objectiveText.textContent = result.objective;
  els.variantText.textContent =
    result.variantCount && result.variantCount > 1 ? `${result.variant} (${result.variantCount} variants)` : result.variant;
  els.wasteText.textContent = `${formatNumber(result.waste)} area units`;

  els.placementRows.replaceChildren(
    ...result.placements.map((item) => {
      const row = document.createElement("tr");
      [item.pieceId, item.binName, `${item.width} x ${item.height}${item.rotated ? " R" : ""}`, `${item.x}, ${item.y}`].forEach(
        (value) => {
          const cell = document.createElement("td");
          cell.textContent = value;
          row.append(cell);
        },
      );
      return row;
    }),
  );

  els.unplacedList.replaceChildren(
    ...(result.unplaced.length
      ? result.unplaced.map((item) => {
          const li = document.createElement("li");
          li.textContent = `${item.id}: ${item.width} x ${item.height}`;
          return li;
        })
      : [emptyListItem("All pieces placed")]),
  );

  els.stepSlider.max = String(result.placements.length);
  setStep(result.placements.length);
}

function emptyListItem(text) {
  const li = document.createElement("li");
  li.textContent = text;
  li.style.background = "#e9f6ee";
  li.style.borderColor = "#b7dfc5";
  li.style.color = "#1d6336";
  return li;
}

function formatNumber(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

function initScene() {
  sceneState.renderer = new THREE.WebGLRenderer({
    canvas: els.canvas,
    antialias: true,
    alpha: false,
    preserveDrawingBuffer: true,
  });
  sceneState.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  sceneState.renderer.setClearColor(0x1f2529, 1);

  sceneState.scene = new THREE.Scene();
  sceneState.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 1000);
  sceneState.camera.position.set(9, 10, 12);

  sceneState.controls = new OrbitControls(sceneState.camera, els.canvas);
  sceneState.controls.enableDamping = true;
  sceneState.controls.target.set(0, 0, 0);

  const ambient = new THREE.HemisphereLight(0xffffff, 0x2b3035, 2.6);
  sceneState.scene.add(ambient);

  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(10, 16, 8);
  sceneState.scene.add(key);

  const fill = new THREE.DirectionalLight(0xc4f0e8, 1.1);
  fill.position.set(-9, 8, -6);
  sceneState.scene.add(fill);

  sceneState.root = new THREE.Group();
  sceneState.scene.add(sceneState.root);

  const resizeObserver = new ResizeObserver(resizeRenderer);
  resizeObserver.observe(els.viewport);
  els.canvas.addEventListener("pointermove", onPointerMove);
  els.canvas.addEventListener("pointerleave", hideTip);
  animate();
}

function resizeRenderer() {
  const width = Math.max(1, els.viewport.clientWidth);
  const height = Math.max(1, els.viewport.clientHeight);
  sceneState.renderer.setSize(width, height, false);
  sceneState.camera.aspect = width / height;
  sceneState.camera.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  sceneState.controls.update();
  sceneState.renderer.render(sceneState.scene, sceneState.camera);
}

function renderScene() {
  sceneState.hoverables = [];
  clearGroup(sceneState.root);

  const result = state.result;
  if (!result) {
    addEmptyScene();
    return;
  }

  const visible = result.placements.slice(0, state.currentStep);
  const layout = createBinLayout(result.bins);

  for (const bin of result.bins) {
    const binLayout = layout.get(bin.id);
    createBinMesh(bin, binLayout);
  }

  visible.forEach((placement, index) => {
    const binLayout = layout.get(placement.binId);
    createPieceMesh(placement, binLayout, index === visible.length - 1);
  });

  frameCamera(layout);
}

function addEmptyScene() {
  const grid = new THREE.GridHelper(8, 8, 0x607078, 0x3c464d);
  sceneState.root.add(grid);
  sceneState.camera.position.set(7, 7, 9);
  sceneState.controls.target.set(0, 0, 0);
}

function clearGroup(group) {
  while (group.children.length) {
    const child = group.children.pop();
    child.traverse((node) => {
      if (node.geometry) node.geometry.dispose();
      if (node.material) {
        if (Array.isArray(node.material)) node.material.forEach((material) => material.dispose());
        else node.material.dispose();
      }
    });
  }
}

function createBinLayout(bins) {
  const maxDim = Math.max(...bins.flatMap((bin) => [bin.width, bin.height]), 1);
  const unit = 7 / maxDim;
  const maxRowWidth = 20;
  const gap = 1.4;
  let cursorX = 0;
  let cursorZ = 0;
  let rowDepth = 0;
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  const layout = new Map();

  bins.forEach((bin) => {
    const worldW = bin.width * unit;
    const worldH = bin.height * unit;
    if (cursorX > 0 && cursorX + worldW > maxRowWidth) {
      cursorX = 0;
      cursorZ += rowDepth + gap;
      rowDepth = 0;
    }
    const info = { originX: cursorX, originZ: cursorZ, worldW, worldH, unit };
    layout.set(bin.id, info);
    minX = Math.min(minX, cursorX);
    maxX = Math.max(maxX, cursorX + worldW);
    minZ = Math.min(minZ, cursorZ);
    maxZ = Math.max(maxZ, cursorZ + worldH);
    cursorX += worldW + gap;
    rowDepth = Math.max(rowDepth, worldH);
  });

  layout.bounds = {
    minX: Number.isFinite(minX) ? minX : -4,
    maxX: Number.isFinite(maxX) ? maxX : 4,
    minZ: Number.isFinite(minZ) ? minZ : -4,
    maxZ: Number.isFinite(maxZ) ? maxZ : 4,
  };
  return layout;
}

function createBinMesh(bin, layout) {
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(layout.worldW, 0.08, layout.worldH),
    new THREE.MeshStandardMaterial({ color: 0xe3ebe7, roughness: 0.85, metalness: 0.05 }),
  );
  base.position.set(layout.originX + layout.worldW / 2, -0.04, layout.originZ + layout.worldH / 2);
  sceneState.root.add(base);

  const edge = new THREE.LineSegments(
    new THREE.EdgesGeometry(base.geometry),
    new THREE.LineBasicMaterial({ color: 0xb7c8c4 }),
  );
  edge.position.copy(base.position);
  sceneState.root.add(edge);

  const border = new THREE.LineSegments(
    createRectLineGeometry(layout.worldW, layout.worldH),
    new THREE.LineBasicMaterial({ color: 0xffffff }),
  );
  border.position.set(layout.originX + layout.worldW / 2, 0.05, layout.originZ + layout.worldH / 2);
  sceneState.root.add(border);

  createGridLines(bin, layout);
}

function createRectLineGeometry(width, height) {
  const hw = width / 2;
  const hh = height / 2;
  const points = [
    new THREE.Vector3(-hw, 0, -hh),
    new THREE.Vector3(hw, 0, -hh),
    new THREE.Vector3(hw, 0, -hh),
    new THREE.Vector3(hw, 0, hh),
    new THREE.Vector3(hw, 0, hh),
    new THREE.Vector3(-hw, 0, hh),
    new THREE.Vector3(-hw, 0, hh),
    new THREE.Vector3(-hw, 0, -hh),
  ];
  return new THREE.BufferGeometry().setFromPoints(points);
}

function createGridLines(bin, layout) {
  const step = chooseGridStep(Math.max(bin.width, bin.height));
  const points = [];
  for (let x = step; x < bin.width; x += step) {
    const wx = layout.originX + x * layout.unit;
    points.push(new THREE.Vector3(wx, 0.055, layout.originZ));
    points.push(new THREE.Vector3(wx, 0.055, layout.originZ + layout.worldH));
  }
  for (let y = step; y < bin.height; y += step) {
    const wz = layout.originZ + y * layout.unit;
    points.push(new THREE.Vector3(layout.originX, 0.055, wz));
    points.push(new THREE.Vector3(layout.originX + layout.worldW, 0.055, wz));
  }
  if (!points.length) return;
  const lines = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color: 0xc9d5d1, transparent: true, opacity: 0.55 }),
  );
  sceneState.root.add(lines);
}

function chooseGridStep(maxDim) {
  if (maxDim <= 40) return 5;
  if (maxDim <= 120) return 10;
  if (maxDim <= 240) return 20;
  return 50;
}

function createPieceMesh(placement, layout, active) {
  const color = palette[placement.groupIndex % palette.length];
  const height = active ? 0.45 : 0.34;
  const geometry = new THREE.BoxGeometry(placement.width * layout.unit, height, placement.height * layout.unit);
  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.58,
    metalness: 0.08,
    emissive: active ? color : 0x000000,
    emissiveIntensity: active ? 0.18 : 0,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(
    layout.originX + (placement.x + placement.width / 2) * layout.unit,
    height / 2,
    layout.originZ + (placement.y + placement.height / 2) * layout.unit,
  );
  mesh.userData.placement = placement;
  sceneState.root.add(mesh);
  sceneState.hoverables.push(mesh);

  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({ color: 0x102025 }));
  edges.position.copy(mesh.position);
  sceneState.root.add(edges);
}

function frameCamera(layout) {
  const bounds = layout.bounds;
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerZ = (bounds.minZ + bounds.maxZ) / 2;
  const spanX = Math.max(1, bounds.maxX - bounds.minX);
  const spanZ = Math.max(1, bounds.maxZ - bounds.minZ);
  const radius = Math.max(spanX, spanZ);
  sceneState.controls.target.set(centerX, 0, centerZ);
  sceneState.camera.position.lerp(new THREE.Vector3(centerX + radius * 0.75, radius * 0.88 + 3, centerZ + radius * 1.05), 0.16);
  sceneState.camera.near = 0.1;
  sceneState.camera.far = 1000;
  sceneState.camera.updateProjectionMatrix();
}

function onPointerMove(event) {
  if (!sceneState.hoverables.length) return hideTip();
  const rect = els.canvas.getBoundingClientRect();
  sceneState.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  sceneState.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  sceneState.raycaster.setFromCamera(sceneState.pointer, sceneState.camera);
  const hit = sceneState.raycaster.intersectObjects(sceneState.hoverables, false)[0];
  if (!hit) return hideTip();
  const item = hit.object.userData.placement;
  sceneState.lastHover = hit.object;
  els.pieceTip.hidden = false;
  els.pieceTip.style.left = `${event.clientX - rect.left + 14}px`;
  els.pieceTip.style.top = `${event.clientY - rect.top + 14}px`;
  els.pieceTip.textContent = `${item.pieceId} | ${item.width} x ${item.height} | ${item.binName} | ${item.x}, ${item.y}`;
}

function hideTip() {
  sceneState.lastHover = null;
  els.pieceTip.hidden = true;
}

function setStep(step) {
  const max = state.result ? state.result.placements.length : 0;
  state.currentStep = Math.max(0, Math.min(step, max));
  els.stepSlider.value = String(state.currentStep);
  els.stepLabel.textContent = `${state.currentStep} / ${max}`;
  renderScene();
}

function play() {
  if (!state.result) return;
  stopPlayback();
  state.playing = true;
  els.playToggle.textContent = "Pause";
  if (state.currentStep >= state.result.placements.length) setStep(0);
  const interval = 900 - state.speed * 130;
  state.playTimer = window.setInterval(() => {
    if (!state.result || state.currentStep >= state.result.placements.length) {
      stopPlayback();
      return;
    }
    setStep(state.currentStep + 1);
  }, Math.max(160, interval));
}

function stopPlayback() {
  state.playing = false;
  els.playToggle.textContent = "Play";
  if (state.playTimer) window.clearInterval(state.playTimer);
  state.playTimer = null;
}

async function optimize() {
  stopPlayback();
  const model = readModel();
  const error = validateModel(model);
  if (error) {
    els.message.textContent = error;
    return;
  }
  els.optimize.disabled = true;
  els.message.textContent = "Solving through backend core...";
  const start = performance.now();
  try {
    const response = await fetch("/api/pack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(model),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Backend request failed.");
    }
    const duration = performance.now() - start;
    state.result = result;
    const notes = result.notes && result.notes.length ? ` ${result.notes.join(" ")}` : "";
    els.message.textContent = `Solved by backend in ${duration.toFixed(1)} ms.${notes}`;
    updateResults(result);
  } catch (error) {
    els.message.textContent = `Backend error: ${error.message}`;
  } finally {
    els.optimize.disabled = false;
  }
}

function bindActions() {
  els.addBin.addEventListener("click", () => {
    state.bins.push({ id: uid("bin"), name: `B${state.bins.length + 1}`, width: 100, height: 100, cost: 100 });
    renderInputs();
  });
  els.addPiece.addEventListener("click", () => {
    state.pieces.push({ id: uid("piece"), name: `P${state.pieces.length + 1}`, width: 20, height: 20, qty: 1 });
    renderInputs();
  });
  els.loadDemo.addEventListener("click", () => {
    loadDemoData();
    void optimize();
  });
  els.optimize.addEventListener("click", () => void optimize());
  els.stepSlider.addEventListener("input", () => {
    stopPlayback();
    setStep(Number(els.stepSlider.value));
  });
  els.prevStep.addEventListener("click", () => {
    stopPlayback();
    setStep(state.currentStep - 1);
  });
  els.nextStep.addEventListener("click", () => {
    stopPlayback();
    setStep(state.currentStep + 1);
  });
  els.playToggle.addEventListener("click", () => {
    if (state.playing) stopPlayback();
    else play();
  });
  els.speedSlider.addEventListener("input", () => {
    state.speed = Number(els.speedSlider.value);
    if (state.playing) play();
  });
}

bindActions();
loadDemoData();
initScene();
void optimize();
