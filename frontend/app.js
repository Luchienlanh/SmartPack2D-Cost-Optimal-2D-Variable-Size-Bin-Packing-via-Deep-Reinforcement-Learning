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
