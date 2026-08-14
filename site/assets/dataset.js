const escapeHtml = value => String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
const pretty = value => String(value).replaceAll("_", " ").replace(/\b\w/g, letter => letter.toUpperCase());

function factRows(object) {
  return Object.entries(object).map(([key, value]) => `<div><dt>${pretty(key)}</dt><dd>${escapeHtml(value === null ? "Not reported" : value)}</dd></div>`).join("");
}

function splitCollectionTokens(value) {
  if (value === null || value === undefined || value === "") return [];
  return String(value)
    .replace(/^[^:\n]+:\s*/, "")
    .split(/\s*[·•|]\s*|\s*,\s*/)
    .map(token => token.trim())
    .filter(Boolean)
    .flatMap(token => {
      const range = token.match(/^(.*?)(\d+)\s*[–-]\s*(\d+)$/);
      if (!range) return [token];
      const [, prefix, startText, endText] = range;
      const start = Number(startText);
      const end = Number(endText);
      if (!Number.isInteger(start) || !Number.isInteger(end) || end < start || end - start > 500) return [token];
      const width = Math.max(startText.length, endText.length);
      return Array.from({ length: end - start + 1 }, (_, index) => `${prefix}${String(start + index).padStart(width, "0")}`.trim());
    });
}

function collectionChipList(items, empty = "Not reported") {
  if (!items.length) return `<span class="collection-empty">${escapeHtml(empty)}</span>`;
  return `<div class="collection-chips">${items.map(item => `<span>${escapeHtml(item)}</span>`).join("")}</div>`;
}

function collectionTaskEntries(collection, dataset) {
  const labels = collection.labels;
  const sharedSettings = splitCollectionTokens(collection.scenario_labels);
  if (labels && typeof labels === "object" && !Array.isArray(labels)) {
    const entries = Object.entries(labels);
    if (!entries.length) return [];
    // Single generic "activity" bag → one expandable task row.
    if (entries.length === 1 && entries[0][0].toLowerCase() === "activity" && (dataset.tasks || []).length === 1) {
      const taskName = (dataset.tasks || []).length === 1
        ? pretty(dataset.tasks[0])
        : "Activity labels";
      return [{
        name: taskName,
        labels: splitCollectionTokens(entries[0][1]),
        summarized: typeof entries[0][1] === "string" && splitCollectionTokens(entries[0][1]).length === 1,
        settings: sharedSettings,
      }];
    }
    return entries.map(([name, value]) => ({
      name: pretty(name),
      labels: splitCollectionTokens(value),
      summarized: typeof value === "string" && splitCollectionTokens(value).length === 1,
      settings: sharedSettings,
    }));
  }
  if (typeof labels === "string" && labels.trim()) {
    return [{
      name: "Labels",
      labels: splitCollectionTokens(labels),
      summarized: splitCollectionTokens(labels).length === 1,
      settings: sharedSettings,
    }];
  }
  return [];
}

function collectionSettingSection(dataset, sample) {
  const collection = dataset.collection;
  if (!(collection && typeof collection === "object")) {
    return `<section class="collection-setting"><p class="section-step">01 · KNOW THE DATA</p><h2>Dataset information</h2><dl class="facts">${factRows({...dataset.settings, ...dataset.hardware})}</dl></section>`;
  }

  const semanticBins = sample?.standardized?.sample_rate_hz == null
    && (sample?.standardized?.axis_order || []).includes("doppler_bin");
  const compactFacts = [
    ["Scenario", collection.scenario || dataset.settings?.scenario],
    ["Distance", collection.distance || "Not reported"],
    ["Environments", dataset.settings?.environments ?? "Not reported"],
    ["Subjects", collection.subjects ?? dataset.settings?.subjects ?? "Not reported"],
    ["Devices", collection.device || dataset.hardware?.platform],
    ["Band", collection.band || dataset.hardware?.band || "Not reported"],
    ["Subcarriers", collection.subcarriers || "Source-dependent"],
    ["Sampling", semanticBins ? "Semantic time bins (rate not reported)" : (collection.sampling_rate_hz || "Source-dependent")],
    ["Clip length", collection.clip_length || "Source-dependent"],
  ].filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(String(value))}</dd></div>`)
    .join("");

  const tasks = collectionTaskEntries(collection, dataset);
  const taskBlocks = tasks.map(task => {
    const labelCount = task.labels.length;
    const labelSummary = task.summarized
      ? "source-defined"
      : `${labelCount} label${labelCount === 1 ? "" : "s"}`;
    const taskName = task.name.replace(/([a-z])([A-Z])/g, "$1 $2");
    return `<article class="collection-task">
      <header class="collection-task-header">
        <span class="collection-task-main">
          <strong>${escapeHtml(taskName)}</strong>
          <span class="collection-task-meta">${labelSummary}</span>
        </span>
      </header>
      <div class="collection-task-body compact-label-body">
        ${collectionChipList(task.labels, "No labels listed")}
      </div>
    </article>`;
  }).join("");

  return `<section id="collection" class="collection-setting">
    <p class="section-step">01 · KNOW THE DATA</p>
    <h2>Dataset information</h2>
    <p class="section-lead">A quick view of the data and recording setup.</p>
    ${compactFacts ? `<dl class="compact-dataset-facts" aria-label="Dataset and capture information">${compactFacts}</dl>` : ""}
    ${tasks.length ? `<div class="collection-tasks">
      <div class="collection-tasks-head">
        <h3>Tasks and labels</h3>
        <p>Labels are shown below.</p>
      </div>
      <div class="compact-task-list">${taskBlocks}</div>
    </div>` : ""}
  </section>`;
}

const formatBytes = bytes => {
  if (bytes === null || bytes === undefined) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

function renderTree(entries, depth = 0) {
  const items = entries.map(entry => {
    if (entry.type === "dir") {
      // Expand only the first folder layer; deeper dirs stay collapsed.
      const openAttr = depth < 1 ? " open" : "";
      return `<li class="tree-dir"><details${openAttr}><summary><span class="tree-name">${escapeHtml(entry.name)}/</span></summary>${renderTree(entry.children || [], depth + 1)}</details></li>`;
    }
    return `<li class="tree-file"><span class="tree-name">${escapeHtml(entry.name)}</span><small>${formatBytes(entry.bytes)}</small></li>`;
  });
  return `<ul class="file-tree">${items.join("")}</ul>`;
}

function structureBlock({ kind, badge, title, hint, body, empty }) {
  return `<article class="structure-block structure-block-${escapeHtml(kind)}"><header><span class="structure-badge">${escapeHtml(badge)}</span><h4>${escapeHtml(title)}</h4><p class="structure-pane-hint">${hint}</p></header><div class="structure-block-body">${body || `<p class="structure-empty">${escapeHtml(empty)}</p>`}</div></article>`;
}

function truncateMiddle(value, max = 52) {
  const text = String(value || "");
  if (text.length <= max) return text;
  const keep = Math.floor((max - 1) / 2);
  return `${text.slice(0, keep)}…${text.slice(-keep)}`;
}

function standardizedSchemaBody(sample) {
  const tree = sample.standardized_file_tree || [];
  const meta = tree[0] || {};
  const pattern = sample.standardized_pattern || meta.pattern || "{label}__{…}__{setting}.npz + .json";
  const example = sample.standardized_example || meta.example || {};
  const clipCount = sample.standardized_file_count ?? meta.clip_count ?? 0;
  if (!example.npz && !clipCount && !tree.length) return "";

  const rows = [];
  if (example.npz) {
    rows.push({
      name: truncateMiddle(example.npz, 44),
      full: example.npz,
      size: formatBytes(example.npz_bytes),
      last: !example.json && clipCount <= 1,
    });
  }
  if (example.json) {
    rows.push({
      name: truncateMiddle(example.json, 44),
      full: example.json,
      size: formatBytes(example.json_bytes),
      last: clipCount <= 1,
    });
  }
  if (clipCount > 1) {
    rows.push({
      name: `… +${clipCount - 1} more clips`,
      full: "",
      size: "same naming",
      last: true,
      muted: true,
    });
  }
  // Fallback abstract pair if example filenames are missing but clips exist.
  if (!rows.length && clipCount) {
    rows.push(
      { name: "clip__….npz", full: "", size: "tensor", last: false },
      { name: "clip__….json", full: "", size: "sidecar", last: true },
    );
  }

  const fileRows = rows.map(item => {
    const branch = item.last ? "└──" : "├──";
    const name = item.full
      ? `<code class="structure-schema-file" title="${escapeHtml(item.full)}">${escapeHtml(item.name)}</code>`
      : `<code class="structure-schema-file${item.muted ? " is-muted" : ""}">${escapeHtml(item.name)}</code>`;
    return `<li><span class="structure-schema-branch">${branch}</span>${name}<small>${escapeHtml(item.size)}</small></li>`;
  }).join("");

  return `<div class="structure-schema">
    <div class="structure-schema-folder"><code>standardized/</code><span>flat · ${clipCount} clip${clipCount === 1 ? "" : "s"}</span></div>
    <ul class="structure-schema-files">${fileRows}</ul>
    <div class="structure-schema-pattern"><span>Naming</span><code>${escapeHtml(pattern)}</code></div>
  </div>`;
}
function packagingSchemaBody(schema) {
  if (!schema) return "";
  const count = schema.folder_count || 0;
  const slot = schema.slot || "label";
  const example = schema.example || {};
  const exampleFiles = example.files || [];
  const leafFiles = schema.leaf_files || ["original.*", "standardized.npz", "metadata.json"];

  const pathRows = [];
  if (schema.nested) {
    pathRows.push({ name: "[task]/", branch: "├──" });
    pathRows.push({ name: `[${slot}]/`, branch: "│   └──" });
    leafFiles.forEach((name, index) => {
      pathRows.push({
        name,
        branch: index === leafFiles.length - 1 ? "│       └──" : "│       ├──",
      });
    });
  } else {
    pathRows.push({ name: `[${slot}]/`, branch: "├──" });
    leafFiles.forEach((name, index) => {
      pathRows.push({
        name,
        branch: index === leafFiles.length - 1 ? "│   └──" : "│   ├──",
      });
    });
  }

  const patternRows = pathRows.map(item => (
    `<li><span class="structure-schema-branch">${item.branch}</span><code class="structure-schema-file">${escapeHtml(item.name)}</code></li>`
  )).join("");

  const exampleFileRows = exampleFiles.map((file, index) => {
    const last = index === exampleFiles.length - 1;
    return `<li><span class="structure-schema-branch">${last ? "└──" : "├──"}</span><code class="structure-schema-file" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</code><small>${formatBytes(file.bytes)}</small></li>`;
  }).join("");

  const more = count > 1
    ? `<p class="structure-schema-more">+${count - 1} more ${escapeHtml(slot)} folder${count - 1 === 1 ? "" : "s"}</p>`
    : "";

  return `<div class="structure-schema">
    <div class="structure-schema-folder"><code>${escapeHtml(schema.root)}/</code><span>${count} folder${count === 1 ? "" : "s"}</span></div>
    <ul class="structure-schema-files">${patternRows}</ul>
    ${example.path ? `<div class="structure-schema-pattern"><span>Example</span><code>${escapeHtml(schema.root)}/${escapeHtml(example.path)}/</code></div>
    <ul class="structure-schema-files">${exampleFileRows}</ul>${more}` : ""}
  </div>`;
}

function usecasePickerBody(sample) {
  const labelSchema = sample.usecase_schema || null;
  const settingSchema = sample.setting_schema || null;
  const hasLabel = Boolean(labelSchema || (sample.usecase_file_tree || []).length);
  const hasSetting = Boolean(settingSchema || (sample.setting_file_tree || []).length);
  if (!hasLabel && !hasSetting) return "";

  const defaultMode = hasLabel ? "label" : "setting";
  const labelCount = sample.usecase_file_count ?? labelSchema?.folder_count ?? 0;
  const settingCount = sample.setting_file_count ?? settingSchema?.folder_count ?? 0;

  const labelBody = labelSchema
    ? packagingSchemaBody(labelSchema)
    : (sample.usecase_file_tree || []).length
      ? renderTree([{ name: "by_label", type: "dir", children: sample.usecase_file_tree }])
      : `<p class="structure-empty">No by_label packages yet.</p>`;
  const settingBody = settingSchema
    ? packagingSchemaBody(settingSchema)
    : (sample.setting_file_tree || []).length
      ? renderTree([{ name: "by_setting", type: "dir", children: sample.setting_file_tree }])
      : `<p class="structure-empty">No by_setting packages yet.</p>`;

  const both = hasLabel && hasSetting;
  const toggle = both
    ? `<div class="structure-mode-toggle" role="tablist" aria-label="Derived packaging mode">
        <button type="button" role="tab" data-mode="label" class="is-active" aria-selected="true">By label <small>${labelCount}</small></button>
        <button type="button" role="tab" data-mode="setting" aria-selected="false">By setting <small>${settingCount}</small></button>
      </div>`
    : `<div class="structure-mode-toggle is-static" role="tablist" aria-label="Derived packaging mode">
        <button type="button" role="tab" data-mode="${defaultMode}" class="is-active" aria-selected="true" disabled>${hasLabel ? "By label" : "By setting"} <small>${hasLabel ? labelCount : settingCount}</small></button>
      </div>`;

  return `<div class="usecase-picker" data-active="${defaultMode}">
    ${toggle}
    <div class="usecase-schema-panel" data-panel="label" ${defaultMode === "label" ? "" : "hidden"}>${hasLabel ? labelBody : ""}</div>
    <div class="usecase-schema-panel" data-panel="setting" ${defaultMode === "setting" ? "" : "hidden"}>${hasSetting ? settingBody : ""}</div>
  </div>`;
}

function bindUsecaseToggle(root) {
  root.querySelectorAll(".usecase-picker").forEach(picker => {
    const toggle = picker.querySelector(".structure-mode-toggle");
    if (!toggle || toggle.classList.contains("is-static")) return;
    toggle.addEventListener("click", event => {
      const button = event.target.closest("button[data-mode]");
      if (!button || button.disabled) return;
      const mode = button.dataset.mode;
      picker.dataset.active = mode;
      toggle.querySelectorAll("button[data-mode]").forEach(item => {
        const active = item === button;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-selected", active ? "true" : "false");
      });
      picker.querySelectorAll(".usecase-schema-panel").forEach(panel => {
        panel.hidden = panel.dataset.panel !== mode;
      });
    });
  });
}

function shapeText(shape) {
  if (!shape) return null;
  if (Array.isArray(shape)) return `[${shape.join(", ")}]`;
  return String(shape);
}

function perSampleSignalShape(sample) {
  const shape = sample?.standardized?.shape;
  const axes = sample?.standardized?.axis_order || [];
  if (!Array.isArray(shape)) return shapeText(shape);
  const perSample = axes[0] === "sample" ? shape.slice(1) : shape;
  return shapeText(perSample);
}

function perSampleSignalAxes(sample, fallback) {
  const axes = sample?.standardized?.axis_order || fallback;
  return axes[0] === "sample" ? axes.slice(1) : axes;
}

function hasCanonicalCsi(sample, dataset = null) {
  const axes = sample?.standardized?.axis_order || [];
  if (axes.includes("subcarrier") && axes.includes("tx_link") && axes.includes("rx_link")) return true;
  const native = dataset?.standardization?.native_shape || {};
  if (native.subcarriers && native.tx_links && native.rx_links) return true;
  return /\[T,\s*S,\s*Tx,\s*Rx\]/i.test(dataset?.conversion_example?.expected_shape || "");
}

function whatWeDoSection(dataset, sample) {
  const explicitAxes = hasCanonicalCsi(sample, dataset);
  const std = dataset.standardization || {};
  const example = dataset.conversion_example || {};
  const window = sample?.preview_window || {};
  const view = sample?.standardized_view?.profile || {};
  const before = sample?.original?.profile || {};
  const beforeDims = sample?.preview_before_dims || {};
  const afterShape = perSampleSignalShape(sample) || example.expected_shape || "—";
  const beforeShape = before.tensor_shape
    || (beforeDims.native_subcarriers && beforeDims.native_time_steps
      ? `${beforeDims.native_subcarriers} × ${beforeDims.native_time_steps} (S × T)`
      : "source-native");
  const rate = window.sample_rate_hz != null
    ? `${window.sample_rate_hz} Hz`
    : (view.sampling_rate || "profile default");
  const duration = view.duration
    || (window.time_steps != null ? `${window.time_steps} steps` : "profile default");
  const layout = perSampleSignalAxes(sample, explicitAxes
    ? ["time", "subcarrier", "tx_link", "rx_link"]
    : ["time", "link", "subcarrier"]).join(", ");
  const primary = example.primary_array || sample?.standardized?.standard_representation || "amplitude";
  const profile = view.task_profile || window.profile || std.profile || "native";
  const profileLabel = explicitAxes ? "task-based processing" : pretty(profile);

  return `<section id="overview" class="what-we-do">
    <p class="eyebrow">WHAT WISENSEHUB DOES HERE</p>
    <h2>Schema first, then task preprocess</h2>
    <p class="setting-help">Two jobs for this dataset: standardize the file contract, then apply <strong>${escapeHtml(profileLabel)}</strong> so comparable clips share a time rule.</p>
    <div class="job-cards">
      <article class="job-card">
        <span>01 · Schema</span>
        <h3>Standardize structure</h3>
        <p>Original release → flat <code>*.npz</code> + <code>*.json</code> with masks and provenance.</p>
      </article>
      <article class="job-card">
        <span>02 · Task view</span>
        <h3>Preprocess by task</h3>
        <p>${explicitAxes ? "Use a task-specific time rate. Keep native subcarriers, Tx, and Rx." : "Resample / crop / pad time only. Links and subcarriers stay native."}</p>
      </article>
    </div>
    <div class="param-strip" aria-label="Standardization parameters">
      <div><dt>Time rule</dt><dd>${explicitAxes ? "Chosen by task" : `<code>${escapeHtml(profile)}</code>`}</dd></div>
      <div><dt>Sampling rate</dt><dd>${escapeHtml(rate)}</dd></div>
      <div><dt>Duration / T</dt><dd>${escapeHtml(duration)}</dd></div>
      <div><dt>Layout</dt><dd><code>[${escapeHtml(layout)}]</code></dd></div>
      <div><dt>Primary array</dt><dd><code>${escapeHtml(primary)}</code></dd></div>
      <div><dt>Shape</dt><dd><span class="param-before">${escapeHtml(beforeShape)}</span><span class="param-arrow">→</span><code>${escapeHtml(afterShape)}</code></dd></div>
    </div>
    ${std.notes ? `<p class="setting-help">${escapeHtml(std.notes)}</p>` : ""}
  </section>`;
}

function sampleStructureSection(sample) {
  const originalTree = sample.original_file_tree || [];
  const standardizedTree = sample.standardized_file_tree || [];
  const hasSplit = originalTree.length || standardizedTree.length;
  if (!hasSplit) {
    if (!(sample.file_tree || []).length) return "";
    return `<section class="structure-compare"><h2>Structure: before → after</h2>${renderTree(sample.file_tree || [])}</section>`;
  }
  const originalCount = sample.original_file_count ?? 0;
  const standardizedCount = sample.standardized_file_count ?? 0;
  const left = structureBlock({
    kind: "original",
    badge: "Before",
    title: "Original release",
    hint: `${sample.kind === "generated-adapter-fixture" ? "Adapter fixture" : "Official layout"} · ${originalCount} file${originalCount === 1 ? "" : "s"}`,
    body: originalTree.length ? renderTree(originalTree) : "",
    empty: "No original sample files hosted.",
  });
  const middle = structureBlock({
    kind: "standardized",
    badge: "After",
    title: "Standardized schema",
    hint: `Flat folder · ${standardizedCount} clip${standardizedCount === 1 ? "" : "s"}`,
    body: standardizedSchemaBody(sample),
    empty: "Run wisensehub prepare to create standardized NPZ + JSON.",
  });
  return `<section id="structure" class="structure-compare">
    <h2>Structure: before → after</h2>
    <p class="compare-caption">Nested release folders → flat clip pairs (<code>npz</code> + <code>json</code>).</p>
    <div class="structure-split">${left}<div class="structure-vs" aria-hidden="true">→</div>${middle}</div>
  </section>`;
}

function packagingAsideSection(sample) {
  const picker = usecasePickerBody(sample);
  if (!picker) return "";
  const usecaseCount = sample.usecase_file_count ?? 0;
  const settingCount = sample.setting_file_count ?? 0;
  const available = [
    usecaseCount ? `${usecaseCount} by label` : null,
    settingCount ? `${settingCount} by setting` : null,
  ].filter(Boolean).join(" · ");
  return `<details class="packaging-aside">
    <summary><h2>Optional sample packaging</h2><span>${escapeHtml(available || "by label / by setting")}</span></summary>
    <p class="setting-help">Convenience folders for browsing—not a third standardization step. Each leaf keeps original + <code>standardized.npz</code> + <code>metadata.json</code>.</p>
    ${picker}
  </details>`;
}

function changedParametersTable(sample) {
  const canonicalCsi = hasCanonicalCsi(sample);
  const before = sample?.original?.profile || {};
  const view = sample?.standardized_view?.profile || {};
  const window = sample?.preview_window || {};
  const beforeDims = sample?.preview_before_dims || {};
  const afterShape = perSampleSignalShape(sample)
    || view.tensor_shape
    || "—";
  const beforeShape = before.tensor_shape
    || (beforeDims.native_subcarriers && beforeDims.native_time_steps
      ? `${beforeDims.native_subcarriers} S × ${beforeDims.native_time_steps} T`
      : "—");
  const beforeRate = before.sampling_rate || "not reported";
  const afterRate = view.sampling_rate
    || (window.sample_rate_hz != null ? `${window.sample_rate_hz} Hz` : "—");
  const beforeDuration = before.duration || (beforeDims.native_time_steps ? `${beforeDims.native_time_steps} packets` : "—");
  const afterDuration = view.duration
    || (window.time_steps != null ? `${window.time_steps} steps` : "—");
  const beforeRep = canonicalCsi ? "adapter-native signal" : "adapter-native processed features";
  const afterRep = view.representation || sample?.standardized?.standard_representation || "amplitude";
  const afterAxes = perSampleSignalAxes(sample, canonicalCsi
    ? ["time", "subcarrier", "tx_link", "rx_link"]
    : ["time", "link", "subcarrier"]).join(", ");
  const beforeAxes = (sample?.original?.dimensions || []).map(item => item.axis).join(", ");
  const sourceName = sample?.preview_source_file || "source file";
  const sourceFormat = sourceName === "csi_data_amp" ? "Zarr array" : (sourceName.includes(".") ? sourceName.split(".").pop().toUpperCase() : "source file");
  const txIndex = (sample?.standardized?.axis_order || []).indexOf("tx_link");
  const rxIndex = (sample?.standardized?.axis_order || []).indexOf("rx_link");
  const stdShape = sample?.standardized?.shape || [];
  const txCount = txIndex >= 0 ? stdShape[txIndex] : null;
  const rxCount = rxIndex >= 0 ? stdShape[rxIndex] : null;
  const originalHasLink = (sample?.original?.dimensions || []).some(item => item.axis === "link");
  const beforeAntenna = canonicalCsi
    ? (originalHasLink ? "flattened link axis" : "source antenna axes")
    : "processed feature axes";
  const afterAntenna = canonicalCsi ? `explicit ${txCount || 1} Tx × ${rxCount || 1} Rx` : "processed feature axes";
  const rows = [
    ["Sampling rate", beforeRate, afterRate],
    ["Duration / T", beforeDuration, afterDuration],
    ["Tensor shape", beforeShape, afterShape],
    ["Axis order", beforeAxes ? `[${beforeAxes}]` : "source-native axes", `[${afterAxes}]`],
    ["Antenna axes", beforeAntenna, afterAntenna],
    ["Validity", "not provided", "valid_mask for every time step"],
    ["File contract", `source ${sourceFormat}`, "NPZ arrays + JSON metadata"],
    ["Representation", beforeRep, afterRep],
  ].map(([label, left, right]) => (
    `<tr><th scope="row">${escapeHtml(label)}</th><td>${escapeHtml(left)}</td><td><strong>${escapeHtml(right)}</strong></td></tr>`
  )).join("");
  return `<div class="table-scroll"><table class="delta-table"><thead><tr><th>Parameter</th><th>Before (adapter-native)</th><th>After (task view)</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function sampleBeforeAfterSection(dataset, sample) {
  const explicitAxes = hasCanonicalCsi(sample, dataset);
  if (dataset.id === "signfi" && sample?.status !== "ok") {
    return `<section id="preview" class="preview-section compact restricted-preview">
      <p class="section-step">04 · PREVIEW</p>
      <h2>Preview stays on your computer</h2>
      <p class="section-lead">No CSI plot is embedded here. SignFi's terms prohibit publishing any constituent part of the dataset, so a real signal preview cannot be hosted without prior written approval.</p>
      <div class="restricted-shape-flow" aria-label="SignFi local schema conversion">
        <article><span>Official MAT tensor</span><strong><code>[200, 30, 3, samples]</code></strong><small>time · subcarrier · Rx · sample</small></article>
        <b aria-hidden="true">→</b>
        <article><span>Local WiSenseHub output</span><strong><code>[T, 30, 1, 3]</code></strong><small>time · subcarrier · Tx · Rx</small></article>
      </div>
      <p class="setting-help">The public project page reports 200 packets, 30 subcarriers, and 3 receive links per clip, but does not report the packet rate. WiSenseHub therefore preserves packet index and labels any requested task rate as an assumption.</p>
    </section>`;
  }
  if (dataset.id === "wifi-80mhz" && sample?.status !== "ok") {
    return `<section id="preview" class="preview-section compact restricted-preview">
      <p class="section-step">04 · PREVIEW</p>
      <h2>Before and after</h2>
      <p class="section-lead">The conversion is ready, but an authentic signal plot is not shown yet. The official archive needs a signed-in IEEE DataPort subscription, and no official MAT file is present in this workspace.</p>
      <div class="restricted-shape-flow three-stage" aria-label="80 MHz CSI schema conversion">
        <article><span>Before · official MAT</span><strong><code>[(T × 4), 242]</code></strong><small>interleaved monitor rows · subcarriers · 173 Hz</small></article>
        <b aria-hidden="true">→</b>
        <article><span>Native WiSenseHub file</span><strong><code>[T, 242, 1, 4]</code></strong><small>time · subcarrier · Tx · Rx</small></article>
        <b aria-hidden="true">→</b>
        <article><span>After · activity view</span><strong><code>[300, 242, 1, 4]</code></strong><small>3 seconds at 100 Hz · carriers and links unchanged</small></article>
      </div>
      <p class="setting-help">The adapter groups each four consecutive source rows into one time step, keeps the complex CSI, amplitude, and phase, and removes only fully empty packet groups. A real four-panel preview will appear after an authentic MAT file is supplied.</p>
    </section>`;
  }
  if (dataset.id === "wipe-fall" && sample?.status !== "ok") {
    return `<section id="preview" class="preview-section compact restricted-preview">
      <p class="section-step">04 · PREVIEW</p>
      <h2>Before and after</h2>
      <p class="section-lead">The real signal preview is waiting for an owner-supplied CSV. The University of Glasgow release is request-only, so the old synthetic fixture has been removed and is not shown as dataset evidence.</p>
      <div class="restricted-shape-flow three-stage" aria-label="WiPE-FaLl CSI schema conversion">
        <article><span>Before · official CSV</span><strong><code>[T, 51]</code></strong><small>time rows · amplitude subcarriers</small></article>
        <b aria-hidden="true">→</b>
        <article><span>Native WiSenseHub file</span><strong><code>[T, 51, 1, 1]</code></strong><small>time · subcarrier · Tx · Rx</small></article>
        <b aria-hidden="true">→</b>
        <article><span>After · fall-risk view</span><strong><code>[up to 300, 51, 1, 1]</code></strong><small>100 Hz task grid · source rate marked unknown</small></article>
      </div>
      <p class="setting-help">WiSenseHub keeps all 51 subcarriers, reads low / medium / high from the six official folders, and preserves the released training-versus-unseen split. Because the README does not report a CSI sampling rate, 100 Hz is explicitly stored as a target-grid assumption.</p>
    </section>`;
  }
  const beforeDims = sample?.preview_before_dims || {};
  const multiRx = (beforeDims.native_tx_links || 1) > 1 || (beforeDims.native_rx_links || 1) > 1;
  const previews = sample.previews || {};
  if (!previews.before && !previews.after) return "";
  const window = sample.preview_window || {};
  const profile = dataset.sample_plan?.profile || window.profile || sample.standardized_view?.profile?.task_profile || dataset.standardization?.profile;
  const previewProfile = explicitAxes ? "" : profile;
  const generated = sample.kind === "generated-adapter-fixture";
  const panel = (title, caption, src, downloadUrl = null) => src
    ? `<figure class="preview-panel compact${multiRx ? " multi-rx-panel" : ""}"><figcaption><strong>${title}</strong><span>${escapeHtml(caption)}</span>${downloadUrl ? `<a class="preview-sample-download" href="${escapeHtml(downloadUrl)}" download>Download this sample</a>` : ""}</figcaption><img src="${escapeHtml(src)}" alt="${title} signal preview" loading="lazy"></figure>`
    : "";
  const labelGroups = new Map();
  (sample.label_previews || []).forEach(item => {
    const task = item.task || "Other labels";
    if (!labelGroups.has(task)) labelGroups.set(task, []);
    labelGroups.get(task).push(item);
  });
  const labelTaskSections = [...labelGroups.entries()].map(([task, items]) => {
    const cards = items.map(item => panel(
      escapeHtml(item.label),
      `${item.kind === "segment" ? "segment" : "clip"}${item.source_file ? ` · ${item.source_file}` : ""}`,
      item.image,
      item.sample_dir ? `${item.sample_dir}/standardized.npz` : null,
    )).join("");
    return `<section class="preview-task-group"><header><h3>${escapeHtml(task)}</h3><span>${items.length} label${items.length === 1 ? "" : "s"}</span></header><div class="preview-label-grid">${cards}</div></section>`;
  }).join("");
  const totalLabelPreviews = [...labelGroups.values()].reduce((total, items) => total + items.length, 0);
  const scaleNote = dataset.id === "widar3"
    ? "Widar BVP is sparse by design. A nonlinear display scale reveals low-power velocity components; downloaded values are unchanged."
    : dataset.id === "wimans"
      ? "Each real antenna link uses its own 2nd–98th percentile display scale, so low-gain links stay visible. Downloaded values are unchanged."
      : "Colors use each clip's 2nd–98th percentile for display, so isolated spikes do not hide the CSI pattern. Downloaded values are unchanged.";
  const labelSectionTitle = dataset.id === "wimans"
    ? `Per-label and setting samples · ${totalLabelPreviews} values in ${labelGroups.size} groups`
    : `Per-label signal samples · ${totalLabelPreviews} labels in ${labelGroups.size} task${labelGroups.size === 1 ? "" : "s"}`;
  const labelSection = labelTaskSections
    ? `<div class="preview-label-details is-direct"><h3 class="preview-label-heading">${escapeHtml(labelSectionTitle)}</h3><p class="setting-help preview-scale-note">${escapeHtml(scaleNote)}</p><div class="preview-task-groups">${labelTaskSections}</div></div>`
    : "";
  return `<section id="preview" class="preview-section compact sample-compare${multiRx ? " multi-rx-preview" : ""}">
    <p class="section-step">04 · PREVIEW</p>
    <h2>Before and after</h2>
    <p class="setting-help preview-lead ${generated ? "synthetic-note" : ""}">${generated ? "Synthetic shape demo—not a recorded dataset measurement. It verifies the real adapter and dimensions. " : "Official mini-sample parsed by the dataset adapter. "}${escapeHtml(sample.preview_source_file || "source file")} → task view.</p>
    <div class="preview-pair compact">${panel(generated ? "Before · synthetic source fixture" : "Before · adapter-native", `${sample?.original?.profile?.tensor_shape || "native parsed axes"} · ${sample?.original?.profile?.sampling_rate || "source rate"}`, previews.before)}${panel(generated ? "After · synthetic task view" : "After · task view", `${previewProfile ? `${previewProfile} · ` : ""}${perSampleSignalShape(sample) || (explicitAxes ? "[T, S, Tx, Rx]" : "feature view")} · ${sample?.standardized_view?.profile?.sampling_rate || "task rate"} + masks + JSON`, previews.after)}</div>
    <details class="preview-change-details"><summary>See the exact schema changes</summary>${changedParametersTable(sample)}</details>
    ${labelSection}
  </section>`;
}

function sampleDownloadSection(dataset, sample, splitConfig) {
  if (!sample || sample.status !== "ok") {
    if (sample?.reason) {
      const officialUrl = sample.source_url || dataset.original?.download_page || dataset.original?.landing_page;
      if (dataset.id === "signfi") {
        return `<section id="sample" class="sample-action-section restricted-sample-section">
          <p class="section-step">02 · USE IT LOCALLY</p>
          <h2>Bring your authorized SignFi copy</h2>
          <p class="section-lead">${escapeHtml(sample.reason)}</p>
          <ol class="restricted-local-steps">
            <li><span>1</span><div><strong>Read and accept the official terms</strong><p>Download the MAT file yourself from the SignFi project page.</p>${officialUrl ? `<a class="button secondary" href="${escapeHtml(officialUrl)}" target="_blank" rel="noreferrer">Open SignFi terms and files</a>` : ""}</div></li>
            <li><span>2</span><div><strong>Place the file here</strong><pre><code>data/signfi/original/dataset_lab_276_dl.mat</code></pre></div></li>
            <li><span>3</span><div><strong>Process it locally</strong><pre><code>wisensehub prepare signfi \\
  --data-root data \\
  --setting official_groups \\
  --profile general-sensing</code></pre><p>The generated NPZ, JSON, and plots remain on your computer. Do not publish them unless the owner gives written permission.</p></div></li>
          </ol>
        </section>`;
      }
      if (dataset.id === "wifi-80mhz") {
        return `<section id="sample" class="sample-action-section restricted-sample-section">
          <p class="section-step">02 · USE IT LOCALLY</p>
          <h2>Bring one official MAT file</h2>
          <p class="section-lead">${escapeHtml(sample.reason)}</p>
          <ol class="restricted-local-steps">
            <li><span>1</span><div><strong>Download from IEEE DataPort</strong><p>Sign in with an account that has DataPort access. The dataset license is CC BY 4.0; the download gate is an access requirement, not a ban on reuse.</p>${officialUrl ? `<a class="button secondary" href="${escapeHtml(officialUrl)}" target="_blank" rel="noreferrer">Open the official dataset</a>` : ""}</div></li>
            <li><span>2</span><div><strong>Place the MAT file here</strong><pre><code>data/wifi-80mhz/original/&lt;official subset&gt;/AR1a_W.mat</code></pre><p>Keep the original subset and filename so the task, setting, and label remain readable.</p></div></li>
            <li><span>3</span><div><strong>Ask the agent, or run locally</strong><pre><code>Use $wisensehub to prepare my WiFi 80 MHz MAT file and explain each step.</code></pre><pre><code>wisensehub prepare wifi-80mhz \
  --data-root data \
  --setting random \
  --profile general-sensing</code></pre><p>WiSenseHub creates native <code>[T, 242, 1, 4]</code> files and 100 Hz activity views without changing the carrier or monitor dimensions.</p></div></li>
          </ol>
        </section>`;
      }
      if (dataset.id === "wipe-fall") {
        return `<section id="sample" class="sample-action-section restricted-sample-section">
          <p class="section-step">02 · GET THE DATA</p>
          <h2>Request the official WiPE-FaLl files</h2>
          <p class="section-lead">${escapeHtml(sample.reason)}</p>
          <ol class="restricted-local-steps">
            <li><span>1</span><div><strong>Use the official Request Data form</strong><p>The dataset is CC BY 4.0, but the University of Glasgow supplies the measurement files only after a request.</p>${officialUrl ? `<a class="button secondary" href="${escapeHtml(officialUrl)}" target="_blank" rel="noreferrer">Open WiPE-FaLl and request data</a>` : ""}</div></li>
            <li><span>2</span><div><strong>Keep the six folders intact</strong><pre><code>data/wipe-fall/original/
├── low/          ├── low_unseen/
├── med/          ├── med_unseen/
├── high/         └── high_unseen/</code></pre><p>The folder name carries both the fall-risk label and the official train/unseen assignment.</p></div></li>
            <li><span>3</span><div><strong>Ask the agent to prepare it</strong><pre><code>Use $wisensehub to prepare my WiPE-FaLl files and explain each step.</code></pre><pre><code>wisensehub prepare wipe-fall \
  --data-root data \
  --setting official_unseen \
  --profile general-sensing</code></pre><p>Each real CSV becomes <code>[T, 51, 1, 1]</code>; the derived fall-risk view keeps all signal dimensions and records the unknown source rate honestly.</p></div></li>
          </ol>
        </section>`;
      }
      return `<section id="sample"><p class="section-step">02 · TRY IT</p><h2>Real data demo</h2><p class="setting-help">${escapeHtml(sample.reason)}</p>${officialUrl ? `<a class="button secondary" href="${escapeHtml(officialUrl)}" target="_blank" rel="noreferrer">Open official data</a>` : ""}</section>`;
    }
    return "";
  }
  const fixture = sample.kind === "generated-adapter-fixture";
  const csiBench = dataset.id === "csi-bench";
  const automaticAxes = hasCanonicalCsi(sample, dataset);
  const plan = dataset.sample_plan || {};
  const setting = plan.split_setting || (csiBench ? "random" : splitConfig.default);
  const profile = plan.profile || "general-sensing";
  const zipUrl = csiBench ? "samples/csi-bench-original-subset.zip" : sample.sample_zip;
  const zipSize = csiBench ? "6.2 MB" : formatBytes(sample.zip_bytes);
  const sampleTitle = fixture ? "Generated adapter demo" : "Official mini-sample";
  const sampleDescription = fixture
    ? `Matches the documented ${dataset.name} source layout and exercises the real adapter. It is not an official measurement.`
    : (sample.note || "A small subset of the official release that preserves its source layout.");
  const downloadLabel = fixture ? "Download demo" : "Download sample";
  const officialFetchOnly = sample.delivery === "official-fetch";
  const labelTaskCounts = new Map();
  (sample.label_previews || []).forEach(item => {
    const task = item.task || "Labels";
    labelTaskCounts.set(task, (labelTaskCounts.get(task) || 0) + 1);
  });
  const labelCoverage = [...labelTaskCounts.entries()]
    .map(([task, count]) => `${count} ${task}`)
    .join(" · ");
  return `<section id="sample" class="sample-action-section">
    <p class="section-step">02 · TRY IT</p>
    <h2>Try this dataset</h2>
    <p class="section-lead">${officialFetchOnly ? "Start with the agent, or fetch the exact official files locally." : "Start with the agent, or download the small package yourself."}</p>
    <div class="sample-action-grid">
      <article class="agent-action-card">
        <span class="sample-ready-badge">Recommended</span>
        <h3>Ask WiSenseHub</h3>
        <p>It downloads, places, processes, checks, and explains the sample.</p>
        <pre><code>Use $wisensehub to prepare the ${escapeHtml(dataset.name)} sample and explain each step.</code></pre>
      </article>
      <article class="download-action-card ${fixture ? "is-fixture" : "is-official"}">
        <span class="sample-ready-badge">${escapeHtml(sampleTitle)}</span>
        <h3>${escapeHtml(dataset.name)} sample</h3>
        <p>${escapeHtml(sampleDescription)}</p>
        ${labelCoverage ? `<p class="sample-label-coverage"><strong>Labeled examples:</strong> ${escapeHtml(labelCoverage)}</p>` : ""}
        ${officialFetchOnly
          ? `<pre><code>python scripts/fetch_samples.py ${escapeHtml(dataset.id)}</code></pre><p class="setting-help">Downloads the exact real label samples from the official source. Raw files stay on your computer.</p>`
          : `<a class="button secondary" href="${escapeHtml(zipUrl)}" download>${downloadLabel} · ${zipSize}</a>`}
      </article>
    </div>
    <details class="compact-command-details"><summary>Show the processing command</summary><pre><code>wisensehub prepare ${escapeHtml(dataset.id)} \\
  --data-root data \\
  --setting ${escapeHtml(setting)}${automaticAxes ? "" : ` \\\n  --profile ${escapeHtml(profile)}`}</code></pre></details>
    ${fixture ? `<p class="sample-license-note"><strong>Demo only:</strong> use the official release for research results and benchmark claims.</p>` : ""}
  </section>`;
}

function setupFigureSection(dataset, sample) {
  if (!sample?.setup_figure) return "";
  const source = sample.figure_source;
  const attribution = source
    ? `<p class="figure-source">Source: <a href="${escapeHtml(source.url)}">${escapeHtml(source.label)}</a></p>`
    : "";
  return `<section class="setup-figure-top"><h2>Experimental Setup</h2><figure class="setup-figure"><img src="${escapeHtml(sample.setup_figure)}" alt="Experimental setup for ${escapeHtml(dataset.name)}" loading="lazy"></figure>${attribution}</section>`;
}

function standardizationPlanSection(dataset, sample) {
  const explicitAxes = hasCanonicalCsi(sample, dataset);
  const tasks = collectionTaskEntries(dataset.collection || {}, dataset);
  const rows = tasks.map(task => {
    const isVitalSign = /breath|vital/i.test(task.name);
    const profile = isVitalSign ? "vital-sign" : "general-sensing";
    const rate = isVitalSign ? "10 Hz" : "100 Hz";
    const duration = "Flexible by task";
    const shape = explicitAxes ? "[T, S, Tx, Rx]" : "task feature axes";
    return `<tr>
      <th scope="row">${escapeHtml(task.name)}</th>
      <td><span class="profile-pill ${isVitalSign ? "vital" : "general"}">${escapeHtml(profile)}</span></td>
      <td>${rate}</td>
      <td>${duration}</td>
      <td><code>${shape}</code></td>
      <td><span class="ready-state">Ready</span></td>
    </tr>`;
  }).join("");
  return `<section id="plan" class="plan-section">
    <p class="section-step">02 · CHOOSE A TASK VIEW</p>
    <h2>Standardization plan by task</h2>
    <p class="section-lead">Every source becomes the same NPZ + JSON schema. Time uses the task rate; ${explicitAxes ? "subcarriers, Tx, and Rx" : "links and subcarriers"} stay native by default.</p>
    <div class="two-step-model" aria-label="Two-step standardization model">
      <div><span>1</span><strong>Standardize structure</strong><small>HDF5 / MAT / CSV → NPZ + metadata</small></div>
      <b aria-hidden="true">→</b>
      <div><span>2</span><strong>Standardize signals</strong><small>${explicitAxes ? "Task rate + flexible [T, S, Tx, Rx]" : "Task rate + documented feature axes"}</small></div>
    </div>
    <div class="table-scroll"><table class="plan-table"><thead><tr><th>Task</th><th>Profile</th><th>Rate</th><th>Duration</th><th>Output shape</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="assumption-note"><strong>Important:</strong> When a source does not report a sampling rate, WiSenseHub records the target-grid assumption used by the selected profile instead of presenting it as measured source metadata.</p>
  </section>`;
}

function datasetProcessingSection(dataset, sample, splitConfig) {
  const csiBench = dataset.id === "csi-bench";
  const explicitAxes = hasCanonicalCsi(sample, dataset);
  const semanticBins = sample?.standardized?.sample_rate_hz == null
    && (sample?.standardized?.axis_order || []).includes("doppler_bin");
  const plan = dataset.sample_plan || {};
  const windowPolicy = plan.window_policy || {};
  const selectedWindow = windowPolicy.selected_window_length || plan.target_length || 300;
  const tasks = collectionTaskEntries(dataset.collection || {}, dataset);
  const profileGroups = new Map();
  tasks.forEach(task => {
    const profile = /breath|vital/i.test(task.name) ? "vital-sign" : "general-sensing";
    if (!profileGroups.has(profile)) profileGroups.set(profile, []);
    profileGroups.get(profile).push(task.name);
  });
  if (!profileGroups.size) {
    profileGroups.set(plan.profile || "general-sensing", ["Dataset sample"]);
  }
  const presets = [...profileGroups.entries()].map(([profile, names]) => {
    const vital = profile === "vital-sign";
    return `<article class="processing-preset">
      <div><span class="profile-pill ${vital ? "vital" : "general"}">${escapeHtml(explicitAxes ? (vital ? "Breathing" : "Other tasks") : profile)}</span><strong>${escapeHtml(names.join(" · "))}</strong></div>
      <code>${semanticBins ? "native time bins" : (vital ? "10 Hz" : "100 Hz")} · ${explicitAxes ? "[T, S, Tx, Rx]" : "task feature axes"}</code>
    </article>`;
  }).join("");
  const sourceFormats = (dataset.original?.formats || ["Source files"]).join(" · ");
  const sourceExample = dataset.conversion_example?.source_path || "source file";
  const adapter = dataset.conversion?.handler || dataset.conversion?.implementation || "dataset adapter";
  const primary = sample?.standardized?.standard_representation || dataset.conversion_example?.primary_array || "amplitude";
  const shape = explicitAxes ? "[T, S, Tx, Rx]" : (shapeText(sample?.standardized?.shape) || dataset.conversion_example?.expected_shape || "task feature axes");
  const standardizedAxes = sample?.standardized?.axis_order || [];
  const standardizedShape = sample?.standardized?.shape || [];
  const sizeFor = axis => {
    const index = standardizedAxes.indexOf(axis);
    return index >= 0 ? standardizedShape[index] : null;
  };
  const nativeShape = dataset.standardization?.native_shape || {};
  const requestedSubcarriers = sizeFor("subcarrier") || nativeShape.subcarriers || Number(dataset.collection?.subcarriers) || 52;
  const requestedTxLinks = sizeFor("tx_link") || nativeShape.tx_links || 1;
  const requestedRxLinks = sizeFor("rx_link") || nativeShape.rx_links || 1;
  const setting = plan.split_setting || splitConfig.default;
  return `<section id="processing" class="dataset-processing-section">
    <p class="section-step">03 · PROCESSING</p>
    <h2>How this dataset is processed</h2>
    <p class="section-lead">Only the adapter and task presets below are specific to ${escapeHtml(dataset.name)}.</p>
    <div class="dataset-pipeline" aria-label="Dataset processing pipeline">
      <article><span>Input</span><strong>${escapeHtml(sourceFormats)}</strong><small>${escapeHtml(sourceExample)}</small></article>
      <b aria-hidden="true">→</b>
      <article><span>Adapter</span><strong>${escapeHtml(pretty(adapter))}</strong><small>Reads this release's source layout</small></article>
      <b aria-hidden="true">→</b>
      <article class="is-primary"><span>Output</span><strong>NPZ + JSON</strong><small><code>${escapeHtml(primary)}</code> · ${escapeHtml(shape)}</small></article>
    </div>
    <div class="processing-summary-row">
      <div><span>Window length</span><strong><code>${explicitAxes ? "Flexible by task" : `${escapeHtml(String(selectedWindow))} ${semanticBins ? "time bins" : "steps"}`}</code></strong></div>
      <div><span>Coverage</span><strong>${plan.full_source_preserved ? "Full source kept" : "Segmented view"}</strong></div>
      <div><span>Sample split</span><strong><code>${escapeHtml(setting)}</code></strong></div>
    </div>
    <p class="segment-policy-note">${semanticBins ? "The BVP cadence is not reported, so WiSenseHub keeps its native time bins and does not claim a sampling rate." : (explicitAxes ? "For each task, the window follows its shortest resampled recording up to the task limit." : "The window is capped by the shortest resampled recording.")} Longer recordings become consecutive windows; only the last remainder is padded and marked by <code>valid_mask</code>.</p>
    <div class="processing-presets"><h3>Task presets used here</h3>${presets}${explicitAxes ? "<p class=\"setting-help\">T = time · S = subcarriers · Tx = tx_link · Rx = rx_link</p>" : ""}</div>
    ${explicitAxes ? `<div class="sample-save-command"><div><strong>Ask for the shape you need</strong><span>The agent selects the task rate and explains every change.</span></div><pre><code>Use $wisensehub to prepare ${escapeHtml(dataset.name)} ${csiBench ? "fall" : "activity"} data with ${requestedSubcarriers} subcarriers, ${requestedTxLinks} Tx, and ${requestedRxLinks} Rx.</code></pre><small>Default: keep native S, Tx, and Rx. Exact request: <code>--subcarriers ${requestedSubcarriers} --tx-links ${requestedTxLinks} --rx-links ${requestedRxLinks}</code>.</small></div>` : ""}
    <div class="shared-guide-callout">
      <div><span>Shared documentation</span><strong>Options, masks, schemas, and Python examples are explained once.</strong></div>
      <a class="button secondary" href="guide.html#preprocessing">Open how-it-works guide →</a>
    </div>
  </section>`;
}

function workflowCommandsSection(dataset, splitConfig) {
  const base = `wisensehub prepare ${dataset.id} \\\n+  --data-root data \\\n+  --setting ${splitConfig.default}`;
  return `<section id="prepare" class="workflow-section">
    <p class="section-step">03 · RUN IT</p>
    <h2>Standardize the dataset</h2>
    <p class="section-lead">Place the intact release in <code>data/${escapeHtml(dataset.id)}/original/</code>, then run the profile that matches the task you are preparing. One command writes the canonical files and the fixed task view.</p>
    <div class="command-setup"><span>Install once</span><code>pip install -e ".[data]"</code></div>
    <div class="profile-commands">
      <article>
        <div class="command-card-head"><div><span class="profile-pill vital">vital-sign</span><h3>Breathing detection</h3></div><small>10 Hz · 30 s · 300 steps</small></div>
        <pre><code>${escapeHtml(base)} \\\n+  --profile vital-sign</code></pre>
      </article>
      <article>
        <div class="command-card-head"><div><span class="profile-pill general">general-sensing</span><h3>All other sensing tasks</h3></div><small>100 Hz · 3 s · 300 steps</small></div>
        <pre><code>${escapeHtml(base)} \\\n+  --profile general-sensing</code></pre>
      </article>
    </div>
  </section>`;
}

function outputGuideSection(dataset, sample) {
  const plan = dataset.sample_plan || {};
  const exampleName = sample?.standardized_example?.npz || `${dataset.id}__sample.npz`;
  const sourceExample = dataset.conversion_example?.source_path || `sample.${(dataset.original?.formats || ["dat"])[0].toLowerCase()}`;
  const setting = plan.split_setting || "random";
  const arrayName = sample?.standardized?.standard_representation || "amplitude";
  const labelExample = sample?.label_samples?.[0];
  const labelFolder = labelExample?.sample_dir ? `site/${labelExample.sample_dir}` : `site/samples/${dataset.id}/by_label/[label]`;
  const labelName = labelExample?.label || "sample label";
  const settingExample = sample?.setting_samples?.[0];
  const settingCard = settingExample ? `<details class="output-use-case" name="output-use-case">
    <summary><span><small>USE CASE 3 · OPTIONAL VIEW</small><strong>Compare collection settings</strong></span><em>${escapeHtml(settingExample.setting || "setting")}</em></summary>
    <div class="use-case-body"><div class="use-case-structure"><span>Folder</span><pre><code>${escapeHtml(settingExample.sample_dir || `site/samples/${dataset.id}/by_setting/`)}</code></pre></div><div class="use-case-demo"><span>What it means</span><p>Each setting folder keeps a small original example, a standardized tensor, and metadata. Use it to compare environments, devices, or difficulty without changing the canonical schema.</p></div></div>
  </details>` : "";
  return `<section id="outputs" class="output-guide">
    <p class="section-step">04 · USE THE OUTPUT</p>
    <h2>What you get</h2>
    <p class="section-lead">WiSenseHub creates one stable NPZ + JSON format. You can then open the same data in different ways based on what you want to do.</p>
    <div class="output-flow">
      <article><span>Input</span><h3>Source layout</h3><pre><code>original/
└── ${escapeHtml(sourceExample)}</code></pre></article>
      <b aria-hidden="true">→</b>
      <article class="is-primary"><span>Canonical output</span><h3>Standardized schema</h3><pre><code>standardized/
├── ${escapeHtml(exampleName)}
└── ${escapeHtml(exampleName.replace(/\.npz$/, ".json"))}</code></pre></article>
      <b aria-hidden="true">→</b>
      <article><span>Flexible access</span><h3>Choose a use case</h3><pre><code>1. Train with a split
2. Inspect an example
3. Compare optional views</code></pre></article>
    </div>
    <div class="use-case-intro">
      <h3>Choose how you want to use the data</h3>
      <p>These examples use this dataset's generated paths and recommended sample split.</p>
    </div>
    <div class="output-use-cases">
      <details class="output-use-case" name="output-use-case" open>
        <summary><span><small>USE CASE 1 · CREATED BY PREPARE</small><strong>Train or evaluate a model</strong></span><em>Use the saved train / val / test split</em></summary>
        <div class="use-case-body">
          <div class="use-case-structure"><span>Folder structure</span><pre><code>data/${escapeHtml(dataset.id)}/
├── standardized/views/
│   └── ${escapeHtml(exampleName)}
└── splits/
    └── ${escapeHtml(setting)}.json</code></pre></div>
          <div class="use-case-demo"><span>Python demo</span><p>Open the first training file listed in the split.</p><pre><code>import json
from pathlib import Path
import numpy as np

root = Path("data/${escapeHtml(dataset.id)}")
split = json.loads((root / "splits/${escapeHtml(setting)}.json").read_text())

sample_id = split["partitions"]["train"][0]
relative_path, separator, window = sample_id.partition("::")
file_path = root / relative_path

with np.load(file_path) as sample:
    signal = sample["${escapeHtml(arrayName)}"]
    mask = sample["valid_mask"]
    if separator:
        signal = signal[int(window)]
        mask = mask[int(window)]

print("file:", file_path.name)
print("shape:", signal.shape)
print("valid steps:", int(mask.sum()))</code></pre></div>
        </div>
      </details>
      <details class="output-use-case" name="output-use-case">
        <summary><span><small>USE CASE 2 · SAMPLE VIEW</small><strong>Inspect one example</strong></span><em>${escapeHtml(labelName)}</em></summary>
        <div class="use-case-body">
          <div class="use-case-structure"><span>Folder structure</span><pre><code>${escapeHtml(labelFolder)}/
├── original.*
├── standardized.npz
└── metadata.json</code></pre></div>
          <div class="use-case-demo"><span>Python demo</span><p>Open the standardized example and read its metadata.</p><pre><code>import json
from pathlib import Path
import numpy as np

folder = Path("${escapeHtml(labelFolder)}")

info = json.loads((folder / "metadata.json").read_text())
with np.load(folder / "standardized.npz") as sample:
    signal = sample["${escapeHtml(arrayName)}"]

print("label:", info.get("label", "${escapeHtml(labelName)}"))
print("shape:", signal.shape)</code></pre></div>
        </div>
      </details>
      ${settingCard}
    </div>
  </section>`;
}

function preprocessingChoicesSection(dataset, splitConfig) {
  const csiBench = dataset.id === "csi-bench";
  const setting = dataset.sample_plan?.split_setting || (dataset.id === "csi-bench" ? "random" : splitConfig.default);
  const settingExplainer = dataset.id === "csi-bench"
    ? `<div class="split-setting-explainer">
        <div class="split-setting-title"><code>--setting</code><span>Data split · not signal preprocessing</span></div>
        <p>Choose how samples are divided into <strong>train</strong>, <strong>validation</strong>, and <strong>test</strong>. This option does not change CSI values or tensor shapes.</p>
        <div class="split-setting-choices">
          <article><code>--setting random</code><strong>Use for the small sample</strong><span>Creates a repeatable 70% / 15% / 15% split.</span></article>
          <article><code>--setting official_id</code><strong>Use for the complete release</strong><span>Reads the official CSI-Bench ID files. They are not included in the small sample.</span></article>
        </div>
        <small>Saved to <code>data/csi-bench/splits/&lt;setting&gt;.json</code></small>
      </div>`
    : `<div class="split-setting-explainer">
        <div class="split-setting-title"><code>--setting</code><span>Data split · not signal preprocessing</span></div>
        <p>Choose how samples are divided into train, validation, and test. This runnable sample uses <code>${escapeHtml(setting)}</code>; the complete release default is <code>${escapeHtml(splitConfig.default)}</code>.</p>
      </div>`;
  const settingChoices = dataset.id === "csi-bench"
    ? "random · official_id"
    : (splitConfig.settings || []).map(item => item.id).join(" · ") || splitConfig.default;
  const rows = [
    ["--setting", settingChoices, "Choose the train / validation / test split; does not change CSI signals"],
    ["--profile", csiBench ? "Automatic by task · optional manual choice" : "vital-sign · general-sensing", "Choose the task time rule"],
    ["--target-rate", "Any positive Hz", "Resample when source rate is known; otherwise record a target-grid assumption"],
    ["--duration", "Any positive seconds", "Crop or pad to a fixed time window"],
    ["--target-length", "Any positive integer", "Force an exact number of time steps"],
    ["--interpolation", "linear · nearest · none", "Choose how time steps are resized"],
    ["--layout", "canonical · flat · link-subcarrier", "Keep [T, S, Tx, Rx] or flatten signal axes"],
    ["--tx-links", "Any positive integer", "Select or zero-pad Tx positions"],
    ["--rx-links", "Any positive integer", "Select or zero-pad Rx positions"],
    ["--subcarriers", "Any positive integer", "Center-crop or pad subcarriers"],
  ].map(([option, choices, effect]) => `<tr><th scope="row"><code>${escapeHtml(option)}</code></th><td>${escapeHtml(choices)}</td><td>${escapeHtml(effect)}</td></tr>`).join("");
  return `<section id="options" class="preprocess-options">
    <p class="section-step">02B · ADJUST IF NEEDED</p>
    <h2>Preprocessing choices</h2>
    <p class="section-lead">Start with a task profile. Use custom options only when your model needs a different time window, shape, or channel count.</p>
    <div class="preprocess-profiles">
      <article><span class="profile-pill vital">vital-sign</span><strong>10 Hz · flexible T</strong><code>[T, S, Tx, Rx]</code><small>Breathing and other vital signals</small></article>
      <article><span class="profile-pill general">general-sensing</span><strong>100 Hz · flexible T</strong><code>[T, S, Tx, Rx]</code><small>Activity, fall, location, identity, and proximity</small></article>
    </div>
    ${settingExplainer}
    <div class="table-scroll"><table class="preprocess-table"><thead><tr><th>Option</th><th>You can choose</th><th>What it changes</th></tr></thead><tbody>${rows}</tbody></table></div>
    <div class="preprocess-example-grid">
      <article class="preprocess-command-card">
        <span>Custom example</span>
        <h3>Make a 4-second view at 50 Hz</h3>
        <pre><code>wisensehub prepare ${escapeHtml(dataset.id)} \\
  --data-root data \\
  --setting ${escapeHtml(setting)} \\
  --target-rate 50 \\
  --duration 4 \\
  --interpolation linear \\
  --layout canonical</code></pre>
        <p>Result: <code>50 × 4 = 200</code> steps, so the CSI shape is <code>[200, S, Tx, Rx]</code>.</p>
      </article>
      <article class="preprocess-rules-card">
        <span>Resize choices</span>
        <dl>
          <div><dt>linear</dt><dd>Smoothly interpolate floating-point CSI values.</dd></div>
          <div><dt>nearest</dt><dd>Copy the closest source time step.</dd></div>
          <div><dt>none</dt><dd>Crop extra steps or add zero padding. The mask marks padding as invalid.</dd></div>
        </dl>
      </article>
    </div>
    <p class="preprocess-note"><strong>Safe by design:</strong> the native standardized NPZ stays unchanged. These options create a derived file under <code>data/${escapeHtml(dataset.id)}/standardized/views/</code>. Custom rate, duration, or length values override the profile defaults.</p>
  </section>`;
}

function learnerCommandsSection(dataset, splitConfig, sample) {
  const plan = dataset.sample_plan || {};
  const setting = plan.split_setting || (dataset.id === "csi-bench" ? "random" : splitConfig.default);
  const csiBench = dataset.id === "csi-bench";
  const recommendedProfile = csiBench ? "automatic task selection" : (plan.profile || (/vital|breath/i.test(dataset.standardization?.profile || "") ? "vital-sign" : "general-sensing"));
  const base = [
    `wisensehub prepare ${dataset.id} \\`,
    "  --data-root data \\",
    `  --setting ${setting} \\`,
  ].join("\n");
  const commandCard = (profile, title, summary, tone) => `<article>
    <div class="command-card-head"><div><span class="profile-pill ${tone}">${profile}</span><h3>${title}</h3></div><small>${summary}</small></div>
    <pre><code>${escapeHtml(base)}
  --profile ${profile}</code></pre>
  </article>`;
  return `<section id="prepare" class="workflow-section">
    <p class="section-step">03 · PREPARE IT</p>
    <h2>Let the agent prepare this sample</h2>
    <p class="section-lead">The recommended sample flow uses <strong>${escapeHtml(setting)}</strong> for the split and <strong>${escapeHtml(recommendedProfile)}</strong> for the task view. The agent downloads, places, processes, checks, and explains the data.</p>
    <div class="sample-save-command"><div><strong>Ask in plain language</strong><span>No flags required.</span></div><pre><code>Use $wisensehub to prepare the ${escapeHtml(dataset.name)} sample and explain each step.</code></pre></div>
    <details class="view-options"><summary>Show reproducible commands</summary>
    <div class="command-setup"><span>Install once</span><code>pip install -e ".[data]"</code></div>
    <div class="profile-commands">
      ${csiBench
        ? `<article><div class="command-card-head"><div><span class="profile-pill general">Automatic</span><h3>All CSI-Bench tasks</h3></div><small>10 Hz breathing · 100 Hz others · native S/Tx/Rx</small></div><pre><code>${escapeHtml(base.slice(0, -2))}</code></pre></article>`
        : commandCard(recommendedProfile, "Recommended sample view", recommendedProfile === "vital-sign" ? "10 Hz · flexible T" : "100 Hz · flexible T", recommendedProfile === "vital-sign" ? "vital" : "general")}
    </div>
    </details>
    ${csiBench ? `<div class="sample-save-command"><div><strong>Ask for an exact model shape</strong><span>The agent chooses the task rate.</span></div><pre><code>Use $wisensehub to prepare CSI-Bench fall data with 64 subcarriers, 1 Tx, and 2 Rx.</code></pre><small>Equivalent shape controls: <code>--subcarriers 64 --tx-links 1 --rx-links 2</code>.</small></div>` : ""}
  </section>`;
}

function evidenceSection(dataset) {
  const evidence = dataset.sources.map(source => `<li><span>${pretty(source.type)}</span><a href="${escapeHtml(source.url)}">${escapeHtml(source.url)}</a></li>`).join("");
  return `<section id="evidence" class="evidence-section">
    <p class="section-step">SOURCE LINKS</p>
    <h2>Evidence</h2>
    <p class="section-lead">Paper, dataset, and implementation sources used to verify this page.</p>
    <ul class="source-list">${evidence}</ul>
  </section>`;
}

async function init() {
  const id = new URLSearchParams(location.search).get("id");
  const catalog = await (await fetch(`data/catalog.json?t=${Date.now()}`, { cache: "no-store" })).json();
  const dataset = catalog.datasets.find(item => item.id === id);
  if (!dataset) throw new Error("Dataset entry not found");
  let sample = null;
  try {
    const samples = await (await fetch(`data/samples.json?t=${Date.now()}`, { cache: "no-store" })).json();
    sample = samples.datasets?.[id] || null;
  } catch { /* samples.json is optional */ }
  document.title = `${dataset.name} — WiSenseHub`;
  const taskNames = dataset.tasks.map(id => catalog.tasks.find(task => task.id === id)?.name || id);
  const sampleActionLabel = sample?.status === "ok"
    ? "Try the sample"
    : (dataset.id === "signfi" ? "Use locally" : "Check data access");
  const splitConfig = dataset.split_settings;
  const example = dataset.conversion_example;
  const settingRows = splitConfig.settings.map(item => `<tr><td><code>${escapeHtml(item.id)}</code>${item.id === splitConfig.default ? " <small>default</small>" : ""}</td><td>${pretty(item.kind)}</td><td>${pretty(item.provenance)}</td><td>${escapeHtml(item.group_by || "—")}</td></tr>`).join("");
  const prepareSection = `<section><h2>Convert and split</h2><div class="example-command compact"><pre><code>pip install -e \".[data]\"\nwisensehub settings ${escapeHtml(dataset.id)}\nwisensehub prepare ${escapeHtml(dataset.id)} --setting ${escapeHtml(splitConfig.default)} --data-root data</code></pre></div><p>Place the intact official release under <code>data/${escapeHtml(dataset.id)}/original/</code>. The <strong>${escapeHtml(dataset.conversion.handler)}</strong> adapter writes native NPZ tensors, sidecars, quality reports, and a reproducible split manifest.</p><details class="view-options"><summary>Optional fixed-size model view</summary><div class="example-command compact"><pre><code>wisensehub prepare ${escapeHtml(dataset.id)} \\\n  --setting ${escapeHtml(splitConfig.default)} \\\n  --data-root data \\\n  --target-length 128 \\\n  --interpolation linear \\\n  --layout link-subcarrier</code></pre></div><p>Native files stay in <code>standardized/</code>. Derived views are written to <code>standardized/views/</code> and record <code>derived_from</code>, target length/rate, interpolation, and layout.</p></details><dl class="facts"><div><dt>Implementation</dt><dd>${pretty(dataset.conversion.implementation)}</dd></div><div><dt>Recognized layout</dt><dd><code>${dataset.conversion.patterns.map(escapeHtml).join("<br>")}</code></dd></div><div><dt>Adapter evidence</dt><dd><a href="${escapeHtml(dataset.conversion.official_reference)}">Official loader or schema</a></dd></div></dl><div class="table-scroll"><table class="settings-table"><thead><tr><th>Setting</th><th>Method</th><th>Provenance</th><th>Group</th></tr></thead><tbody>${settingRows}</tbody></table></div><p class="setting-help">For cross-group protocols, use <code>--holdout 3</code> (or another official group ID). If filenames do not encode the group, add <code>original/metadata.csv</code>.</p></section>`;
  const inspectCommand = `python - <<'PY'\nfrom pathlib import Path\nimport numpy as np\np = next(Path(\"data/${dataset.id}/standardized\").glob(\"*.npz\"))\nx = np.load(p)\nprint(p)\nfor name in x.files:\n    print(name, x[name].shape, x[name].dtype)\nPY`;
  const conversionExampleSection = `<section class="conversion-example"><p class="eyebrow">DATASET-SPECIFIC WALKTHROUGH</p><h2>Conversion example</h2><ol class="example-steps"><li><strong>Place one official source</strong><pre><code>data/${escapeHtml(dataset.id)}/original/${escapeHtml(example.source_path)}</code></pre></li><li><strong>Run the adapter and default split</strong><pre><code>wisensehub prepare ${escapeHtml(dataset.id)} \\\n  --data-root data \\\n  --setting ${escapeHtml(splitConfig.default)} \\\n  --limit 1</code></pre></li><li><strong>Inspect the generated tensor</strong><pre><code>${escapeHtml(inspectCommand)}</code></pre></li></ol><dl class="facts example-output"><div><dt>Primary array</dt><dd><code>${escapeHtml(example.primary_array)}</code></dd></div><div><dt>Expected shape</dt><dd><code>${escapeHtml(example.expected_shape)}</code></dd></div></dl><p class="setting-help">${escapeHtml(example.note)}</p></section>`;
  const storySections = `${collectionSettingSection(dataset, sample)}
    ${sampleDownloadSection(dataset, sample, splitConfig)}
    ${datasetProcessingSection(dataset, sample, splitConfig)}
    ${sampleBeforeAfterSection(dataset, sample || {})}`;
  document.querySelector("#dataset-detail").innerHTML = `
    <a class="back-link" href="index.html#datasets">← All datasets</a>
    <div class="detail-intro ${sample?.setup_figure ? "has-figure" : "no-figure"}">
      <section class="detail-hero ${dataset.name.length > 52 ? "long-title" : ""}">
        <p class="eyebrow">DATASET · ${dataset.year}</p>
        <h1>${escapeHtml(dataset.name)}</h1>
        <p>${escapeHtml(dataset.summary)}</p>
        <div class="detail-hero-meta"><span>${taskNames.length} sensing tasks</span><span>${escapeHtml(dataset.original.formats.join(" · ") || "Format not confirmed")}</span><span>${pretty(dataset.original.access)} access</span></div>
        <div class="task-tags large">${taskNames.map(name => `<span>${escapeHtml(name)}</span>`).join("")}</div>
        <div class="hero-actions dataset-hero-actions"><a class="button primary" href="#sample">${escapeHtml(sampleActionLabel)}</a><a class="button secondary" href="#processing">See processing</a></div>
      </section>
      ${setupFigureSection(dataset, sample)}
    </div>
    <div class="detail-grid">
      <div class="detail-main">
        ${storySections}
      </div>
      <aside class="access-panel">
        <div class="access-heading"><p class="eyebrow">SOURCE DATA</p><h2>Original release</h2><span class="access-state">${pretty(dataset.original.access)}</span></div>
        <dl><div><dt>License</dt><dd>${escapeHtml(dataset.original.license)}</dd></div><div><dt>Redistribution</dt><dd>${pretty(dataset.original.redistribution)}</dd></div><div><dt>Formats</dt><dd>${escapeHtml(dataset.original.formats.join(", ") || "Not confirmed")}</dd></div></dl>
        <a class="button primary full" href="${escapeHtml(dataset.original.download_page || dataset.original.landing_page)}">Open original source <span aria-hidden="true">↗</span></a>
        <div class="access-evidence"><span>Evidence</span><ul>${dataset.sources.map(source => `<li><a href="${escapeHtml(source.url)}">${escapeHtml(pretty(source.type))}</a></li>`).join("")}</ul></div>
        <nav class="page-nav" aria-label="On this page"><span>On this page</span><a href="#collection">Overview</a><a href="#sample">Try sample</a><a href="#processing">Processing</a><a href="#preview">Preview</a><a href="guide.html">How WiSenseHub works ↗</a></nav>
        <p class="verification">Metadata verified ${dataset.verified_at}</p>
      </aside>
    </div>`;
  bindUsecaseToggle(document.querySelector("#dataset-detail"));
}

init().catch(error => { document.querySelector("#dataset-detail").innerHTML = `<a class="back-link" href="index.html">← Home</a><h1>Dataset unavailable</h1><p>${escapeHtml(error.message)}</p>`; });
