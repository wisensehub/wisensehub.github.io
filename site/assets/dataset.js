const escapeHtml = value => String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
const pretty = value => String(value).replaceAll("_", " ").replace(/\b\w/g, letter => letter.toUpperCase());

function factRows(object) {
  return Object.entries(object).map(([key, value]) => `<div><dt>${pretty(key)}</dt><dd>${escapeHtml(value === null ? "Not reported" : value)}</dd></div>`).join("");
}

function formatCollectionValue(value) {
  if (value === null || value === undefined || value === "") return "Not reported";
  if (typeof value !== "object" || Array.isArray(value)) return escapeHtml(String(value));
  return Object.entries(value)
    .map(([key, item]) => `<div class="label-line"><span class="label-key">${escapeHtml(pretty(key))}:</span> ${escapeHtml(String(item))}</div>`)
    .join("");
}

function collectionSettingSection(dataset) {
  const collection = dataset.collection;
  if (collection && typeof collection === "object") {
    const order = [
      ["device", "Device"],
      ["distance", "Distance"],
      ["subjects", "Subjects"],
      ["labels", "Labels"],
      ["scenario_labels", "Scenario"],
      ["band", "Band"],
      ["subcarriers", "Subcarriers"],
      ["subcarrier_spacing", "Subcarrier spacing"],
      ["sampling_rate_hz", "Sampling rate"],
      ["clip_length", "Clip length"],
    ];
    const rows = order
      .filter(([key]) => collection[key] !== undefined && collection[key] !== null && collection[key] !== "")
      .map(([key, label]) => {
        const raw = collection[key];
        let display = raw;
        if (key === "sampling_rate_hz" && typeof raw === "number") display = `${raw} Hz`;
        return `<tr><th scope="row">${escapeHtml(label)}</th><td>${formatCollectionValue(display)}</td></tr>`;
      })
      .join("");
    return `<section class="collection-setting"><h2>Collection setting</h2><div class="table-scroll"><table class="collection-table"><tbody>${rows}</tbody></table></div></section>`;
  }
  return `<section class="collection-setting"><h2>Collection setting</h2><dl class="facts">${factRows({...dataset.settings, ...dataset.hardware})}</dl></section>`;
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

function structureBlock({ kind, badge, title, hint, tree, empty }) {
  return `<article class="structure-block structure-block-${escapeHtml(kind)}"><header><span class="structure-badge">${escapeHtml(badge)}</span><h4>${escapeHtml(title)}</h4><p class="structure-pane-hint">${hint}</p></header><div class="structure-block-body">${tree.length ? renderTree(tree) : `<p class="structure-empty">${escapeHtml(empty)}</p>`}</div></article>`;
}

function sampleStructureSection(sample) {
  const originalTree = sample.original_file_tree || [];
  const standardizedTree = sample.standardized_file_tree || [];
  const usecaseTree = sample.usecase_file_tree || [];
  const hasSplit = originalTree.length || standardizedTree.length || usecaseTree.length;
  if (!hasSplit) {
    return `<details class="sample-structure" open><summary><h3>Sample structure</h3><span class="structure-hint">${sample.file_count || 0} files — click folders to expand</span></summary>${renderTree(sample.file_tree || [])}</details>`;
  }
  const originalCount = sample.original_file_count ?? 0;
  const standardizedCount = sample.standardized_file_count ?? 0;
  const usecaseCount = sample.usecase_file_count ?? 0;
  const left = structureBlock({
    kind: "original",
    badge: "1 · Original",
    title: "Original dataset",
    hint: `Official release layout · ${originalCount} file${originalCount === 1 ? "" : "s"}`,
    tree: originalTree,
    empty: "No original sample files hosted.",
  });
  const middle = structureBlock({
    kind: "standardized",
    badge: "2 · Standardized",
    title: "Standardized structure",
    hint: `Prepare output under data/…/standardized/ · ${standardizedCount} file${standardizedCount === 1 ? "" : "s"}`,
    tree: standardizedTree,
    empty: "Run wisensehub prepare to create standardized NPZ + JSON.",
  });
  const right = structureBlock({
    kind: "usecase",
    badge: "3 · Use case",
    title: "by_label packaging",
    hint: `One derived example: one folder per label · ${usecaseCount} file${usecaseCount === 1 ? "" : "s"}`,
    tree: usecaseTree.length ? [{ name: "by_label", type: "dir", children: usecaseTree }] : [],
    empty: "No by_label use-case samples yet.",
  });
  return `<details class="sample-structure" open><summary><h3>Sample structure</h3><span class="structure-hint">original → standardized → one use case</span></summary><div class="structure-split structure-split-3">${left}<div class="structure-vs" aria-hidden="true">→</div>${middle}<div class="structure-vs" aria-hidden="true">→</div>${right}</div></details>`;
}

function dimTable(dimensions, caption) {
  const rows = (dimensions || []).map(item => `<tr><td><code>${escapeHtml(item.axis)}</code></td><td><code>${item.size}</code></td><td>${escapeHtml(item.meaning)}</td></tr>`).join("");
  if (!rows) return "";
  return `<div class="preview-dim-block"><h4>${escapeHtml(caption)}</h4><table class="compact-table"><thead><tr><th>Axis</th><th>Size</th><th>Meaning</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function metaCard(title, block) {
  if (!block) return "";
  const profile = block.profile || {};
  return `<article class="preview-meta-card"><h3>${escapeHtml(title)}</h3><dl class="facts compact">${factRows(profile)}</dl>${dimTable(block.dimensions, "Tensor dimensions")}</article>`;
}

function setupFigureSection(dataset, sample) {
  if (!sample?.setup_figure) return "";
  const source = sample.figure_source;
  const attribution = source
    ? `<p class="figure-source">Source: <a href="${escapeHtml(source.url)}">${escapeHtml(source.label)}</a></p>`
    : "";
  return `<section class="setup-figure-top"><h2>Experimental Setup</h2><figure class="setup-figure"><img src="${escapeHtml(sample.setup_figure)}" alt="Experimental setup for ${escapeHtml(dataset.name)}" loading="lazy"></figure>${attribution}</section>`;
}

function sampleSections(dataset, sample) {
  if (!sample) return "";
  const sections = [];
  if (sample.status !== "ok") {
    if (sample.reason) {
      sections.push(`<section><h2>Data sample</h2><p class="setting-help">No downloadable sample is hosted for this dataset yet: ${escapeHtml(sample.reason)}</p></section>`);
    }
    return sections.join("");
  }
  const licenseNote = dataset.original.redistribution === "allowed"
    ? "The original license permits redistribution."
    : "Check the original license before reusing this sample beyond local evaluation.";
  const packageParts = [
    sample.original_file_count ? "original" : null,
    sample.standardized_file_count ? "standardized" : null,
    sample.usecase_file_count ? "by_label use case" : null,
  ].filter(Boolean).join(" · ");
  sections.push(`<section><h2>Data sample</h2>
    <p>${escapeHtml(sample.note || "A small subset of the official release, preserving the original directory structure.")}</p>
    <div class="sample-download"><a class="button primary" href="${escapeHtml(sample.sample_zip)}" download>Download sample (${formatBytes(sample.zip_bytes)})</a><small>${sample.file_count} file${sample.file_count === 1 ? "" : "s"} · ${formatBytes(sample.sample_bytes)} uncompressed${packageParts ? ` · includes ${packageParts}` : ""} · ${escapeHtml(licenseNote)}</small></div>
    ${sampleStructureSection(sample)}</section>`);
  const previews = sample.previews || {};
  if (previews.before || previews.after) {
    const window = sample.preview_window || {};
    const gridNote = window.profile
      ? `Task profile <strong>${escapeHtml(window.profile)}</strong>${window.sample_rate_hz ? ` @ <strong>${window.sample_rate_hz} Hz</strong>` : ""}${window.time_steps ? `, T=${window.time_steps}` : ""}${window.links != null ? `, L=${window.links}` : ""}${window.subcarriers ? `, S=${window.subcarriers}` : ""}. Links/subcarriers stay native.`
      : (window.subcarriers && window.time_steps
        ? `Standardized view grid: <strong>${window.subcarriers} subcarriers × ${window.time_steps} time steps</strong>${window.sample_rate_hz ? ` at <strong>${window.sample_rate_hz} Hz</strong>` : ""}.`
        : "");
    const panel = (title, caption, src) => src
      ? `<figure class="preview-panel compact"><figcaption><strong>${title}</strong><span>${escapeHtml(caption)}</span></figcaption><img src="${escapeHtml(src)}" alt="${title} CSI amplitude heatmap" loading="lazy"></figure>`
      : "";
    const labelPanels = (sample.label_previews || []).map(item => panel(
      escapeHtml(item.label),
      `${item.kind === "segment" ? "segment" : "clip"}${item.source_file ? ` · ${item.source_file}` : ""}`,
      item.image,
    )).join("");
    const labelSection = labelPanels
      ? `<div class="preview-label-section"><h3>Per-label CSI samples</h3><div class="preview-label-grid">${labelPanels}</div></div>`
      : "";
    sections.push(`<section class="preview-section compact"><h2>CSI preview</h2>
      <p class="setting-help preview-lead">Paper-style amplitude heatmaps: <strong>x = time</strong>, <strong>y = subcarrier</strong>. ${gridNote}</p>
      <div class="preview-compare-meta">${metaCard("Original release", sample.original)}${metaCard("Standardized view", sample.standardized_view)}</div>
      <div class="preview-pair compact">${panel("Original", sample.preview_source_file || "source file", previews.before)}${panel("Standardized", "task-profile view", previews.after)}</div>
      ${labelSection}</section>`);
  }
  return sections.join("");
}

async function init() {
  const id = new URLSearchParams(location.search).get("id");
  const catalog = await (await fetch("data/catalog.json")).json();
  const dataset = catalog.datasets.find(item => item.id === id);
  if (!dataset) throw new Error("Dataset entry not found");
  let sample = null;
  try {
    const samples = await (await fetch("data/samples.json")).json();
    sample = samples.datasets?.[id] || null;
  } catch { /* samples.json is optional */ }
  document.title = `${dataset.name} — WiSenseHub`;
  const taskNames = dataset.tasks.map(id => catalog.tasks.find(task => task.id === id)?.name || id);
  const splitConfig = dataset.split_settings;
  const example = dataset.conversion_example;
  const settingRows = splitConfig.settings.map(item => `<tr><td><code>${escapeHtml(item.id)}</code>${item.id === splitConfig.default ? " <small>default</small>" : ""}</td><td>${pretty(item.kind)}</td><td>${pretty(item.provenance)}</td><td>${escapeHtml(item.group_by || "—")}</td></tr>`).join("");
  const prepareSection = `<section><h2>Convert and split</h2><div class="example-command compact"><pre><code>pip install -e \".[data]\"\nwisensehub settings ${escapeHtml(dataset.id)}\nwisensehub prepare ${escapeHtml(dataset.id)} --setting ${escapeHtml(splitConfig.default)} --data-root data</code></pre></div><p>Place the intact official release under <code>data/${escapeHtml(dataset.id)}/original/</code>. The <strong>${escapeHtml(dataset.conversion.handler)}</strong> adapter writes native NPZ tensors, sidecars, quality reports, and a reproducible split manifest.</p><details class="view-options"><summary>Optional fixed-size model view</summary><div class="example-command compact"><pre><code>wisensehub prepare ${escapeHtml(dataset.id)} \\\n  --setting ${escapeHtml(splitConfig.default)} \\\n  --data-root data \\\n  --target-length 128 \\\n  --interpolation linear \\\n  --layout link-subcarrier</code></pre></div><p>Native files stay in <code>standardized/</code>. Derived views are written to <code>standardized/views/</code> and record <code>derived_from</code>, target length/rate, interpolation, and layout.</p></details><dl class="facts"><div><dt>Implementation</dt><dd>${pretty(dataset.conversion.implementation)}</dd></div><div><dt>Recognized layout</dt><dd><code>${dataset.conversion.patterns.map(escapeHtml).join("<br>")}</code></dd></div><div><dt>Adapter evidence</dt><dd><a href="${escapeHtml(dataset.conversion.official_reference)}">Official loader or schema</a></dd></div></dl><div class="table-scroll"><table class="settings-table"><thead><tr><th>Setting</th><th>Method</th><th>Provenance</th><th>Group</th></tr></thead><tbody>${settingRows}</tbody></table></div><p class="setting-help">For cross-group protocols, use <code>--holdout 3</code> (or another official group ID). If filenames do not encode the group, add <code>original/metadata.csv</code>.</p></section>`;
  const inspectCommand = `python - <<'PY'\nfrom pathlib import Path\nimport numpy as np\np = next(Path(\"data/${dataset.id}/standardized\").glob(\"*.npz\"))\nx = np.load(p)\nprint(p)\nfor name in x.files:\n    print(name, x[name].shape, x[name].dtype)\nPY`;
  const conversionExampleSection = `<section class="conversion-example"><p class="eyebrow">DATASET-SPECIFIC WALKTHROUGH</p><h2>Conversion example</h2><ol class="example-steps"><li><strong>Place one official source</strong><pre><code>data/${escapeHtml(dataset.id)}/original/${escapeHtml(example.source_path)}</code></pre></li><li><strong>Run the adapter and default split</strong><pre><code>wisensehub prepare ${escapeHtml(dataset.id)} \\\n  --data-root data \\\n  --setting ${escapeHtml(splitConfig.default)} \\\n  --limit 1</code></pre></li><li><strong>Inspect the generated tensor</strong><pre><code>${escapeHtml(inspectCommand)}</code></pre></li></ol><dl class="facts example-output"><div><dt>Primary array</dt><dd><code>${escapeHtml(example.primary_array)}</code></dd></div><div><dt>Expected shape</dt><dd><code>${escapeHtml(example.expected_shape)}</code></dd></div></dl><p class="setting-help">${escapeHtml(example.note)}</p></section>`;
  const outputPreviewSection = `<section><h2>Standardized output preview</h2><dl class="facts example-output"><div><dt>Native NPZ</dt><dd><code>data/${escapeHtml(dataset.id)}/standardized/*.npz</code></dd></div><div><dt>Derived view</dt><dd><code>data/${escapeHtml(dataset.id)}/standardized/views/*.npz</code></dd></div><div><dt>Primary array</dt><dd><code>${escapeHtml(example.primary_array)}</code></dd></div><div><dt>Typical shape</dt><dd><code>${escapeHtml(example.expected_shape)}</code></dd></div></dl><p class="setting-help">Canonical CSI uses <code>[T,L,S]</code> or <code>[N,T,L,S]</code>. <code>--layout link-subcarrier</code> exports <code>[T,F]</code> or <code>[N,T,F]</code> for model-ready features.</p></section>`;
  document.querySelector("#dataset-detail").innerHTML = `
    <a class="back-link" href="index.html#datasets">← All datasets</a>
    <section class="detail-hero"><p class="eyebrow">DATASET · ${dataset.year}</p><h1>${escapeHtml(dataset.name)}</h1><p>${escapeHtml(dataset.summary)}</p><div class="task-tags large">${taskNames.map(name => `<span>${escapeHtml(name)}</span>`).join("")}</div></section>
    ${setupFigureSection(dataset, sample)}
    <div class="detail-grid">
      <div class="detail-main">
        ${collectionSettingSection(dataset)}
        <section><h2>Standardization plan</h2><div class="standard-note"><strong>${pretty(dataset.standardization.status)}</strong><code>${escapeHtml(dataset.standardization.profile)}</code><p>${escapeHtml(dataset.standardization.notes)}</p></div></section>${sampleSections(dataset, sample)}${prepareSection}${conversionExampleSection}${outputPreviewSection}
        <section><h2>Evidence</h2><ul class="source-list">${dataset.sources.map(source => `<li><span>${pretty(source.type)}</span><a href="${escapeHtml(source.url)}">${escapeHtml(source.url)}</a></li>`).join("")}</ul></section>
      </div>
      <aside class="access-panel"><h2>Original release</h2><dl><div><dt>Access</dt><dd>${pretty(dataset.original.access)}</dd></div><div><dt>License</dt><dd>${escapeHtml(dataset.original.license)}</dd></div><div><dt>Redistribution</dt><dd>${pretty(dataset.original.redistribution)}</dd></div><div><dt>Formats</dt><dd>${escapeHtml(dataset.original.formats.join(", ") || "Not confirmed")}</dd></div></dl><a class="button primary full" href="${escapeHtml(dataset.original.download_page || dataset.original.landing_page)}">Open original source</a><p class="verification">Metadata verified ${dataset.verified_at}</p></aside>
    </div>`;
}

init().catch(error => { document.querySelector("#dataset-detail").innerHTML = `<a class="back-link" href="index.html">← Home</a><h1>Dataset unavailable</h1><p>${escapeHtml(error.message)}</p>`; });
