const state = { catalog: null, task: "", query: "", access: "", samples: {} };

const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
const labelForAccess = { direct: "Direct", account: "Account", application: "Application", unclear: "Verify access" };

function renderTasks() {
  const container = document.querySelector("#task-filters");
  const buttons = [{id: "", name: "All tasks", dataset_count: state.catalog.datasets.length}, ...state.catalog.tasks];
  container.innerHTML = buttons.map(task => `<button class="task-chip ${state.task === task.id ? "active" : ""}" data-task="${task.id}">${escapeHtml(task.name)} <span>${task.dataset_count}</span></button>`).join("");
  container.querySelectorAll("button").forEach(button => button.addEventListener("click", () => { state.task = button.dataset.task; renderTasks(); renderDatasets(); }));
}

function datasetCard(dataset) {
  const tasks = dataset.tasks.slice(0, 3).map(id => state.catalog.tasks.find(task => task.id === id)?.name || id);
  const subjects = dataset.settings.subjects === null ? "Subjects not reported" : `${dataset.settings.subjects} subjects`;
  const environments = dataset.settings.environments === null ? "Environments vary" : `${dataset.settings.environments} environment${dataset.settings.environments === 1 ? "" : "s"}`;
  const statusLabel = dataset.standardization.status === "adapter-ready" ? "Adapter ready" : dataset.standardization.status === "verified" ? "Verified" : "Planned";
  const sampleBadge = state.samples[dataset.id]?.status === "ok" ? `<span class="sample-badge">Sample</span>` : "";
  return `<article class="dataset-card">
    <div class="card-top"><span class="year">${dataset.year}</span><span>${sampleBadge}<span class="standard-status ${dataset.standardization.status}">${statusLabel}</span><span class="access ${dataset.original.access}">${labelForAccess[dataset.original.access]}</span></span></div>
    <h3><a href="dataset.html?id=${encodeURIComponent(dataset.id)}">${escapeHtml(dataset.name)}</a></h3>
    <p>${escapeHtml(dataset.summary)}</p>
    <div class="task-tags">${tasks.map(task => `<span>${escapeHtml(task)}</span>`).join("")}</div>
    <dl><div><dt>Hardware</dt><dd>${escapeHtml(dataset.hardware.chipset || dataset.hardware.platform)}</dd></div><div><dt>Setting</dt><dd>${subjects} · ${environments}</dd></div></dl>
    <a class="card-link" href="dataset.html?id=${encodeURIComponent(dataset.id)}">View evidence & standardization <span>→</span></a>
  </article>`;
}

function renderDatasets() {
  const query = state.query.trim().toLowerCase();
  const filtered = state.catalog.datasets.filter(dataset => {
    const searchable = JSON.stringify(dataset).toLowerCase();
    return (!state.task || dataset.tasks.includes(state.task)) && (!state.access || dataset.original.access === state.access) && (!query || searchable.includes(query));
  });
  document.querySelector("#dataset-grid").innerHTML = filtered.map(datasetCard).join("");
  document.querySelector("#result-count").textContent = `${filtered.length} dataset${filtered.length === 1 ? "" : "s"}`;
  document.querySelector("#empty-state").hidden = filtered.length !== 0;
}

async function init() {
  const response = await fetch("data/catalog.json");
  state.catalog = await response.json();
  try {
    const samples = await (await fetch("data/samples.json")).json();
    state.samples = samples.datasets || {};
  } catch { /* samples.json is optional */ }
  document.querySelector("#stat-datasets").textContent = state.catalog.stats.datasets;
  document.querySelector("#stat-tasks").textContent = state.catalog.stats.tasks;
  document.querySelector("#stat-hardware").textContent = state.catalog.stats.hardware_platforms;
  document.querySelector("#stat-open").textContent = state.catalog.stats.open_or_direct;
  document.querySelector("#verified-date").textContent = state.catalog.generated_at;
  document.querySelector("#search").addEventListener("input", event => { state.query = event.target.value; renderDatasets(); });
  document.querySelector("#access-filter").addEventListener("change", event => { state.access = event.target.value; renderDatasets(); });
  renderTasks(); renderDatasets();
}

init().catch(error => { document.querySelector("#dataset-grid").innerHTML = `<p>Catalog could not be loaded: ${escapeHtml(error.message)}</p>`; });
