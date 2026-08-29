(() => {
  "use strict";

  const state = { data: null, selectedRun: null };
  const $ = (selector) => document.querySelector(selector);

  function text(value) {
    return document.createTextNode(value == null ? "" : String(value));
  }

  function el(tag, className, content) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.append(content);
    return node;
  }

  function formatAge(seconds) {
    if (seconds < 60) return "just now";
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
  }

  function formatValue(value) {
    if (!Number.isFinite(value)) return "—";
    if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < .001)) return value.toExponential(2);
    return value.toFixed(Math.abs(value) >= 10 ? 1 : 3).replace(/\.000$/, "");
  }

  function setConnection(status, label) {
    const node = $("#connection-status");
    node.className = `connection-status is-${status}`;
    node.textContent = label;
  }

  function renderSummary(data) {
    const summary = data.summary || {};
    $("#stat-runs").textContent = summary.runs ?? 0;
    $("#stat-active").textContent = summary.activeRuns ?? 0;
    $("#stat-checkpoints").textContent = summary.checkpoints ?? 0;
    $("#stat-media").textContent = summary.media ?? 0;
    $("#run-count").textContent = `${summary.runs ?? 0} ${summary.runs === 1 ? "run" : "runs"}`;
    $("#last-updated").textContent = data.generatedAt ? `Updated ${new Date(data.generatedAt).toLocaleTimeString()}` : "Waiting for first update";
  }

  function renderRuns(data) {
    const runs = data.runs || [];
    const grid = $("#run-grid");
    grid.replaceChildren();
    $("#runs-empty").hidden = runs.length !== 0;
    const selected = state.selectedRun && runs.some((run) => run.id === state.selectedRun) ? state.selectedRun : runs[0]?.id;
    state.selectedRun = selected || null;
    for (const run of runs) {
      const card = el("article", `run-card${run.id === state.selectedRun ? " is-selected" : ""}`);
      const header = el("div", "run-card-header");
      const title = el("div");
      title.append(el("h3", null, text(run.name)), el("p", "run-task", text(run.task)));
      const badge = el("span", `status-badge ${run.status}`, text(run.status));
      badge.setAttribute("aria-label", `Run status: ${run.status}`);
      header.append(title, badge);
      card.append(header);
      const meta = el("div", "run-meta");
      meta.append(el("span", null, text(run.experiment)), el("span", null, text(formatAge(run.ageSeconds || 0))));
      card.append(meta);
      const checkpoints = run.checkpoints || [];
      if (checkpoints.length) {
        const list = el("div", "checkpoint-list");
        for (const checkpoint of checkpoints.slice(0, 3)) {
          const row = el("div", "checkpoint-row");
          row.append(el("span", null, text(checkpoint.name)), el("span", null, text(checkpoint.size)));
          list.append(row);
        }
        if (checkpoints.length > 3) list.append(el("div", "checkpoint-row", text(`+ ${checkpoints.length - 3} more checkpoints`)));
        card.append(list);
      }
      card.tabIndex = 0;
      card.setAttribute("role", "button");
      card.setAttribute("aria-label", `Select run ${run.name}`);
      card.addEventListener("click", () => selectRun(run.id));
      card.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectRun(run.id); } });
      grid.append(card);
    }
    const select = $("#run-select");
    select.replaceChildren();
    for (const run of runs) {
      const option = el("option", null, text(run.name));
      option.value = run.id;
      option.selected = run.id === state.selectedRun;
      select.append(option);
    }
    select.disabled = runs.length === 0;
  }

  function selectRun(runId) {
    state.selectedRun = runId;
    if (state.data) {
      renderRuns(state.data);
      renderMetrics(state.data);
    }
  }

  function makeSparkline(points) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "sparkline");
    svg.setAttribute("viewBox", "0 0 240 36");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Metric trend");
    const values = points.map((point) => Number(point[1])).filter(Number.isFinite);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const spread = max - min || 1;
    const coords = values.map((value, index) => `${(index / Math.max(1, values.length - 1)) * 240},${33 - ((value - min) / spread) * 28}`).join(" ");
    const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    polyline.setAttribute("points", coords);
    svg.append(polyline);
    return svg;
  }

  function renderMetrics(data) {
    const run = (data.runs || []).find((item) => item.id === state.selectedRun);
    const list = $("#metric-list");
    list.replaceChildren();
    const metrics = Object.entries(run?.metrics || {}).sort((a, b) => a[0].localeCompare(b[0]));
    $("#metrics-empty").hidden = metrics.length !== 0;
    for (const [name, metric] of metrics.slice(0, 18)) {
      const row = el("article", "metric-row");
      const header = el("div", "metric-row-header");
      header.append(el("span", "metric-name", text(name)), el("strong", "metric-value", text(formatValue(metric.latest))));
      row.append(header);
      if (metric.points?.length > 1) row.append(makeSparkline(metric.points));
      list.append(row);
    }
  }

  function renderMedia(data) {
    const grid = $("#media-grid");
    grid.replaceChildren();
    const media = (data.media || []).filter((item) => item.kind === "video");
    $("#media-empty").hidden = media.length !== 0;
    for (const item of media) {
      const card = el("article", "media-card");
      const preview = el("div", "media-preview");
      const video = document.createElement("video");
      video.controls = true;
      video.preload = "metadata";
      video.src = item.url;
      video.setAttribute("aria-label", item.name);
      preview.append(video);
      const meta = el("div", "media-meta");
      const location = item.source === "runs" ? item.path : `${item.source}/${item.path || item.name}`;
      meta.append(el("span", "media-name", text(item.name)), el("span", "media-source", text(location)));
      card.append(preview, meta);
      grid.append(card);
    }
  }

  function render(data) {
    state.data = data;
    renderSummary(data);
    renderRuns(data);
    renderMetrics(data);
    renderMedia(data);
  }

  async function refresh() {
    try {
      const response = await fetch(`/api/state?at=${Date.now()}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`);
      render(await response.json());
      setConnection("live", "Live");
      $("#error-banner").hidden = true;
    } catch (error) {
      setConnection("error", "Offline");
      const banner = $("#error-banner");
      banner.textContent = `Could not refresh training data: ${error.message}`;
      banner.hidden = false;
    }
  }

  $("#refresh-button").addEventListener("click", refresh);
  $("#run-select").addEventListener("change", (event) => selectRun(event.target.value));
  refresh();
  window.setInterval(refresh, 10000);
})();
