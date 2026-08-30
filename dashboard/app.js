(() => {
  "use strict";

  const state = { mediaObserver: null, data: null, collectionId: null };
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

  function setConnection(status, label) {
    const node = $("#connection-status");
    node.className = `connection-status is-${status}`;
    node.textContent = label;
  }

  function hasNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function formatCount(value) {
    return hasNumber(value) ? new Intl.NumberFormat().format(value) : "Not reported";
  }

  function formatPercent(value) {
    if (!hasNumber(value)) return "Not reported";
    const normalized = Math.abs(value) <= 1 ? value : value / 100;
    return new Intl.NumberFormat(undefined, {
      style: "percent",
      maximumFractionDigits: 1,
    }).format(normalized);
  }

  function formatSeconds(value) {
    return hasNumber(value) ? `${value.toFixed(2)} s` : "Not reported";
  }

  function formatDistance(value) {
    return hasNumber(value) ? `${value.toFixed(2)} m` : "Not reported";
  }

  function gateBadge(label, value) {
    const badge = el("span", `gate-badge ${value === true ? "is-pass" : value === false ? "is-fail" : "is-unknown"}`);
    badge.append(el("span", "gate-name", text(label)), text(value === true ? "Pass" : value === false ? "Fail" : "Pending"));
    return badge;
  }

  function renderRollAudit(data) {
    const panel = $("#roll-audit");
    panel.hidden = state.collectionId !== "roll-sprint";
    const evaluation = data.rollSprintEvaluation || {};
    const content = $("#audit-content");
    const empty = $("#audit-empty");
    content.hidden = !evaluation.available;
    empty.hidden = Boolean(evaluation.available);
    if (!evaluation.available) return;

    const checkpointBits = [evaluation.checkpoint || evaluation.file];
    if (hasNumber(evaluation.checkpointIteration)) {
      checkpointBits.push(`iteration ${evaluation.checkpointIteration}`);
    }
    if (evaluation.modified) {
      checkpointBits.push(new Date(evaluation.modified).toLocaleString());
    }
    $("#audit-checkpoint").textContent = checkpointBits.filter(Boolean).join(" · ");

    const overall = evaluation.passes?.overall;
    const overallBadge = $("#audit-overall");
    overallBadge.className = `gate-badge ${overall === true ? "is-pass" : overall === false ? "is-fail" : "is-unknown"}`;
    overallBadge.textContent = overall === true ? "Ready" : overall === false ? "Not ready" : "Pending";

    const attempts = evaluation.selfRightAttempts;
    const successes = evaluation.selfRightSuccesses;
    $("#metric-self-right").textContent = hasNumber(attempts) && hasNumber(successes)
      ? `${formatCount(successes)} / ${formatCount(attempts)}`
      : "Not reported";
    $("#metric-self-right-rate").textContent = hasNumber(evaluation.selfRightSuccessRate)
      ? `${formatPercent(evaluation.selfRightSuccessRate)} success`
      : "Attempts / successes";
    $("#metric-recovery-latency").textContent = [
      formatSeconds(evaluation.recoveryLatencyMeanS),
      formatSeconds(evaluation.recoveryLatencyP95S),
    ].join(" / ");
    $("#metric-recovered-rerolls").textContent = formatCount(evaluation.selfRightThenRerollCount);
    $("#metric-recovered-rerolls-rate").textContent = hasNumber(evaluation.selfRightThenRerollRate)
      ? `${formatPercent(evaluation.selfRightThenRerollRate)} of recoveries rerolled`
      : "Self-right then valid roll";
    $("#metric-frontier-after-recovery").textContent = formatDistance(evaluation.frontierAfterRecoveryM);
    $("#metric-lane-reposition").textContent = [
      formatCount(evaluation.laneRepositionCount),
      formatSeconds(evaluation.laneRepositionLatencyMeanS),
    ].join(" / ");

    const gateNames = {
      recovery: "Recovery",
      reroll: "Recovered reroll",
      raceFrontier: "Race frontier",
      straightLane: "Straight lane",
      target20m: "20 m target",
      lateralDrift: "Lateral drift",
      finite: "NaN / OOB",
    };
    const gateList = $("#audit-gates");
    gateList.replaceChildren();
    for (const [key, label] of Object.entries(gateNames)) {
      gateList.append(gateBadge(label, evaluation.passes?.[key]));
    }

    const orientationBody = $("#orientation-results");
    orientationBody.replaceChildren();
    const orientations = evaluation.orientations || [];
    orientationBody.closest(".orientation-table-wrap").hidden = orientations.length === 0;
    for (const item of orientations) {
      const row = document.createElement("tr");
      const recovered = hasNumber(item.successes) && hasNumber(item.attempts)
        ? `${formatCount(item.successes)} / ${formatCount(item.attempts)}`
        : "Not reported";
      const latency = [formatSeconds(item.latencyMeanS), formatSeconds(item.latencyP95S)].join(" / ");
      const cells = [
        item.label || item.id,
        recovered,
        formatPercent(item.successRate),
        latency,
        formatCount(item.rerollCount),
        formatDistance(item.frontierAfterRecoveryM),
      ];
      for (const value of cells) row.append(el("td", "", text(value)));
      const gateCell = document.createElement("td");
      gateCell.append(gateBadge("", item.pass));
      row.append(gateCell);
      orientationBody.append(row);
    }
  }

  function renderCollectionSelector(data) {
    const selector = $("#video-collection");
    const collections = data.videoCollections || [];
    const selected = state.collectionId || data.defaultVideoCollection || collections[0]?.id;
    selector.replaceChildren();
    for (const collection of collections) {
      const option = document.createElement("option");
      option.value = collection.id;
      option.textContent = `${collection.label} (${collection.videoCount || 0})`;
      selector.append(option);
    }
    if (collections.some((collection) => collection.id === selected)) {
      state.collectionId = selected;
      selector.value = selected;
    }
    const active = collections.find((collection) => collection.id === state.collectionId);
    $("#gallery-copy").textContent = active?.description || "A rollout from the selected policy run.";
  }

  function renderMedia(data, collectionId) {
    const media = (data.media || [])
      .filter((item) => item.kind === "video" && item.collection === collectionId)
      .sort((left, right) => String(right.modified || "").localeCompare(String(left.modified || "")));
    const grid = $("#media-grid");
    grid.replaceChildren();
    if (state.mediaObserver) state.mediaObserver.disconnect();
    state.mediaObserver = null;
    $("#media-empty").hidden = media.length !== 0;
    $("#video-count").textContent = `${media.length} ${media.length === 1 ? "video" : "videos"}`;

    const loadVideo = (video) => {
      if (video.src || !video.dataset.src) return;
      video.src = video.dataset.src;
      video.load();
    };
    if ("IntersectionObserver" in window) {
      state.mediaObserver = new IntersectionObserver((entries, observer) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          loadVideo(entry.target);
          observer.unobserve(entry.target);
        }
      }, { rootMargin: "360px 0px" });
    }

    for (const item of media) {
      const card = el("article", "media-card");
      const preview = el("div", "media-preview");
      const video = document.createElement("video");
      video.controls = true;
      video.preload = "none";
      video.dataset.src = item.url;
      video.setAttribute("aria-label", item.name);
      video.addEventListener("pointerdown", () => loadVideo(video), { once: true });
      video.addEventListener("focus", () => loadVideo(video), { once: true });
      if (state.mediaObserver) state.mediaObserver.observe(video);
      else loadVideo(video);
      preview.append(video);
      const meta = el("div", "media-meta");
      meta.append(
        el("span", "media-name", text(item.name)),
        el("time", "media-source", text(new Date(item.modified).toLocaleString())),
      );
      card.append(preview, meta);
      grid.append(card);
    }
  }

  async function loadGallery() {
    try {
      const response = await fetch("/api/state", { cache: "no-store" });
      if (!response.ok) throw new Error(`Gallery API returned ${response.status}`);
      const data = await response.json();
      state.data = data;
      renderCollectionSelector(data);
      renderRollAudit(data);
      renderMedia(data, state.collectionId);
      $("#loaded-at").textContent = `Loaded ${new Date(data.generatedAt).toLocaleTimeString()}`;
      setConnection("live", "Loaded");
    } catch (error) {
      setConnection("error", "Offline");
      const banner = $("#error-banner");
      banner.textContent = `Could not load training videos: ${error.message}`;
      banner.hidden = false;
    }
  }

  $("#video-collection").addEventListener("change", (event) => {
    state.collectionId = event.target.value;
    const collection = (state.data?.videoCollections || []).find(
      (item) => item.id === state.collectionId,
    );
    $("#gallery-copy").textContent = collection?.description || "A rollout from the selected policy run.";
    renderRollAudit(state.data || {});
    renderMedia(state.data || {}, state.collectionId);
  });

  loadGallery();
})();
