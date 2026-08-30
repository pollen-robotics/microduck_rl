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
    renderMedia(state.data || {}, state.collectionId);
  });

  loadGallery();
})();
