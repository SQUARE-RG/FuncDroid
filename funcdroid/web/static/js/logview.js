/* Live run.log viewer with tag colors, search, and filtering */
const LogView = (() => {
  let logEl = null;
  let tagBarEl = null;
  let searchInput = null;
  let autoScrollCheckbox = null;

  let allItems = [];
  let sinceTotal = 0;
  let knownTags = new Set();
  let activeTag = "All";
  let searchTerm = "";

  const TAG_RE = /^\[([^\]]+)\]\s*(.*)$/;
  const ROUND_RE = /^={5,}$/;

  function escapeHtml(str) {
    return String(str ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function parseLine(raw) {
    if (ROUND_RE.test(raw.trim())) {
      return { raw, tag: "", cls: "dim", isSep: false };
    }
    if (/^\s*(?:Iteration\s*\d+|\u7b2c\s*\d+\s*\u8f6e)/.test(raw)) {
      return { raw, tag: "Iteration", cls: "round-sep", isSep: true };
    }
    const m = TAG_RE.exec(raw);
    if (m) {
      const tag = m[1].replace(/\s+/g, "_");
      return { raw, tag, cls: "" };
    }
    return { raw, tag: "", cls: "" };
  }

  function matchesFilter(item) {
    if (activeTag !== "All" && item.tag !== activeTag) return false;
    if (searchTerm && !item.raw.toLowerCase().includes(searchTerm)) return false;
    return true;
  }

  function renderLineHtml(item) {
    const tagHtml = item.tag ? `<span class="tag">[${escapeHtml(item.tag)}]</span> ` : "";
    const rest = item.tag ? item.raw.replace(TAG_RE, "$2") : item.raw;
    const classes = ["log-line"];
    if (item.cls) classes.push(item.cls);
    if (item.tag) classes.push(`tag-${item.tag}`);
    return `<div class="${classes.join(" ")}">${tagHtml}${escapeHtml(rest)}</div>`;
  }

  function rebuildTagBar() {
    const tags = ["All", ...Array.from(knownTags).sort()];
    tagBarEl.innerHTML = tags
      .map((t) => `<button type="button" class="log-tag-btn${t === activeTag ? " active" : ""}" data-tag="${escapeHtml(t)}">${escapeHtml(t)}</button>`)
      .join("");
    tagBarEl.querySelectorAll(".log-tag-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        activeTag = btn.getAttribute("data-tag");
        rebuildTagBar();
        renderAll();
      });
    });
  }

  function renderAll() {
    const html = allItems.filter(matchesFilter).map(renderLineHtml).join("");
    logEl.innerHTML = html;
    maybeScroll();
  }

  function appendItems(items) {
    const matching = items.filter(matchesFilter);
    if (matching.length) {
      logEl.insertAdjacentHTML("beforeend", matching.map(renderLineHtml).join(""));
      maybeScroll();
    }
  }

  function maybeScroll() {
    if (autoScrollCheckbox && autoScrollCheckbox.checked) {
      logEl.scrollTop = logEl.scrollHeight;
    }
  }

  function init(logContainer, tagBarContainer, search, autoScrollEl) {
    logEl = logContainer;
    tagBarEl = tagBarContainer;
    searchInput = search;
    autoScrollCheckbox = autoScrollEl;
    rebuildTagBar();
    searchInput.addEventListener("input", () => {
      searchTerm = searchInput.value.trim().toLowerCase();
      renderAll();
    });
  }

  function reset() {
    allItems = [];
    sinceTotal = 0;
    knownTags = new Set();
    activeTag = "All";
    if (searchInput) searchInput.value = "";
    searchTerm = "";
    if (logEl) logEl.innerHTML = "";
    rebuildTagBar();
  }

  async function poll(run) {
    const data = await Api.log(run, sinceTotal);
    if (!data || !Array.isArray(data.lines) || !data.lines.length) {
      sinceTotal = data ? data.total : sinceTotal;
      return;
    }
    const parsed = data.lines.map(parseLine);
    let tagsChanged = false;
    parsed.forEach((p) => {
      if (p.tag && !knownTags.has(p.tag)) {
        knownTags.add(p.tag);
        tagsChanged = true;
      }
    });
    allItems.push(...parsed);
    sinceTotal = data.total;
    if (tagsChanged) rebuildTagBar();
    appendItems(parsed);
  }

  return { init, reset, poll };
})();
