/* FFG visualization of functionalities, pages, and their relationships using vis-network */
const GraphView = (() => {
  let network = null;
  let nodesDS = null;
  let edgesDS = null;
  let rawFfg = { functions: [], pages: [], page_transitions: [], function_dependencies: [] };
  let liveState = {};
  let currentRun = "";
  let hasBaseline = false;
  const recentNewNodes = new Map();
  let container = null;
  let detailPanel = null;

  // Semantic status colors for the dark graph canvas
  const STATUS_COLOR = {
    not_started: "#86868b",
    unknown: "#86868b",
    in_progress: "#2997ff",
    completed: "#30d158",
    blocked: "#ff9f0a",
    deferred: "#ff9f0a",
    unavailable: "#ff453a",
  };
  const PAGE_COLOR = "#bf5af2";

  const filters = {
    showPages: true,
    showTransitions: true,
    showDependencies: true,
    search: "",
  };

  function statusColor(status) {
    return STATUS_COLOR[status] || STATUS_COLOR.not_started;
  }

  function truncate(text, n) {
    if (!text) return "";
    text = String(text);
    return text.length > n ? text.slice(0, n) + "…" : text;
  }

  function buildDatasets() {
    const nodes = [];
    const edges = [];
    const searchTerm = filters.search.trim().toLowerCase();
    const currentPageId = String(liveState.current_page_id || "");
    const tracePageIds = new Set((liveState.trace || []).map((step) => String(step.page_id || "")).filter(Boolean));
    const now = Date.now();
    for (const [id, expiresAt] of recentNewNodes.entries()) {
      if (expiresAt <= now) recentNewNodes.delete(id);
    }
    const tracePairs = new Set();
    let previousTracePage = "";
    (liveState.trace || []).forEach((step) => {
      const pageId = String(step.page_id || "");
      if (previousTracePage && pageId) tracePairs.add(`${previousTracePage}>${pageId}`);
      if (pageId) previousTracePage = pageId;
    });

    const funcMatch = (fn) =>
      !searchTerm ||
      (fn.name || "").toLowerCase().includes(searchTerm) ||
      (fn.id || "").toLowerCase().includes(searchTerm) ||
      (fn.description || "").toLowerCase().includes(searchTerm);
    const pageMatch = (pg) =>
      !searchTerm ||
      (pg.name || "").toLowerCase().includes(searchTerm) ||
      (pg.id || "").toLowerCase().includes(searchTerm) ||
      (pg.activity || "").toLowerCase().includes(searchTerm);

    const visibleFuncIds = new Set();
    (rawFfg.functions || []).forEach((fn) => {
      const visible = funcMatch(fn);
      if (!visible && searchTerm) return;
      visibleFuncIds.add(fn.id);
      const isNew = recentNewNodes.has(`f:${fn.id}`);
      const coverage = Math.max(0, Math.min(100, Number(fn.coverage_percent || 0)));
      nodes.push({
        id: `f:${fn.id}`,
        label: `${truncate(fn.name || fn.id, 22)}\n${coverage}%`,
        shape: "box",
        margin: 10,
        color: {
          background: statusColor(fn.test_status),
          border: statusColor(fn.test_status),
          highlight: { background: statusColor(fn.test_status), border: "#ffffff" },
        },
        font: { color: "#1d1d1f", size: 13, bold: { color: "#1d1d1f" } },
        borderWidth: isNew ? 3 : 0,
        shadow: isNew
          ? { enabled: true, color: "rgba(48,209,88,.85)", size: 24 }
          : fn.test_status === "in_progress" ? { enabled: true, color: "rgba(41,151,255,.55)", size: 14 } : false,
        group: "function",
        raw: fn,
      });
    });

    const visiblePageIds = new Set();
    if (filters.showPages) {
      (rawFfg.pages || []).forEach((pg) => {
        const visible = pageMatch(pg);
        if (!visible && searchTerm) return;
        visiblePageIds.add(pg.id);
        const isCurrent = pg.id === currentPageId;
        const isTrace = tracePageIds.has(pg.id);
        const isNew = recentNewNodes.has(`p:${pg.id}`);
        nodes.push({
          id: `p:${pg.id}`,
          label: truncate(pg.name || pg.id, 18),
          shape: isCurrent ? "dot" : "ellipse",
          size: isCurrent ? 17 : undefined,
          color: {
            background: isCurrent ? "#ffffff" : isTrace ? "#2997ff" : PAGE_COLOR,
            border: isCurrent ? "#30d158" : isTrace ? "#8bc7ff" : PAGE_COLOR,
            highlight: { background: "#ffffff", border: "#30d158" },
          },
          borderWidth: isCurrent ? 5 : isNew ? 4 : isTrace ? 3 : 1,
          shadow: isNew
            ? { enabled: true, color: "rgba(48,209,88,.85)", size: 25 }
            : isCurrent ? { enabled: true, color: "rgba(48,209,88,.8)", size: 20 } : false,
          font: { color: "#1d1d1f", size: isCurrent ? 13 : 11, vadjust: isCurrent ? -30 : 0 },
          group: "page",
          raw: pg,
        });
      });

      (rawFfg.functions || []).forEach((fn) => {
        if (!visibleFuncIds.has(fn.id)) return;
        (fn.page_ids || []).forEach((pid) => {
          if (!visiblePageIds.has(pid)) return;
          edges.push({
            id: `contains:${fn.id}:${pid}`,
            from: `f:${fn.id}`,
            to: `p:${pid}`,
            color: { color: "rgba(255,255,255,0.28)", opacity: 0.6 },
            dashes: true,
            arrows: "to",
            width: 1,
            group: "contains",
          });
        });
      });

      if (filters.showTransitions) {
        // Multiple transitions can share a source/destination pair, one per action.
        // Aggregate them into one edge whose tooltip lists every observed action.
        const transitionGroups = new Map();
        (rawFfg.page_transitions || []).forEach((t) => {
          if (!visiblePageIds.has(t.src_page_id) || !visiblePageIds.has(t.dst_page_id)) return;
          const key = `${t.src_page_id}>${t.dst_page_id}`;
          if (!transitionGroups.has(key)) transitionGroups.set(key, []);
          transitionGroups.get(key).push(t);
        });
        transitionGroups.forEach((transitions, key) => {
          const [srcId, dstId] = key.split(">");
          const onTrace = tracePairs.has(key);
          const ops = [...new Set(transitions.map((t) => t.operation || "Page transition"))];
          edges.push({
            id: `trans:${srcId}:${dstId}`,
            from: `p:${srcId}`,
            to: `p:${dstId}`,
            title: ops.map(escapeHtml).join("<br/>"),
            color: { color: onTrace ? "#2997ff" : "rgba(255,255,255,0.22)", opacity: onTrace ? 0.95 : 0.5 },
            arrows: "to",
            width: onTrace ? 3 : 1,
            group: "transition",
          });
        });
      }
    }

    if (filters.showDependencies) {
      (rawFfg.function_dependencies || []).forEach((d, idx) => {
        if (!visibleFuncIds.has(d.src_function_id) || !visibleFuncIds.has(d.dst_function_id)) return;
        edges.push({
          id: `dep:${idx}:${d.src_function_id}:${d.dst_function_id}`,
          from: `f:${d.src_function_id}`,
          to: `f:${d.dst_function_id}`,
          label: truncate(d.resource_variable, 14),
          font: { size: 9, color: "#ff9f0a", strokeWidth: 0 },
          color: { color: "#ff9f0a", opacity: 0.85 },
          arrows: "to",
          width: 2,
          group: "dependency",
        });
      });
    }

    return { nodes, edges };
  }

  // Include the close button in each rendered detail panel because innerHTML replaces it.
  // Recreate it with the content so selecting a node cannot remove the close control.
  function detailShell(bodyHtml) {
    return `<button class="detail-close" aria-label="Close details">×</button>${bodyHtml}`;
  }

  function renderDetailFunction(fn) {
    const scenarios = fn.scenarios || [];
    detailPanel.innerHTML = detailShell(`
      <div class="detail-title">${escapeHtml(fn.name || fn.id)}</div>
      <div class="detail-sub">Functionality · ${escapeHtml(fn.id)} · <span class="status-pill status-${fn.test_status || 'unknown'}">${escapeHtml(fn.test_status || 'unknown')}</span></div>
      <div class="detail-section">
        <div class="detail-section-title">Description</div>
        <div class="detail-desc">${escapeHtml(fn.description || '(No description)')}</div>
      </div>
      <div class="detail-section">
        <div class="detail-section-title">Coverage score</div>
        <div class="detail-desc">${escapeHtml(String(fn.coverage_percent ?? 0))}%</div>
      </div>
      <div class="detail-section">
        <div class="detail-section-title">State/resource variables (${(fn.resource_variables||[]).length})</div>
        <div class="chip-list">${(fn.resource_variables||[]).map(v => `<span class="chip">${escapeHtml(v)}</span>`).join('') || '<span class="detail-desc">None</span>'}</div>
      </div>
      <div class="detail-section">
        <div class="detail-section-title">Recorded scenarios (${scenarios.length})</div>
        <div class="detail-desc">${scenarios.map(s => `<div style="margin-bottom:6px;">• ${escapeHtml(typeof s === 'string' ? s : JSON.stringify(s))}</div>`).join('') || 'None'}</div>
      </div>
      <div class="detail-section">
        <div class="detail-section-title">Associated pages (${(fn.page_ids||[]).length})</div>
        <div class="chip-list">${(fn.page_ids||[]).map(v => `<span class="chip">${escapeHtml(v)}</span>`).join('') || '<span class="detail-desc">None</span>'}</div>
      </div>
    `);
  }

  function renderDetailPage(pg) {
    const shot = pg.representative_screenshot ? `<img class="shot" src="${Api.fileUrl(currentRun, 'screenshots/' + pg.representative_screenshot)}" onerror="this.style.display='none'"/>` : '';
    detailPanel.innerHTML = detailShell(`
      <div class="detail-title">${escapeHtml(pg.name || pg.id)}</div>
      <div class="detail-sub">Page · ${escapeHtml(pg.id)}</div>
      ${shot}
      <div class="detail-section">
        <div class="detail-section-title">Activity</div>
        <div class="detail-desc">${escapeHtml(pg.activity || '-')}</div>
      </div>
      <div class="detail-section">
        <div class="detail-section-title">Description</div>
        <div class="detail-desc">${escapeHtml(pg.description || '(No description)')}</div>
      </div>
      <div class="detail-section">
        <div class="detail-section-title">Widgets (${(pg.controls||[]).length})</div>
        <div class="chip-list">${(pg.controls||[]).slice(0, 40).map(v => `<span class="chip">${escapeHtml(truncate(v, 30))}</span>`).join('') || 'None'}</div>
      </div>
      <div class="detail-section">
        <div class="detail-section-title">Executed operations (${(pg.operations||[]).length})</div>
        <div class="chip-list">${(pg.operations||[]).slice(0, 40).map(v => `<span class="chip">${escapeHtml(truncate(v, 30))}</span>`).join('') || 'None'}</div>
      </div>
    `);
  }

  function escapeHtml(str) {
    return String(str ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function init(containerEl, detailEl) {
    container = containerEl;
    detailPanel = detailEl;
    nodesDS = new vis.DataSet([]);
    edgesDS = new vis.DataSet([]);
    const options = {
      autoResize: true,
      layout: { improvedLayout: true },
      physics: {
        enabled: true,
        solver: "forceAtlas2Based",
        forceAtlas2Based: { gravitationalConstant: -60, springLength: 110, springConstant: 0.06, avoidOverlap: 0.6 },
        stabilization: { enabled: true, iterations: 200, fit: true },
        minVelocity: 0.75,
      },
      interaction: { hover: true, tooltipDelay: 150 },
      nodes: { borderWidth: 1, shadow: false },
      // Disable dynamic smoothing, which introduces unconstrained support nodes.
      // Those nodes can keep the physics simulation moving indefinitely.
      edges: { smooth: false, selectionWidth: 2 },
    };
    network = new vis.Network(container, { nodes: nodesDS, edges: edgesDS }, options);
    network.on("click", (params) => {
      if (!params.nodes.length) return;
      const node = nodesDS.get(params.nodes[0]);
      if (!node) return;
      if (node.group === "function") renderDetailFunction(node.raw);
      else if (node.group === "page") renderDetailPage(node.raw);
      detailPanel.classList.add("open");
    });
    // Delegate close events because detail rendering replaces the button.
    detailPanel.addEventListener("click", (event) => {
      if (event.target.closest(".detail-close")) {
        detailPanel.classList.remove("open");
      }
    });
    // Disable continuous physics after initial stabilization to prevent drift.
    network.once("stabilizationIterationsDone", () => {
      network.setOptions({ physics: false });
    });
  }

  function refresh() {
    const { nodes, edges } = buildDatasets();
    const oldNodeIds = nodesDS.getIds();
    const oldEdgeIds = edgesDS.getIds();
    const oldNodeIdSet = new Set(oldNodeIds);
    const oldEdgeIdSet = new Set(oldEdgeIds);
    const newNodeIdSet = new Set(nodes.map((n) => n.id));
    const newEdgeIdSet = new Set(edges.map((e) => e.id));

    const nodesToRemove = oldNodeIds.filter((id) => !newNodeIdSet.has(id));
    const edgesToRemove = oldEdgeIds.filter((id) => !newEdgeIdSet.has(id));
    const addedNodeCount = nodes.filter((n) => !oldNodeIdSet.has(n.id)).length;
    const addedEdgeCount = edges.filter((e) => !oldEdgeIdSet.has(e.id)).length;
    const structureChanged = nodesToRemove.length > 0 || edgesToRemove.length > 0 || addedNodeCount > 0 || addedEdgeCount > 0;

    // Use incremental upserts to preserve unchanged node positions.
    // Do not reset all coordinates on every polling update.
    if (edgesToRemove.length) edgesDS.remove(edgesToRemove);
    if (nodesToRemove.length) nodesDS.remove(nodesToRemove);
    if (nodes.length) nodesDS.update(nodes);
    if (edges.length) edgesDS.update(edges);

    // Run bounded stabilization only when nodes or edges are added or removed.
    // Position new nodes while keeping unchanged topology still.
    if (structureChanged && network) {
      network.stabilize(120);
    }
  }

  function setData(ffgData, runName, live) {
    const nextFfg = ffgData || rawFfg;
    if (runName !== currentRun) {
      hasBaseline = false;
      recentNewNodes.clear();
    }
    if (hasBaseline) {
      const oldFunctions = new Set((rawFfg.functions || []).map((fn) => fn.id));
      const oldPages = new Set((rawFfg.pages || []).map((page) => page.id));
      const expiresAt = Date.now() + 12000;
      (nextFfg.functions || []).forEach((fn) => {
        if (!oldFunctions.has(fn.id)) recentNewNodes.set(`f:${fn.id}`, expiresAt);
      });
      (nextFfg.pages || []).forEach((page) => {
        if (!oldPages.has(page.id)) recentNewNodes.set(`p:${page.id}`, expiresAt);
      });
    }
    rawFfg = nextFfg;
    currentRun = runName;
    liveState = live || {};
    hasBaseline = true;
    refresh();
  }

  function closeDetail() {
    if (detailPanel) detailPanel.classList.remove("open");
  }

  function setFilter(partial) {
    Object.assign(filters, partial);
    refresh();
  }

  function fit() {
    if (network) network.fit({ animation: { duration: 400 } });
  }

  function counts() {
    return {
      functions: (rawFfg.functions || []).length,
      pages: (rawFfg.pages || []).length,
      transitions: (rawFfg.page_transitions || []).length,
      dependencies: (rawFfg.function_dependencies || []).length,
    };
  }

  return { init, setData, setFilter, fit, counts, closeDetail };
})();
