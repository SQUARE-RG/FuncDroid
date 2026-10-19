/* Bug list and reproduction trace viewer */
const BugsView = (() => {
  let listEl = null;
  let detailEl = null;
  let currentRun = "";
  let bugs = [];
  let selectedBugId = null;
  let emptyHint = "No bugs detected yet";

  function escapeHtml(str) {
    return String(str ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function severityClass(sev) {
    const s = String(sev || "").toLowerCase();
    if (s.includes("high") || s.includes("critical") || s.includes("\u4e25\u91cd") || s.includes("\u9ad8")) return "badge-sev-high";
    if (s.includes("medium") || s.includes("\u4e2d")) return "badge-sev-medium";
    return "badge-sev-low";
  }
  function confidenceClass(conf) {
    const c = String(conf || "").toLowerCase();
    if (c.includes("high") || c.includes("\u9ad8")) return "badge-conf-high";
    if (c.includes("medium") || c.includes("\u4e2d")) return "badge-conf-medium";
    return "badge-conf-low";
  }

  function renderList() {
    if (!bugs.length) {
      listEl.innerHTML = `<div class="bug-empty">${escapeHtml(emptyHint)}</div>`;
      return;
    }
    listEl.innerHTML = bugs
      .map((bug) => {
        const id = bug.bug_id || "";
        const selected = id === selectedBugId ? " selected" : "";
        return `<div class="bug-card${selected}" data-bug-id="${escapeHtml(id)}">
          <div class="bug-card-title">${escapeHtml(id)} · ${escapeHtml(bug.type || "Unclassified")}</div>
          <div class="bug-card-badges">
            <span class="badge ${severityClass(bug.severity)}">${escapeHtml(bug.severity || "-")}</span>
            <span class="badge ${confidenceClass(bug.confidence)}">Confidence: ${escapeHtml(bug.confidence || "-")}</span>
          </div>
          <div class="bug-card-desc">${escapeHtml(truncate(bug.description, 80))}</div>
        </div>`;
      })
      .join("");
    listEl.querySelectorAll(".bug-card").forEach((card) => {
      card.addEventListener("click", () => selectBug(card.getAttribute("data-bug-id")));
    });
  }

  function truncate(text, n) {
    if (!text) return "";
    text = String(text);
    return text.length > n ? text.slice(0, n) + "…" : text;
  }

  function renderTraceStep(step, dirName) {
    const page = step.page || {};
    const shotRel = page.screenshot ? `bugs/${dirName}/${page.screenshot}` : "";
    const img = shotRel ? `<img src="${Api.fileUrl(currentRun, shotRel)}" onerror="this.style.display='none'" />` : "";
    const flags = [];
    if (step.action_failed) flags.push('<span class="chip" style="color:#ff3b30;">Action failed</span>');
    if (step.is_interruption) flags.push('<span class="chip">Interrupted</span>');
    if (step.external_context) flags.push('<span class="chip">External context</span>');
    if (step.foreground_recovery) flags.push('<span class="chip">Foreground recovery</span>');
    const op = step.operation || {};
    const params = op.parameters ? Object.entries(op.parameters).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ") : "";
    return `<div class="trace-step">
      ${img}
      <div class="trace-step-body">
        <div class="trace-step-idx">Step ${step.step_index ?? "-"} · ${escapeHtml(step.trace_phase || "")} ${page.activity ? "· " + escapeHtml(page.activity) : ""}</div>
        <div class="trace-step-action">${escapeHtml(op.action_name || "-")}${params ? ` <span style="color:var(--text-dim);font-weight:400;font-size:12px;">(${escapeHtml(params)})</span>` : ""}</div>
        ${step.thought ? `<div class="trace-step-thought">${escapeHtml(step.thought)}</div>` : ""}
        ${step.action_error ? `<div class="trace-step-thought" style="color:#ff3b30;">Error: ${escapeHtml(step.action_error)}</div>` : ""}
        <div class="trace-step-flags">${flags.join("")}</div>
      </div>
    </div>`;
  }

  async function selectBug(bugId) {
    selectedBugId = bugId;
    renderList();
    detailEl.innerHTML = '<div class="detail-empty">Loading...</div>';
    try {
      const detail = await Api.bugDetail(currentRun, bugId);
      const bug = detail.bug || {};
      const trace = Array.isArray(detail.trace) ? detail.trace : [];
      const evidenceHtml = (bug.evidence || [])
        .map((e) => `<span class="chip">step ${e.global_step} / ${escapeHtml(e.role)}</span>`)
        .join("");
      detailEl.innerHTML = `
        <div class="detail-title">${escapeHtml(bug.bug_id || bugId)} · ${escapeHtml(bug.type || "")}</div>
        <div class="detail-sub">
          <span class="badge ${severityClass(bug.severity)}">${escapeHtml(bug.severity || "-")}</span>
          <span class="badge ${confidenceClass(bug.confidence)}">Confidence ${escapeHtml(bug.confidence || "-")}</span>
          <span class="chip">${escapeHtml(bug.evidence_category || "")}</span>
        </div>
        <div class="detail-section"><div class="detail-section-title">Scenario</div><div class="detail-desc">${escapeHtml(bug.scenario || "-")}</div></div>
        <div class="detail-section"><div class="detail-section-title">Problem description</div><div class="detail-desc">${escapeHtml(bug.description || "-")}</div></div>
        <div class="detail-section"><div class="detail-section-title">Expected behavior</div><div class="detail-desc">${escapeHtml(bug.expected || "-")}</div></div>
        <div class="detail-section"><div class="detail-section-title">Observed behavior</div><div class="detail-desc">${escapeHtml(bug.actual || "-")}</div></div>
        <div class="detail-section"><div class="detail-section-title">Associated functionalities</div><div class="chip-list">
          ${bug.target_function_id ? `<span class="chip">target: ${escapeHtml(bug.target_function_id)}</span>` : ""}
          ${bug.src_function_id ? `<span class="chip">src: ${escapeHtml(bug.src_function_id)}</span>` : ""}
          ${bug.dst_function_id ? `<span class="chip">dst: ${escapeHtml(bug.dst_function_id)}</span>` : ""}
        </div></div>
        <div class="detail-section"><div class="detail-section-title">Evidence steps</div><div class="chip-list">${evidenceHtml || '<span class="detail-desc">None</span>'}</div></div>
        <div class="detail-section">
          <div class="detail-section-title">Reproduction trace (${trace.length} steps)</div>
          ${trace.map((s) => renderTraceStep(s, detail.dir_name)).join("") || '<div class="detail-desc">No trace data</div>'}
        </div>
      `;
    } catch (err) {
      detailEl.innerHTML = `<div class="detail-empty">Failed to load bug details: ${escapeHtml(err.message)}</div>`;
    }
  }

  function init(listContainer, detailContainer) {
    listEl = listContainer;
    detailEl = detailContainer;
  }

  function setRun(run) {
    currentRun = run;
    selectedBugId = null;
    detailEl.innerHTML = '<div class="detail-empty">Select a bug to view its reproduction trace</div>';
  }

  function setBugs(newBugs, hint) {
    bugs = Array.isArray(newBugs) ? newBugs : [];
    if (hint) emptyHint = hint;
    renderList();
  }

  return { init, setRun, setBugs };
})();
