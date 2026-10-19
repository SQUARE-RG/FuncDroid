/* Statistics: model usage, errors, artifacts, and observed activities */
const StatsView = (() => {
  function escapeHtml(str) {
    return String(str ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  const ERROR_LABELS = {
    navigation_failure_count: "Navigation failures",
    scenario_exec_error_count: "Execution errors",
    scenario_plan_error_count: "Planning errors",
    ffg_update_error_count: "FFG update errors",
    screenshot_unavailable_count: "Screenshot unavailable",
    target_app_unavailable_count: "Target app unavailable",
  };

  function kvItem(label, value) {
    return `<div class="kv-item"><div class="kv-label">${escapeHtml(label)}</div><div class="kv-value">${escapeHtml(value)}</div></div>`;
  }

  function renderLlmTotals(totals) {
    const el = document.getElementById("llmTotals");
    totals = totals || {};
    el.innerHTML = [
      kvItem("Requests", totals.requests ?? 0),
      kvItem("Responses", totals.responses ?? 0),
      kvItem("Failures", totals.failures ?? 0),
      kvItem("Prompt Tokens", totals.prompt_tokens ?? 0),
      kvItem("Completion Tokens", totals.completion_tokens ?? 0),
      kvItem("Total tokens", totals.total_tokens ?? 0),
    ].join("");
  }

  function renderByModel(byModel) {
    const tbody = document.querySelector("#llmByModelTable tbody");
    const entries = Object.entries(byModel || {});
    tbody.innerHTML =
      entries
        .map(
          ([key, v]) => `<tr>
        <td>${escapeHtml(v.model || key)}</td>
        <td>${v.requests ?? 0}</td>
        <td>${v.failures ?? 0}</td>
        <td>${v.prompt_tokens ?? 0}</td>
        <td>${v.completion_tokens ?? 0}</td>
        <td>${v.total_tokens ?? 0}</td>
      </tr>`
        )
        .join("") || `<tr><td colspan="6" style="color:var(--text-dim);">No data</td></tr>`;
  }

  function renderByOperation(byOp) {
    const tbody = document.querySelector("#llmByOpTable tbody");
    const entries = Object.entries(byOp || {});
    tbody.innerHTML =
      entries
        .map(
          ([key, v]) => `<tr>
        <td>${escapeHtml(key)}</td>
        <td>${v.requests ?? 0}</td>
        <td>${v.failures ?? 0}</td>
        <td>${v.total_tokens ?? 0}</td>
      </tr>`
        )
        .join("") || `<tr><td colspan="4" style="color:var(--text-dim);">No data</td></tr>`;
  }

  function renderErrorCounts(errorCounts) {
    const el = document.getElementById("errorCounts");
    errorCounts = errorCounts || {};
    el.innerHTML = Object.entries(ERROR_LABELS)
      .map(([key, label]) => kvItem(label, errorCounts[key] ?? 0))
      .join("");
  }

  function renderManifest(manifest) {
    const el = document.getElementById("manifestInfo");
    manifest = manifest || {};
    const errors = Array.isArray(manifest.errors) ? manifest.errors : [];
    const warnings = Array.isArray(manifest.warnings) ? manifest.warnings : [];
    let html = `${kvItem("Errors", errors.length)}${kvItem("Warnings", warnings.length)}`;
    el.innerHTML = html;
    const extra = [];
    if (errors.length) extra.push(`<div class="detail-section-title" style="margin-top:10px;color:#ff3b30;">Error</div>` + errors.map((e) => `<div class="detail-desc">• ${escapeHtml(e)}</div>`).join(""));
    if (warnings.length) extra.push(`<div class="detail-section-title" style="margin-top:10px;color:#ff9500;">Warning</div>` + warnings.map((w) => `<div class="detail-desc">• ${escapeHtml(w)}</div>`).join(""));
    if (extra.length) el.insertAdjacentHTML("beforeend", `<div style="grid-column: 1 / -1;">${extra.join("")}</div>`);
  }

  function renderActivities(activities) {
    const el = document.getElementById("activityList");
    activities = Array.isArray(activities) ? activities : [];
    el.innerHTML = activities.map((a) => `<span class="chip">${escapeHtml(a)}</span>`).join("") || '<span class="detail-desc">None</span>';
  }

  function renderSeed(seed) {
    const card = document.getElementById("seedInfoCard");
    const el = document.getElementById("seedInfo");
    if (!card || !el) return;
    seed = seed || {};
    if (!seed.source_output) {
      card.hidden = true;
      el.innerHTML = "";
      return;
    }
    card.hidden = false;
    const sourceName = String(seed.source_output).split("/").filter(Boolean).pop();
    el.innerHTML = [
      kvItem("Source run", sourceName || "-"),
      kvItem("Source mode", seed.source_mode || "-"),
      kvItem("Inherited functionalities", seed.inherited_functions ?? 0),
      kvItem("Inherited pages", seed.inherited_pages ?? 0),
      kvItem("Coverage reset", seed.coverage_reset ? "Yes" : "No"),
      kvItem("FFG fingerprint", String(seed.ffg_sha256 || "").slice(0, 12) || "-"),
    ].join("");
  }

  function render(summary, timeline) {
    const runResult = (summary && summary.run_result) || {};
    const llmUsage = (summary && summary.llm_usage) || {};
    renderLlmTotals(llmUsage.totals);
    renderByModel(llmUsage.by_model);
    renderByOperation(llmUsage.by_operation);
    renderErrorCounts(runResult.error_counts);
    renderManifest(summary && summary.manifest);
    renderSeed(runResult.seed_provenance);
    renderActivities((timeline && timeline.activities) || (runResult.activity_discovery || {}).activities);
  }

  return { render };
})();
