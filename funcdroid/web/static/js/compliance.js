/* Compliance rule verdicts, evidence summaries, and traces from compliance_report.json */
const ComplianceView = (() => {
  const VERDICT_META = {
    violation: { label: "Violation", cls: "verdict-violation" },
    compliant: { label: "Compliant", cls: "verdict-compliant" },
    undetermined: { label: "Insufficient evidence", cls: "verdict-undetermined" },
    not_applicable: { label: "Not applicable", cls: "verdict-not-applicable" },
  };
  const VERDICT_ORDER = { violation: 0, undetermined: 1, compliant: 2, not_applicable: 3 };
  const GROUP_LABELS = {
    first_launch_flow: "Initial launch and consent",
    permission_denial_flow: "Permission refusal",
    consent_observation: "Separate sensitive-data consent",
    registration_form: "Registration data minimization",
    settings_search: "Settings disclosures and controls",
    rights_entry_search: "User-rights entrypoints",
    policy_text_review: "Privacy-policy text review",
  };

  let currentRun = "";
  let report = null;
  let reportSignature = "";
  let activeVerdict = "all";
  let searchTerm = "";
  const traceCache = new Map();

  let summaryEl = null;
  let listEl = null;
  let filterEl = null;
  let searchEl = null;

  function escapeHtml(str) {
    return String(str ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function verdictMeta(verdict) {
    return VERDICT_META[verdict] || { label: verdict || "Unknown", cls: "verdict-undetermined" };
  }

  function init() {
    summaryEl = document.getElementById("complianceSummary");
    listEl = document.getElementById("complianceChecks");
    filterEl = document.getElementById("complianceFilters");
    searchEl = document.getElementById("complianceSearch");
    if (searchEl) {
      searchEl.addEventListener("input", () => {
        searchTerm = searchEl.value.trim().toLowerCase();
        renderList();
      });
    }
    rebuildFilters();
  }

  function rebuildFilters() {
    if (!filterEl) return;
    const options = [["all", "All"], ["violation", "Violation"], ["undetermined", "Insufficient evidence"], ["compliant", "Compliant"], ["not_applicable", "Not applicable"]];
    filterEl.innerHTML = options
      .map(([key, label]) => `<button type="button" class="verdict-filter${key === activeVerdict ? " active" : ""}" data-verdict="${key}">${label}</button>`)
      .join("");
    filterEl.querySelectorAll(".verdict-filter").forEach((btn) => {
      btn.addEventListener("click", () => {
        activeVerdict = btn.getAttribute("data-verdict");
        rebuildFilters();
        renderList();
      });
    });
  }

  function setRun(run) {
    if (run !== currentRun) {
      currentRun = run;
      report = null;
      reportSignature = "";
      activeVerdict = "all";
      searchTerm = "";
      traceCache.clear();
      if (searchEl) searchEl.value = "";
      rebuildFilters();
    }
  }

  function reportFingerprint(data) {
    if (!data) return "none";
    const checks = (data.checks || [])
      .map((c) => [c.rule_id, c.verdict, c.attempts, c.description, c.trace_path].join("~"))
      .join("|");
    return `${JSON.stringify(data.summary || {})}::${checks}`;
  }

  async function render() {
    if (!currentRun) return;
    let data = null;
    try {
      data = await Api.compliance(currentRun);
    } catch (err) {
      data = null;
    }
    report = data && Array.isArray(data.checks) ? data : null;
    // Preserve expanded sections and scroll position when polling returns unchanged data.
    const signature = reportFingerprint(report);
    if (signature === reportSignature) return;
    reportSignature = signature;
    renderSummary();
    renderList();
  }

  function renderSummary() {
    if (!summaryEl) return;
    if (!report) {
      summaryEl.innerHTML = '<div class="compliance-empty">This run has no compliance report; it may use a different mode.</div>';
      return;
    }
    const s = report.summary || {};
    const total = s.gui_total ?? (report.checks || []).length;
    const items = [
      ["Total rules", total, ""],
      ["Violation", s.violation ?? 0, "stat-violation"],
      ["Insufficient evidence", s.undetermined ?? 0, "stat-undetermined"],
      ["Compliant", s.compliant ?? 0, "stat-compliant"],
      ["Not applicable", s.not_applicable ?? 0, "stat-not-applicable"],
      ["Not processed", s.untested ?? s.pending ?? 0, ""],
    ];
    summaryEl.innerHTML = items
      .map(([label, value, cls]) => `
        <div class="compliance-stat ${cls}">
          <div class="compliance-stat-value">${value}</div>
          <div class="compliance-stat-label">${label}</div>
        </div>`)
      .join("");
  }

  function matchesFilter(check) {
    const verdict = String(check.verdict || "undetermined");
    if (activeVerdict !== "all" && verdict !== activeVerdict) return false;
    if (!searchTerm) return true;
    const haystack = [check.rule_id, check.clause, check.description, check.regulation]
      .map((v) => String(v || "").toLowerCase())
      .join(" ");
    return haystack.includes(searchTerm);
  }

  function sortChecks(a, b) {
    const va = VERDICT_ORDER[String(a.verdict || "")] ?? 9;
    const vb = VERDICT_ORDER[String(b.verdict || "")] ?? 9;
    if (va !== vb) return va - vb;
    return String(a.rule_id || "").localeCompare(String(b.rule_id || ""));
  }

  function renderList() {
    if (!listEl) return;
    if (!report) {
      listEl.innerHTML = "";
      return;
    }
    const checks = (report.checks || []).filter(matchesFilter).sort(sortChecks);
    if (!checks.length) {
      listEl.innerHTML = '<div class="compliance-empty">No rules match the current filters.</div>';
      return;
    }
    listEl.innerHTML = checks.map(renderCheckCard).join("");
    listEl.querySelectorAll(".check-card").forEach((card) => {
      const head = card.querySelector(".check-head");
      if (head) head.addEventListener("click", () => toggleCard(card));
    });
  }

  function renderCheckCard(check) {
    const verdict = String(check.verdict || "undetermined");
    const meta = verdictMeta(verdict);
    const group = GROUP_LABELS[check.group] || check.group || "";
    const attempts = Number(check.attempts || 0);
    return `
      <div class="check-card" data-verdict="${escapeHtml(verdict)}" data-rule-id="${escapeHtml(check.rule_id || "")}">
        <div class="check-head">
          <span class="check-id">${escapeHtml(check.rule_id || "-")}</span>
          <span class="verdict-pill ${meta.cls}">${meta.label}</span>
          ${group ? `<span class="check-group">${escapeHtml(group)}</span>` : ""}
          <span class="check-toggle">▾</span>
        </div>
        <div class="check-clause">${escapeHtml(check.clause || "")}</div>
        ${check.description ? `<div class="check-desc">${escapeHtml(check.description)}</div>` : ""}
        <div class="check-meta">
          ${escapeHtml(check.regulation || "")}${attempts ? ` · Attempts ${attempts}` : ""}${check.trace_path ? ` · <span class="check-trace-hint">Evidence trace available</span>` : ""}
        </div>
        <div class="check-evidence" hidden></div>
      </div>`;
  }

  async function toggleCard(card) {
    const evidenceEl = card.querySelector(".check-evidence");
    if (!evidenceEl) return;
    const open = card.classList.toggle("open");
    if (!open) {
      evidenceEl.hidden = true;
      return;
    }
    evidenceEl.hidden = false;
    if (evidenceEl.dataset.loaded === "1") return;
    evidenceEl.dataset.loaded = "1";
    const ruleId = card.dataset.ruleId;
    const check = (report.checks || []).find((c) => String(c.rule_id) === ruleId);
    if (!check || !check.trace_path) {
      evidenceEl.innerHTML = '<div class="evidence-empty">This verdict has no locatable execution trace.</div>';
      return;
    }
    evidenceEl.innerHTML = '<div class="evidence-empty">Loading evidence trace...</div>';
    try {
      let trace = traceCache.get(check.trace_path);
      if (!trace) {
        trace = await Api.complianceTrace(currentRun, check.trace_path);
        traceCache.set(check.trace_path, trace);
      }
      evidenceEl.innerHTML = renderEvidence(check, trace);
    } catch (err) {
      delete evidenceEl.dataset.loaded;  // Allow a retry on the next click.
      evidenceEl.innerHTML = `<div class="evidence-empty">Failed to read trace: ${escapeHtml(err && err.message ? err.message : err)}</div>`;
    }
  }

  function renderEvidence(check, trace) {
    const steps = Array.isArray(trace.steps) ? trace.steps : [];
    const scenarios = trace.scenario ? `<div class="evidence-scenario">${escapeHtml(String(trace.scenario).slice(0, 220))}</div>` : "";
    if (!steps.length) {
      return `${scenarios}<div class="evidence-empty">The trace file contains no steps.</div>`;
    }
    const traceDir = String(check.trace_path).replace(/\/[^/]+$/, "");
    const evidenceItems = (check.evidence || [])
      .map((ev) => {
        const step = steps.find((s) => Number(s.step_index) === Number(ev.global_step));
        return { ev, step };
      })
      .filter((item) => item.step);
    const displayItems = evidenceItems.length
      ? evidenceItems
      : steps.slice(-3).map((step) => ({ ev: { global_step: step.step_index, role: "context" }, step }));
    const html = displayItems
      .map(({ ev, step }) => {
        const page = step.page || {};
        const screenshot = page.screenshot || "";
        const shotPath = screenshot ? `${traceDir}/${screenshot}` : "";
        const image = shotPath
          ? `<img src="${Api.fileUrl(currentRun, shotPath)}" alt="Step ${step.step_index} screenshot" loading="lazy" onerror="this.style.display='none'"/>`
          : '<div class="evidence-no-shot">No screenshot</div>';
        const operation = step.operation || {};
        const params = operation.parameters && Object.keys(operation.parameters).length
          ? ` ${escapeHtml(JSON.stringify(operation.parameters))}`
          : "";
        return `
          <div class="evidence-step">
            <div class="evidence-shot">${image}</div>
            <div class="evidence-body">
              <div class="evidence-action">Step ${escapeHtml(step.step_index ?? "-")} · ${escapeHtml(operation.action_name || "Action")}${params} <span class="evidence-role">${escapeHtml(ev.role || "")}</span></div>
              <div class="evidence-thought">${escapeHtml(step.thought || step.action_description || "(No reasoning summary)")}</div>
              <div class="evidence-page">${escapeHtml(page.id || "")} ${escapeHtml(page.activity || "")}</div>
            </div>
          </div>`;
      })
      .join("");
    return `${scenarios}<div class="evidence-steps">${html}</div>`;
  }

  return { init, setRun, render };
})();
