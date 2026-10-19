/* Run selection, polling, tabs, and summary metrics */
(function () {
  const state = {
    currentRun: "",
    refreshMs: 3000,
    autoRefresh: true,
    timer: null,
    runs: [],
  };

  const el = {
    runSelect: document.getElementById("runSelect"),
    runStatusBadge: document.getElementById("runStatusBadge"),
    autoRefreshToggle: document.getElementById("autoRefreshToggle"),
    refreshInterval: document.getElementById("refreshInterval"),
    connDot: document.getElementById("connDot"),
    tabs: document.querySelectorAll(".tab-btn"),
    panels: document.querySelectorAll(".panel"),
    mStatus: document.getElementById("mStatus"),
    mIteration: document.getElementById("mIteration"),
    mElapsed: document.getElementById("mElapsed"),
    mFuncPage: document.getElementById("mFuncPage"),
    mCompletion: document.getElementById("mCompletion"),
    mBugs: document.getElementById("mBugs"),
    mLLM: document.getElementById("mLLM"),
    toast: document.getElementById("toast"),
  };

  function showToast(msg) {
    el.toast.textContent = msg;
    el.toast.classList.add("show");
    setTimeout(() => el.toast.classList.remove("show"), 2500);
  }

  function setConn(online) {
    el.connDot.classList.toggle("online", !!online);
    el.connDot.classList.toggle("offline", !online);
  }

  function formatElapsed(seconds) {
    seconds = Math.round(Number(seconds) || 0);
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  function statusBadgeClass(status) {
    if (status === "running") return "badge-running";
    if (status === "success") return "badge-success";
    if (status === "failed") return "badge-failed";
    return "badge-muted";
  }

  // ---------- Tabs ----------
  function initTabs() {
    el.tabs.forEach((btn) => {
      btn.addEventListener("click", () => {
        el.tabs.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const target = btn.getAttribute("data-tab");
        el.panels.forEach((p) => p.classList.toggle("active", p.id === `panel-${target}`));
        if (target === "graph") GraphView.fit();
        if (target === "evolution") EvolutionView.resize();
      });
    });
  }

  // ---------- Run selector ----------
  async function refreshRunList(preserveSelection) {
    try {
      const data = await Api.listRuns();
      state.runs = data.runs || [];
      setConn(true);
      const prev = preserveSelection ? el.runSelect.value : "";
      el.runSelect.innerHTML = state.runs
        .map((r) => `<option value="${r.name}">${r.name} ${r.status === "running" ? "🟢" : ""}</option>`)
        .join("");
      if (!state.runs.length) {
        el.runSelect.innerHTML = '<option value="">(No runs available)</option>';
        return;
      }
      let target = prev && state.runs.some((r) => r.name === prev) ? prev : state.runs[0].name;
      el.runSelect.value = target;
      if (target !== state.currentRun) {
        switchRun(target);
      }
    } catch (err) {
      setConn(false);
    }
  }

  function switchRun(name) {
    state.currentRun = name;
    LogView.reset();
    BugsView.setRun(name);
    ComplianceView.setRun(name);
    fetchAndRenderAll();
  }

  // ---------- Metrics ----------
  function updateMetrics(summary) {
    const runResult = (summary && summary.run_result) || {};
    const ffgCounts = runResult.ffg_counts || {};
    const testStatus = ffgCounts.test_status || {};
    const completed = testStatus.completed || 0;
    const totalFunctions = ffgCounts.functions || 0;
    const rate = totalFunctions ? Math.round((completed / totalFunctions) * 1000) / 10 : 0;
    const llm = runResult.llm_usage || {};

    el.mStatus.textContent = runResult.status || "unknown";
    el.runStatusBadge.textContent = runResult.status || "Unknown";
    el.runStatusBadge.className = `badge ${statusBadgeClass(runResult.status)}`;
    el.mIteration.textContent = runResult.iteration_count ?? "-";
    el.mElapsed.textContent = formatElapsed(runResult.elapsed_seconds);
    el.mFuncPage.textContent = `${ffgCounts.functions ?? 0} / ${ffgCounts.pages ?? 0}`;
    el.mCompletion.textContent = `${rate}%`;
    el.mBugs.textContent = `${runResult.high_confidence_bug_count ?? 0} (total ${summary.bug_count ?? 0})`;
    el.mLLM.textContent = `${llm.requests ?? 0} / ${llm.total_tokens ?? 0}`;
  }

  // ---------- Fetch & render everything for current run ----------
  async function fetchAndRenderAll() {
    if (!state.currentRun) return;
    const run = state.currentRun;
    try {
      const [summary, ffgData, bugsData, timeline, metrics, live] = await Promise.all([
        Api.summary(run),
        Api.ffg(run),
        Api.bugs(run),
        Api.timeline(run),
        Api.metrics(run),
        Api.live(run),
      ]);
      setConn(true);
      updateMetrics(summary);
      GraphView.setData(ffgData, run, live);
      renderLiveState(live, run);
      const runInfo = summary.run_result || {};
      const isComplianceRun = String(runInfo.mode || "") === "compliance" || !!runInfo.compliance_detection_enabled;
      const bugDetectionEnabled = runInfo.bug_detection_enabled !== false;
      BugsView.setBugs(
        bugsData.bugs || [],
        isComplianceRun
          ? "Bug detection was not run in this mode. See the Compliance tab for rule verdicts."
          : bugDetectionEnabled
            ? "No bugs were detected in this run."
            : "Bug detection is disabled for this run (--no-bug-detection)."
      );
      EvolutionView.render(metrics, timeline);
      StatsView.render(summary, timeline);
      await ComplianceView.render();
      await LogView.poll(run);
    } catch (err) {
      setConn(false);
      console.error(err);
    }
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function stageText(stage) {
    return ({
      initializing: "Initializing",
      planning: "Planning",
      executing: "Executing",
      updating_ffg: "Updating FFG",
      round_complete: "Iteration complete",
    })[stage] || stage || "Waiting for data";
  }

  function phaseText(phase) {
    return phase === "navigation" ? "Navigation" : phase === "test" ? "Test" : phase || "Execution";
  }

  function renderLiveState(live, run) {
    live = live || {};
    const runInfo = state.runs.find((item) => item.name === run) || {};
    const running = runInfo.status === "running";
    const page = live.current_page || {};
    const delta = live.delta || {};
    const trace = live.trace || [];
    const shownRound = live.display_round || live.latest_round || 0;

    document.getElementById("livePulse").classList.toggle("active", running);
    document.getElementById("liveRound").textContent = `Iteration ${live.latest_round || shownRound || "-"}`;
    document.getElementById("liveStage").textContent = stageText(live.stage);
    document.getElementById("liveStage").className = `stage-pill stage-${live.stage || "unknown"}`;
    const scenarioPrefix = live.trace_is_historical ? `Latest execution · Iteration ${live.trace_round} · ` : "";
    document.getElementById("liveScenario").textContent = scenarioPrefix + (live.scenario || `${live.mode || "Unified planning"} · ${live.scope || "Waiting for exploration scope"}`);

    const deltaItems = [
      ["functions", "Functionality"], ["pages", "Page"], ["operations", "Action"], ["transitions", "Transitions"],
    ].filter(([key]) => Number(delta[key] || 0) > 0);
    document.getElementById("liveDelta").innerHTML = deltaItems.length
      ? `<span class="delta-label">${live.trace_is_historical ? "Latest execution changes" : "Changes this iteration"}</span>${deltaItems.map(([key, label]) => `<span class="delta-chip">+${delta[key]} ${label}</span>`).join("")}`
      : `<span class="delta-empty">${live.trace_is_historical ? "No structural changes in the latest execution" : "No structural changes this iteration"}</span>`;

    const pageMatch = document.getElementById("currentPageMatch");
    const exactPage = live.current_page_match === "exact";
    pageMatch.textContent = exactPage ? "Matched FFG page" : page.id ? "Inferred from activity" : "Activity only";
    pageMatch.className = exactPage ? "matched" : "unmatched";
    const screenshot = page.representative_screenshot
      ? `<div class="phone-shot"><img src="${Api.fileUrl(run, "screenshots/" + page.representative_screenshot)}" alt="${escapeHtml(page.name || page.id)} representative FFG screenshot" /></div>`
      : '<div class="phone-shot phone-empty">No representative screenshot</div>';
    document.getElementById("currentPageView").innerHTML = `
      ${screenshot}
      <div class="current-page-name">${escapeHtml(page.name || live.current_page_id || "Unmatched page")}</div>
      <div class="current-page-id">${escapeHtml(page.id || "Unknown FFG page")}</div>
      <div class="activity-label">Activity</div>
      <div class="activity-value" title="${escapeHtml(live.current_activity || "")}">${escapeHtml(live.current_activity || page.activity || "-")}</div>
      <div class="page-note">${exactPage ? "The screenshot represents the stored FFG page; the activity is from the latest observation." : page.id ? "This older run did not store a final page ID. The page is inferred from the latest FFG page with the same activity." : "The current activity cannot be mapped to an FFG page."}</div>`;

    document.getElementById("traceMeta").textContent = `${live.trace_is_historical ? `Latest execution: Iteration ${live.trace_round} · ` : ""}${trace.length} steps`;
    document.getElementById("liveTrace").innerHTML = trace.length ? trace.map((step, index) => `
      <div class="live-step phase-${escapeHtml(step.phase)} ${index === trace.length - 1 ? "latest" : ""}">
        <div class="step-rail"><span>${step.sequence}</span></div>
        <div class="step-content">
          <div class="step-top"><span class="phase-label">${phaseText(step.phase)}</span><b>${escapeHtml(step.action)}</b></div>
          <div class="step-page">${escapeHtml(step.page_id || step.activity || "Page match pending")}${step.page_inferred ? " · Inferred from activity" : ""}</div>
          <div class="step-thought">${escapeHtml(step.thought || "No reasoning summary")}</div>
        </div>
      </div>`).join("") : '<div class="detail-empty">No actions executed this iteration</div>';
    const traceEl = document.getElementById("liveTrace");
    if (running && trace.length) traceEl.scrollTop = traceEl.scrollHeight;
  }

  async function tick() {
    if (!state.autoRefresh || !state.currentRun) return;
    await refreshRunList(true);
    await fetchAndRenderAll();
  }

  function restartTimer() {
    if (state.timer) clearInterval(state.timer);
    state.timer = setInterval(tick, state.refreshMs);
  }

  // ---------- Graph toolbar ----------
  function initGraphToolbar() {
    document.getElementById("graphSearch").addEventListener("input", (e) => {
      GraphView.setFilter({ search: e.target.value });
    });
    document.getElementById("toggleTransitions").addEventListener("change", (e) => {
      GraphView.setFilter({ showTransitions: e.target.checked });
    });
    document.getElementById("toggleDependencies").addEventListener("change", (e) => {
      GraphView.setFilter({ showDependencies: e.target.checked });
    });
    document.getElementById("togglePages").addEventListener("change", (e) => {
      GraphView.setFilter({ showPages: e.target.checked });
    });
    document.getElementById("graphFitBtn").addEventListener("click", () => GraphView.fit());
  }

  function init() {
    initTabs();
    initGraphToolbar();
    GraphView.init(document.getElementById("ffgGraph"), document.getElementById("detailPanel"));
    BugsView.init(document.getElementById("bugsList"), document.getElementById("bugDetail"));
    ComplianceView.init();
    LogView.init(
      document.getElementById("logView"),
      document.getElementById("logTagFilters"),
      document.getElementById("logSearch"),
      document.getElementById("logAutoScroll")
    );

    el.runSelect.addEventListener("change", () => switchRun(el.runSelect.value));
    el.autoRefreshToggle.addEventListener("change", (e) => {
      state.autoRefresh = e.target.checked;
    });
    el.refreshInterval.addEventListener("change", (e) => {
      state.refreshMs = Number(e.target.value) || 3000;
      restartTimer();
    });

    refreshRunList(false);
    restartTimer();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
