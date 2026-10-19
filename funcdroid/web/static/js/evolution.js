/* Plot FFG size and completion from iteration metrics in run.log,
   and observed activity counts over time from activity_timeline.json */
const EvolutionView = (() => {
  let chartGrowth = null;
  let chartCompletion = null;
  let chartActivity = null;
  let lastMetrics = null;
  let lastTimeline = null;
  let growthSignature = "";
  let completionSignature = "";
  let activitySignature = "";
  let tableSignature = "";

  // Do not create charts in a hidden panel whose dimensions cannot be measured.
  function panelVisible() {
    const panel = document.getElementById("panel-evolution");
    return !!panel && panel.classList.contains("active");
  }

  function renderAllCharts() {
    const iterations = (lastMetrics && lastMetrics.iterations) || [];
    renderGrowthChart(iterations);
    renderCompletionChart(iterations);
    const points = (lastTimeline && lastTimeline.points) || [];
    renderActivityChart(points);
  }

  // Use blue as the primary chart color with restrained secondary colors.
  const CHART_COLORS = {
    functions: "#0066cc",
    pages: "#5ac8fa",
    completed: "#34c759",
    completion: "#ff9500",
    activity: "#bf5af2",
  };

  function baseOptions(extra) {
    return Object.assign(
      {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        responsiveAnimationDuration: 0,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color: "#1d1d1f", font: { size: 11 }, boxWidth: 12, boxHeight: 2 } },
          tooltip: {
            mode: "index",
            intersect: false,
            backgroundColor: "rgba(255,255,255,0.96)",
            titleColor: "#1d1d1f",
            bodyColor: "#333333",
            borderColor: "#e0e0e0",
            borderWidth: 1,
            padding: 10,
            cornerRadius: 8,
          },
        },
        scales: {
          x: {
            border: { display: false },
            ticks: { color: "#86868b", font: { size: 10 } },
            grid: { color: "rgba(0,0,0,0.05)" },
          },
          y: {
            beginAtZero: true,
            border: { display: false },
            ticks: { color: "#86868b", font: { size: 10 }, precision: 0 },
            grid: { color: "rgba(0,0,0,0.05)" },
          },
        },
      },
      extra || {}
    );
  }

  function dataSignature(labels, datasets) {
    return JSON.stringify([labels, ...datasets]);
  }

  function updateChartData(chart, labels, datasets) {
    chart.data.labels = labels;
    datasets.forEach((data, index) => {
      chart.data.datasets[index].data = data;
    });
    chart.update("none");
  }

  function renderGrowthChart(iterations) {
    const canvas = document.getElementById("chartFfgGrowth");
    const labels = iterations.map((it) => `#${it.iteration ?? "?"}`);
    const dataFunctions = iterations.map((it) => it.functions ?? 0);
    const dataPages = iterations.map((it) => it.pages ?? 0);
    const dataCompleted = iterations.map((it) => it.completed_functions ?? 0);

    const signature = dataSignature(labels, [dataFunctions, dataPages, dataCompleted]);
    if (!chartGrowth) {
      chartGrowth = new Chart(canvas, {
        type: "line",
        data: {
          labels,
          datasets: [
            { label: "Functionalities", data: dataFunctions, borderColor: CHART_COLORS.functions, backgroundColor: CHART_COLORS.functions, tension: 0.25, pointRadius: 2 },
            { label: "Pages", data: dataPages, borderColor: CHART_COLORS.pages, backgroundColor: CHART_COLORS.pages, tension: 0.25, pointRadius: 2 },
            { label: "Completed functionalities", data: dataCompleted, borderColor: CHART_COLORS.completed, backgroundColor: CHART_COLORS.completed, tension: 0.25, pointRadius: 2 },
          ],
        },
        options: baseOptions(),
      });
    } else if (signature !== growthSignature) {
      updateChartData(chartGrowth, labels, [dataFunctions, dataPages, dataCompleted]);
    }
    growthSignature = signature;
  }

  function renderCompletionChart(iterations) {
    const canvas = document.getElementById("chartCompletion");
    const labels = iterations.map((it) => `#${it.iteration ?? "?"}`);
    const data = iterations.map((it) => Math.round((it.completion_rate ?? 0) * 1000) / 10);

    const signature = dataSignature(labels, [data]);
    if (!chartCompletion) {
      chartCompletion = new Chart(canvas, {
        type: "line",
        data: {
          labels,
          datasets: [
            { label: "Completion (%)", data, borderColor: CHART_COLORS.completion, backgroundColor: "rgba(255,149,0,0.12)", fill: true, tension: 0.25, pointRadius: 2 },
          ],
        },
        options: baseOptions({ scales: { y: { min: 0, max: 100, border: { display: false }, ticks: { color: "#86868b" }, grid: { color: "rgba(0,0,0,0.05)" } }, x: { border: { display: false }, ticks: { color: "#86868b", font: { size: 10 } }, grid: { color: "rgba(0,0,0,0.05)" } } } }),
      });
    } else if (signature !== completionSignature) {
      updateChartData(chartCompletion, labels, [data]);
    }
    completionSignature = signature;
  }

  function renderActivityChart(points) {
    const canvas = document.getElementById("chartActivity");
    const labels = points.map((p) => `${(p.elapsed_minutes ?? 0).toFixed ? p.elapsed_minutes.toFixed(1) : p.elapsed_minutes}min`);
    const data = points.map((p) => p.activity_count ?? 0);

    const signature = dataSignature(labels, [data]);
    if (!chartActivity) {
      chartActivity = new Chart(canvas, {
        type: "line",
        data: {
          labels,
          datasets: [
            { label: "Observed activities", data, borderColor: CHART_COLORS.activity, backgroundColor: "rgba(191,90,242,0.12)", fill: true, stepped: true, pointRadius: 1 },
          ],
        },
        options: baseOptions(),
      });
    } else if (signature !== activitySignature) {
      updateChartData(chartActivity, labels, [data]);
    }
    activitySignature = signature;
  }

  function renderTable(iterations) {
    const tbody = document.querySelector("#iterTable tbody");
    tbody.innerHTML = iterations
      .slice()
      .reverse()
      .map((it) => {
        const rate = it.completion_rate != null ? `${Math.round(it.completion_rate * 1000) / 10}%` : "-";
        return `<tr>
          <td>${it.iteration ?? "-"}</td>
          <td>${it.outcome ?? "-"}</td>
          <td>${it.scenario_type ?? "-"}</td>
          <td title="${it.scope ?? ""}">${truncate(it.scope, 18)}</td>
          <td>${it.page ?? "-"}</td>
          <td>${it.elapsed_seconds ?? "-"}</td>
          <td>${it.llm_requests ?? "-"}</td>
          <td>${it.llm_tokens ?? "-"}</td>
          <td>${it.functions ?? "-"}</td>
          <td>${it.pages ?? "-"}</td>
          <td>${it.completed_functions ?? "-"}</td>
          <td>${rate}</td>
          <td>${it.scenario_status ?? "-"}</td>
        </tr>`;
      })
      .join("");
  }

  function truncate(text, n) {
    if (text == null) return "-";
    text = String(text);
    return text.length > n ? text.slice(0, n) + "…" : text;
  }

  function render(metricsData, timelineData) {
    lastMetrics = metricsData;
    lastTimeline = timelineData;
    const iterations = (metricsData && metricsData.iterations) || [];
    const nextTableSignature = JSON.stringify(iterations);
    if (nextTableSignature !== tableSignature) {
      renderTable(iterations);
      tableSignature = nextTableSignature;
    }
    if (!panelVisible()) return; // Update tables while hidden; create charts when the tab becomes visible.
    renderAllCharts();
  }

  function resize() {
    if (lastMetrics || lastTimeline) {
      // Hidden panels can receive updates; unchanged signatures avoid a chart redraw.
      renderAllCharts();
    }
    if (chartGrowth && chartCompletion && chartActivity) {
      // Resize existing charts when visible instead of destroying them.
      chartGrowth.resize();
      chartCompletion.resize();
      chartActivity.resize();
    }
  }

  return { render, resize };
})();
