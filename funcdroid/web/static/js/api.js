/* API client for web/server.py */
const Api = (() => {
  async function getJson(url) {
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`RequestsFailed ${resp.status}: ${text}`);
    }
    return resp.json();
  }

  return {
    listRuns: () => getJson("/api/runs"),
    summary: (run) => getJson(`/api/runs/${encodeURIComponent(run)}/summary`),
    ffg: (run) => getJson(`/api/runs/${encodeURIComponent(run)}/ffg`),
    bugs: (run) => getJson(`/api/runs/${encodeURIComponent(run)}/bugs`),
    bugDetail: (run, bugId) => getJson(`/api/runs/${encodeURIComponent(run)}/bugs/${encodeURIComponent(bugId)}`),
    compliance: (run) => getJson(`/api/runs/${encodeURIComponent(run)}/compliance`),
    complianceTrace: (run, relPath) => getJson(`/api/runs/${encodeURIComponent(run)}/compliance/trace?path=${encodeURIComponent(relPath)}`),
    timeline: (run) => getJson(`/api/runs/${encodeURIComponent(run)}/timeline`),
    metrics: (run) => getJson(`/api/runs/${encodeURIComponent(run)}/metrics`),
    live: (run) => getJson(`/api/runs/${encodeURIComponent(run)}/live`),
    log: (run, since) => getJson(`/api/runs/${encodeURIComponent(run)}/log?since=${since || 0}&limit=800`),
    fileUrl: (run, relPath) => `/api/runs/${encodeURIComponent(run)}/file?path=${encodeURIComponent(relPath)}`,
  };
})();
