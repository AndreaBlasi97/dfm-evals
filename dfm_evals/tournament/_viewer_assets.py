from textwrap import dedent

VIEWER_HTML = dedent(
    """\
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Tournament Viewer</title>
        <link rel="stylesheet" href="/app.css">
      </head>
      <body>
        <div class="background-grid"></div>
        <div class="background-glow background-glow-a"></div>
        <div class="background-glow background-glow-b"></div>
        <div class="app-shell">
          <header class="hero">
            <div class="hero-copy">
              <p class="eyebrow">Tournament Viewer</p>
              <div class="hero-heading-row">
                <h1 id="project-title">Tournament Viewer</h1>
                <span id="status-pill" class="status-pill">Connecting</span>
              </div>
              <p id="hero-subtitle" class="hero-subtitle">
                Ratings, head-to-heads, and prompt evidence.
              </p>
              <div class="hero-meta">
                <span id="run-label" class="meta-pill">Latest run</span>
                <span id="run-context" class="meta-pill meta-pill-muted"></span>
              </div>
            </div>
            <div class="hero-actions">
              <button id="refresh-button" class="primary-button" type="button">
                Refresh
              </button>
              <p id="refresh-meta" class="hero-note">Auto refresh every 10s</p>
            </div>
          </header>

          <section id="summary-grid" class="summary-grid"></section>

          <nav class="tab-bar" aria-label="Tournament sections">
            <button class="tab-button is-active" data-tab="standings" type="button">
              Standings
            </button>
            <button class="tab-button" data-tab="pairwise" type="button">
              Head-to-Head
            </button>
            <button class="tab-button" data-tab="matches" type="button">
              Matches
            </button>
            <button class="tab-button" data-tab="prompts" type="button">
              Prompts
            </button>
            <button class="tab-button" data-tab="models" type="button">
              Models
            </button>
          </nav>

          <main class="tab-panels">
            <section id="panel-standings" class="tab-panel is-active">
              <div class="section-header">
                <h2>Standings</h2>
                <p>Rank by conservative score.</p>
              </div>
              <div id="standings-table" class="panel-card panel-card-scroll"></div>
            </section>

            <section id="panel-pairwise" class="tab-panel">
              <div class="section-header">
                <h2>Head-to-Head</h2>
                <p>Score plus record. Click a cell to filter matches.</p>
              </div>
              <div id="pairwise-table" class="panel-card panel-card-scroll"></div>
            </section>

            <section id="panel-matches" class="tab-panel">
              <div class="section-header">
                <h2>Matches</h2>
                <p>Filter rows, then open a match for full responses.</p>
              </div>
              <div class="panel-card">
                <form id="matches-filters" class="filter-grid">
                  <label>
                    <span>Model</span>
                    <select id="filter-model">
                      <option value="">Any</option>
                    </select>
                  </label>
                  <label>
                    <span>Opponent</span>
                    <select id="filter-opponent">
                      <option value="">Any</option>
                    </select>
                  </label>
                  <label>
                    <span>Category</span>
                    <select id="filter-category">
                      <option value="">Any</option>
                    </select>
                  </label>
                  <label>
                    <span>Outcome</span>
                    <select id="filter-outcome">
                      <option value="">Any</option>
                      <option value="decisive">Decisive</option>
                      <option value="tie">Tie</option>
                      <option value="invalid">Invalid</option>
                    </select>
                  </label>
                  <label>
                    <span>Batch</span>
                    <input id="filter-batch" type="text" placeholder="batch-000001">
                  </label>
                  <div class="filter-actions">
                    <button id="clear-filters" class="secondary-button" type="button">
                      Clear
                    </button>
                    <button class="primary-button" type="submit">Apply</button>
                  </div>
                </form>
              </div>
              <div id="matches-meta" class="panel-meta"></div>
              <div id="matches-table" class="panel-card panel-card-scroll"></div>
            </section>

            <section id="panel-prompts" class="tab-panel">
              <div class="section-header">
                <h2>Prompts</h2>
                <p>Coverage and prompt-level drilldown.</p>
              </div>
              <div id="prompts-grid" class="card-grid"></div>
            </section>

            <section id="panel-models" class="tab-panel">
              <div class="section-header">
                <h2>Models</h2>
                <p>Ratings, coverage, and prompt responses.</p>
              </div>
              <div id="models-grid" class="card-grid"></div>
            </section>
          </main>
        </div>

        <aside id="detail-drawer" class="detail-drawer" aria-hidden="true">
          <div class="detail-backdrop" data-close-drawer="1"></div>
          <div class="detail-panel">
            <button id="drawer-close" class="drawer-close" type="button" aria-label="Close">
              Close
            </button>
            <div id="drawer-content"></div>
          </div>
        </aside>

        <script src="/app.js"></script>
      </body>
    </html>
    """
)

VIEWER_CSS = dedent(
    """\
    :root {
      --bg: #f3f5f8;
      --surface: #ffffff;
      --surface-strong: #ffffff;
      --surface-muted: #f7f9fc;
      --ink: #142235;
      --ink-soft: #36485d;
      --muted: #6a7b8d;
      --line: rgba(20, 34, 53, 0.1);
      --accent: #1763d1;
      --accent-strong: #114ea8;
      --accent-soft: rgba(23, 99, 209, 0.08);
      --success: #0d8f63;
      --danger: #c34b4b;
      --shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
      --shadow-soft: 0 2px 10px rgba(15, 23, 42, 0.035);
      --radius: 14px;
      --radius-small: 10px;
      --font-display: "Segoe UI Variable", "Segoe UI", "Helvetica Neue", sans-serif;
      --font-ui: "Segoe UI Variable", "Segoe UI", "Helvetica Neue", sans-serif;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: var(--font-ui);
      background: var(--bg);
    }

    body.modal-open {
      overflow: hidden;
    }

    .background-grid {
      display: none;
    }

    .background-glow {
      display: none;
    }

    .background-glow-a {
      display: none;
    }

    .background-glow-b {
      display: none;
    }

    .app-shell {
      position: relative;
      z-index: 1;
      max-width: 1460px;
      margin: 0 auto;
      padding: 10px 12px 18px;
    }

    .hero {
      position: relative;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--surface-strong);
      box-shadow: var(--shadow-soft);
    }

    .hero-copy,
    .hero-actions {
      position: relative;
      z-index: 1;
    }

    .hero-copy {
      display: grid;
      gap: 2px;
      min-width: 0;
    }

    .eyebrow {
      display: none;
    }

    h1,
    h2,
    h3 {
      margin: 0;
      font-family: var(--font-display);
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    h1 {
      font-size: clamp(1.28rem, 2vw, 1.7rem);
      line-height: 1;
      font-weight: 700;
    }

    h2 {
      font-size: 1.05rem;
      font-weight: 700;
    }

    .hero-subtitle,
    .section-header p,
    .card-meta,
    .muted {
      color: var(--muted);
    }

    .hero-heading-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }

    .hero-subtitle {
      max-width: 980px;
      margin: 1px 0 0;
      font-size: 0.76rem;
      line-height: 1.24;
    }

    .hero-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 3px;
    }

    .meta-pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      max-width: 100%;
      border-radius: 999px;
      padding: 0.12rem 0.42rem;
      border: 1px solid rgba(23, 99, 209, 0.14);
      background: var(--accent-soft);
      color: var(--accent-strong);
      font-size: 0.69rem;
      font-weight: 600;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .meta-pill-muted {
      color: var(--muted);
      border-color: var(--line);
      background: var(--surface-muted);
      font-weight: 600;
    }

    .hero-actions {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      justify-content: center;
      gap: 6px;
      min-width: auto;
    }

    .hero-note {
      margin: 0;
      color: var(--muted);
      font-size: 0.68rem;
      font-weight: 600;
      white-space: nowrap;
    }

    .primary-button,
    .secondary-button,
    .drawer-close,
    .tab-button,
    .matrix-cell {
      font: inherit;
      cursor: pointer;
      transition:
        transform 120ms ease,
        box-shadow 120ms ease,
        background-color 120ms ease,
        border-color 120ms ease,
        color 120ms ease;
    }

    .primary-button,
    .secondary-button,
    .drawer-close {
      border-radius: 999px;
      padding: 0.38rem 0.66rem;
      border: 1px solid transparent;
      font-size: 0.78rem;
      font-weight: 600;
    }

    .primary-button {
      color: #f8fbff;
      background: var(--accent);
      box-shadow: none;
    }

    .secondary-button,
    .drawer-close {
      color: var(--ink);
      background: var(--surface);
      border-color: var(--line);
    }

    .primary-button:hover,
    .secondary-button:hover,
    .drawer-close:hover,
    .tab-button:hover,
    .matrix-cell:hover {
      transform: translateY(-1px);
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      min-height: 22px;
      padding: 0.14rem 0.42rem;
      border-radius: 999px;
      background: var(--surface-muted);
      border: 1px solid var(--line);
      color: var(--ink-soft);
      font-size: 0.69rem;
      font-weight: 600;
    }

    .status-pill.is-ok {
      color: var(--success);
    }

    .status-pill.is-error {
      color: var(--danger);
      background: rgba(255, 239, 239, 0.94);
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 6px;
      margin: 6px 0 8px;
    }

    .summary-card,
    .panel-card,
    .info-card {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--surface);
      box-shadow: var(--shadow-soft);
    }

    .summary-card {
      padding: 8px 9px 7px;
    }

    .summary-card .label {
      font-size: 0.62rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 3px;
    }

    .summary-card .value {
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: -0.01em;
    }

    .summary-card .subvalue {
      margin-top: 2px;
      font-size: 0.68rem;
      color: var(--muted);
      line-height: 1.2;
    }

    .tab-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 6px;
      padding: 0;
      background: transparent;
      border: 0;
      box-shadow: none;
      width: auto;
      max-width: 100%;
    }

    .tab-button {
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink-soft);
      padding: 0.34rem 0.58rem;
      border-radius: 999px;
      font-size: 0.77rem;
      font-weight: 600;
    }

    .tab-button.is-active {
      color: var(--accent-strong);
      background: var(--accent-soft);
      border-color: rgba(23, 99, 209, 0.22);
      box-shadow: none;
    }

    .tab-panel {
      display: none;
      animation: panel-in 180ms ease-out;
    }

    .tab-panel.is-active {
      display: block;
    }

    @keyframes panel-in {
      from {
        opacity: 0;
        transform: translateY(6px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 8px;
      margin: 0 0 4px;
    }

    .section-header p {
      margin: 0;
      max-width: none;
      line-height: 1.22;
      font-size: 0.74rem;
    }

    .panel-card {
      padding: 8px;
      overflow: hidden;
    }

    .panel-card-scroll {
      overflow: auto;
    }

    .panel-meta {
      margin: 4px 0 6px;
      color: var(--muted);
      font-size: 0.74rem;
    }

    .filter-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 6px;
      align-items: end;
    }

    .filter-grid label {
      display: flex;
      flex-direction: column;
      gap: 3px;
      font-size: 0.74rem;
      font-weight: 600;
    }

    .filter-grid select,
    .filter-grid input {
      width: 100%;
      border-radius: var(--radius-small);
      border: 1px solid var(--line);
      background: var(--surface);
      padding: 0.4rem 0.5rem;
      font: inherit;
      color: var(--ink);
      font-size: 0.78rem;
    }

    .filter-actions {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      align-items: center;
    }

    .table-wrap {
      overflow: auto;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 620px;
    }

    th,
    td {
      padding: 0.42rem 0.44rem;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 0.78rem;
    }

    th {
      position: sticky;
      top: 0;
      z-index: 1;
      font-size: 0.63rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      background: var(--surface-muted);
    }

    td.numeric,
    th.numeric {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    tbody tr {
      transition: background-color 120ms ease;
    }

    tbody tr:hover {
      background: rgba(24, 102, 224, 0.04);
    }

    .row-button,
    .plain-link {
      color: inherit;
      text-decoration: none;
      background: none;
      border: 0;
      padding: 0;
      font: inherit;
      cursor: pointer;
      text-align: left;
    }

    .row-button strong,
    .plain-link strong {
      color: var(--accent-strong);
    }

    .badge-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--surface-muted);
      padding: 0.1rem 0.34rem;
      font-size: 0.66rem;
      color: var(--muted);
    }

    .badge.is-strong {
      color: var(--accent-strong);
      border-color: rgba(24, 102, 224, 0.18);
      background: var(--accent-soft);
    }

    .card-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 6px;
    }

    .info-card {
      padding: 8px 9px;
      transition: border-color 140ms ease;
    }

    .info-card:hover {
      border-color: rgba(24, 102, 224, 0.14);
    }

    .info-card h3 {
      font-size: 0.88rem;
      margin-bottom: 3px;
    }

    .card-kicker {
      margin: 0 0 2px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.58rem;
      color: var(--muted);
    }

    .card-body {
      margin: 4px 0 0;
      color: var(--ink);
      line-height: 1.24;
      font-size: 0.76rem;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .metric-line {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-top: 3px;
      font-size: 0.74rem;
    }

    .metric-line span:last-child {
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }

    .matrix {
      min-width: 760px;
    }

    .matrix th:first-child,
    .matrix td:first-child {
      position: sticky;
      left: 0;
      z-index: 2;
      background: rgba(246, 249, 253, 0.99);
    }

    .matrix-cell {
      min-width: 84px;
      width: 84px;
      border-radius: 8px;
      border: 1px solid rgba(16, 32, 51, 0.08);
      padding: 5px;
      background: var(--surface);
      text-align: left;
    }

    .matrix-cell.is-empty {
      cursor: default;
      background: rgba(255, 255, 255, 0.52);
      color: var(--muted);
    }

    .matrix-score {
      display: block;
      font-size: 0.78rem;
      font-weight: 700;
    }

    .matrix-record {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 0.61rem;
    }

    .detail-drawer {
      position: fixed;
      inset: 0;
      display: none;
      z-index: 12;
      align-items: center;
      justify-content: center;
      padding: 8px;
    }

    .detail-drawer.is-open {
      display: flex;
    }

    .detail-backdrop {
      position: absolute;
      inset: 0;
      background: rgba(11, 20, 32, 0.34);
      backdrop-filter: blur(4px);
    }

    .detail-panel {
      position: relative;
      z-index: 1;
      width: min(1240px, calc(100vw - 12px));
      height: min(95vh, calc(100vh - 12px));
      padding: 10px 12px 14px;
      overflow: auto;
      background: var(--surface-strong);
      box-shadow: var(--shadow);
      border: 1px solid var(--line);
      border-radius: 14px;
    }

    .drawer-close {
      position: sticky;
      top: 0;
      z-index: 2;
      margin-left: auto;
      display: block;
      margin-bottom: 6px;
      background: var(--surface);
    }

    .drawer-section {
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
    }

    .drawer-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 6px;
      margin-top: 6px;
    }

    .drawer-card {
      padding: 8px;
      border-radius: 10px;
      border: 1px solid rgba(16, 32, 51, 0.08);
      background: var(--surface-muted);
    }

    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 0.76rem;
      line-height: 1.32;
    }

    details {
      margin-top: 5px;
      border-radius: 10px;
      border: 1px solid rgba(16, 32, 51, 0.08);
      background: var(--surface-muted);
      overflow: hidden;
    }

    summary {
      cursor: pointer;
      padding: 6px 8px;
      font-size: 0.78rem;
      font-weight: 600;
    }

    details > div {
      padding: 0 8px 8px;
    }

    .empty-state {
      padding: 8px;
      color: var(--muted);
      text-align: center;
      font-size: 0.76rem;
    }

    @media (max-width: 900px) {
      .hero {
        flex-direction: column;
        align-items: flex-start;
      }

      .hero-actions {
        align-items: flex-start;
      }

      .hero-heading-row {
        align-items: flex-start;
      }

      .section-header {
        flex-direction: column;
        align-items: flex-start;
      }

      .detail-drawer {
        padding: 6px;
      }

      .detail-panel {
        width: min(100vw - 8px, 1280px);
        height: min(100vh - 8px, 1000px);
        padding: 10px 10px 14px;
      }

      .drawer-grid {
        grid-template-columns: 1fr;
      }
    }
    """
)

VIEWER_JS = dedent(
    """\
    const state = {
      summary: null,
      standings: [],
      pairwise: null,
      prompts: [],
      models: [],
      matches: { items: [], total: 0, limit: 100, offset: 0 },
      detail: { kind: "", id: "" },
      detailRequestId: 0,
      refreshPromise: null,
      lastRefreshAt: null,
      autoRefreshMs: 10000,
      filters: {
        model: "",
        opponent: "",
        category: "",
        outcome: "",
        batch_id: "",
      },
    };

    const staticData = window.__TOURNAMENT_STATIC_DATA__ || null;
    if (staticData) {
      state.autoRefreshMs = 0;
    }

    function el(id) {
      return document.getElementById(id);
    }

    function isStaticMode() {
      return staticData !== null;
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function truncate(value, length = 140) {
      const text = String(value ?? "").trim();
      if (text.length <= length) {
        return text;
      }
      return `${text.slice(0, Math.max(0, length - 1)).trimEnd()}...`;
    }

    function formatNumber(value, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
      }
      return Number(value).toFixed(digits);
    }

    function formatPercent(value, digits = 1) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
      }
      return `${(Number(value) * 100).toFixed(digits)}%`;
    }

    function promptLabel(prompt) {
      return prompt.title || prompt.prompt_id;
    }

    function matchOutcomeLabel(match) {
      if (match.canonical_decision === "TIE") {
        return "Tie";
      }
      if (match.canonical_decision === "INVALID") {
        return "Invalid";
      }
      return match.winner_model_name || "Decisive";
    }

    async function fetchJson(path) {
      if (isStaticMode()) {
        return resolveStaticJson(path);
      }

      const url = new URL(path, window.location.href);
      url.searchParams.set("_ts", String(Date.now()));
      const response = await fetch(url, {
        headers: {
          Accept: "application/json",
        },
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      return response.json();
    }

    function parsePositiveInt(value, defaultValue) {
      if (value === null || value === undefined || String(value).trim() === "") {
        return defaultValue;
      }
      const parsed = Number.parseInt(String(value), 10);
      return Number.isFinite(parsed) && parsed > 0 ? parsed : defaultValue;
    }

    function parseNonNegativeInt(value, defaultValue) {
      if (value === null || value === undefined || String(value).trim() === "") {
        return defaultValue;
      }
      const parsed = Number.parseInt(String(value), 10);
      return Number.isFinite(parsed) && parsed >= 0 ? parsed : defaultValue;
    }

    function staticMatchFilter(record, filters) {
      const modelFilter = (filters.model || "").trim();
      if (
        modelFilter &&
        ![
          record.model_a_id,
          record.model_b_id,
          record.model_a_name,
          record.model_b_name,
        ].includes(modelFilter)
      ) {
        return false;
      }

      const opponentFilter = (filters.opponent || "").trim();
      if (
        opponentFilter &&
        ![
          record.model_a_id,
          record.model_b_id,
          record.model_a_name,
          record.model_b_name,
        ].includes(opponentFilter)
      ) {
        return false;
      }

      const promptFilter = (filters.prompt_id || "").trim();
      if (promptFilter && promptFilter !== record.prompt_id) {
        return false;
      }

      const categoryFilter = (filters.category || "").trim();
      if (categoryFilter && categoryFilter !== (record.category || "")) {
        return false;
      }

      const winnerFilter = (filters.winner || "").trim();
      if (winnerFilter) {
        if (["TIE", "INVALID"].includes(winnerFilter)) {
          if (record.canonical_decision !== winnerFilter) {
            return false;
          }
        } else if (![record.winner_model_id, record.winner_model_name].includes(winnerFilter)) {
          return false;
        }
      }

      const outcomeFilter = (filters.outcome || "").trim();
      if (outcomeFilter === "decisive" && !["A", "B"].includes(record.canonical_decision)) {
        return false;
      }
      if (outcomeFilter === "tie" && record.canonical_decision !== "TIE") {
        return false;
      }
      if (outcomeFilter === "invalid" && record.canonical_decision !== "INVALID") {
        return false;
      }

      const batchFilter = (filters.batch_id || "").trim();
      if (batchFilter && batchFilter !== record.batch_id) {
        return false;
      }

      return true;
    }

    function staticMatchesResponse(searchParams) {
      const filters = Object.fromEntries(searchParams.entries());
      const limit = parsePositiveInt(searchParams.get("limit"), 100);
      const offset = parseNonNegativeInt(searchParams.get("offset"), 0);
      const allMatches = staticData.matches?.items || [];
      const items = allMatches.filter((record) => staticMatchFilter(record, filters));
      return {
        items: items.slice(offset, offset + limit),
        total: items.length,
        limit,
        offset,
      };
    }

    function requireStaticId(searchParams, key) {
      const value = (searchParams.get(key) || "").trim();
      if (!value) {
        throw new Error(`Missing required query parameter: ${key}`);
      }
      return value;
    }

    function parseStaticRequest(path) {
      const text = String(path || "");
      const questionIndex = text.indexOf("?");
      if (questionIndex === -1) {
        return {
          pathname: text || "/",
          searchParams: new URLSearchParams(),
        };
      }
      return {
        pathname: text.slice(0, questionIndex) || "/",
        searchParams: new URLSearchParams(text.slice(questionIndex + 1)),
      };
    }

    function resolveStaticJson(path) {
      const request = parseStaticRequest(path);
      if (request.pathname === "/api/summary") {
        return staticData.summary;
      }
      if (request.pathname === "/api/standings") {
        return staticData.standings;
      }
      if (request.pathname === "/api/pairwise") {
        return staticData.pairwise;
      }
      if (request.pathname === "/api/prompts") {
        return staticData.prompts;
      }
      if (request.pathname === "/api/models") {
        return staticData.models;
      }
      if (request.pathname === "/api/matches") {
        return staticMatchesResponse(request.searchParams);
      }
      if (request.pathname === "/api/match") {
        const matchId = requireStaticId(request.searchParams, "match_id");
        const match = staticData.match_details?.[matchId];
        if (!match) {
          throw new Error(`Unknown match id: ${matchId}`);
        }
        return match;
      }
      if (request.pathname === "/api/prompt") {
        const promptId = requireStaticId(request.searchParams, "prompt_id");
        const prompt = staticData.prompt_details?.[promptId];
        if (!prompt) {
          throw new Error(`Unknown prompt id: ${promptId}`);
        }
        return prompt;
      }
      if (request.pathname === "/api/model") {
        const modelId = requireStaticId(request.searchParams, "model_id");
        const model = staticData.model_details?.[modelId];
        if (!model) {
          throw new Error(`Unknown model id: ${modelId}`);
        }
        return model;
      }
      throw new Error(`Unknown route: ${request.pathname}`);
    }

    function setStatus(message, kind = "neutral") {
      const pill = el("status-pill");
      pill.textContent = message;
      pill.classList.remove("is-ok", "is-error");
      if (kind === "ok") {
        pill.classList.add("is-ok");
      } else if (kind === "error") {
        pill.classList.add("is-error");
      }
    }

    function setRefreshButtonBusy(isBusy) {
      const button = el("refresh-button");
      button.disabled = isBusy;
      button.textContent = isBusy ? "Refreshing..." : "Refresh";
    }

    function isAutoRefreshPaused() {
      return document.body.classList.contains("modal-open");
    }

    function updateRefreshMeta() {
      const meta = el("refresh-meta");
      if (isStaticMode()) {
        meta.textContent = "Static export \u00b7 no live refresh";
        return;
      }
      if (!state.lastRefreshAt) {
        meta.textContent = isAutoRefreshPaused()
          ? "Auto refresh paused while detail view is open"
          : `Auto refresh every ${Math.round(state.autoRefreshMs / 1000)}s`;
        return;
      }
      const lastFetch = state.lastRefreshAt.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
      meta.textContent = isAutoRefreshPaused()
        ? `Last fetch ${lastFetch} · Auto refresh paused`
        : `Last fetch ${lastFetch} · Auto ${Math.round(state.autoRefreshMs / 1000)}s`;
    }

    function setActiveTab(name) {
      document
        .querySelectorAll(".tab-button")
        .forEach((button) => button.classList.toggle("is-active", button.dataset.tab === name));
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("is-active", panel.id === `panel-${name}`);
      });
    }

    function buildQuery(filters) {
      const params = new URLSearchParams();
      for (const [key, value] of Object.entries(filters)) {
        if (value) {
          params.set(key, value);
        }
      }
      params.set("limit", "100");
      return params.toString();
    }

    async function loadMatches() {
      const query = buildQuery(state.filters);
      state.matches = await fetchJson(`/api/matches?${query}`);
      renderMatches();
    }

    async function refreshOpenDetail() {
      if (state.detail.kind === "" || state.detail.id === "") {
        return;
      }
      if (!document.body.classList.contains("modal-open")) {
        return;
      }
      if (state.detail.kind === "match") {
        await openMatch(state.detail.id, { preserveSelection: true });
        return;
      }
      if (state.detail.kind === "prompt") {
        await openPrompt(state.detail.id, { preserveSelection: true });
        return;
      }
      if (state.detail.kind === "model") {
        await openModel(state.detail.id, { preserveSelection: true });
      }
    }

    async function refreshAll({ silent = false } = {}) {
      if (state.refreshPromise) {
        return state.refreshPromise;
      }

      state.refreshPromise = (async () => {
        if (!silent) {
          setStatus("Refreshing", "neutral");
        }
        setRefreshButtonBusy(true);

        const [summary, standings, pairwise, prompts, models] = await Promise.all([
          fetchJson("/api/summary"),
          fetchJson("/api/standings"),
          fetchJson("/api/pairwise"),
          fetchJson("/api/prompts"),
          fetchJson("/api/models"),
        ]);

        state.summary = summary;
        state.standings = standings.items;
        state.pairwise = pairwise;
        state.prompts = prompts.items;
        state.models = models.items;
        state.lastRefreshAt = new Date();

        renderSummary();
        renderStandings();
        renderPairwise();
        renderPrompts();
        renderModels();
        populateFilterControls();
        await loadMatches();
        await refreshOpenDetail();

        updateRefreshMeta();
        setStatus(summary.run_status || "ready", "ok");
        document.title = `${summary.project_id || "Tournament"} · Tournament Viewer`;
        el("project-title").textContent = summary.project_id || "Tournament Viewer";
        el("hero-subtitle").textContent =
          `${summary.total_models} models · ${summary.total_prompts} prompts · judge ${truncate(summary.judge_model || "-", 56)}`;
        el("run-label").textContent = summary.run_label || "Tournament run";
        el("run-label").title = summary.run_label || "";
        el("run-context").textContent = summary.last_updated_at
          ? `Updated ${summary.last_updated_at}`
          : "State loaded";
        el("run-context").title = summary.run_dir || summary.state_dir || "";
      })();

      try {
        await state.refreshPromise;
      } finally {
        state.refreshPromise = null;
        setRefreshButtonBusy(false);
      }
    }

    function renderSummary() {
      if (!state.summary) {
        return;
      }

      const summary = state.summary;
      const items = [
        {
          label: "Run status",
          value: summary.run_status,
          subvalue:
            summary.stop_reasons && summary.stop_reasons.length > 0
              ? summary.stop_reasons.join(", ")
              : "no explicit stop reason",
        },
        {
          label: "Coverage",
          value: formatPercent(summary.response_coverage, 0),
          subvalue: `${summary.response_count}/${summary.expected_responses} responses`,
        },
        {
          label: "Rated matches",
          value: String(summary.rated_matches),
          subvalue: `${summary.total_matches} total, ${summary.batch_count} batches`,
        },
        {
          label: "Models",
          value: String(summary.total_models),
          subvalue: `${summary.total_prompts} prompts`,
        },
        {
          label: "Judge",
          value: summary.judge_model,
          subvalue: `${summary.judgment_count} side judgments`,
        },
        {
          label: "Last update",
          value: summary.last_updated_at || "-",
          subvalue: summary.run_label || summary.project_id || "",
        },
      ];

      el("summary-grid").innerHTML = items
        .map(
          (item) => `
            <article class="summary-card">
              <div class="label">${escapeHtml(item.label)}</div>
              <div class="value">${escapeHtml(item.value)}</div>
              <div class="subvalue">${escapeHtml(item.subvalue)}</div>
            </article>
          `
        )
        .join("");
    }

    function renderStandings() {
      const container = el("standings-table");
      if (!state.standings.length) {
        container.innerHTML = '<div class="empty-state">No standings yet.</div>';
        return;
      }

      container.innerHTML = `
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th class="numeric">Rank</th>
                <th>Model</th>
                <th class="numeric">Conservative</th>
                <th class="numeric">Elo-like</th>
                <th class="numeric">Mu</th>
                <th class="numeric">Sigma</th>
                <th class="numeric">Games</th>
                <th class="numeric">W</th>
                <th class="numeric">L</th>
                <th class="numeric">T</th>
              </tr>
            </thead>
            <tbody>
              ${state.standings
                .map(
                  (item) => `
                    <tr>
                      <td class="numeric">${item.rank}</td>
                      <td>
                        <button class="row-button" data-open-model="${escapeHtml(item.model_id)}" type="button">
                          <strong>${escapeHtml(item.model_name)}</strong>
                        </button>
                      </td>
                      <td class="numeric">${formatNumber(item.conservative, 2)}</td>
                      <td class="numeric">${formatNumber(item.elo_like, 1)}</td>
                      <td class="numeric">${formatNumber(item.mu, 2)}</td>
                      <td class="numeric">${formatNumber(item.sigma, 2)}</td>
                      <td class="numeric">${item.games}</td>
                      <td class="numeric">${item.wins}</td>
                      <td class="numeric">${item.losses}</td>
                      <td class="numeric">${item.ties}</td>
                    </tr>
                  `
                )
                .join("")}
            </tbody>
          </table>
        </div>
      `;
    }

    function renderPairwise() {
      const container = el("pairwise-table");
      if (!state.pairwise || !state.pairwise.models.length) {
        container.innerHTML = '<div class="empty-state">No pairwise results yet.</div>';
        return;
      }

      const headerCells = state.pairwise.models
        .map((model) => `<th>${escapeHtml(model.model_name)}</th>`)
        .join("");
      const bodyRows = state.pairwise.rows
        .map((row) => {
          const cells = row.cells
            .map((cell) => {
              if (!cell) {
                return '<td><div class="matrix-cell is-empty">-</div></td>';
              }
              if (!cell.has_data) {
                return '<td><div class="matrix-cell is-empty">No matches</div></td>';
              }
              return `
                <td>
                  <button
                    class="matrix-cell"
                    type="button"
                    data-pair-model="${escapeHtml(row.model_id)}"
                    data-pair-opponent="${escapeHtml(cell.opponent_model_id)}"
                  >
                    <span class="matrix-score">${formatPercent(cell.score, 0)}</span>
                    <span class="matrix-record">${cell.wins}-${cell.losses}-${cell.ties} (${cell.rated_games})</span>
                  </button>
                </td>
              `;
            })
            .join("");
          return `
            <tr>
              <th>${escapeHtml(row.model_name)}</th>
              ${cells}
            </tr>
          `;
        })
        .join("");

      container.innerHTML = `
        <div class="table-wrap">
          <table class="matrix">
            <thead>
              <tr>
                <th>Model</th>
                ${headerCells}
              </tr>
            </thead>
            <tbody>${bodyRows}</tbody>
          </table>
        </div>
      `;
    }

    function populateFilterControls() {
      const modelOptions = state.models
        .map(
          (model) => `
            <option value="${escapeHtml(model.model_id)}">${escapeHtml(model.model_name)}</option>
          `
        )
        .join("");
      const categories = [...new Set(state.prompts.map((prompt) => prompt.category).filter(Boolean))].sort();
      const categoryOptions = categories
        .map(
          (category) => `
            <option value="${escapeHtml(category)}">${escapeHtml(category)}</option>
          `
        )
        .join("");

      el("filter-model").innerHTML = '<option value="">Any</option>' + modelOptions;
      el("filter-opponent").innerHTML = '<option value="">Any</option>' + modelOptions;
      el("filter-category").innerHTML = '<option value="">Any</option>' + categoryOptions;

      el("filter-model").value = state.filters.model;
      el("filter-opponent").value = state.filters.opponent;
      el("filter-category").value = state.filters.category;
      el("filter-outcome").value = state.filters.outcome;
      el("filter-batch").value = state.filters.batch_id;
    }

    function renderMatches() {
      const container = el("matches-table");
      const meta = el("matches-meta");
      meta.textContent = `${state.matches.total} matches shown across current filters.`;

      if (!state.matches.items.length) {
        container.innerHTML = '<div class="empty-state">No matches match the current filters.</div>';
        return;
      }

      container.innerHTML = `
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Match</th>
                <th>Prompt</th>
                <th>Pair</th>
                <th>Outcome</th>
                <th>Judge note</th>
              </tr>
            </thead>
            <tbody>
              ${state.matches.items
                .map(
                  (match) => `
                    <tr>
                      <td>
                        <button class="row-button" data-open-match="${escapeHtml(match.match_id)}" type="button">
                          <strong>${escapeHtml(match.batch_id)}</strong><br>
                          <span class="muted">round ${match.round_index}</span>
                        </button>
                      </td>
                      <td>
                        <button class="row-button" data-open-prompt="${escapeHtml(match.prompt_id)}" type="button">
                          <strong>${escapeHtml(match.prompt_title || match.prompt_id)}</strong><br>
                          <span class="muted">${escapeHtml(match.category || "uncategorized")}</span>
                        </button>
                      </td>
                      <td>
                        <button class="row-button" data-open-model="${escapeHtml(match.model_a_id)}" type="button">
                          ${escapeHtml(match.model_a_name)}
                        </button>
                        <span class="muted"> vs </span>
                        <button class="row-button" data-open-model="${escapeHtml(match.model_b_id)}" type="button">
                          ${escapeHtml(match.model_b_name)}
                        </button>
                      </td>
                      <td>
                        <div>${escapeHtml(matchOutcomeLabel(match))}</div>
                        <div class="muted">${escapeHtml(match.canonical_decision)}</div>
                      </td>
                      <td>${escapeHtml(match.explanation_preview || "-")}</td>
                    </tr>
                  `
                )
                .join("")}
            </tbody>
          </table>
        </div>
      `;
    }

    function renderPrompts() {
      const container = el("prompts-grid");
      if (!state.prompts.length) {
        container.innerHTML = '<div class="empty-state">No prompts found.</div>';
        return;
      }

      container.innerHTML = state.prompts
        .map(
          (prompt) => `
            <article class="info-card">
              <p class="card-kicker">${escapeHtml(prompt.category || "Prompt")}</p>
              <h3>
                <button class="row-button" data-open-prompt="${escapeHtml(prompt.prompt_id)}" type="button">
                  <strong>${escapeHtml(promptLabel(prompt))}</strong>
                </button>
              </h3>
              <p class="card-meta">${escapeHtml(prompt.source || prompt.prompt_id)}</p>
              <p class="card-body">${escapeHtml(truncate(prompt.prompt_text, 220))}</p>
              <div class="metric-line"><span>Coverage</span><span>${prompt.response_count}/${prompt.expected_responses}</span></div>
              <div class="metric-line"><span>Rated matches</span><span>${prompt.rated_match_count}</span></div>
            </article>
          `
        )
        .join("");
    }

    function renderModels() {
      const container = el("models-grid");
      if (!state.models.length) {
        container.innerHTML = '<div class="empty-state">No models found.</div>';
        return;
      }

      container.innerHTML = state.models
        .map(
          (model) => `
            <article class="info-card">
              <p class="card-kicker">Rank ${model.rank}</p>
              <h3>
                <button class="row-button" data-open-model="${escapeHtml(model.model_id)}" type="button">
                  <strong>${escapeHtml(model.model_name)}</strong>
                </button>
              </h3>
              <div class="badge-row">
                <span class="badge is-strong">${formatNumber(model.conservative, 2)} conservative</span>
                <span class="badge">${model.games} games</span>
                <span class="badge">${model.response_count}/${model.expected_responses} prompts</span>
              </div>
              <div class="metric-line"><span>Wins / losses / ties</span><span>${model.wins} / ${model.losses} / ${model.ties}</span></div>
              <div class="metric-line"><span>Sigma</span><span>${formatNumber(model.sigma, 2)}</span></div>
            </article>
          `
        )
        .join("");
    }

    function captureDrawerState() {
      const panel = document.querySelector("#detail-drawer .detail-panel");
      const openKeys = [...document.querySelectorAll("#drawer-content details[data-detail-key]")]
        .filter((item) => item.open)
        .map((item) => item.dataset.detailKey || "");
      return {
        openKeys,
        scrollTop: panel instanceof HTMLElement ? panel.scrollTop : 0,
      };
    }

    function restoreDrawerState(snapshot) {
      if (!snapshot) {
        return;
      }
      const openKeySet = new Set(snapshot.openKeys || []);
      document
        .querySelectorAll("#drawer-content details[data-detail-key]")
        .forEach((item) => {
          if (!(item instanceof HTMLDetailsElement)) {
            return;
          }
          item.open = openKeySet.has(item.dataset.detailKey || "");
        });
      const panel = document.querySelector("#detail-drawer .detail-panel");
      if (panel instanceof HTMLElement) {
        panel.scrollTop = Number(snapshot.scrollTop || 0);
      }
    }

    function openDrawer(title, bodyHtml, { preserveState = false } = {}) {
      const snapshot = preserveState ? captureDrawerState() : null;
      el("drawer-content").innerHTML = `
        <section>
          <h2>${escapeHtml(title)}</h2>
          ${bodyHtml}
        </section>
      `;
      restoreDrawerState(snapshot);
      el("detail-drawer").classList.add("is-open");
      el("detail-drawer").setAttribute("aria-hidden", "false");
      document.body.classList.add("modal-open");
      updateRefreshMeta();
    }

    function closeDrawer() {
      state.detail = { kind: "", id: "" };
      state.detailRequestId += 1;
      el("detail-drawer").classList.remove("is-open");
      el("detail-drawer").setAttribute("aria-hidden", "true");
      document.body.classList.remove("modal-open");
      updateRefreshMeta();
      refreshAll({ silent: true }).catch(handleError);
    }

    async function openMatch(matchId, { preserveSelection = false } = {}) {
      const requestId = state.detailRequestId + 1;
      state.detailRequestId = requestId;
      state.detail = { kind: "match", id: matchId };
      const match = await fetchJson(`/api/match?match_id=${encodeURIComponent(matchId)}`);
      if (
        state.detailRequestId !== requestId ||
        state.detail.kind !== "match" ||
        state.detail.id !== matchId
      ) {
        return;
      }
      openDrawer(
        `${match.batch_id} · ${match.prompt_title || match.prompt_id}`,
        `
          <div class="badge-row">
            <span class="badge is-strong">${escapeHtml(matchOutcomeLabel(match))}</span>
            <span class="badge">${escapeHtml(match.canonical_decision)}</span>
            <span class="badge">${escapeHtml(match.judge_model || "-")}</span>
          </div>
          <div class="drawer-section">
            <h3>Prompt</h3>
            <pre>${escapeHtml(match.prompt_text)}</pre>
          </div>
          <div class="drawer-section">
            <h3>Responses</h3>
            <div class="drawer-grid">
              <div class="drawer-card">
                <p class="card-kicker">${escapeHtml(match.model_a_name)}</p>
                <pre>${escapeHtml(match.response_a_text)}</pre>
              </div>
              <div class="drawer-card">
                <p class="card-kicker">${escapeHtml(match.model_b_name)}</p>
                <pre>${escapeHtml(match.response_b_text)}</pre>
              </div>
            </div>
          </div>
          <div class="drawer-section">
            <h3>Judgments</h3>
            ${["ab", "ba"]
              .filter((side) => match.judgments[side])
              .map(
                (side) => `
                  <details data-detail-key="judgment-${escapeHtml(side)}" ${side === "ab" ? "open" : ""}>
                    <summary>${escapeHtml(side.toUpperCase())} · ${escapeHtml(match.judgments[side].decision)}</summary>
                    <div>
                      <p class="card-meta">${escapeHtml(match.judgments[side].judge_model || "-")}</p>
                      <pre>${escapeHtml(match.judgments[side].explanation || match.judgments[side].raw_completion || "-")}</pre>
                    </div>
                  </details>
                `
              )
              .join("")}
          </div>
        `,
        { preserveState: preserveSelection }
      );
      if (!preserveSelection) {
        state.detail = { kind: "match", id: matchId };
      }
    }

    async function openPrompt(promptId, { preserveSelection = false } = {}) {
      const requestId = state.detailRequestId + 1;
      state.detailRequestId = requestId;
      state.detail = { kind: "prompt", id: promptId };
      const prompt = await fetchJson(`/api/prompt?prompt_id=${encodeURIComponent(promptId)}`);
      if (
        state.detailRequestId !== requestId ||
        state.detail.kind !== "prompt" ||
        state.detail.id !== promptId
      ) {
        return;
      }
      openDrawer(
        prompt.title || prompt.prompt_id,
        `
          <div class="badge-row">
            <span class="badge is-strong">${prompt.response_count}/${prompt.expected_responses} responses</span>
            <span class="badge">${prompt.rated_match_count} rated matches</span>
            <span class="badge">${escapeHtml(prompt.category || "uncategorized")}</span>
          </div>
          <div class="drawer-section">
            <h3>Prompt text</h3>
            <pre>${escapeHtml(prompt.prompt_text)}</pre>
          </div>
          <div class="drawer-section">
            <h3>Responses</h3>
            <div class="drawer-grid">
              ${prompt.responses
                .map(
                  (response) => `
                    <div class="drawer-card">
                      <p class="card-kicker">${escapeHtml(response.model_name)}</p>
                      <pre>${escapeHtml(response.response_text || "(missing)")}</pre>
                    </div>
                  `
                )
                .join("")}
            </div>
          </div>
          <div class="drawer-section">
            <h3>Matches</h3>
            <div class="badge-row">
              ${prompt.matches
                .map(
                  (match) => `
                    <button class="secondary-button" type="button" data-open-match="${escapeHtml(match.match_id)}">
                      ${escapeHtml(match.model_a_name)} vs ${escapeHtml(match.model_b_name)} · ${escapeHtml(matchOutcomeLabel(match))}
                    </button>
                  `
                )
                .join("")}
            </div>
          </div>
        `,
        { preserveState: preserveSelection }
      );
      if (!preserveSelection) {
        state.detail = { kind: "prompt", id: promptId };
      }
    }

    async function openModel(modelId, { preserveSelection = false } = {}) {
      const requestId = state.detailRequestId + 1;
      state.detailRequestId = requestId;
      state.detail = { kind: "model", id: modelId };
      const model = await fetchJson(`/api/model?model_id=${encodeURIComponent(modelId)}`);
      if (
        state.detailRequestId !== requestId ||
        state.detail.kind !== "model" ||
        state.detail.id !== modelId
      ) {
        return;
      }
      openDrawer(
        model.model_name,
        `
          <div class="badge-row">
            <span class="badge is-strong">Rank ${model.rank}</span>
            <span class="badge">${formatNumber(model.conservative, 2)} conservative</span>
            <span class="badge">${model.response_count}/${model.expected_responses} prompt coverage</span>
          </div>
          <div class="drawer-section">
            <h3>Head-to-head</h3>
            <div class="drawer-grid">
              ${model.pairwise
                .map(
                  (row) => `
                    <div class="drawer-card">
                      <p class="card-kicker">${escapeHtml(row.opponent_model_name)}</p>
                      <div class="metric-line"><span>Score</span><span>${formatPercent(row.score, 0)}</span></div>
                      <div class="metric-line"><span>Record</span><span>${row.wins}-${row.losses}-${row.ties}</span></div>
                      <div class="metric-line"><span>Invalid</span><span>${row.invalid}</span></div>
                    </div>
                  `
                )
                .join("")}
            </div>
          </div>
          <div class="drawer-section">
            <h3>Prompt responses</h3>
            ${model.responses
              .map(
                (response) => `
                  <details data-detail-key="response-${escapeHtml(response.prompt_id)}">
                    <summary>${escapeHtml(response.title || response.prompt_id)}</summary>
                    <div><pre>${escapeHtml(response.response_text || "(missing)")}</pre></div>
                  </details>
                `
              )
              .join("")}
          </div>
        `,
        { preserveState: preserveSelection }
      );
      if (!preserveSelection) {
        state.detail = { kind: "model", id: modelId };
      }
    }

    function bindStaticEvents() {
      el("refresh-button").addEventListener("click", () => {
        refreshAll().catch(handleError);
      });

      el("drawer-close").addEventListener("click", closeDrawer);
      el("detail-drawer").addEventListener("click", (event) => {
        if (event.target instanceof HTMLElement && event.target.dataset.closeDrawer === "1") {
          closeDrawer();
        }
      });
      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
          closeDrawer();
        }
      });

      document.querySelectorAll(".tab-button").forEach((button) => {
        button.addEventListener("click", () => setActiveTab(button.dataset.tab));
      });

      el("matches-filters").addEventListener("submit", (event) => {
        event.preventDefault();
        state.filters.model = el("filter-model").value;
        state.filters.opponent = el("filter-opponent").value;
        state.filters.category = el("filter-category").value;
        state.filters.outcome = el("filter-outcome").value;
        state.filters.batch_id = el("filter-batch").value.trim();
        loadMatches().catch(handleError);
      });

      el("clear-filters").addEventListener("click", () => {
        state.filters = {
          model: "",
          opponent: "",
          category: "",
          outcome: "",
          batch_id: "",
        };
        populateFilterControls();
        loadMatches().catch(handleError);
      });

      document.body.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) {
          return;
        }
        const modelId = target.closest("[data-open-model]")?.getAttribute("data-open-model");
        if (modelId) {
          openModel(modelId).catch(handleError);
          return;
        }
        const promptId = target.closest("[data-open-prompt]")?.getAttribute("data-open-prompt");
        if (promptId) {
          openPrompt(promptId).catch(handleError);
          return;
        }
        const matchId = target.closest("[data-open-match]")?.getAttribute("data-open-match");
        if (matchId) {
          openMatch(matchId).catch(handleError);
          return;
        }
        const pairButton = target.closest("[data-pair-model]");
        if (pairButton instanceof HTMLElement) {
          state.filters.model = pairButton.getAttribute("data-pair-model") || "";
          state.filters.opponent = pairButton.getAttribute("data-pair-opponent") || "";
          state.filters.category = "";
          state.filters.outcome = "";
          state.filters.batch_id = "";
          populateFilterControls();
          setActiveTab("matches");
          loadMatches().catch(handleError);
        }
      });
    }

    function handleError(error) {
      console.error(error);
      setStatus(error.message || "Request failed", "error");
    }

    document.addEventListener("DOMContentLoaded", () => {
      bindStaticEvents();
      updateRefreshMeta();
      refreshAll().catch(handleError);
      if (!isStaticMode()) {
        document.addEventListener("visibilitychange", () => {
          if (document.visibilityState === "visible" && !isAutoRefreshPaused()) {
            refreshAll({ silent: true }).catch(handleError);
          }
        });
        window.setInterval(() => {
          if (document.visibilityState === "visible" && !isAutoRefreshPaused()) {
            refreshAll({ silent: true }).catch(handleError);
          }
        }, state.autoRefreshMs);
      }
    });
    """
)
