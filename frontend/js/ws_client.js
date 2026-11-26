/**
 * WebSocket Client, Navigation Controller & Live Telemetry Dispatcher.
 *
 * Connects to ws://${window.location.host}/ws/telemetry with auto-reconnect backoff,
 * routes incoming typed messages (prediction, drift_alert, multi_city_drift_alert, model_reload),
 * synchronizes all dashboard sections (Current Model, Model Performances, Audit Stream),
 * and manages fixed sidebar scroll navigation.
 */

class DashboardWebSocketClient {
  constructor() {
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectDelay = 10000;
    this.eventCount = 0;
    this.latencies = [];

    // Configuration & Thresholds (Dynamically fetched from backend)
    this.psiThreshold = 0.20;
    this.multiCityThreshold = 4;

    // Spatial Zones Metadata & Live Status Dictionary
    this.zones = [
      { zone_id: "ZONE_KODAGU", name: "Kodagu (Coorg)", lat: 12.25, lon: 75.82 },
      { zone_id: "ZONE_DAKSHINA_KANNADA", name: "Dakshina Kannada (Mangaluru)", lat: 12.91, lon: 75.08 },
      { zone_id: "ZONE_CHIKMAGALUR", name: "Chikmagalur", lat: 13.48, lon: 75.80 },
      { zone_id: "ZONE_SHIMOGA_AGUMBE", name: "Shimoga / Agumbe", lat: 14.08, lon: 75.28 },
      { zone_id: "ZONE_UTTARA_KANNADA", name: "Uttara Kannada (Karwar)", lat: 14.76, lon: 74.53 },
      { zone_id: "ZONE_WAYANAD_NILGIRIS", name: "Wayanad / Nilgiris Foothills", lat: 11.65, lon: 76.25 },
    ];

    // Live City Status Cache
    this.cityStatus = {};
    this.zones.forEach((z) => {
      this.cityStatus[z.zone_id] = {
        name: z.name,
        lat: z.lat,
        lon: z.lon,
        hazard_probability: 0.0,
        predicted_label: 0,
        latency_ms: 0.0,
        last_updated: null,
        current_psi: 0.0,
        drift_detected: false,
        retrain_triggered: false,
      };
    });

    // DOM Elements - Navigation & Badges
    this.wsBadge = document.getElementById("ws-badge");
    this.wsStatusText = document.getElementById("ws-status-text");
    this.modelBadge = document.getElementById("model-badge");
    this.modelVersionText = document.getElementById("model-version-text");
    this.driftBadge = document.getElementById("drift-badge");
    this.driftStatusText = document.getElementById("drift-status-text");
    this.multiCityBadge = document.getElementById("multi-city-badge");
    this.multiCityStatusText = document.getElementById("multi-city-status-text");
    this.eventCounterEl = document.getElementById("event-counter");
    this.avgLatencyEl = document.getElementById("avg-latency");
    this.auditLogBody = document.getElementById("audit-log-body");
    this.multiCityBanner = document.getElementById("multi-city-drift-banner");
    this.bannerIcon = document.getElementById("multi-city-banner-icon");
    this.bannerTitle = document.getElementById("multi-city-banner-title");
    this.bannerMessage = document.getElementById("multi-city-banner-message");
    this.bannerStatusBadge = document.getElementById("multi-city-banner-status-badge");
    this.bannerCountdownBadge = document.getElementById("banner-countdown-badge");
    this.bannerCountdownSeconds = document.getElementById("banner-countdown-seconds");

    // Alert Reset Timer & Independent Event Tracking
    this.alertResetTimer = null;
    this.countdownInterval = null;
    this.activeDriftEventId = null;
    this.resetSecondsRemaining = 60;

    // DOM Elements - Current Model Section
    this.cmVersion = document.getElementById("cm-version");
    this.ovModelVersion = document.getElementById("ov-model-version");
    this.cmAlias = document.getElementById("cm-alias");
    this.cmSource = document.getElementById("cm-source");
    this.cmLoadedAt = document.getElementById("cm-loaded-at");
    this.cmTrainedAt = document.getElementById("cm-trained-at");
    this.cmMaxRisk = document.getElementById("cm-max-risk");
    this.cmHighRiskCount = document.getElementById("cm-high-risk-count");
    this.cityRiskTableBody = document.getElementById("city-risk-table-body");

    // DOM Elements - Model Performances Section
    this.perfAccuracy = document.getElementById("perf-accuracy");
    this.perfPrecision = document.getElementById("perf-precision");
    this.perfRecall = document.getElementById("perf-recall");
    this.perfF1 = document.getElementById("perf-f1");
    this.perfRocAuc = document.getElementById("perf-roc-auc");
    this.perfSourceLabel = document.getElementById("perf-source-label");
    this.cityDriftTableBody = document.getElementById("city-drift-table-body");
    this.matrixDriftCount = document.getElementById("matrix-drift-count");
    this.evalHistoryTableBody = document.getElementById("eval-history-table-body");
    this.evalHistoryCount = document.getElementById("eval-history-count");
    this.driftEventTableBody = document.getElementById("drift-event-table-body");
    this.driftEventCount = document.getElementById("drift-event-count");
  }

  async init() {
    this.initNavigation();
    this.renderInitialCityRiskTable();
    this.renderInitialCityDriftTable();

    // 1. Fetch initial performance & model metadata
    await this.fetchModelPerformance();

    // 2. Connect WebSocket
    this.connect();
  }

  // ==========================================================================
  // Navigation & Scrollspy
  // ==========================================================================
  initNavigation() {
    const navLinks = document.querySelectorAll(".sidebar-menu .nav-item");
    const mainContent = document.querySelector(".main-content");
    const sections = document.querySelectorAll(".content-section");

    // Click navigation handler
    navLinks.forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        const targetId = link.getAttribute("href").substring(1);
        const targetEl = document.getElementById(targetId);
        if (targetEl && mainContent) {
          const headerOffset = 70;
          const targetPos = targetEl.offsetTop - headerOffset;
          mainContent.scrollTo({ top: targetPos, behavior: "smooth" });

          navLinks.forEach((l) => l.classList.remove("active"));
          link.classList.add("active");
        }
      });
    });

    // Scrollspy handler on main content scroll
    if (mainContent) {
      mainContent.addEventListener("scroll", () => {
        const scrollPos = mainContent.scrollTop + 120;
        sections.forEach((sec) => {
          const top = sec.offsetTop;
          const height = sec.offsetHeight;
          const id = sec.getAttribute("id");
          if (scrollPos >= top && scrollPos < top + height) {
            navLinks.forEach((l) => {
              if (l.getAttribute("data-section") === id) {
                l.classList.add("active");
              } else {
                l.classList.remove("active");
              }
            });
          }
        });
      });
    }
  }

  // ==========================================================================
  // REST API Telemetry Fetching (/api/model/performance)
  // ==========================================================================
  async fetchModelPerformance() {
    try {
      const resp = await fetch("/api/model/performance");
      if (!resp.ok) {
        console.warn(`[WSClient] /api/model/performance returned HTTP ${resp.status}`);
        return;
      }
      const data = await resp.json();

      // 1. Thresholds
      if (data.psi_threshold) {
        this.psiThreshold = data.psi_threshold;
      }
      if (data.multi_city_drift_threshold) {
        this.multiCityThreshold = data.multi_city_drift_threshold;
      }

      // 2. Active Model Metadata
      const meta = data.model_metadata || {};
      const activeVersion = data.active_model_version || meta.model_version || "v1";
      this.updateModelVersion(activeVersion);

      if (this.cmVersion) this.cmVersion.textContent = activeVersion;
      if (this.ovModelVersion) this.ovModelVersion.textContent = `${activeVersion} (Production)`;
      if (this.cmAlias) this.cmAlias.textContent = meta.alias || "Production";
      if (this.cmSource) this.cmSource.textContent = meta.source || "MLflow: weather-hazard-classifier";

      if (this.cmLoadedAt) {
        this.cmLoadedAt.textContent = meta.loaded_at ? new Date(meta.loaded_at).toLocaleString() : "Active in memory";
      }
      if (this.cmTrainedAt) {
        this.cmTrainedAt.textContent = meta.trained_at ? new Date(meta.trained_at).toLocaleString() : "Recorded";
      }

      // 3. Active Quality Metrics (Distinguish unavailable data with 'N/A')
      const metrics = data.active_metrics || meta.metrics || {};
      this.renderMetricsCards(metrics, activeVersion);

      // 4. Latest Zone Status
      if (data.latest_zone_status) {
        Object.keys(data.latest_zone_status).forEach((zid) => {
          if (this.cityStatus[zid]) {
            const zs = data.latest_zone_status[zid];
            this.cityStatus[zid].lat = zs.lat || this.cityStatus[zid].lat;
            this.cityStatus[zid].lon = zs.lon || this.cityStatus[zid].lon;
            this.cityStatus[zid].hazard_probability = zs.hazard_probability || 0.0;
            this.cityStatus[zid].predicted_label = zs.predicted_label || 0;
            this.cityStatus[zid].latency_ms = zs.last_latency_ms || 0.0;
            this.cityStatus[zid].last_updated = zs.latest_timestamp;
          }
        });
        this.renderInitialCityRiskTable();
        this.updateRiskOverviewStats();
      }

      // 5. Evaluation History & Performance Chart
      const history = data.evaluation_history || [];
      this.renderEvaluationHistoryTable(history);

      // 6. Significant Multi-City Drift Event History
      const driftHistory = data.drift_event_history || [];
      this.renderDriftEventHistoryTable(driftHistory);

      if (window.performanceChartController && history.length > 0) {
        // Find latest record with evaluation metrics
        const latestRecord = history[history.length - 1];
        window.performanceChartController.updateComparison(latestRecord);
      }

      console.log(`[WSClient] Synchronized model performance data: active=${activeVersion}, history=${history.length} runs, drift_events=${driftHistory.length}.`);
    } catch (err) {
      console.error("[WSClient] Failed to fetch /api/model/performance:", err);
    }
  }

  renderMetricsCards(metrics, version) {
    const fmt = (val) => {
      if (val === null || val === undefined) return "N/A";
      return `${(val * 100).toFixed(2)}%`;
    };

    if (this.perfAccuracy) this.perfAccuracy.textContent = fmt(metrics.accuracy);
    if (this.perfPrecision) this.perfPrecision.textContent = fmt(metrics.precision);
    if (this.perfRecall) this.perfRecall.textContent = fmt(metrics.recall);
    if (this.perfF1) this.perfF1.textContent = fmt(metrics.f1_score);
    if (this.perfRocAuc) this.perfRocAuc.textContent = fmt(metrics.roc_auc);

    if (this.perfSourceLabel) {
      this.perfSourceLabel.textContent = `Model ${version} • MLflow Registry Evaluated`;
    }
  }

  // ==========================================================================
  // Render Data Tables
  // ==========================================================================
  renderInitialCityRiskTable() {
    if (!this.cityRiskTableBody) return;
    this.cityRiskTableBody.innerHTML = "";

    this.zones.forEach((z) => {
      const s = this.cityStatus[z.zone_id];
      const probPct = ((s.hazard_probability || 0) * 100).toFixed(1);
      const isSevere = (s.hazard_probability || 0) >= 0.70;
      const isModerate = (s.hazard_probability || 0) >= 0.40 && !isSevere;

      let statusBadge = `<span class="badge badge-green" style="font-size:0.68rem;">Nominal</span>`;
      let probColor = "var(--green)";
      if (isSevere) {
        statusBadge = `<span class="badge badge-red pulse" style="font-size:0.68rem;">Severe Hazard</span>`;
        probColor = "var(--red)";
      } else if (isModerate) {
        statusBadge = `<span class="badge badge-amber" style="font-size:0.68rem;">Moderate Risk</span>`;
        probColor = "var(--amber)";
      }

      const tr = document.createElement("tr");
      tr.id = `risk-row-${z.zone_id}`;
      tr.innerHTML = `
        <td>
          <strong>${z.name}</strong><br>
          <span style="font-size: 0.68rem; color: var(--text-dim);">${z.zone_id}</span>
        </td>
        <td class="font-mono">${s.lat.toFixed(2)}, ${s.lon.toFixed(2)}</td>
        <td class="font-mono"><strong style="color: ${probColor};">${probPct}%</strong></td>
        <td>${statusBadge}</td>
        <td class="font-mono">${s.latency_ms ? s.latency_ms.toFixed(1) + ' ms' : '--'}</td>
      `;
      this.cityRiskTableBody.appendChild(tr);
    });
  }

  renderInitialCityDriftTable() {
    if (!this.cityDriftTableBody) return;
    this.cityDriftTableBody.innerHTML = "";

    let driftedCount = 0;
    this.zones.forEach((z) => {
      const s = this.cityStatus[z.zone_id];
      const psi = (s.current_psi || 0.0).toFixed(4);
      const isDrifted = s.drift_detected === true || s.current_psi > this.psiThreshold;
      if (isDrifted) driftedCount++;

      const statusBadge = isDrifted
        ? `<span class="badge badge-red" style="font-size: 0.68rem;">DRIFTED (PSI &gt; 0.20)</span>`
        : `<span class="badge badge-green" style="font-size: 0.68rem;">NOMINAL</span>`;

      const retrainBadge = s.retrain_triggered
        ? `<span class="badge badge-purple" style="font-size: 0.68rem;">Airflow Triggered</span>`
        : `<span style="font-size: 0.72rem; color: var(--text-dim);">Monitoring</span>`;

      const tr = document.createElement("tr");
      tr.id = `drift-row-${z.zone_id}`;
      tr.innerHTML = `
        <td><strong>${z.name}</strong></td>
        <td class="font-mono" style="font-size: 0.72rem; color: var(--text-dim);">${z.zone_id}</td>
        <td class="font-mono"><strong style="color: ${isDrifted ? 'var(--red)' : 'var(--text-main)'};">${psi}</strong></td>
        <td class="font-mono">${this.psiThreshold.toFixed(2)}</td>
        <td>${statusBadge}</td>
        <td>${retrainBadge}</td>
      `;
      this.cityDriftTableBody.appendChild(tr);
    });

    if (this.matrixDriftCount) {
      this.matrixDriftCount.textContent = `${driftedCount} of ${this.zones.length} Drifted`;
      this.matrixDriftCount.className = driftedCount >= this.multiCityThreshold ? "badge badge-red" : "badge badge-blue";
    }
  }

  renderEvaluationHistoryTable(history) {
    if (!this.evalHistoryTableBody) return;
    this.evalHistoryTableBody.innerHTML = "";

    if (this.evalHistoryCount) {
      this.evalHistoryCount.textContent = `${history.length} Evaluations`;
    }

    if (history.length === 0) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="6" style="text-align: center; color: var(--text-dim);">No model evaluations recorded yet.</td>`;
      this.evalHistoryTableBody.appendChild(tr);
      return;
    }

    // Display in reverse chronological order (newest first)
    const reversed = [...history].reverse();

    reversed.forEach((rec) => {
      const timeStr = rec.timestamp ? new Date(rec.timestamp).toLocaleString() : "--";
      const champ = rec.champion_version || "Champion";
      const chall = rec.challenger_version || "None";
      const isApproved = rec.promoted === true || rec.status === "APPROVED";

      const decisionBadge = isApproved
        ? `<span class="badge badge-green" style="font-size:0.68rem;">PROMOTED</span>`
        : `<span class="badge badge-amber" style="font-size:0.68rem;">REJECTED</span>`;

      let deltaStr = "Baseline";
      let deltaColor = "var(--text-dim)";
      if (rec.f1_improvement !== undefined && rec.challenger_metrics) {
        const d = (rec.f1_improvement * 100).toFixed(2);
        deltaStr = `${d >= 0 ? '+' : ''}${d}%`;
        deltaColor = d >= 1.0 ? "var(--green)" : (d >= 0 ? "var(--amber)" : "var(--red)");
      }

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="font-mono" style="font-size: 0.72rem;">${timeStr}</td>
        <td><strong class="font-mono">${champ}</strong></td>
        <td><strong class="font-mono" style="color: var(--purple);">${chall}</strong></td>
        <td class="font-mono"><strong style="color: ${deltaColor};">${deltaStr}</strong></td>
        <td>${decisionBadge}</td>
        <td style="font-size: 0.74rem; line-height: 1.3;">${rec.reason || "Automated model evaluation run."}</td>
      `;
      this.evalHistoryTableBody.appendChild(tr);
    });
  }

  renderDriftEventHistoryTable(driftHistory) {
    if (!this.driftEventTableBody) return;
    this.driftEventTableBody.innerHTML = "";

    if (this.driftEventCount) {
      this.driftEventCount.textContent = `${driftHistory.length} Events`;
    }

    if (!driftHistory || driftHistory.length === 0) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="7" style="text-align: center; color: var(--text-dim); padding: 18px;">No multi-city drift events recorded yet.</td>`;
      this.driftEventTableBody.appendChild(tr);
      return;
    }

    // Display in reverse chronological order (newest first)
    const reversed = [...driftHistory].reverse();

    reversed.forEach((rec) => {
      // 1. Trigger Time
      const triggerTimeStr = rec.trigger_timestamp ? new Date(rec.trigger_timestamp).toLocaleString() : "--";

      // 2. Affected Cities & Recorded PSI values
      let citiesHtml = "";
      if (rec.city_psi_values && Object.keys(rec.city_psi_values).length > 0) {
        citiesHtml = Object.entries(rec.city_psi_values)
          .map(([zid, psi]) => {
            const z = this.zones.find((zone) => zone.zone_id === zid);
            const name = z ? z.name.split(" ")[0] : zid.replace("ZONE_", "");
            const psiVal = typeof psi === "number" ? psi.toFixed(2) : psi;
            return `<span class="city-psi-tag">${name}: <strong>${psiVal}</strong></span>`;
          })
          .join(" ");
      } else if (rec.affected_cities && rec.affected_cities.length > 0) {
        citiesHtml = rec.affected_cities
          .map((zid) => {
            const z = this.zones.find((zone) => zone.zone_id === zid);
            const name = z ? z.name.split(" ")[0] : zid.replace("ZONE_", "");
            return `<span class="city-psi-tag">${name}</span>`;
          })
          .join(" ");
      } else {
        citiesHtml = `<span style="color: var(--text-dim);">--</span>`;
      }

      // 3. Threshold & Cities
      const count = rec.affected_city_count || (rec.affected_cities ? rec.affected_cities.length : 0);
      const thresh = rec.psi_threshold !== undefined ? Number(rec.psi_threshold).toFixed(2) : "0.20";
      const threshHtml = `<strong>${count}</strong> / 6 Cities<div class="time-label">PSI &gt; ${thresh}</div>`;

      // 4. Model Version at Trigger Time
      const modelAtTrigger = rec.active_model_version || "--";

      // 5. Challenger Version
      const challenger = rec.challenger_version || "--";

      // 6. Champion–Challenger Decision & Decision Timestamp
      let decisionBadge = `<span class="badge badge-purple pulse" style="font-size: 0.68rem;">EVALUATING</span>`;
      const dec = (rec.decision || rec.champion_challenger_decision || "").toLowerCase();
      if (dec === "approved" || rec.promoted === true || rec.status === "PROMOTED_AND_DEPLOYED") {
        decisionBadge = `<span class="badge badge-green" style="font-size: 0.68rem;">APPROVED</span>`;
      } else if (dec === "rejected" || rec.status === "REJECTED_CHAMPION_RETAINED") {
        decisionBadge = `<span class="badge badge-amber" style="font-size: 0.68rem;">REJECTED</span>`;
      }

      const decisionTime = rec.decision_timestamp
        ? new Date(rec.decision_timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
        : null;
      const decisionHtml = `
        ${decisionBadge}
        ${decisionTime ? `<div class="time-label font-mono">Decided: ${decisionTime}</div>` : ""}
      `;

      // 7. Final Production Model & Deployment Timestamp
      const finalModel = rec.final_production_model_version || (dec === "approved" ? challenger : modelAtTrigger);
      const deployTime = rec.deployment_timestamp
        ? new Date(rec.deployment_timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
        : null;

      let deployHtml = `<strong class="font-mono">${finalModel}</strong>`;
      if (deployTime) {
        deployHtml += `<div class="time-label font-mono" style="color: var(--green);">Deployed: ${deployTime}</div>`;
      } else if (dec === "rejected") {
        deployHtml += `<div class="time-label" style="color: var(--amber);">Champion Retained</div>`;
      } else if (dec === "pending" || !dec) {
        deployHtml += `<div class="time-label" style="color: var(--purple);">Retraining...</div>`;
      }

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="font-mono" style="font-size: 0.72rem;">
          ${triggerTimeStr}
          <div class="time-label">Trigger Time</div>
        </td>
        <td>${citiesHtml}</td>
        <td>${threshHtml}</td>
        <td class="font-mono"><strong>${modelAtTrigger}</strong></td>
        <td class="font-mono" style="color: var(--purple);"><strong>${challenger}</strong></td>
        <td>${decisionHtml}</td>
        <td>${deployHtml}</td>
      `;
      this.driftEventTableBody.appendChild(tr);
    });
  }

  updateRiskOverviewStats() {
    let maxRisk = 0.0;
    let highRiskCount = 0;

    Object.values(this.cityStatus).forEach((s) => {
      const p = s.hazard_probability || 0.0;
      if (p > maxRisk) maxRisk = p;
      if (p >= 0.70) highRiskCount++;
    });

    if (this.cmMaxRisk) {
      this.cmMaxRisk.textContent = `${(maxRisk * 100).toFixed(1)}%`;
      this.cmMaxRisk.style.color = maxRisk >= 0.70 ? "var(--red)" : (maxRisk >= 0.40 ? "var(--amber)" : "var(--green)");
    }
    if (this.cmHighRiskCount) {
      this.cmHighRiskCount.textContent = `${highRiskCount} of ${this.zones.length}`;
      this.cmHighRiskCount.style.color = highRiskCount > 0 ? "var(--red)" : "var(--text-main)";
    }
  }

  // ==========================================================================
  // WebSocket Lifecycle & Message Routing
  // ==========================================================================
  connect() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

    console.log(`[WSClient] Connecting to telemetry stream: ${wsUrl}...`);
    this.setConnectionState("connecting");

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log("[WSClient] WebSocket connection established.");
        this.reconnectAttempts = 0;
        this.setConnectionState("connected");
        this.appendAuditLog("SYSTEM", "WEBSOCKET", "Connected to real-time telemetry stream");
      };

      this.ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          this.handleMessage(payload);
        } catch (err) {
          console.error("[WSClient] Error parsing message payload:", err);
        }
      };

      this.ws.onclose = () => {
        console.warn("[WSClient] WebSocket closed. Scheduling reconnect...");
        this.setConnectionState("disconnected");
        this.scheduleReconnect();
      };

      this.ws.onerror = (err) => {
        console.error("[WSClient] WebSocket error:", err);
        this.ws.close();
      };
    } catch (e) {
      console.error("[WSClient] Exception initiating WebSocket:", e);
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts), this.maxReconnectDelay);
    console.log(`[WSClient] Reconnecting in ${(delay / 1000).toFixed(1)}s (attempt ${this.reconnectAttempts})...`);
    setTimeout(() => this.connect(), delay);
  }

  setConnectionState(state) {
    if (!this.wsBadge || !this.wsStatusText) return;

    if (state === "connected") {
      this.wsBadge.className = "badge badge-green";
      this.wsStatusText.textContent = "Live Stream";
    } else if (state === "connecting") {
      this.wsBadge.className = "badge badge-amber";
      this.wsStatusText.textContent = "Connecting...";
    } else {
      this.wsBadge.className = "badge badge-red";
      this.wsStatusText.textContent = "Reconnecting";
    }
  }

  handleMessage(msg) {
    const type = msg.type;
    const data = msg.data || {};

    switch (type) {
      case "prediction":
        this.handlePredictionEvent(data);
        break;

      case "drift_alert":
        this.handleDriftAlertEvent(data);
        break;

      case "multi_city_drift_alert":
        this.handleMultiCityDriftAlert(data);
        break;

      case "retrain_outcome":
        this.handleRetrainOutcome(data);
        break;

      case "model_reload":
        this.handleModelReloadEvent(data);
        break;

      case "pong":
        break;

      default:
        console.debug("[WSClient] Unhandled message type:", type);
    }
  }

  // ==========================================================================
  // Handlers for Specific Typed Message Envelopes
  // ==========================================================================
  handlePredictionEvent(pred) {
    const zoneId = pred.zone_id;

    // 1. Update Map Marker
    if (window.mapController) {
      window.mapController.updateSensorMarker(pred);
    }

    // 2. Cache City Status
    if (this.cityStatus[zoneId]) {
      this.cityStatus[zoneId].hazard_probability = pred.hazard_probability || 0.0;
      this.cityStatus[zoneId].predicted_label = pred.predicted_label || 0;
      this.cityStatus[zoneId].latency_ms = pred.inference_latency_ms || 0.0;
      this.cityStatus[zoneId].lat = pred.lat || this.cityStatus[zoneId].lat;
      this.cityStatus[zoneId].lon = pred.lon || this.cityStatus[zoneId].lon;

      // Update Table Row
      this.updateCityRiskRow(zoneId);
    }

    // 3. Update Model Version Badge
    if (pred.model_version) {
      this.updateModelVersion(pred.model_version);
    }

    // 4. Update Event Counter and Latency
    this.eventCount++;
    if (this.eventCounterEl) {
      this.eventCounterEl.textContent = this.eventCount.toLocaleString();
    }

    const latency = pred.inference_latency_ms || 0.0;
    this.latencies.push(latency);
    if (this.latencies.length > 50) this.latencies.shift();

    if (this.avgLatencyEl) {
      const avg = this.latencies.reduce((a, b) => a + b, 0) / this.latencies.length;
      this.avgLatencyEl.textContent = `${avg.toFixed(1)} ms`;
    }

    this.updateRiskOverviewStats();

    // 5. Log Severe Hazards
    if (pred.predicted_label === 1 && pred.hazard_probability >= 0.70) {
      const probPct = (pred.hazard_probability * 100).toFixed(1);
      const zoneName = this.cityStatus[zoneId]?.name || zoneId;
      this.appendAuditLog(
        "HAZARD",
        zoneName,
        `Severe Hazard Detected: ${probPct}% (Latency: ${latency.toFixed(1)}ms)`
      );
    }
  }

  updateCityRiskRow(zoneId) {
    const s = this.cityStatus[zoneId];
    if (!s) return;
    const row = document.getElementById(`risk-row-${zoneId}`);
    if (!row) return;

    const probPct = ((s.hazard_probability || 0) * 100).toFixed(1);
    const isSevere = (s.hazard_probability || 0) >= 0.70;
    const isModerate = (s.hazard_probability || 0) >= 0.40 && !isSevere;

    let statusBadge = `<span class="badge badge-green" style="font-size:0.68rem;">Nominal</span>`;
    let probColor = "var(--green)";
    if (isSevere) {
      statusBadge = `<span class="badge badge-red pulse" style="font-size:0.68rem;">Severe Hazard</span>`;
      probColor = "var(--red)";
    } else if (isModerate) {
      statusBadge = `<span class="badge badge-amber" style="font-size:0.68rem;">Moderate Risk</span>`;
      probColor = "var(--amber)";
    }

    row.innerHTML = `
      <td>
        <strong>${s.name}</strong><br>
        <span style="font-size: 0.68rem; color: var(--text-dim);">${zoneId}</span>
      </td>
      <td class="font-mono">${s.lat.toFixed(2)}, ${s.lon.toFixed(2)}</td>
      <td class="font-mono"><strong style="color: ${probColor};">${probPct}%</strong></td>
      <td>${statusBadge}</td>
      <td class="font-mono">${s.latency_ms ? s.latency_ms.toFixed(1) + ' ms' : '--'}</td>
    `;
  }

  handleDriftAlertEvent(alert) {
    const isDrifted = alert.drift_detected === true;
    const zoneId = alert.zone_id;
    const maxPsi = parseFloat(alert.max_psi || 0.0);

    // 1. Push to Live PSI Chart
    if (window.chartController) {
      window.chartController.pushDriftPoint(alert);
    }

    // 2. Highlight/Unhighlight Zone on GIS Map
    if (window.mapController) {
      window.mapController.highlightZoneDrift(zoneId, isDrifted);
    }

    // 3. Update Status Cache and Matrix Table
    if (this.cityStatus[zoneId]) {
      this.cityStatus[zoneId].current_psi = maxPsi;
      this.cityStatus[zoneId].drift_detected = isDrifted;
      this.cityStatus[zoneId].retrain_triggered = alert.retrain_triggered || false;
      this.updateCityDriftRow(zoneId);
    }

    // 4. Evaluate Multi-City Drift Count across all 6 zones
    let totalDrifted = 0;
    Object.values(this.cityStatus).forEach((s) => {
      if (s.drift_detected || s.current_psi > this.psiThreshold) totalDrifted++;
    });

    if (this.matrixDriftCount) {
      this.matrixDriftCount.textContent = `${totalDrifted} of ${this.zones.length} Drifted`;
      this.matrixDriftCount.className = totalDrifted >= this.multiCityThreshold ? "badge badge-red" : "badge badge-blue";
    }

    // 5. Update Status Badges
    if (this.driftBadge && this.driftStatusText) {
      if (alert.retrain_triggered) {
        this.driftBadge.className = "badge badge-purple";
        this.driftStatusText.textContent = "Retrain Triggered";
      } else if (isDrifted) {
        this.driftBadge.className = "badge badge-amber";
        this.driftStatusText.textContent = `Drift: ${zoneId.replace("ZONE_", "")}`;
      } else {
        this.driftBadge.className = "badge badge-green";
        this.driftStatusText.textContent = "Drift: Nominal";
      }
    }

    // 6. Multi-City Drift Gate Evaluation
    if (totalDrifted >= this.multiCityThreshold) {
      this.showMultiCityBanner(totalDrifted, maxPsi);
    }

    // 7. Audit Log
    const zoneName = this.cityStatus[zoneId]?.name || zoneId;
    if (isDrifted) {
      this.appendAuditLog(
        "DRIFT",
        zoneName,
        `Distribution shift detected (Max PSI: ${maxPsi.toFixed(4)} > ${this.psiThreshold.toFixed(2)})`
      );
    }
    if (alert.retrain_triggered) {
      this.appendAuditLog(
        "RETRAIN",
        zoneName,
        `Airflow 'automated_retrain_dag' triggered via REST API (Multi-city threshold met)`
      );
    }
  }

  updateCityDriftRow(zoneId) {
    const s = this.cityStatus[zoneId];
    if (!s) return;
    const row = document.getElementById(`drift-row-${zoneId}`);
    if (!row) return;

    const isDrifted = s.drift_detected === true || s.current_psi > this.psiThreshold;
    const statusBadge = isDrifted
      ? `<span class="badge badge-red" style="font-size: 0.68rem;">DRIFTED (PSI &gt; 0.20)</span>`
      : `<span class="badge badge-green" style="font-size: 0.68rem;">NOMINAL</span>`;

    const retrainBadge = s.retrain_triggered
      ? `<span class="badge badge-purple" style="font-size: 0.68rem;">Airflow Triggered</span>`
      : `<span style="font-size: 0.72rem; color: var(--text-dim);">Monitoring</span>`;

    row.innerHTML = `
      <td><strong>${s.name}</strong></td>
      <td class="font-mono" style="font-size: 0.72rem; color: var(--text-dim);">${zoneId}</td>
      <td class="font-mono"><strong style="color: ${isDrifted ? 'var(--red)' : 'var(--text-main)'};">${s.current_psi.toFixed(4)}</strong></td>
      <td class="font-mono">${this.psiThreshold.toFixed(2)}</td>
      <td>${statusBadge}</td>
      <td>${retrainBadge}</td>
    `;
  }

  handleMultiCityDriftAlert(data) {
    console.warn("[WSClient] Multi-City Drift Alert received:", data);

    // Cancel any active reset timer from an earlier event (independent event handling)
    this.clearAlertResetTimers();

    const eventId = data.event_id || data.alert_id || (`drift_evt_${Date.now()}`);
    this.activeDriftEventId = eventId;

    const count = data.drifted_city_count || (data.drifted_zones ? data.drifted_zones.length : this.multiCityThreshold);
    const maxPsi = data.max_psi || 0.25;
    this.showMultiCityBanner(count, maxPsi, data.drifted_zones, eventId);

    // Re-fetch model performance to immediately display initial event in drift history table
    this.fetchModelPerformance();

    this.appendAuditLog(
      "RETRAIN",
      `MULTI-CITY (${count} Zones)`,
      `Airflow retrain workflow triggered. Production model serving protected by champion-challenger gate.`
    );
  }

  showMultiCityBanner(count, maxPsi, zonesList = null, eventId = null) {
    if (this.multiCityBanner) {
      this.multiCityBanner.classList.remove("hidden");
      this.multiCityBanner.classList.remove("outcome-approved", "outcome-rejected");

      if (this.bannerIcon) this.bannerIcon.textContent = "⚠️";
      if (this.bannerTitle) {
        this.bannerTitle.textContent = `SIGNIFICANT MULTI-CITY SPATIAL DRIFT DETECTED (${count} OF 6 CITIES DRIFTED)`;
      }
      if (this.bannerCountdownBadge) {
        this.bannerCountdownBadge.classList.add("hidden");
      }
      if (this.bannerStatusBadge) {
        this.bannerStatusBadge.className = "badge badge-red pulse";
        this.bannerStatusBadge.textContent = "Retrain Triggered";
      }

      if (this.bannerMessage) {
        const zonesDesc = zonesList ? `(${zonesList.join(', ')})` : "";
        this.bannerMessage.innerHTML = `
          Population Stability Index exceeded the 0.20 threshold across <strong>${count} monitored cities</strong> ${zonesDesc} (Peak PSI: <strong>${maxPsi.toFixed(4)}</strong>).
          The automated retraining pipeline has been dispatched to Airflow. 
          <strong>Important MLOps Safety Constraint:</strong> The currently deployed Champion model remains in active production and will only be replaced if a retrained Challenger model achieves verified <strong>+1.00% F1 improvement</strong> during gating evaluation.
        `;
      }
    }

    if (this.multiCityBadge && this.multiCityStatusText) {
      this.multiCityBadge.className = "badge badge-red pulse";
      this.multiCityStatusText.textContent = `Multi-City Alert (${count}/6)`;
    }
  }

  handleRetrainOutcome(data) {
    console.log("[WSClient] Champion-Challenger retrain outcome received:", data);

    const eventId = data.event_id;
    // Check if an independent newer drift event was triggered after this one
    if (eventId && this.activeDriftEventId && eventId !== this.activeDriftEventId) {
      console.warn(`[WSClient] Ignoring outcome for older event ${eventId}; newer active event is ${this.activeDriftEventId}`);
      this.fetchModelPerformance();
      return;
    }

    const isApproved = data.promoted === true || data.decision === "approved";
    const finalModel = data.final_production_model_version || (isApproved ? data.challenger_version : data.champion_version);
    const champ = data.champion_version || "Champion";
    const chall = data.challenger_version || "Challenger";
    const f1Imp = data.f1_improvement !== undefined ? (Number(data.f1_improvement) * 100).toFixed(2) : null;
    const reason = data.reason || data.gating_rationale || "";

    // 1. Reflect active model in production
    if (isApproved) {
      this.updateModelVersion(finalModel, true);
      this.appendAuditLog(
        "PROMOTION",
        `Model ${finalModel}`,
        `Champion-Challenger gate APPROVED (${f1Imp !== null ? '+' + f1Imp + '% F1' : 'met threshold'}). Promoted to Production with zero downtime.`
      );
    } else {
      this.updateModelVersion(finalModel, false);
      this.appendAuditLog(
        "GATE_REJECTED",
        `Model ${champ}`,
        `Challenger ${chall} rejected: ${reason || 'did not meet +1% F1 gate'}. Active Champion ${finalModel} safely retained.`
      );
    }

    // 2. Update Multi-City Banner UI to reflect decision outcome
    if (this.multiCityBanner) {
      this.multiCityBanner.classList.remove("hidden");
      this.multiCityBanner.classList.remove("outcome-approved", "outcome-rejected");

      if (isApproved) {
        this.multiCityBanner.classList.add("outcome-approved");
        if (this.bannerIcon) this.bannerIcon.textContent = "✅";
        if (this.bannerTitle) {
          this.bannerTitle.textContent = `CHAMPION-CHALLENGER GATE APPROVED: MODEL ${finalModel} PROMOTED`;
        }
        if (this.bannerStatusBadge) {
          this.bannerStatusBadge.className = "badge badge-green";
          this.bannerStatusBadge.textContent = `Promoted & Deployed (${finalModel})`;
        }
        if (this.bannerMessage) {
          this.bannerMessage.innerHTML = `
            Challenger model <strong>${chall}</strong> achieved verified <strong>${f1Imp !== null ? '+' + f1Imp + '%' : '> +1.0%'} F1 improvement</strong> over Champion (<strong>${champ}</strong>).
            The Challenger has passed all production quality gates and has been promoted to <strong>Production</strong> alias with zero-downtime hot-reload.
          `;
        }
      } else {
        this.multiCityBanner.classList.add("outcome-rejected");
        if (this.bannerIcon) this.bannerIcon.textContent = "🛡️";
        if (this.bannerTitle) {
          this.bannerTitle.textContent = `CHALLENGER REJECTED: CHAMPION MODEL (${finalModel}) RETAINED`;
        }
        if (this.bannerStatusBadge) {
          this.bannerStatusBadge.className = "badge badge-amber";
          this.bannerStatusBadge.textContent = `Champion Retained (${finalModel})`;
        }
        if (this.bannerMessage) {
          this.bannerMessage.innerHTML = `
            Challenger model <strong>${chall}</strong> did not achieve the required +1.00% F1 improvement (${reason || 'insufficient quality gain'}).
            <strong>MLOps safety constraint enforced:</strong> The currently deployed Champion model (<strong>${finalModel}</strong>) safely remains in active production.
          `;
        }
      }
    }

    // 3. Update Sidebar Multi-City Badge
    if (this.multiCityBadge && this.multiCityStatusText) {
      if (isApproved) {
        this.multiCityBadge.className = "badge badge-green";
        this.multiCityStatusText.textContent = `Promoted (${finalModel})`;
      } else {
        this.multiCityBadge.className = "badge badge-amber";
        this.multiCityStatusText.textContent = `Retained (${finalModel})`;
      }
    }

    // 4. Start 60-Second Timers for Alert Reset & Message Fade-Away ONLY NOW
    this.start60SecondAlertReset(eventId);

    // 5. Refresh tables (Evaluation history and Drift Event History)
    this.fetchModelPerformance();
  }

  start60SecondAlertReset(eventId) {
    this.clearAlertResetTimers();
    this.resetSecondsRemaining = 60;

    if (this.bannerCountdownSeconds) {
      this.bannerCountdownSeconds.textContent = "60";
    }
    if (this.bannerCountdownBadge) {
      this.bannerCountdownBadge.classList.remove("hidden");
    }

    this.countdownInterval = setInterval(() => {
      // If an independent newer drift event was triggered, cancel this countdown
      if (eventId && this.activeDriftEventId && eventId !== this.activeDriftEventId) {
        this.clearAlertResetTimers();
        return;
      }

      this.resetSecondsRemaining--;
      if (this.bannerCountdownSeconds) {
        this.bannerCountdownSeconds.textContent = Math.max(0, this.resetSecondsRemaining).toString();
      }

      if (this.resetSecondsRemaining <= 0) {
        this.clearAlertResetTimers();
        this.resetVisualAlerts(eventId);
      }
    }, 1000);
  }

  clearAlertResetTimers() {
    if (this.countdownInterval) {
      clearInterval(this.countdownInterval);
      this.countdownInterval = null;
    }
    if (this.alertResetTimer) {
      clearTimeout(this.alertResetTimer);
      this.alertResetTimer = null;
    }
  }

  resetVisualAlerts(eventId) {
    // Check if another drift event occurred in the meantime
    if (eventId && this.activeDriftEventId && eventId !== this.activeDriftEventId) {
      return;
    }

    // 1. Hide the banner and remove outcome styling
    if (this.multiCityBanner) {
      this.multiCityBanner.classList.add("hidden");
      this.multiCityBanner.classList.remove("outcome-approved", "outcome-rejected");
    }
    if (this.bannerCountdownBadge) {
      this.bannerCountdownBadge.classList.add("hidden");
    }

    // 2. Return sidebar multi-city badge to normal state
    if (this.multiCityBadge && this.multiCityStatusText) {
      this.multiCityBadge.className = "badge badge-green";
      this.multiCityStatusText.textContent = "4-City Gate: Normal";
    }

    // 3. Return drift badge to normal state
    if (this.driftBadge && this.driftStatusText) {
      this.driftBadge.className = "badge badge-green";
      this.driftStatusText.textContent = "Drift: Nominal";
    }

    // 4. Reset GIS Map spatial zone drift outlines back to standard boundaries
    if (window.mapController && typeof window.mapController.resetAllZones === "function") {
      window.mapController.resetAllZones();
    }

    // 5. Reset City Drift Table state
    Object.keys(this.cityStatus).forEach((zid) => {
      if (this.cityStatus[zid]) {
        this.cityStatus[zid].drift_detected = false;
        this.cityStatus[zid].current_psi = 0.0;
        this.cityStatus[zid].retrain_triggered = false;
      }
    });
    this.renderInitialCityDriftTable();

    this.appendAuditLog(
      "SYSTEM",
      "STATUS_RESET",
      "60-second post-decision window elapsed: Temporary alerts returned to normal. Event history and evaluation audit trail preserved."
    );

    this.activeDriftEventId = null;
  }

  handleModelReloadEvent(reload) {
    const prev = reload.previous_version;
    const active = reload.active_version;

    console.log(`[WSClient] Model Hot-Reload received: ${prev} -> ${active}`);
    this.updateModelVersion(active, true);

    // Refresh full performance telemetry & evaluation history
    this.fetchModelPerformance();

    this.appendAuditLog(
      "RELOAD",
      `Model ${active}`,
      `Zero-downtime hot-reload completed (${prev} -> ${active}). Deployed to active production traffic.`
    );

    // Update Drift Badge to Nominal
    if (this.driftBadge && this.driftStatusText) {
      this.driftBadge.className = "badge badge-green";
      this.driftStatusText.textContent = "Model Updated";
    }
  }

  updateModelVersion(versionStr, animated = false) {
    if (this.modelVersionText) {
      this.modelVersionText.textContent = `Model: ${versionStr}`;
    }
    if (this.cmVersion) {
      this.cmVersion.textContent = versionStr;
    }
    if (this.ovModelVersion) {
      this.ovModelVersion.textContent = `${versionStr} (Production)`;
    }

    if (this.modelBadge) {
      if (versionStr !== "v1") {
        this.modelBadge.className = "badge badge-green";
      } else {
        this.modelBadge.className = "badge badge-purple";
      }

      if (animated) {
        this.modelBadge.classList.add("pulse");
        setTimeout(() => this.modelBadge.classList.remove("pulse"), 4000);
      }
    }
  }

  appendAuditLog(eventType, entity, detail) {
    if (!this.auditLogBody) return;

    const timeStr = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    let tagClass = "tag-pred";
    if (eventType === "HAZARD") tagClass = "tag-hazard";
    else if (eventType === "DRIFT") tagClass = "tag-drift";
    else if (eventType === "RETRAIN") tagClass = "tag-retrain";
    else if (eventType === "RELOAD" || eventType === "PROMOTION") tagClass = "tag-reload";
    else if (eventType === "GATE_REJECTED" || eventType === "REJECT") tagClass = "tag-hazard";
    else if (eventType === "SYSTEM" || eventType === "STATUS_RESET" || eventType === "ALERT_RESET") tagClass = "tag-pred";

    const tr = document.createElement("tr");
    tr.className = "audit-row";
    tr.innerHTML = `
      <td class="font-mono">${timeStr}</td>
      <td><span class="audit-tag ${tagClass}">${eventType}</span></td>
      <td><strong>${entity}</strong></td>
      <td style="line-height: 1.3;">${detail}</td>
    `;

    // Prepend latest on top
    this.auditLogBody.insertBefore(tr, this.auditLogBody.firstChild);

    // Keep log trimmed to 60 rows
    while (this.auditLogBody.children.length > 60) {
      this.auditLogBody.removeChild(this.auditLogBody.lastChild);
    }
  }
}

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", async () => {
  if (window.mapController) {
    await window.mapController.init();
  }
  if (window.chartController) {
    await window.chartController.init();
  }
  if (window.performanceChartController) {
    window.performanceChartController.init();
  }

  window.wsClient = new DashboardWebSocketClient();
  await window.wsClient.init();
});
