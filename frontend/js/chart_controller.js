/**
 * Chart.js Controllers for:
 * 1. ChartController: Live Spatial-Temporal PSI Drift Monitoring (#drift-chart).
 * 2. PerformanceChartController: Champion vs. Challenger Model Metrics Comparison (#performance-chart).
 *
 * Keeps PSI monitoring and model performance visualizations strictly separate.
 */

// ============================================================================
// 1. Live Spatial-Temporal PSI Drift Monitor
// ============================================================================
class ChartController {
  constructor() {
    this.chart = null;
    this.maxPoints = 30;
    this.psiThreshold = 0.20;

    this.zoneColors = {
      ZONE_KODAGU: "#10b981",          // Emerald
      ZONE_DAKSHINA_KANNADA: "#3b82f6", // Blue
      ZONE_CHIKMAGALUR: "#a855f7",     // Purple
      ZONE_SHIMOGA_AGUMBE: "#f97316",  // Orange
      ZONE_UTTARA_KANNADA: "#06b6d4",  // Cyan
      ZONE_WAYANAD_NILGIRIS: "#eab308",// Yellow
    };

    this.labels = [];
    this.zoneData = {};
    Object.keys(this.zoneColors).forEach((z) => {
      this.zoneData[z] = [];
    });
  }

  async init() {
    // 1. Fetch threshold dynamically from backend API
    try {
      const resp = await fetch("/api/zones");
      if (resp.ok) {
        const data = await resp.json();
        if (data.psi_threshold) {
          this.psiThreshold = data.psi_threshold;
          const badge = document.getElementById("psi-threshold-badge");
          if (badge) {
            badge.textContent = `Threshold: ${this.psiThreshold.toFixed(2)}`;
          }
        }
      }
    } catch (e) {
      console.warn("[ChartController] Using default PSI threshold 0.20", e);
    }

    // 2. Build Chart.js Datasets
    const datasets = Object.keys(this.zoneColors).map((zoneId) => ({
      label: zoneId.replace("ZONE_", "").replace(/_/g, " "),
      data: this.zoneData[zoneId],
      borderColor: this.zoneColors[zoneId],
      backgroundColor: this.zoneColors[zoneId],
      borderWidth: 2,
      tension: 0.35,
      pointRadius: 3,
      pointHoverRadius: 5,
    }));

    // Add Configured PSI Threshold Reference Line Dataset
    datasets.push({
      label: `PSI Threshold (${this.psiThreshold.toFixed(2)})`,
      data: [],
      borderColor: "#ef4444",
      borderWidth: 2,
      borderDash: [6, 4],
      pointRadius: 0,
      fill: false,
    });

    const canvasEl = document.getElementById("drift-chart");
    if (!canvasEl) return;
    const ctx = canvasEl.getContext("2d");

    this.chart = new Chart(ctx, {
      type: "line",
      data: {
        labels: this.labels,
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              boxWidth: 10,
              boxHeight: 10,
              font: { size: 9, family: "-apple-system, sans-serif" },
              color: "#94a3b8",
            },
          },
          tooltip: {
            mode: "index",
            intersect: false,
            backgroundColor: "#0f172a",
            titleColor: "#94a3b8",
            bodyColor: "#f8fafc",
            borderColor: "#334155",
            borderWidth: 1,
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(34, 48, 73, 0.4)" },
            ticks: { color: "#64748b", font: { size: 9 }, maxTicksLimit: 8 },
          },
          y: {
            min: 0,
            suggestedMax: 0.4,
            grid: { color: "rgba(34, 48, 73, 0.4)" },
            ticks: { color: "#64748b", font: { size: 9 } },
          },
        },
      },
    });

    console.log("[ChartController] Initialized live PSI drift chart.");
  }

  pushDriftPoint(driftAlert) {
    if (!this.chart || !driftAlert.zone_id) return;

    const zoneId = driftAlert.zone_id;
    const cleanLabel = zoneId.replace("ZONE_", "").replace(/_/g, " ");
    const psiVal = parseFloat(driftAlert.max_psi || 0.0);
    const timeLabel = new Date(driftAlert.timestamp || Date.now()).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    // Append label
    this.labels.push(timeLabel);
    if (this.labels.length > this.maxPoints) {
      this.labels.shift();
    }

    // Append threshold value
    const thresholdDataset = this.chart.data.datasets.find((d) => d.label.includes("Threshold"));
    if (thresholdDataset) {
      thresholdDataset.data.push(this.psiThreshold);
      if (thresholdDataset.data.length > this.maxPoints) {
        thresholdDataset.data.shift();
      }
    }

    // Append zone data
    this.chart.data.datasets.forEach((ds) => {
      if (ds.label === cleanLabel) {
        ds.data.push(psiVal);
        if (ds.data.length > this.maxPoints) {
          ds.data.shift();
        }
      } else if (!ds.label.includes("Threshold")) {
        // Pad with last known value or 0 to maintain dataset synchronization
        const lastVal = ds.data.length > 0 ? ds.data[ds.data.length - 1] : 0.0;
        ds.data.push(lastVal);
        if (ds.data.length > this.maxPoints) {
          ds.data.shift();
        }
      }
    });

    this.chart.update("none");
  }
}

// ============================================================================
// 2. Champion vs. Challenger Model Performance Comparison Chart
// ============================================================================
class PerformanceChartController {
  constructor() {
    this.chart = null;
    this.metricLabels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"];
  }

  init() {
    const canvasEl = document.getElementById("performance-chart");
    if (!canvasEl) return;
    const ctx = canvasEl.getContext("2d");

    this.chart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: this.metricLabels,
        datasets: [
          {
            label: "Champion Model",
            data: [0, 0, 0, 0, 0],
            backgroundColor: "rgba(59, 130, 246, 0.7)", // Blue
            borderColor: "#3b82f6",
            borderWidth: 1.5,
            borderRadius: 4,
          },
          {
            label: "Challenger Model",
            data: [0, 0, 0, 0, 0],
            backgroundColor: "rgba(139, 92, 246, 0.7)", // Purple
            borderColor: "#8b5cf6",
            borderWidth: 1.5,
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              boxWidth: 12,
              boxHeight: 12,
              font: { size: 10, family: "-apple-system, sans-serif" },
              color: "#94a3b8",
            },
          },
          tooltip: {
            mode: "index",
            intersect: false,
            backgroundColor: "#0f172a",
            titleColor: "#94a3b8",
            bodyColor: "#f8fafc",
            borderColor: "#334155",
            borderWidth: 1,
            callbacks: {
              label: (item) => {
                const val = item.raw;
                return `${item.dataset.label}: ${val !== null && val !== undefined ? (val * 100).toFixed(2) + '%' : 'N/A'}`;
              },
            },
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(34, 48, 73, 0.3)" },
            ticks: { color: "#94a3b8", font: { size: 10 } },
          },
          y: {
            min: 0.0,
            max: 1.0,
            grid: { color: "rgba(34, 48, 73, 0.3)" },
            ticks: {
              color: "#64748b",
              font: { size: 9 },
              callback: (v) => `${Math.round(v * 100)}%`,
            },
          },
        },
      },
    });

    console.log("[PerformanceChartController] Initialized model performance comparison chart.");
  }

  updateComparison(latestEval) {
    if (!this.chart || !latestEval) return;

    const champ = latestEval.champion_metrics || {};
    const chall = latestEval.challenger_metrics || {};

    const champVersion = latestEval.champion_version || "Champion";
    const challVersion = latestEval.challenger_version || "Challenger";

    const champVals = [
      champ.accuracy ?? null,
      champ.precision ?? null,
      champ.recall ?? null,
      champ.f1_score ?? null,
      champ.roc_auc ?? null,
    ];

    const challVals = [
      chall.accuracy ?? null,
      chall.precision ?? null,
      chall.recall ?? null,
      chall.f1_score ?? null,
      chall.roc_auc ?? null,
    ];

    // Update labels and data
    this.chart.data.datasets[0].label = `Champion (${champVersion})`;
    this.chart.data.datasets[0].data = champVals;

    if (latestEval.challenger_metrics) {
      this.chart.data.datasets[1].label = `Challenger (${challVersion})`;
      this.chart.data.datasets[1].data = challVals;
      this.chart.data.datasets[1].hidden = false;
    } else {
      this.chart.data.datasets[1].label = `Challenger (None)`;
      this.chart.data.datasets[1].data = [0, 0, 0, 0, 0];
      this.chart.data.datasets[1].hidden = true;
    }

    const subEl = document.getElementById("perf-chart-subtitle");
    if (subEl) {
      if (latestEval.challenger_metrics) {
        const delta = ((latestEval.f1_improvement || 0) * 100).toFixed(2);
        const sign = delta >= 0 ? "+" : "";
        subEl.textContent = `Latest Evaluation: ${champVersion} vs. ${challVersion} (F1 Delta: ${sign}${delta}% • Status: ${latestEval.status || 'EVALUATED'})`;
      } else {
        subEl.textContent = `Baseline Model ${champVersion} Performance Metrics (No challenger yet evaluated)`;
      }
    }

    this.chart.update();
  }
}

// Global instances
window.chartController = new ChartController();
window.performanceChartController = new PerformanceChartController();
