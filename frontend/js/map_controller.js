/**
 * Leaflet GIS Map Controller for Western Ghats & Karnataka Sensor Stations.
 *
 * Manages Leaflet canvas, draws bounding spatial zone polygons from /api/zones,
 * updates color-coded sensor prediction markers, and highlights drifted spatial zones.
 */

class MapController {
  constructor() {
    this.map = null;
    this.zoneLayers = {};
    this.sensorMarkers = {};
    this.defaultCenter = [13.25, 75.60];
    this.defaultZoom = 8;
    this.zonesMeta = {};
  }

  async init() {
    // 1. Initialize Leaflet Map
    const mapEl = document.getElementById("gis-map");
    if (!mapEl) return;

    this.map = L.map("gis-map", {
      center: this.defaultCenter,
      zoom: this.defaultZoom,
      zoomControl: true,
      attributionControl: false,
    });

    // 2. Add CartoDB DarkMatter Raster Tiles (clean dark-mode cartography)
    L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}{r}.png?key=cb1_5g81_1_f6ed0764db2a05dd16ded3e0",
      {
        maxZoom: 18,
        subdomains: "abcd",
      }
    ).addTo(this.map);

    // 3. Fetch and Render Spatial Zones from API
    await this.loadSpatialZones();
  }

  async loadSpatialZones() {
    try {
      const response = await fetch("/api/zones");
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} loading /api/zones`);
      }
      const data = await response.json();
      const zones = data.zones || [];

      zones.forEach((z) => {
        this.zonesMeta[z.zone_id] = z;

        const bounds = [
          [z.min_lat, z.min_lon],
          [z.max_lat, z.max_lon],
        ];

        const rect = L.rectangle(bounds, {
          color: "#3b82f6",
          weight: 1.5,
          fillColor: "#3b82f6",
          fillOpacity: 0.08,
          dashArray: null,
        }).addTo(this.map);

        rect.bindTooltip(
          `<strong>${z.name}</strong><br><span style="font-size: 10px; color: #94a3b8;">${z.zone_id}</span>`,
          { permanent: false, direction: "center", className: "zone-tooltip" }
        );

        this.zoneLayers[z.zone_id] = rect;
      });

      console.log(`[MapController] Loaded ${zones.length} spatial zone boundaries.`);
    } catch (err) {
      console.error("[MapController] Failed to load spatial zones:", err);
    }
  }

  updateSensorMarker(prediction) {
    if (!this.map || !prediction.lat || !prediction.lon) return;

    const lat = prediction.lat;
    const lon = prediction.lon;
    const zoneId = prediction.zone_id || "UNKNOWN";
    const markerKey = `${zoneId}_${lat.toFixed(2)}_${lon.toFixed(2)}`;
    const prob = prediction.hazard_probability || 0.0;
    const label = prediction.predicted_label === 1 ? "SEVERE HAZARD" : "NOMINAL";

    // Color mapping based on probability thresholds
    let color = "#10b981"; // Green (< 0.40)
    let riskLevel = "Low Risk";
    if (prob >= 0.70) {
      color = "#ef4444";   // Red (>= 0.70)
      riskLevel = "Severe Hazard";
    } else if (prob >= 0.40) {
      color = "#f59e0b";   // Amber (0.40 - 0.70)
      riskLevel = "Moderate Hazard";
    }

    const zoneName = this.zonesMeta[zoneId]?.name || zoneId.replace("ZONE_", "").replace(/_/g, " ");

    const popupHtml = `
      <div style="font-family: -apple-system, sans-serif; font-size: 12px; color: #0f172a; min-width: 180px;">
        <div style="font-weight: 700; font-size: 13px; color: #1e293b; margin-bottom: 4px;">
          ${zoneName}
        </div>
        <div>Coords: <strong>${lat.toFixed(3)}, ${lon.toFixed(3)}</strong></div>
        <div>Hazard Risk: <strong style="color: ${color};">${(prob * 100).toFixed(1)}%</strong></div>
        <div>Status: <strong>${label}</strong> (${riskLevel})</div>
        <div style="margin-top: 4px; font-size: 10px; color: #64748b;">
          Model: <code>${prediction.model_version || 'v1'}</code> | ${prediction.inference_latency_ms || 0} ms
        </div>
      </div>
    `;

    if (this.sensorMarkers[markerKey]) {
      const marker = this.sensorMarkers[markerKey];
      marker.setLatLng([lat, lon]);
      marker.setStyle({ fillColor: color });
      marker.setPopupContent(popupHtml);
    } else {
      const marker = L.circleMarker([lat, lon], {
        radius: 7,
        fillColor: color,
        color: "#ffffff",
        weight: 1.5,
        opacity: 0.9,
        fillOpacity: 0.85,
      }).addTo(this.map);

      marker.bindPopup(popupHtml);
      this.sensorMarkers[markerKey] = marker;
    }
  }

  highlightZoneDrift(zoneId, isDrifted) {
    const rect = this.zoneLayers[zoneId];
    if (!rect) return;

    if (isDrifted) {
      rect.setStyle({
        color: "#ef4444",
        weight: 2.5,
        dashArray: "6, 6",
        fillColor: "#ef4444",
        fillOpacity: 0.22,
      });
    } else {
      rect.setStyle({
        color: "#3b82f6",
        weight: 1.5,
        dashArray: null,
        fillColor: "#3b82f6",
        fillOpacity: 0.08,
      });
    }
  }

  resetAllZones() {
    Object.keys(this.zoneLayers).forEach((zid) => {
      this.highlightZoneDrift(zid, false);
    });
  }
}

// Global instance
window.mapController = new MapController();
