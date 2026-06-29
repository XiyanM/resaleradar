/* =========================================================
   ResaleRadar — app.js
   ========================================================= */

"use strict";

const API_BASE = "http://127.0.0.1:8000";
const CURRENT_CPI = 102.814; //May 2026 data
const BASE_CPI = 102.052;

const TOWNS = [
  "ANG MO KIO",
  "BEDOK",
  "BISHAN",
  "BUKIT BATOK",
  "BUKIT MERAH",
  "BUKIT PANJANG",
  "BUKIT TIMAH",
  "CENTRAL AREA",
  "CHOA CHU KANG",
  "CLEMENTI",
  "GEYLANG",
  "HOUGANG",
  "JURONG EAST",
  "JURONG WEST",
  "KALLANG/WHAMPOA",
  "MARINE PARADE",
  "PASIR RIS",
  "PUNGGOL",
  "QUEENSTOWN",
  "SEMBAWANG",
  "SENGKANG",
  "SERANGOON",
  "TAMPINES",
  "TOA PAYOH",
  "WOODLANDS",
  "YISHUN",
];

const FLAT_MODELS = [
  "Improved",
  "New Generation",
  "DBSS",
  "Standard",
  "Apartment",
  "Simplified",
  "Model A",
  "Premium Apartment",
  "Adjoined flat",
  "Model A-Maisonette",
  "Maisonette",
  "Type S1",
  "Type S2",
  "Model A2",
  "Terrace",
  "Improved-Maisonette",
  "Premium Maisonette",
  "Multi Generation",
  "Premium Apartment Loft",
  "2-room",
  "3Gen",
];

let lastPrediction = null;
let lastPayload = null;
let leafletMap = null;
let flatMarker = null;

function initScrollAnimations() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.12,
      rootMargin: "0px 0px -40px 0px",
    }
  );

  document.querySelectorAll(".fade-up").forEach((el) => observer.observe(el));
}

function initDropdowns() {
  const townSelect = document.getElementById("town");
  TOWNS.forEach((t) => {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = toTitleCase(t);
    townSelect.appendChild(opt);
  });

  const modelSelect = document.getElementById("flat_model");
  FLAT_MODELS.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    modelSelect.appendChild(opt);
  });
}

function toTitleCase(str) {
  return str.toLowerCase().replace(/(?:^|\s|\/)\S/g, (c) => c.toUpperCase());
}

function storeyMidpoint(rangeStr) {
  const parts = rangeStr.split(" TO ");
  if (parts.length !== 2) return null;
  const lo = parseInt(parts[0], 10);
  const hi = parseInt(parts[1], 10);
  if (isNaN(lo) || isNaN(hi)) return null;
  return (lo + hi) / 2;
}

function remainingLease(leaseStart, txYear) {
  return 99 - (txYear - leaseStart);
}

async function geocodeAddress(address) {
  const url = `https://www.onemap.gov.sg/api/common/elastic/search?searchVal=${encodeURIComponent(
    address
  )}&returnGeom=Y&getAddrDetails=Y&pageNum=1`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("OneMap request failed");
  const data = await res.json();
  if (!data.results || data.results.length === 0) {
    throw new Error(
      'Address not found. Try a more specific address, e.g. "123 Ang Mo Kio Ave 3".'
    );
  }
  const top = data.results[0];
  return { lat: parseFloat(top.LATITUDE), lon: parseFloat(top.LONGITUDE) };
}

function buildPayload(lat, lon) {
  const txMonth = document.getElementById("transaction_month").value;
  if (!txMonth) throw new Error("Transaction month is required.");
  const [txYear, txMonthNum] = txMonth.split("-").map(Number);

  const leaseStart = parseInt(document.getElementById("lease_start").value, 10);
  if (isNaN(leaseStart))
    throw new Error("Lease commencement year is required.");

  const storeyRange = document.getElementById("storey").value;
  if (!storeyRange) throw new Error("Storey range is required.");
  const storey_midpoint = storeyMidpoint(storeyRange);

  const floor_area_sqm = parseFloat(
    document.getElementById("floor_area").value
  );
  if (isNaN(floor_area_sqm)) throw new Error("Floor area is required.");

  const town = document.getElementById("town").value;
  if (!town) throw new Error("Town is required.");

  const flat_type = document.getElementById("flat_type").value;
  if (!flat_type) throw new Error("Flat type is required.");

  const flat_model = document.getElementById("flat_model").value;
  if (!flat_model) throw new Error("Flat model is required.");

  const remaining_lease_years = remainingLease(leaseStart, txYear);
  if (remaining_lease_years < 0 || remaining_lease_years > 99) {
    throw new Error(
      "Remaining lease is out of range. Check lease commencement year."
    );
  }

  return {
    town,
    flat_type,
    flat_model,
    storey_midpoint,
    floor_area_sqm,
    lease_commence_date: leaseStart,
    transaction_year: txYear,
    transaction_month: txMonthNum,
    lat,
    lon,
  };
}

function animatePrice(targetValue) {
  const el = document.getElementById("result-price");
  const duration = 850;
  const start = performance.now();

  function step(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(targetValue * eased);
    el.textContent = "SGD " + current.toLocaleString("en-SG");
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function renderSHAP(shapValues) {
  const entries = Object.entries(shapValues)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 10)
    .reverse();

  const labels = entries.map(([k]) => formatFeatureName(k));
  const values = entries.map(([, v]) => Math.round(v));
  const colors = values.map((v) => (v >= 0 ? "#00A896" : "#C0392B"));

  const trace = {
    type: "bar",
    orientation: "h",
    x: values,
    y: labels,
    marker: { color: colors },
    hovertemplate: "<b>%{y}</b><br>SGD %{x:+,.0f}<extra></extra>",
    hoverlabel: {
      bgcolor: "#0F1923",
      bordercolor: "#0F1923",
      font: { family: "Inter", size: 13, color: "#ffffff" },
      align: "left",
    },
  };

  const layout = {
    margin: { t: 8, b: 44, l: 180, r: 24 },
    xaxis: {
      title: {
        text: "SHAP value (SGD)",
        font: { size: 12, family: "Inter", color: "#6B7A8D" },
      },
      tickfont: { size: 12, family: "Inter", color: "#6B7A8D" },
      gridcolor: "#E2E8EF",
      zeroline: true,
      zerolinecolor: "#CBD5DF",
      zerolinewidth: 1.5,
    },
    yaxis: {
      tickfont: { size: 12, family: "Inter", color: "#0F1923" },
      automargin: true,
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    height: 320,
    font: { family: "Inter", color: "#0F1923" },
    showlegend: false,
    bargap: 0.35,
  };

  Plotly.react("shap-chart", [trace], layout, {
    responsive: true,
    displayModeBar: false,
  });
}

function formatFeatureName(name) {
  const map = {
    floor_area_sqm: "Floor area (sqm)",
    storey_midpoint: "Storey",
    remaining_lease_years: "Remaining lease (yrs)",
    dist_to_cbd: "Distance to CBD",
    dist_nearest_mrt: "Dist. to nearest MRT",
    dist_nearest_school: "Dist. to nearest school",
    dist_nearest_mall: "Dist. to nearest mall",
    dist_nearest_hawker: "Dist. to nearest hawker",
    dist_nearest_bus_stop: "Dist. to nearest bus stop",
    dist_nearest_expressway: "Dist. to expressway",
    num_mrt_within_1km: "MRT within 1km",
    num_mrt_within_2km: "MRT within 2km",
    num_schools_within_1km: "Schools within 1km",
    num_malls_within_2km: "Malls within 2km",
    num_hawkers_within_500m: "Hawkers within 500m",
    num_bus_stops_within_300m: "Bus stops within 300m",
    num_primary_schools_within_1km: "Primary schools within 1km",
    dist_nearest_primary_school: "Dist. to nearest primary school",
    is_mature_estate: "Mature estate",
    transaction_year: "Transaction year",
    transaction_month: "Transaction month",
    lease_commence_date: "Lease commencement",
    flat_type_encoded: "Flat type",
  };
  if (map[name]) return map[name];
  if (name.startsWith("town_"))
    return "Town: " + toTitleCase(name.replace("town_", "").replace(/_/g, " "));
  if (name.startsWith("flat_model_"))
    return "Model: " + name.replace("flat_model_", "").replace(/_/g, " ");
  return name.replace(/_/g, " ");
}

function initOrUpdateMap(lat, lon, nearestAmenities) {
  if (!leafletMap) {
    leafletMap = L.map("map", {
      zoomControl: true,
      scrollWheelZoom: false,
    }).setView([lat, lon], 15);
    L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
      {
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: "abcd",
        maxZoom: 19,
      }
    ).addTo(leafletMap);
  } else {
    leafletMap.setView([lat, lon], 15);
    if (flatMarker) leafletMap.removeLayer(flatMarker);
    if (leafletMap._amenityMarkers) {
      leafletMap._amenityMarkers.forEach((m) => leafletMap.removeLayer(m));
    }
  }

  // Flat marker
  const flatIcon = L.divIcon({
    className: "",
    html: `<div style="width:16px;height:16px;border-radius:50%;background:#00A896;border:2.5px solid #fff;box-shadow:0 0 0 1.5px #00A896;cursor:pointer;"></div>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8],
  });
  flatMarker = L.marker([lat, lon], { icon: flatIcon })
    .bindPopup("<strong>Selected flat</strong>")
    .addTo(leafletMap);

  // Amenity markers
  const amenityConfigs = {
    mrt: { color: "#E74C3C", label: "MRT" },
    school: { color: "#3498DB", label: "School" },
    hawker: { color: "#F39C12", label: "Hawker" },
    mall: { color: "#9B59B6", label: "Mall" },
  };

  leafletMap._amenityMarkers = [];

  if (nearestAmenities) {
    Object.entries(amenityConfigs).forEach(([type, config]) => {
      const amenity = nearestAmenities[type];
      if (!amenity || amenity.lat === null) return;

      const icon = L.divIcon({
        className: "",
        html: `<div style="width:16px;height:16px;border-radius:50%;background:${config.color};border:2px solid #fff;box-shadow:0 0 0 1.5px ${config.color};cursor:pointer;"></div>`,
        iconSize: [16, 16],
        iconAnchor: [8, 8],
      });

      const marker = L.marker([amenity.lat, amenity.lon], { icon })
        .bindPopup(`<strong>${config.label}</strong><br>${amenity.name}`)
        .addTo(leafletMap);

      leafletMap._amenityMarkers.push(marker);
    });
  }

  // Fit map to show all markers
  const allPoints = [
    [lat, lon],
    ...Object.values(nearestAmenities || {})
      .filter((a) => a && a.lat !== null)
      .map((a) => [a.lat, a.lon]),
  ];
  if (allPoints.length > 1) {
    leafletMap.fitBounds(allPoints, { padding: [30, 30], maxZoom: 16 });
  }
}

async function runPrediction(overridePayload = null) {
  const btn = document.getElementById("predict-btn");
  const btnText = document.getElementById("predict-btn-text");
  const spinner = document.getElementById("predict-spinner");
  const errorDiv = document.getElementById("error-msg");

  errorDiv.classList.add("hidden");
  errorDiv.textContent = "";
  btn.disabled = true;
  btnText.textContent = "Predicting…";
  spinner.classList.remove("hidden");

  try {
    let payload;
    let lat, lon;

    if (overridePayload) {
      payload = { ...lastPayload, ...overridePayload };
      lat = payload.lat;
      lon = payload.lon;
    } else {
      const address = document.getElementById("address").value.trim();
      if (!address) throw new Error("Address is required.");
      const coords = await geocodeAddress(address);
      lat = coords.lat;
      lon = coords.lon;
      payload = buildPayload(lat, lon);
      lastPayload = payload;
    }

    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    lastPrediction = data;
    displayResults(data, lat, lon, payload);
  } catch (err) {
    errorDiv.textContent = err.message;
    errorDiv.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btnText.textContent = "Predict price";
    spinner.classList.add("hidden");
  }
}

function displayResults(data, lat, lon, payload) {
  const factor = CURRENT_CPI / BASE_CPI;
  const price = Math.round(data.predicted_price * factor);
  const lower = Math.round(data.lower_bound * factor);
  const upper = Math.round(data.upper_bound * factor);

  document.getElementById("empty-state").classList.add("hidden");
  document.getElementById("results").classList.remove("hidden");

  animatePrice(price);

  document.getElementById(
    "result-interval"
  ).innerHTML = `80% confidence interval: <strong>SGD ${lower.toLocaleString(
    "en-SG"
  )}</strong> — <strong>SGD ${upper.toLocaleString("en-SG")}</strong>`;

  const warn = document.getElementById("crossing-warning");
  data.quantile_crossing
    ? warn.classList.remove("hidden")
    : warn.classList.add("hidden");

  syncSliders(payload);
  if (data.shap_values) renderSHAP(data.shap_values);
  initOrUpdateMap(lat, lon, data.nearest_amenities);
}

function syncSliders(payload) {
  const set = (id, valId, val) => {
    const el = document.getElementById(id);
    if (el) el.value = val;
    const label = document.getElementById(valId);
    if (label) label.textContent = val;
  };

  set("slider-floor-area", "val-floor-area", payload.floor_area_sqm);
  set("slider-storey", "val-storey", payload.storey_midpoint);

  if (lastPrediction && lastPrediction.feature_values) {
    const fv = lastPrediction.feature_values;
    set("slider-mrt", "val-mrt", fv.dist_nearest_mrt.toFixed(2));
    set("slider-cbd", "val-cbd", fv.dist_to_cbd.toFixed(1));
  }
}

function onSliderChange() {
  const fa = document.getElementById("slider-floor-area").value;
  const st = document.getElementById("slider-storey").value;
  const ls = document.getElementById("slider-lease").value;
  const mrt = parseFloat(document.getElementById("slider-mrt").value).toFixed(
    2
  );
  const cbd = parseFloat(document.getElementById("slider-cbd").value).toFixed(
    1
  );

  document.getElementById("val-floor-area").textContent = fa;
  document.getElementById("val-storey").textContent = st;
  document.getElementById("val-lease").textContent = ls;
  document.getElementById("val-mrt").textContent = mrt;
  document.getElementById("val-cbd").textContent = cbd;

  clearTimeout(onSliderChange._timer);
  onSliderChange._timer = setTimeout(() => {
    if (!lastPayload) return;
    runPrediction({
      floor_area_sqm: parseFloat(fa),
      storey_midpoint: parseFloat(st),
      remaining_lease_years: parseFloat(ls),
      feature_overrides: {
        dist_nearest_mrt: parseFloat(mrt),
        dist_to_cbd: parseFloat(cbd),
      },
    });
  }, 400);
}

const TOWN_MEDIANS = {
  Queenstown: 785000,
  "Bukit Merah": 745000,
  Bishan: 720000,
  "Toa Payoh": 700000,
  "Marine Parade": 698000,
  "Central Area": 690000,
  "Kallang/Whampoa": 680000,
  Clementi: 658000,
  "Bukit Timah": 640000,
  Serangoon: 610000,
  "Ang Mo Kio": 598000,
  Geylang: 580000,
  Tampines: 568000,
  Bedok: 558000,
  Hougang: 548000,
  "Pasir Ris": 540000,
  "Jurong East": 532000,
  Punggol: 525000,
  Sengkang: 518000,
  "Bukit Panjang": 515000,
  Yishun: 505000,
  "Bukit Batok": 495000,
  Woodlands: 480000,
  "Choa Chu Kang": 475000,
  "Jurong West": 468000,
  Sembawang: 455000,
};

const TREND_DATA = {
  labels: [
    "2017-01",
    "2017-07",
    "2018-01",
    "2018-07",
    "2019-01",
    "2019-07",
    "2020-01",
    "2020-07",
    "2021-01",
    "2021-07",
    "2022-01",
    "2022-07",
    "2023-01",
    "2023-07",
    "2024-01",
    "2024-07",
    "2025-01",
    "2025-07",
    "2026-01",
  ],
  values: [
    420000, 418000, 415000, 412000, 418000, 422000, 428000, 435000, 460000,
    510000, 565000, 610000, 645000, 660000, 665000, 668000, 672000, 678000,
    685000,
  ],
};

const HOVER_LABEL = {
  bgcolor: "#0F1923",
  bordercolor: "#0F1923",
  font: { family: "Inter", size: 13, color: "#ffffff" },
  align: "left",
};

function renderExplorerCharts() {
  const towns = Object.keys(TOWN_MEDIANS).sort(
    (a, b) => TOWN_MEDIANS[b] - TOWN_MEDIANS[a]
  );
  const prices = towns.map((t) => TOWN_MEDIANS[t]);
  const maxPrice = Math.max(...prices);

  Plotly.newPlot(
    "town-chart",
    [
      {
        type: "bar",
        x: prices,
        y: towns,
        orientation: "h",
        marker: { color: "#00A896" },
        hovertemplate: "<b>%{y}</b><br>SGD %{x:,.0f}<extra></extra>",
        hoverlabel: HOVER_LABEL,
      },
    ],
    {
      margin: { t: 8, b: 48, l: 148, r: 24 },
      xaxis: {
        range: [0, maxPrice * 1.05],
        tickfont: { size: 12, family: "Inter", color: "#6B7A8D" },
        gridcolor: "#E2E8EF",
        tickformat: ",.0f",
        title: {
          text: "Median resale price (SGD)",
          font: { size: 12, family: "Inter", color: "#6B7A8D" },
        },
      },
      yaxis: {
        tickfont: { size: 12, family: "Inter", color: "#0F1923" },
        automargin: true,
        ticklabelposition: "outside left",
        ticksuffix: "  ",
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      height: 460,
      font: { family: "Inter", color: "#0F1923" },
      showlegend: false,
      bargap: 0.3,
    },
    { responsive: true, displayModeBar: false }
  );

  Plotly.newPlot(
    "trend-chart",
    [
      {
        type: "scatter",
        mode: "lines",
        x: TREND_DATA.labels,
        y: TREND_DATA.values,
        line: { color: "#00A896", width: 2.5 },
        hovertemplate: "<b>%{x}</b><br>SGD %{y:,.0f}<extra></extra>",
        hoverlabel: HOVER_LABEL,
      },
    ],
    {
      margin: { t: 8, b: 48, l: 80, r: 24 },
      xaxis: {
        tickfont: { size: 12, family: "Inter", color: "#6B7A8D" },
        gridcolor: "#E2E8EF",
      },
      yaxis: {
        tickfont: { size: 12, family: "Inter", color: "#6B7A8D" },
        tickformat: ",.0f",
        gridcolor: "#E2E8EF",
        title: {
          text: "Median resale price (SGD)",
          font: { size: 12, family: "Inter", color: "#6B7A8D" },
        },
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      height: 460,
      font: { family: "Inter", color: "#0F1923" },
      showlegend: false,
    },
    { responsive: true, displayModeBar: false }
  );
}

document.addEventListener("DOMContentLoaded", () => {
  initDropdowns();
  renderExplorerCharts();
  initScrollAnimations();

  const now = new Date();
  const yyyy = now.getFullYear();
  const mm = String(now.getMonth() + 1).padStart(2, "0");
  document.getElementById("transaction_month").value = `${yyyy}-${mm}`;
});
