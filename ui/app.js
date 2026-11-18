// PathSnap evaluator UI. Talks only to the API: GET /health, GET /trips, POST /predict.
(function () {
  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);

  // API address: ?api= in the URL, else the --api given to ui/serve.py, else localhost
  let API = (params.get("api") || window.PATHSNAP_API || "http://localhost:8000").replace(/\/$/, "");
  $("api-url").value = API;

  // ---------- map ----------
  const map = L.map("map", { preferCanvas: true }).setView([20, 0], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  const layers = { gps: L.layerGroup(), hmm: L.layerGroup(), clf: L.layerGroup() };
  layers.hmm.addTo(map); layers.gps.addTo(map);
  L.control.layers(null, {
    "Predicted route (HMM)": layers.hmm,
    "Classifier only": layers.clf,
    "GPS observations": layers.gps,
  }, { collapsed: false }).addTo(map);

  // join consecutive arcs that share an endpoint into one polyline
  function chains(route) {
    const out = [];
    let cur = null;
    for (const seg of route) {
      const [a, b] = seg.coords;
      if (cur && cur[cur.length - 1][0] === a[0] && cur[cur.length - 1][1] === a[1]) cur.push(b);
      else { cur = [a, b]; out.push(cur); }
    }
    return out;
  }

  // ---------- helpers ----------
  const pct = (x) => (100 * x).toFixed(2) + "%";
  function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = !msg; }
  async function getJSON(path, opts) {
    const r = await fetch(API + path, opts);
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status} ${await r.text()}`);
    return r.json();
  }

  // ---------- API status + trips ----------
  async function connect() {
    showError("");
    $("api-dot").className = "dot"; $("api-status").textContent = "Connecting to " + API + "…";
    try {
      const h = await getJSON("/health");
      $("api-dot").className = "dot ok";
      $("api-status").textContent = "API online";
      $("api-model").textContent = `${h.model} · ${h.features} features · HMM beta=${h.hmm.beta} m · ${h.trips} trips`;
      const trips = await getJSON("/trips");
      const box = $("trips"); box.innerHTML = "";
      for (const t of trips) {
        const b = document.createElement("button");
        b.className = "trip"; b.dataset.trip = t.trip_id;
        b.innerHTML = `<span class="score"></span><b>${t.trip_id}</b><span>${t.gps_points.toLocaleString()} GPS fixes</span>`;
        b.onclick = () => run(t.trip_id);
        box.appendChild(b);
      }
      if (params.get("trip")) run(params.get("trip"));
    } catch (err) {
      $("api-dot").className = "dot bad";
      $("api-status").textContent = "API not reachable";
      $("trips").innerHTML = '<div class="muted small">No trips: check the API address.</div>';
      showError(`Could not reach the API at ${API} (${err.message || err}). Check that it is running with ` +
                "--host 0.0.0.0, that both laptops are on the same network, and that port 8000 is allowed " +
                "through the firewall on Laptop 1.");
    }
  }
  $("api-save").onclick = () => { API = $("api-url").value.trim().replace(/\/$/, ""); connect(); };

  // ---------- predict ----------
  let busy = false;
  async function run(tid) {
    if (busy) return;
    busy = true; showError("");
    document.querySelectorAll(".trip").forEach((b) => b.classList.toggle("active", b.dataset.trip === tid));
    $("loading").hidden = false;
    try {
      const r = await getJSON("/predict", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trip_id: tid }),
      });
      draw(r);
    } catch (err) {
      showError(String(err.message || err));
    } finally {
      $("loading").hidden = true; busy = false;
    }
  }

  function draw(r) {
    Object.values(layers).forEach((g) => g.clearLayers());
    for (const [lat, lon] of r.gps) {
      L.circleMarker([lat, lon], { radius: 2.5, color: "#222", weight: 1, fillOpacity: 0.6 }).addTo(layers.gps);
    }
    for (const c of chains(r.classifier_route)) {
      L.polyline(c, { color: "#ef8f00", weight: 4, dashArray: "6 6", opacity: 0.9 }).addTo(layers.clf);
    }
    for (const c of chains(r.route)) {
      L.polyline(c, { color: "#1565c0", weight: 5, opacity: 0.85 }).addTo(layers.hmm);
    }
    if (r.gps.length) {
      L.marker(r.gps[0], { title: "start" }).bindTooltip("start").addTo(layers.gps);
      L.marker(r.gps[r.gps.length - 1], { title: "end" }).bindTooltip("end").addTo(layers.gps);
      map.fitBounds(L.latLngBounds(r.gps), { padding: [30, 30] });
    }

    const m = r.metrics, s = r.stats, t = r.timings;
    $("results").hidden = false;
    $("r-trip").textContent = r.trip_id;
    $("m-top1").textContent = pct(m.top1_hmm);
    $("m-clf").textContent = pct(m.top1_classifier);
    $("m-correct").textContent = `${m.correct_hmm} / ${m.observations_evaluated}`;
    $("m-cont").textContent = m.route_continuity.toFixed(4);
    $("m-prec").textContent = m.route_arc_precision.toFixed(4);
    $("m-rec").textContent = m.route_arc_recall.toFixed(4);
    $("m-miss").textContent = pct(m.candidate_miss_rate);
    $("m-obs").textContent = `${s.gps_fixes.toLocaleString()} / ${s.observations}`;
    $("m-restarts").textContent = s.chain_restarts;
    const rows = [["Load files", t.load_s], ["Features", t.features_s], ["Classifier", t.classifier_s],
                  ["HMM / Viterbi", t.hmm_s], ["Evaluation", t.evaluation_s], ["Total", t.total_s]];
    $("timings").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${(v * 1000).toFixed(0)} ms</td></tr>`).join("");
    const btn = document.querySelector(`.trip[data-trip="${r.trip_id}"] .score`);
    if (btn) btn.textContent = pct(m.top1_hmm);
  }

  connect();
})();
