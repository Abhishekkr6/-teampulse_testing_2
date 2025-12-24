const routingTableBody = document.querySelector("#routing-table tbody");
const routeTemplate = document.getElementById("route-template");
const routeControlsTemplate = document.getElementById("route-controls-template");
const spawnRouteButton = document.getElementById("spawn-route");
const reseedChartButton = document.getElementById("reseed-chart");
const missionClock = document.getElementById("mission-clock");
const diagnosticConsole = document.getElementById("diagnostic-console");
const composerForm = document.getElementById("composer-form");
const queueList = document.getElementById("transmission-queue");
const flushQueueButton = document.getElementById("flush-queue");
const alertFeed = document.getElementById("alert-feed");
const alertTemplate = document.getElementById("alert-template");
const ackAlertsButton = document.getElementById("ack-alerts");
const noiseCanvas = document.getElementById("noise-canvas");
const noiseCtx = noiseCanvas.getContext("2d");

const ROUTE_NAMES = [
    "Lyra-West",
    "Cygnus-Ring",
    "Polaris-Gate",
    "Auriga-Node",
    "Cassio Relay",
    "Vela Array",
    "Eridanus Spoke",
    "Delphi Crown"
];

const CHANNELS = ["Ka", "Ku", "X", "S"];

const STATUS_POOL = [
    { label: "Nominal", color: "#4ade80" },
    { label: "Watch", color: "#f4b942" },
    { label: "Critical", color: "#ff6b6b" }
];

const ALERT_MESSAGES = [
    { type: "info", message: "Passive scan complete. Noise floor within tolerance." },
    { type: "warn", message: "Telemetry variance detected on outer band channels." },
    { type: "critical", message: "Relay Lyra-West reporting burst packet loss." },
    { type: "info", message: "Orbit alignment window opens in 12 minutes." },
    { type: "warn", message: "Spectrum analyzer exceeds baseline by 18%." }
];

const routes = [];
const transmissions = [];
const alerts = [];
const bootTime = Date.now();

class Diagnostic {
    static write(message) {
        const stamp = new Date().toLocaleTimeString();
        const line = document.createElement("p");
        line.className = "log-line";
        line.innerHTML = `<span class="log-line__time">[${stamp}]</span> ${message}`;
        diagnosticConsole.appendChild(line);
        diagnosticConsole.scrollTop = diagnosticConsole.scrollHeight;
    }
}

function pickRandom(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
}

function createRoute() {
    const baseLatency = (Math.random() * 120 + 20).toFixed(1);
    return {
        relay: pickRandom(ROUTE_NAMES),
        channel: pickRandom(CHANNELS),
        latency: `${baseLatency} ms`,
        status: pickRandom(STATUS_POOL)
    };
}

function renderRoutes() {
    routingTableBody.innerHTML = "";
    routes.forEach((route, index) => {
        const fragment = routeTemplate.content.cloneNode(true);
        fragment.querySelector('[data-cell="relay"]').textContent = route.relay;
        fragment.querySelector('[data-cell="channel"]').textContent = route.channel;
        fragment.querySelector('[data-cell="latency"]').textContent = route.latency;

        const statusCell = fragment.querySelector('[data-cell="status"]');
        statusCell.textContent = route.status.label;
        statusCell.style.color = route.status.color;

        const controlsFragment = routeControlsTemplate.content.cloneNode(true);
        const controlsCell = fragment.querySelector('[data-cell="controls"]');
        controlsCell.appendChild(controlsFragment);

        controlsCell.querySelector('[data-action="boost"]').addEventListener("click", () => boostRoute(index));
        controlsCell.querySelector('[data-action="halt"]').addEventListener("click", () => haltRoute(index));

        routingTableBody.appendChild(fragment);
    });
}

function boostRoute(index) {
    const route = routes[index];
    route.latency = `${Math.max(3, Number.parseFloat(route.latency) * 0.65).toFixed(1)} ms`;
    route.status = STATUS_POOL[0];
    renderRoutes();
    Diagnostic.write(`Boost applied to ${route.relay} ${route.channel}, latency trimmed.`);
}

function haltRoute(index) {
    const route = routes[index];
    route.status = STATUS_POOL[2];
    renderRoutes();
    Diagnostic.write(`Route ${route.relay} ${route.channel} flagged as critical.`);
    pushAlert({ type: "critical", message: `${route.relay} halted for inspection.` });
}

function injectRoute() {
    const route = createRoute();
    routes.push(route);
    renderRoutes();
    Diagnostic.write(`Route injected: ${route.relay} ${route.channel}, ${route.latency}.`);
}

function updateMissionClock() {
    const elapsed = Date.now() - bootTime;
    const hours = Math.floor(elapsed / 3_600_000);
    const minutes = Math.floor((elapsed % 3_600_000) / 60_000).toString().padStart(2, "0");
    const seconds = Math.floor((elapsed % 60_000) / 1000).toString().padStart(2, "0");
    missionClock.textContent = `${hours.toString().padStart(2, "0")}:${minutes}:${seconds}`;
}

function drawNoiseChart() {
    noiseCtx.clearRect(0, 0, noiseCanvas.width, noiseCanvas.height);
    const samples = generateNoiseSamples(36);
    noiseCtx.beginPath();
    noiseCtx.lineWidth = 2;
    noiseCtx.strokeStyle = "#5a98ff";

    samples.forEach((value, index) => {
        const x = (noiseCanvas.width / (samples.length - 1)) * index;
        const y = noiseCanvas.height - value * noiseCanvas.height;
        if (index === 0) {
            noiseCtx.moveTo(x, y);
        } else {
            noiseCtx.lineTo(x, y);
        }
    });
    noiseCtx.stroke();

    noiseCtx.fillStyle = "rgba(90, 152, 255, 0.15)";
    noiseCtx.lineTo(noiseCanvas.width, noiseCanvas.height);
    noiseCtx.lineTo(0, noiseCanvas.height);
    noiseCtx.closePath();
    noiseCtx.fill();
}

function generateNoiseSamples(count) {
    const samples = [];
    let cursor = Math.random() * 0.4 + 0.2;
    for (let i = 0; i < count; i += 1) {
        cursor += Math.random() * 0.1 - 0.05;
        cursor = Math.max(0.05, Math.min(0.95, cursor));
        samples.push(cursor);
    }
    return samples;
}

function handleTransmissionSubmit(event) {
    event.preventDefault();
    const formData = new FormData(composerForm);
    const entry = {
        id: crypto.randomUUID(),
        relay: formData.get("relay"),
        channel: formData.get("channel"),
        payload: formData.get("payload"),
        enqueued: Date.now(),
        status: "Queued"
    };
    transmissions.push(entry);
    renderQueue();
    Diagnostic.write(`Transmission queued for ${entry.relay} on ${entry.channel}.`);
    composerForm.reset();
}

function renderQueue() {
    queueList.innerHTML = "";
    transmissions
        .slice()
        .reverse()
        .forEach(item => {
            const li = document.createElement("li");
            li.className = "queue__item";

            const header = document.createElement("div");
            header.className = "queue__item-header";
            header.innerHTML = `<span>${item.relay} · ${item.channel}</span><span>${item.status}</span>`;

            const payload = document.createElement("pre");
            payload.className = "queue__payload";
            payload.textContent = item.payload;

            const meta = document.createElement("div");
            meta.className = "queue__meta";
            meta.style.fontSize = "0.8rem";
            meta.style.color = "var(--muted)";
            meta.textContent = new Date(item.enqueued).toLocaleString();

            li.appendChild(header);
            li.appendChild(payload);
            li.appendChild(meta);
            queueList.appendChild(li);
        });
}

function flushQueue() {
    transmissions.length = 0;
    renderQueue();
    Diagnostic.write("Transmission queue flushed by operator request.");
}

function pushAlert({ type, message }) {
    const alert = {
        id: crypto.randomUUID(),
        type,
        message,
        timestamp: Date.now()
    };
    alerts.unshift(alert);
    if (alerts.length > 12) {
        alerts.pop();
    }
    renderAlerts();
}

function renderAlerts() {
    alertFeed.innerHTML = "";
    alerts.forEach(alert => {
        const fragment = alertTemplate.content.cloneNode(true);
        const badge = fragment.querySelector('[data-cell="type"]');
        badge.textContent = alert.type.toUpperCase();
        badge.dataset.severity = alert.type;
        fragment.querySelector('[data-cell="message"]').textContent = alert.message;
        fragment.querySelector('[data-cell="timestamp"]').textContent = new Date(alert.timestamp).toLocaleTimeString();
        alertFeed.appendChild(fragment);
    });
}

function acknowledgeAlerts() {
    alerts.length = 0;
    renderAlerts();
    Diagnostic.write("Alert feed acknowledged and cleared.");
}

function seedRoutes(count = 6) {
    for (let i = 0; i < count; i += 1) {
        routes.push(createRoute());
    }
    renderRoutes();
}

function seedAlerts() {
    for (let i = 0; i < 3; i += 1) {
        pushAlert(pickRandom(ALERT_MESSAGES));
    }
}

function periodicTasks() {
    if (Math.random() < 0.5) {
        const message = pickRandom(ALERT_MESSAGES);
        pushAlert({ type: message.type, message: message.message });
        Diagnostic.write(`Alert generated: ${message.message}`);
    }
    if (routes.length && Math.random() < 0.6) {
        const route = pickRandom(routes);
        const jitter = (Math.random() * 14 - 7).toFixed(1);
        const currentLatency = Number.parseFloat(route.latency) + Number.parseFloat(jitter);
        route.latency = `${Math.max(2, currentLatency).toFixed(1)} ms`;
        if (currentLatency > 120) {
            route.status = STATUS_POOL[2];
        } else if (currentLatency > 80) {
            route.status = STATUS_POOL[1];
        } else {
            route.status = STATUS_POOL[0];
        }
        renderRoutes();
        Diagnostic.write(`Telemetry update: ${route.relay} latency jitter ${jitter} ms.`);
    }
}

function initialize() {
    seedRoutes();
    seedAlerts();
    renderQueue();
    drawNoiseChart();
    updateMissionClock();
    Diagnostic.write("Signal Relay Nexus boot sequence complete.");
}

spawnRouteButton.addEventListener("click", injectRoute);
reseedChartButton.addEventListener("click", () => {
    drawNoiseChart();
    Diagnostic.write("Spectral noise chart reseeded.");
});
composerForm.addEventListener("submit", handleTransmissionSubmit);
flushQueueButton.addEventListener("click", flushQueue);
ackAlertsButton.addEventListener("click", acknowledgeAlerts);

setInterval(updateMissionClock, 1_000);
setInterval(drawNoiseChart, 9_000);
setInterval(periodicTasks, 11_000);

initialize();
