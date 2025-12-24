const queueTableBody = document.querySelector("#queue-table tbody");
const queueRowTemplate = document.getElementById("row-template");
const telemetryGrid = document.getElementById("telemetry-grid");
const telemetryTemplate = document.getElementById("telemetry-template");
const timelineContainer = document.getElementById("timeline");
const enqueueButton = document.getElementById("enqueue-scan");
const timelineRangeSelect = document.getElementById("timeline-range");
const refreshTelemetryButton = document.getElementById("refresh-telemetry");
const uptimePill = document.getElementById("uptime-pill");
const logForm = document.getElementById("log-form");
const logEntries = document.getElementById("log-entries");
const exportButton = document.getElementById("export-log");
const exportDialog = document.getElementById("export-dialog");
const exportText = document.getElementById("export-text");

const TARGETS = [
    "Sirius Array",
    "Mimas Ridge",
    "Ganymede Iceband",
    "Tycho Crater",
    "Orion Filament",
    "Cygnus Gate",
    "Helios Swarm",
    "Vela Drift"
];

const BANDS = ["X-ray", "Infrared", "Ultraviolet", "Radio", "Optical"];

const PRIORITY_LEVELS = [
    { label: "Low", badge: "badge--low" },
    { label: "Medium", badge: "badge--medium" },
    { label: "High", badge: "badge--high" }
];

const TELEMETRY_KEYS = [
    { id: "dish", label: "Dish Array", unit: "%" },
    { id: "coolant", label: "Coolant", unit: "K" },
    { id: "battery", label: "Reserve", unit: "%" },
    { id: "uplink", label: "Uplink", unit: "Mbps" },
    { id: "weather", label: "Atmos", unit: "hPa" },
    { id: "tracking", label: "Tracking", unit: "deg" }
];

const queue = [];
const logs = [];
const timelineEvents = [];
const bootTime = Date.now();

function pickRandom(array) {
    return array[Math.floor(Math.random() * array.length)];
}

function randomWindow() {
    const startOffset = Math.floor(Math.random() * 120);
    const duration = Math.floor(Math.random() * 50) + 10;
    return {
        start: startOffset,
        end: startOffset + duration
    };
}

function formatWindow(window) {
    return `${window.start}m - ${window.end}m`;
}

function renderQueue() {
    queueTableBody.innerHTML = "";
    queue.forEach((entry, index) => {
        const row = queueRowTemplate.content.cloneNode(true);
        row.querySelector('[data-cell="target"]').textContent = entry.target;
        row.querySelector('[data-cell="band"]').textContent = entry.band;
        row.querySelector('[data-cell="priority"]').textContent = entry.priority.label;
        row.querySelector('[data-cell="window"]').textContent = formatWindow(entry.window);

        row.querySelector('[data-action="promote"]').addEventListener("click", () => promoteEntry(index));
        row.querySelector('[data-action="drop"]').addEventListener("click", () => dropEntry(index));
        queueTableBody.appendChild(row);
    });
}

function scheduleRandomScan() {
    const entry = {
        target: pickRandom(TARGETS),
        band: pickRandom(BANDS),
        priority: pickRandom(PRIORITY_LEVELS),
        window: randomWindow()
    };
    queue.push(entry);
    queue.sort((a, b) => PRIORITY_LEVELS.indexOf(b.priority) - PRIORITY_LEVELS.indexOf(a.priority));
    renderQueue();
    addTimelineEvent({
        label: `Scheduled ${entry.target}`,
        priority: entry.priority.label,
        timestamp: Date.now(),
        band: entry.band
    });
}

function promoteEntry(index) {
    if (index <= 0) return;
    const [entry] = queue.splice(index, 1);
    queue.splice(index - 1, 0, entry);
    renderQueue();
    addTimelineEvent({
        label: `Promoted ${entry.target}`,
        priority: entry.priority.label,
        timestamp: Date.now(),
        band: entry.band
    });
}

function dropEntry(index) {
    const [entry] = queue.splice(index, 1);
    renderQueue();
    addTimelineEvent({
        label: `Dropped ${entry.target}`,
        priority: entry.priority.label,
        timestamp: Date.now(),
        band: entry.band
    });
}

function badgeClass(priority) {
    const level = PRIORITY_LEVELS.find(level => level.label === priority);
    return level ? level.badge : PRIORITY_LEVELS[0].badge;
}

function addTimelineEvent(event) {
    timelineEvents.unshift(event);
    if (timelineEvents.length > 40) {
        timelineEvents.pop();
    }
    renderTimeline();
}

function filterTimeline(rangeHours) {
    const now = Date.now();
    const horizon = now - rangeHours * 60 * 60 * 1000;
    return timelineEvents.filter(event => event.timestamp >= horizon);
}

function renderTimeline() {
    const range = Number(timelineRangeSelect.value);
    const events = filterTimeline(range);
    timelineContainer.innerHTML = "";
    events.forEach(event => {
        const node = document.createElement("article");
        node.className = "timeline-event";

        const time = document.createElement("div");
        time.textContent = new Date(event.timestamp).toLocaleTimeString();

        const detail = document.createElement("div");
        detail.innerHTML = `<strong>${event.label}</strong><br><span>${event.band} band</span>`;

        const priorityBadge = document.createElement("div");
        priorityBadge.className = `badge ${badgeClass(event.priority)}`;
        priorityBadge.textContent = event.priority;

        node.appendChild(time);
        node.appendChild(detail);
        node.appendChild(priorityBadge);
        timelineContainer.appendChild(node);
    });
}

function generateTelemetry() {
    const now = Date.now();
    return TELEMETRY_KEYS.map(({ id, label, unit }) => {
        const base = Math.abs(Math.sin(now / (7000 + id.length * 123))) * 100;
        const variance = Math.random() * 15 - 7.5;
        const value = Math.max(0, base + variance);
        const trend = variance >= 0 ? "Rising" : "Falling";
        return { label, value: value.toFixed(1) + unit, trend };
    });
}

function renderTelemetry(cards) {
    telemetryGrid.innerHTML = "";
    cards.forEach(card => {
        const fragment = telemetryTemplate.content.cloneNode(true);
        fragment.querySelector('[data-cell="label"]').textContent = card.label;
        fragment.querySelector('[data-cell="value"]').textContent = card.value;
        fragment.querySelector('[data-cell="trend"]').textContent = card.trend;
        telemetryGrid.appendChild(fragment);
    });
}

function updateUptime() {
    const elapsed = Date.now() - bootTime;
    const seconds = Math.floor(elapsed / 1000);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainder = seconds % 60;
    uptimePill.textContent = `Uptime: ${hours}h ${minutes}m ${remainder}s`;
}

function renderLogs() {
    logEntries.innerHTML = "";
    logs
        .slice()
        .reverse()
        .forEach(entry => {
            const item = document.createElement("li");
            item.className = "log-entry";

            const heading = document.createElement("h4");
            heading.textContent = `${entry.author} — ${entry.summary}`;

            const timestamp = document.createElement("p");
            timestamp.textContent = new Date(entry.timestamp).toLocaleString();

            const body = document.createElement("p");
            body.textContent = entry.body;

            item.appendChild(heading);
            item.appendChild(timestamp);
            item.appendChild(body);
            logEntries.appendChild(item);
        });
}

function handleLogSubmit(event) {
    event.preventDefault();
    const formData = new FormData(logForm);
    const entry = {
        author: formData.get("log-author"),
        summary: formData.get("log-summary"),
        body: formData.get("log-body"),
        timestamp: Date.now()
    };
    logs.push(entry);
    renderLogs();
    logForm.reset();
    logForm.elements[0].focus();
}

function exportLogs() {
    if (logs.length === 0) {
        exportText.value = "No entries available.";
    } else {
        const csvLines = logs.map(entry => {
            const cells = [
                new Date(entry.timestamp).toISOString(),
                entry.author,
                entry.summary,
                entry.body.replace(/\n/g, " ")
            ];
            return cells.map(cell => `"${cell.replace(/"/g, '""')}"`).join(",");
        });
        exportText.value = ["timestamp,author,summary,body", ...csvLines].join("\n");
    }
    if (typeof exportDialog.showModal === "function") {
        exportDialog.showModal();
    } else {
        alert(exportText.value);
    }
}

function seedInitialData() {
    for (let i = 0; i < 5; i += 1) {
        scheduleRandomScan();
    }
    const seededLogs = [
        {
            author: "Astra",
            summary: "Ridge Sweep",
            body: "Completed a low pass over the ridge line. Detected mild auroral residue.",
            timestamp: Date.now() - 3600 * 1000
        },
        {
            author: "Lyric",
            summary: "Array Sync",
            body: "Synced sector dishes to the Orion filament; phase variance under 0.03 degrees.",
            timestamp: Date.now() - 5400 * 1000
        }
    ];
    logs.push(...seededLogs);
    renderLogs();
    renderTelemetry(generateTelemetry());
}

enqueueButton.addEventListener("click", scheduleRandomScan);
timelineRangeSelect.addEventListener("change", renderTimeline);
refreshTelemetryButton.addEventListener("click", () => {
    renderTelemetry(generateTelemetry());
});
logForm.addEventListener("submit", handleLogSubmit);
exportButton.addEventListener("click", exportLogs);

setInterval(() => {
    renderTelemetry(generateTelemetry());
}, 12_000);

setInterval(() => {
    updateUptime();
}, 1_000);

seedInitialData();
renderTimeline();
updateUptime();
