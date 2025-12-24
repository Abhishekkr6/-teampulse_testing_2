const canvas = document.getElementById("humidity-chart");
const ctx = canvas.getContext("2d");
const growthLog = document.getElementById("growth-log");
const toggleButton = document.getElementById("toggle-stats");
const statsPanel = document.getElementById("stats-panel");
const lastSync = document.getElementById("last-sync");

const mossNames = ["Verdant-7", "Nimbus Fern", "Echo Moss", "Zephyr Cap", "Aurora Puff"];

function randomHumidityPoints() {
    return new Array(30).fill(0).map(() => 40 + Math.random() * 40);
}

function drawChart(points) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#8cf5b0";
    ctx.lineWidth = 3;
    ctx.beginPath();

    const step = canvas.width / (points.length - 1);
    points.forEach((value, index) => {
        const x = index * step;
        const y = canvas.height - (value / 100) * canvas.height;
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();
}

function addGrowthEntry() {
    const entry = document.createElement("li");
    const name = mossNames[Math.floor(Math.random() * mossNames.length)];
    const growth = (Math.random() * 1.5 + 0.2).toFixed(2);
    entry.textContent = `${name} expanded ${growth} cm overnight.`;
    growthLog.prepend(entry);
}

function syncMossNetwork() {
    drawChart(randomHumidityPoints());
    addGrowthEntry();
    lastSync.textContent = new Date().toLocaleTimeString();
}

setInterval(syncMossNetwork, 7000);
syncMossNetwork();

toggleButton.addEventListener("click", () => {
    statsPanel.classList.toggle("hidden");
});
