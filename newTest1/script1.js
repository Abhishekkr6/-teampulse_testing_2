const joinButton = document.getElementById("join-button");
const timerElement = document.getElementById("timer");

// Generate a countdown to a fictional tasting event.
const eventTime = Date.now() + 86_400_000; // 24 hours from load

function formatTime(ms) {
    const totalSeconds = Math.max(0, Math.floor(ms / 1000));
    const hours = String(Math.floor(totalSeconds / 3600)).padStart(2, "0");
    const minutes = String(Math.floor((totalSeconds % 3600) / 60)).padStart(2, "0");
    const seconds = String(totalSeconds % 60).padStart(2, "0");
    return `${hours}:${minutes}:${seconds}`;
}

function updateTimer() {
    const remaining = eventTime - Date.now();
    timerElement.textContent = formatTime(remaining);
    if (remaining <= 0) {
        clearInterval(timerInterval);
        timerElement.textContent = "It's tasting time!";
    }
}

const timerInterval = setInterval(updateTimer, 1000);
updateTimer();

joinButton.addEventListener("click", () => {
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = "Orbit invite dispatched to your inbox.";
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.classList.add("visible");
    }, 50);

    setTimeout(() => {
        toast.classList.remove("visible");
        setTimeout(() => toast.remove(), 600);
    }, 3200);
});
