const stratus = document.getElementById("stratus");
const cirrus = document.getElementById("cirrus");
const stormButton = document.getElementById("roll-storm");
const stormOutput = document.getElementById("storm-output");
const sessionNotes = document.getElementById("session-notes");

function describePatch() {
    const density = Number(stratus.value);
    const altitude = Number(cirrus.value);

    const densityMood = density > 70 ? "heavy fog cascade" : density > 40 ? "layered shimmer" : "feathered haze";
    const altitudeMood = altitude > 70 ? "high-altitude sparkle" : altitude > 40 ? "mid-sky resonance" : "ground-hug hum";

    return `Patch mix: ${densityMood} with ${altitudeMood}.`;
}

function updateNotesHint() {
    sessionNotes.placeholder = describePatch();
}

stratus.addEventListener("input", updateNotesHint);
cirrus.addEventListener("input", updateNotesHint);
updateNotesHint();

const stormDescriptors = [
    "braided lightning arpeggios",
    "sub-bass thunder rolls",
    "reverb-soaked cloud bursts",
    "granular hail texture",
    "polyrhythmic drizzle pulses"
];

stormButton.addEventListener("click", () => {
    const descriptor = stormDescriptors[Math.floor(Math.random() * stormDescriptors.length)];
    const intensity = (Math.random() * 9 + 1).toFixed(1);
    stormOutput.textContent = `Forecast seed: ${descriptor} at intensity ${intensity}.`;
});
