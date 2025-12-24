type Beacon = {
    id: string;
    phase: number;
};

function oscillate(beacon: Beacon): string {
    const phaseShift = Math.sin(beacon.phase) * 180;
    return `Beacon ${beacon.id} resonates at ${phaseShift.toFixed(2)}°`;
}

const sample: Beacon = { id: "delta-aurora", phase: Math.PI / 3 };
console.log(oscillate(sample));
