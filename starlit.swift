import Foundation

struct Pulse {
    let hue: String
    let amplitude: Double
}

func blend(_ pulses: [Pulse]) -> Double {
    return pulses.reduce(0.0) { $0 + $1.amplitude }
}

let pulses = [
    Pulse(hue: "cerulean", amplitude: 1.4),
    Pulse(hue: "rose", amplitude: 0.9),
    Pulse(hue: "amber", amplitude: 1.1)
]

print("Auroral blend: \(blend(pulses)) quanta")
