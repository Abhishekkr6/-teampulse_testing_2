local glyphs = {"nova", "aurora", "plasma", "spectrum"}

math.randomseed(os.time())

local function emit()
    local idx = math.random(#glyphs)
    local energy = math.random(15, 75)
    return string.format("Pulse %s radiates %d units", glyphs[idx], energy)
end

print(emit())
