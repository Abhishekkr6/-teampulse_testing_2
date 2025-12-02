# Whisper synthesizes a playful rune from random glyphs.
GLYPHS = %w[sol rune flux mica arc ion]

class Whisper
  def initialize(seed)
    @rng = Random.new(seed)
  end

  def conjure
    parts = Array.new(4) { GLYPHS.sample(random: @rng) }
    "«#{parts.join('-')}»"
  end
end

if __FILE__ == $PROGRAM_NAME
  puts Whisper.new(42).conjure
end
