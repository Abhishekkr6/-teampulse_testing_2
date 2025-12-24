defmodule OrbitMap do
  def sample(nodes) do
    Enum.map(nodes, fn {name, radius} ->
      {name, Float.round(radius * :math.pi(), 2)}
    end)
  end
end

nodes = [{"Lyra", 1.2}, {"Kepler", 2.8}, {"Vega", 3.6}]
IO.inspect(OrbitMap.sample(nodes), label: "Orbital circumference")
