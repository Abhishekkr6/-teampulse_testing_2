object Cascade extends App {
  case class Wave(label: String, amplitude: Int)

  val waves = List(
    Wave("alpha", 5),
    Wave("beta", 7),
    Wave("gamma", 9)
  )

  val total = waves.map(_.amplitude * 3).sum
  println(s"Cascade resonance: $total")
}
