data class Node(val label: String, val charge: Int)

fun shimmer(nodes: List<Node>): Int {
    return nodes.sumOf { it.charge * it.label.length }
}

fun main() {
    val cluster = listOf(
        Node("ion", 3),
        Node("flux", 7),
        Node("nova", 5)
    )
    println("Lattice shimmer score: ${shimmer(cluster)}")
}
