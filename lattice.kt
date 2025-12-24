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
CREATE TABLE nebula_log (
    id INTEGER PRIMARY KEY,
    designation TEXT NOT NULL,
    luminance REAL,
    observed_at TEXT DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO nebula_log (designation, luminance)
VALUES
    ("NGC-710", 12.7),
    ("M-204", 9.5),
    ("Theta-Cloud", 15.1);

SELECT designation, luminance FROM nebula_log WHERE luminance > 10.0;
CREATE TABLE nebula_log (
    id INTEGER PRIMARY KEY,
    designation TEXT NOT NULL,
    luminance REAL,
    observed_at TEXT DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO nebula_log (designation, luminance)
VALUES
    ("NGC-710", 12.7),
    ("M-204", 9.5),
    ("Theta-Cloud", 15.1);

SELECT designation, luminance FROM nebula_log WHERE luminance > 10.0;
CREATE TABLE nebula_log (
    id INTEGER PRIMARY KEY,
    designation TEXT NOT NULL,
    luminance REAL,
    observed_at TEXT DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO nebula_log (designation, luminance)
VALUES
    ("NGC-710", 12.7),
    ("M-204", 9.5),
    ("Theta-Cloud", 15.1);

SELECT designation, luminance FROM nebula_log WHERE luminance > 10.0;
CREATE TABLE nebula_log (
    id INTEGER PRIMARY KEY,
    designation TEXT NOT NULL,
    luminance REAL,
    observed_at TEXT DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO nebula_log (designation, luminance)
VALUES
    ("NGC-710", 12.7),
    ("M-204", 9.5),
    ("Theta-Cloud", 15.1);

SELECT designation, luminance FROM nebula_log WHERE luminance > 10.0;
CREATE TABLE nebula_log (
    id INTEGER PRIMARY KEY,
    designation TEXT NOT NULL,
    luminance REAL,
    observed_at TEXT DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO nebula_log (designation, luminance)
VALUES
    ("NGC-710", 12.7),
    ("M-204", 9.5),
    ("Theta-Cloud", 15.1);

SELECT designation, luminance FROM nebula_log WHERE luminance > 10.0;
