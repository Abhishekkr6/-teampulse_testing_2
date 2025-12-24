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
