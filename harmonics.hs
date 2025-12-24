module Main where

harmonics :: [Double]
harmonics = [1.2, 2.4, 3.6, 5.0]

flux :: Double
flux = sum (map (* 0.9) harmonics)

main :: IO ()
main = putStrLn ("Harmonic flux: " ++ show flux)


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
