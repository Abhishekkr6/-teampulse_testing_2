module Main where

harmonics :: [Double]
harmonics = [1.2, 2.4, 3.6, 5.0]

flux :: Double
flux = sum (map (* 0.9) harmonics)

main :: IO ()
main = putStrLn ("Harmonic flux: " ++ show flux)
