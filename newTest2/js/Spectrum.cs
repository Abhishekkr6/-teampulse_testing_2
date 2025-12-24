using System;

namespace Spectrum
{
    class Analyzer
    {
        static readonly string[] bands = { "infra", "crimson", "cerulean", "ultra" };

        static void Main()
        {
            var rng = new Random(31415);
            string band = bands[rng.Next(bands.Length)];
            int energy = rng.Next(90, 181);
            Console.WriteLine($"Spectrum analyzer locks on {band} with {energy} keV");
        }
    }
}
