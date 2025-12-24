<?php
$tones = ["drift", "pulse", "shine", "glow"];
$level = rand(10, 99);
$choice = $tones[array_rand($tones)];
echo "Echo chamber registers {$choice} at level {$level}.";
