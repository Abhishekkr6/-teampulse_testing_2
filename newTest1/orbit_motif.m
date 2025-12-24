% Orbit motif generator with simple frequency mix
frequencies = [220, 330, 495, 742.5];
weights = [0.4, 0.3, 0.2, 0.1];
blend = sum(frequencies .* weights);
fprintf('Orbit motif weighted frequency: %.2f Hz\n', blend);
