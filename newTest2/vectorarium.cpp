#include <iostream>
#include <vector>
#include <numeric>

int main() {
    std::vector<int> waveform{3, -1, 4, 1, 5, -9, 2, 6, 5};
    int total = std::accumulate(waveform.begin(), waveform.end(), 0);
    std::cout << "Vectorarium checksum: " << total << "\n";
    for (std::size_t i = 0; i < waveform.size(); ++i) {
        std::cout << "idx " << i << " => " << waveform[i] * waveform[i] << '\n';
    }
    return 0;
}
