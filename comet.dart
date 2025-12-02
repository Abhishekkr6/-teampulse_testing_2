import 'dart:math';

class Comet {
  final String id;
  final double velocity;

  Comet(this.id, this.velocity);

  @override
  String toString() => 'Comet $id streaks at ${velocity.toStringAsFixed(2)} km/s';
}

void main() {
  final rng = Random();
  final comet = Comet('Halcyon', rng.nextDouble() * 72 + 28);
  print(comet);
}
