fn main() {
    let harmonics = [1.5_f32, 2.25, 3.75, 6.0];
    let drift: f32 = harmonics.iter().map(|n| n * 1.2).sum();
    println!("Flux channel stabilized at {:.2} quanta", drift);
}
