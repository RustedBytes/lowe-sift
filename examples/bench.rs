use lowe_sift::{GrayImage, Sift};
use std::time::Instant;

fn main() {
    let mut data = vec![0.0f32; 1024 * 1024];
    // Generate some synthetic structure (sine waves, noise, blobs)
    for y in 0..1024 {
        for x in 0..1024 {
            let val = 0.5
                + 0.25 * ((x as f32 * 0.05).sin() + (y as f32 * 0.05).sin())
                + 0.1 * ((x as f32 * 0.2).cos() * (y as f32 * 0.2).cos());
            data[y * 1024 + x] = val.clamp(0.0, 1.0);
        }
    }
    let image = GrayImage::new(1024, 1024, data).unwrap();
    let sift = Sift::default();
    
    // Warmup
    let features = sift.detect_and_compute(&image);
    println!("Detected {} features in warmup", features.len());

    let start = Instant::now();
    let iterations = 5;
    for _ in 0..iterations {
        let features = sift.detect_and_compute(&image);
        assert!(!features.is_empty());
    }
    let duration = start.elapsed();
    println!("Average time per detect_and_compute: {:?}", duration / iterations as u32);
}
