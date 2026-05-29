use std::{env, error::Error};

use lowe_sift::{GrayImage, Sift};

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: cargo run --example extract -- <image>")?;

    let image = image::open(&path)?;
    let gray = GrayImage::from_dynamic_image(&image);
    let sift = Sift::default();
    let features = sift.detect_and_compute(&gray);

    println!("image: {path}");
    println!("features: {}", features.len());
    if let Some(first) = features.first() {
        println!(
            "first keypoint: x={:.2} y={:.2} scale={:.2} angle={:.3} response={:.5}",
            first.keypoint.x,
            first.keypoint.y,
            first.keypoint.scale,
            first.keypoint.angle,
            first.keypoint.response
        );
    }

    Ok(())
}
