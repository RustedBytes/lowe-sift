use std::{env, error::Error};

use lowe_sift::{GrayImage, Sift, match_features};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let Some(query_path) = args.next() else {
        return Err(
            "usage: cargo run --example match -- <query-image> <train-image> [ratio]".into(),
        );
    };
    let Some(train_path) = args.next() else {
        return Err(
            "usage: cargo run --example match -- <query-image> <train-image> [ratio]".into(),
        );
    };
    let ratio = args.next().as_deref().unwrap_or("0.8").parse::<f32>()?;

    let query = GrayImage::from_dynamic_image(&image::open(&query_path)?);
    let train = GrayImage::from_dynamic_image(&image::open(&train_path)?);

    let sift = Sift::default();
    let query_features = sift.detect_and_compute(&query);
    let train_features = sift.detect_and_compute(&train);
    let matches = match_features(&query_features, &train_features, ratio);

    println!("query features: {}", query_features.len());
    println!("train features: {}", train_features.len());
    println!("accepted matches at ratio {ratio}: {}", matches.len());

    Ok(())
}
