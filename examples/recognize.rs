use std::{env, error::Error, process};

use lowe_sift::{
    BbfConfig, GrayImage, HoughConfig, ModelDatabase, ObjectModel, Sift, cluster_matches_hough,
    match_features_bbf, verify_hough_clusters,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <query-image> <model-image>", args[0]);
        process::exit(2);
    }

    let query = GrayImage::from_dynamic_image(&image::open(&args[1])?);
    let model_image = GrayImage::from_dynamic_image(&image::open(&args[2])?);

    let sift = Sift::default();
    let query_features = sift.detect_and_compute(&query);
    let model_features = sift.detect_and_compute(&model_image);

    let model = ObjectModel::new(
        1,
        model_image.width() as f32,
        model_image.height() as f32,
        model_features,
    )?;
    let database = ModelDatabase::new(vec![model])?;

    let matches = match_features_bbf(
        &query_features,
        database.train_features(),
        BbfConfig::default(),
    )?;
    let clusters =
        cluster_matches_hough(&matches, &query_features, &database, HoughConfig::default())?;
    let hypotheses = verify_hough_clusters(
        &matches,
        &query_features,
        &database,
        &clusters,
        HoughConfig::default(),
    )?;

    println!("query features: {}", query_features.len());
    println!("model features: {}", database.train_features().len());
    println!("ratio-test matches: {}", matches.len());
    println!("Hough clusters: {}", clusters.len());
    println!("verified hypotheses: {}", hypotheses.len());

    for hypothesis in hypotheses.iter().take(10) {
        println!(
            "model {}: {} inliers, rmse {:.3}, affine {:?}",
            hypothesis.model_id,
            hypothesis.inlier_match_indices.len(),
            hypothesis.rmse,
            hypothesis.affine
        );
    }

    Ok(())
}
