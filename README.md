# lowe-sift

[![Crates.io Version](https://img.shields.io/crates/v/lowe-sift)](https://crates.io/crates/lowe-sift)

A dependency-light Rust crate implementing Lowe-style SIFT feature extraction,
matching, and object-recognition helpers from David G. Lowe, “Distinctive Image
Features from Scale-Invariant Keypoints,” *International Journal of Computer
Vision*, 2004.

The crate implements:

- Gaussian scale-space construction with optional input-image doubling.
- Difference-of-Gaussian extrema detection over 26 neighbors.
- 3D quadratic keypoint localization in `(x, y, scale)`.
- Low-contrast rejection and edge-response rejection.
- Orientation assignment from a 36-bin local gradient histogram.
- Lowe's 128-dimensional `4 x 4 x 8` descriptor with trilinear interpolation,
  normalization, `0.2` clipping, and renormalization.
- Exact nearest-neighbor descriptor matching with Lowe's distance-ratio test.
- Approximate Best-Bin-First k-d-tree descriptor matching with a default cap of
  200 checked candidates per query.
- Generalized Hough clustering over model id, location, scale, and orientation.
- Least-squares affine verification with iterative outlier rejection and a
  top-down pass that adds further model matches consistent with the fitted pose.

This is a readable reference-style implementation, not a SIMD-optimized extractor.

## Install

```toml
[dependencies]
lowe-sift = { version = "0.1.0" }
```

By default the crate enables an `image` feature for conversion from the `image` crate.
For a core-only build:

```toml
lowe-sift = { version = "0.1.0", default-features = false }
```

## Extract features

```rust
use lowe_sift::{GrayImage, Sift};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = image::open("scene.jpg")?;
    let gray = GrayImage::from_dynamic_image(&input);

    let sift = Sift::default();
    let features = sift.detect_and_compute(&gray);

    println!("{} SIFT features", features.len());
    Ok(())
}
```

Run the included example:

```bash
cargo run --example extract -- scene.jpg
```

## Match two images exactly

```rust
use lowe_sift::{match_features, GrayImage, Sift};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = GrayImage::from_dynamic_image(&image::open("a.jpg")?);
    let b = GrayImage::from_dynamic_image(&image::open("b.jpg")?);

    let sift = Sift::default();
    let features_a = sift.detect_and_compute(&a);
    let features_b = sift.detect_and_compute(&b);

    let matches = match_features(&features_a, &features_b, 0.8);
    println!("{} exact ratio-test matches", matches.len());
    Ok(())
}
```

Run the included matcher:

```bash
cargo run --example match -- query.jpg train.jpg 0.8
```

## Match with Best-Bin-First search

```rust
use lowe_sift::{match_features_bbf, BbfConfig, GrayImage, Sift};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let query = GrayImage::from_dynamic_image(&image::open("query.jpg")?);
    let train = GrayImage::from_dynamic_image(&image::open("train.jpg")?);

    let sift = Sift::default();
    let q = sift.detect_and_compute(&query);
    let t = sift.detect_and_compute(&train);

    let matches = match_features_bbf(&q, &t, BbfConfig::default())?;
    println!("{} approximate ratio-test matches", matches.len());
    Ok(())
}
```

`BbfConfig::default()` uses `max_candidates = 200`, `leaf_size = 16`, and
`ratio_threshold = 0.8`.

## Estimate an affine pose

```rust
use lowe_sift::{estimate_affine_train_to_query, match_features, GrayImage, Sift};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let query = GrayImage::from_dynamic_image(&image::open("query.jpg")?);
    let train = GrayImage::from_dynamic_image(&image::open("train.jpg")?);

    let sift = Sift::default();
    let q = sift.detect_and_compute(&query);
    let t = sift.detect_and_compute(&train);
    let matches = match_features(&q, &t, 0.8);

    if matches.len() >= 3 {
        let affine = estimate_affine_train_to_query(&matches, &q, &t)?;
        println!("train -> query affine: {affine:?}");
    }
    Ok(())
}
```

## Recognize objects with Hough clustering

```rust
use lowe_sift::{
    cluster_matches_hough, match_features_bbf, verify_hough_clusters, BbfConfig,
    GrayImage, HoughConfig, ModelDatabase, ObjectModel, Sift,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_image = image::open("object.jpg")?;
    let query_image = image::open("scene.jpg")?;
    let train_gray = GrayImage::from_dynamic_image(&train_image);
    let query_gray = GrayImage::from_dynamic_image(&query_image);

    let sift = Sift::default();
    let model_features = sift.detect_and_compute(&train_gray);
    let query_features = sift.detect_and_compute(&query_gray);

    let model = ObjectModel::new(
        1,
        train_gray.width() as f32,
        train_gray.height() as f32,
        model_features,
    )?;
    let database = ModelDatabase::new(vec![model])?;

    let matches = match_features_bbf(
        &query_features,
        database.train_features(),
        BbfConfig::default(),
    )?;
    let clusters = cluster_matches_hough(
        &matches,
        &query_features,
        &database,
        HoughConfig::default(),
    )?;
    let hypotheses = verify_hough_clusters(
        &matches,
        &query_features,
        &database,
        &clusters,
        HoughConfig::default(),
    )?;

    for hypothesis in hypotheses {
        println!(
            "model {} with {} inliers, rmse {:.3}",
            hypothesis.model_id,
            hypothesis.inlier_match_indices.len(),
            hypothesis.rmse
        );
    }
    Ok(())
}
```

## Tunable parameters

```rust
use lowe_sift::{Sift, SiftConfig};

let mut config = SiftConfig::default();
config.contrast_threshold = 0.04;
config.edge_threshold = 12.0;
let sift = Sift::new(config).expect("valid SIFT configuration");
```

Important defaults:

| Parameter | Default |
| --- | ---: |
| Intervals per octave | `3` |
| Initial sigma | `1.6` |
| Assumed input blur | `0.5` |
| Input image doubling | `true` |
| Contrast threshold | `0.03` |
| Edge threshold | `10.0` |
| Orientation histogram bins | `36` |
| Secondary orientation peak ratio | `0.8` |
| Descriptor length | `128` |
| Descriptor clipping | `0.2` |
| Exact match ratio | caller supplied; paper commonly uses `0.8` |
| BBF checked candidates | `200` |
| Hough orientation bin width | `π / 6` radians |
| Hough scale bin width | `1.0` in `log2(scale)` |
| Hough location bin factor | `0.25` of projected max model dimension |
| Minimum Hough votes | `3` |

## Notes

- Coordinates in public `Keypoint` values are expressed in the original input image,
  even when the implementation uses a doubled base image internally.
- Pixel intensities should be normalized to `[0, 1]` for the paper's contrast threshold
  to have the intended meaning.
- The BBF matcher is approximate. Use `match_features` or `match_descriptors` when an
  exhaustive correctness baseline is more important than throughput.
- The Hough helpers implement model-id/location/scale/orientation clustering and affine
  verification. They do not implement Lowe's final Bayesian object-presence probability;
  callers can apply their own acceptance threshold to inlier count and reprojection error.
