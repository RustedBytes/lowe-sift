//! Lowe-style SIFT feature extraction and matching.
//!
//! This crate implements the core feature pipeline from David Lowe's 2004 IJCV
//! paper, *Distinctive Image Features from Scale-Invariant Keypoints*: Gaussian
//! scale space, difference-of-Gaussian extrema, quadratic keypoint localization,
//! orientation assignment, 128-dimensional descriptors, and Lowe's nearest-neighbor
//! distance-ratio matcher, approximate Best-Bin-First lookup, generalized
//! Hough pose clustering, and affine geometric verification.
//!
//! The crate is intentionally small and dependency-light. The algorithm works on
//! [`GrayImage`] values with pixels represented as `f32` in `[0, 1]`; enabling the
//! default `image` feature adds conversion helpers for the `image` crate.
//!
//! # Example
//!
//! ```no_run
//! use lowe_sift::{GrayImage, Sift};
//!
//! # #[cfg(feature = "image")]
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let input = image::open("scene.jpg")?;
//! let gray = GrayImage::from_dynamic_image(&input);
//! let features = Sift::default().detect_and_compute(&gray);
//! println!("{} SIFT features", features.len());
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod gray;
mod sift;

pub mod bbf;
pub mod geometry;
pub mod matching;
pub mod recognition;

pub use crate::bbf::{
    ApproxNeighbor, BbfConfig, BbfConfigError, BbfDescriptorIndex, match_descriptors_bbf,
    match_features_bbf,
};
pub use crate::geometry::{
    Affine2, GeometryError, estimate_affine_from_pairs, estimate_affine_train_to_query,
};
pub use crate::gray::{GrayImage, GrayImageError};
pub use crate::matching::{DescriptorMatch, match_descriptors, match_features};
pub use crate::recognition::{
    HoughBin, HoughCluster, HoughConfig, ModelDatabase, ModelFeatureRecord, ObjectHypothesis,
    ObjectModel, RecognitionError, SimilarityPose, cluster_matches_hough, predict_similarity_pose,
    verify_hough_clusters,
};
pub use crate::sift::{
    DESCRIPTOR_LEN, Descriptor, Feature, Keypoint, Sift, SiftConfig, SiftConfigError,
};
