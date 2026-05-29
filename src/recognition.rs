use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::{
    Feature,
    geometry::{Affine2, GeometryError, estimate_affine_from_pairs},
    matching::DescriptorMatch,
};

const TWO_PI: f32 = std::f32::consts::PI * 2.0;
const DEFAULT_ORIENTATION_BIN_WIDTH: f32 = std::f32::consts::PI / 6.0;
const DEFAULT_SCALE_BIN_WIDTH: f32 = 1.0;
const DEFAULT_LOCATION_BIN_FACTOR: f32 = 0.25;
const DEFAULT_AFFINE_RESIDUAL_FACTOR: f32 = 0.125;
const DEFAULT_MIN_VOTES: usize = 3;
const DEFAULT_MAX_VERIFICATION_ITERATIONS: usize = 8;
const EPSILON: f32 = 1.0e-7;

/// A set of SIFT features belonging to one reference object or scene.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectModel {
    /// Caller-defined object identifier. Identifiers must be unique within a [`ModelDatabase`].
    pub id: u32,
    /// Width of the training image or model extent in pixels.
    pub width: f32,
    /// Height of the training image or model extent in pixels.
    pub height: f32,
    /// Features extracted from the training image or model.
    pub features: Vec<Feature>,
}

impl ObjectModel {
    /// Creates an object model after validating dimensions.
    pub fn new(
        id: u32,
        width: f32,
        height: f32,
        features: Vec<Feature>,
    ) -> Result<Self, RecognitionError> {
        if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
            return Err(RecognitionError::InvalidModelDimension { model_id: id });
        }
        Ok(Self {
            id,
            width,
            height,
            features,
        })
    }

    /// Returns the larger model dimension.
    #[inline]
    pub fn max_dimension(&self) -> f32 {
        self.width.max(self.height)
    }
}

/// A flattened database of model features used for descriptor matching and pose clustering.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelDatabase {
    models: Vec<ObjectModel>,
    train_features: Vec<Feature>,
    records: Vec<ModelFeatureRecord>,
}

impl ModelDatabase {
    /// Builds a database from object models.
    pub fn new(models: Vec<ObjectModel>) -> Result<Self, RecognitionError> {
        let mut ids = HashSet::new();
        let mut train_features = Vec::new();
        let mut records = Vec::new();

        for (model_index, model) in models.iter().enumerate() {
            if !model.width.is_finite()
                || !model.height.is_finite()
                || model.width <= 0.0
                || model.height <= 0.0
            {
                return Err(RecognitionError::InvalidModelDimension { model_id: model.id });
            }
            if !ids.insert(model.id) {
                return Err(RecognitionError::DuplicateModelId(model.id));
            }

            for (feature_index, feature) in model.features.iter().enumerate() {
                records.push(ModelFeatureRecord {
                    model_id: model.id,
                    model_index,
                    feature_index,
                });
                train_features.push(feature.clone());
            }
        }

        Ok(Self {
            models,
            train_features,
            records,
        })
    }

    /// Returns all models in insertion order.
    #[inline]
    pub fn models(&self) -> &[ObjectModel] {
        &self.models
    }

    /// Returns the flattened training features used as the descriptor-matching train set.
    #[inline]
    pub fn train_features(&self) -> &[Feature] {
        &self.train_features
    }

    /// Returns metadata for each flattened training feature.
    #[inline]
    pub fn records(&self) -> &[ModelFeatureRecord] {
        &self.records
    }

    /// Looks up a model by identifier.
    pub fn model(&self, model_id: u32) -> Option<&ObjectModel> {
        self.models.iter().find(|model| model.id == model_id)
    }

    fn record_for_train_index(
        &self,
        train_index: usize,
    ) -> Result<&ModelFeatureRecord, RecognitionError> {
        self.records
            .get(train_index)
            .ok_or(RecognitionError::MatchIndexOutOfBounds)
    }

    fn feature_for_train_index(&self, train_index: usize) -> Result<&Feature, RecognitionError> {
        self.train_features
            .get(train_index)
            .ok_or(RecognitionError::MatchIndexOutOfBounds)
    }

    fn model_for_record(
        &self,
        record: &ModelFeatureRecord,
    ) -> Result<&ObjectModel, RecognitionError> {
        self.models
            .get(record.model_index)
            .ok_or(RecognitionError::ModelIndexOutOfBounds)
    }
}

/// Metadata for a feature in the flattened [`ModelDatabase`] train set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModelFeatureRecord {
    /// Identifier of the model that owns this feature.
    pub model_id: u32,
    /// Index of the model in [`ModelDatabase::models`].
    pub model_index: usize,
    /// Index of the feature within the owning [`ObjectModel`].
    pub feature_index: usize,
}

/// Configuration for generalized Hough clustering and affine verification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoughConfig {
    /// Minimum number of distinct matches required for a pose cluster.
    pub min_votes: usize,
    /// Orientation bin width in radians. Lowe uses 30 degrees.
    pub orientation_bin_width: f32,
    /// Scale bin width in base-2 logarithmic scale. A value of 1.0 is a factor of two.
    pub scale_bin_width: f32,
    /// Location bin size as a fraction of the projected maximum model dimension.
    pub location_bin_factor: f32,
    /// Verification residual threshold as a fraction of the projected maximum model dimension.
    pub affine_residual_factor: f32,
    /// Maximum number of outlier-removal and affine-refit iterations per cluster.
    pub max_verification_iterations: usize,
}

impl Default for HoughConfig {
    fn default() -> Self {
        Self {
            min_votes: DEFAULT_MIN_VOTES,
            orientation_bin_width: DEFAULT_ORIENTATION_BIN_WIDTH,
            scale_bin_width: DEFAULT_SCALE_BIN_WIDTH,
            location_bin_factor: DEFAULT_LOCATION_BIN_FACTOR,
            affine_residual_factor: DEFAULT_AFFINE_RESIDUAL_FACTOR,
            max_verification_iterations: DEFAULT_MAX_VERIFICATION_ITERATIONS,
        }
    }
}

impl HoughConfig {
    /// Validates that all configuration values are usable.
    pub fn validate(&self) -> Result<(), RecognitionError> {
        if self.min_votes == 0 {
            return Err(RecognitionError::InvalidConfig(
                "min_votes must be greater than zero",
            ));
        }
        if !self.orientation_bin_width.is_finite()
            || self.orientation_bin_width <= 0.0
            || self.orientation_bin_width > TWO_PI
        {
            return Err(RecognitionError::InvalidConfig(
                "orientation_bin_width must be in (0, 2π]",
            ));
        }
        if !self.scale_bin_width.is_finite() || self.scale_bin_width <= 0.0 {
            return Err(RecognitionError::InvalidConfig(
                "scale_bin_width must be greater than zero",
            ));
        }
        if !self.location_bin_factor.is_finite() || self.location_bin_factor <= 0.0 {
            return Err(RecognitionError::InvalidConfig(
                "location_bin_factor must be greater than zero",
            ));
        }
        if !self.affine_residual_factor.is_finite() || self.affine_residual_factor <= 0.0 {
            return Err(RecognitionError::InvalidConfig(
                "affine_residual_factor must be greater than zero",
            ));
        }
        if self.max_verification_iterations == 0 {
            return Err(RecognitionError::InvalidConfig(
                "max_verification_iterations must be greater than zero",
            ));
        }
        Ok(())
    }
}

/// A similarity pose predicted by a single matched SIFT feature.
///
/// The pose maps points from the model image into the query image using a
/// uniform scale, an orientation, and the projected location of the model origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimilarityPose {
    /// Predicted x coordinate of the model origin in the query image.
    pub x: f32,
    /// Predicted y coordinate of the model origin in the query image.
    pub y: f32,
    /// Predicted uniform scale from model pixels to query pixels.
    pub scale: f32,
    /// Predicted orientation in radians in `[0, 2π)`.
    pub orientation: f32,
}

impl SimilarityPose {
    /// Projects a model point into the query image using this similarity pose.
    pub fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        let cos = self.orientation.cos();
        let sin = self.orientation.sin();
        (
            self.x + self.scale * (cos * x - sin * y),
            self.y + self.scale * (sin * x + cos * y),
        )
    }
}

/// A generalized-Hough accumulator bin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HoughBin {
    /// Identifier of the voted model.
    pub model_id: u32,
    /// Discrete x-location bin.
    pub x_bin: i32,
    /// Discrete y-location bin.
    pub y_bin: i32,
    /// Discrete log-scale bin.
    pub scale_bin: i32,
    /// Discrete orientation bin.
    pub orientation_bin: i32,
}

/// A cluster of matches voting for a consistent object pose.
#[derive(Clone, Debug, PartialEq)]
pub struct HoughCluster {
    /// Identifier of the model supported by this cluster.
    pub model_id: u32,
    /// Accumulator bin that produced the cluster.
    pub bin: HoughBin,
    /// Indices into the match slice that voted for this cluster.
    pub match_indices: Vec<usize>,
    /// Number of distinct matches in the cluster.
    pub votes: usize,
    /// Average similarity pose predicted by the cluster's matches.
    pub pose: SimilarityPose,
}

/// An affine-verified object hypothesis.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectHypothesis {
    /// Identifier of the recognized model.
    pub model_id: u32,
    /// Index of the Hough cluster that produced this hypothesis.
    pub cluster_index: usize,
    /// Least-squares affine transform from model coordinates to query-image coordinates.
    pub affine: Affine2,
    /// Indices into the match slice that remain after affine outlier rejection.
    pub inlier_match_indices: Vec<usize>,
    /// Root-mean-square reprojection error of the inliers, in query-image pixels.
    pub rmse: f32,
}

/// Errors returned by object-recognition helpers.
#[derive(Clone, Debug, PartialEq)]
pub enum RecognitionError {
    /// A model identifier was used more than once in a database.
    DuplicateModelId(u32),
    /// A model has a non-finite or non-positive image extent.
    InvalidModelDimension {
        /// Identifier of the invalid model.
        model_id: u32,
    },
    /// A matched keypoint has a non-finite or non-positive scale.
    InvalidFeatureScale,
    /// A match referenced a missing query or train feature.
    MatchIndexOutOfBounds,
    /// Internal model metadata referenced a missing model.
    ModelIndexOutOfBounds,
    /// A configuration field was invalid.
    InvalidConfig(&'static str),
    /// Affine estimation failed during geometric verification.
    Geometry(GeometryError),
}

impl fmt::Display for RecognitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateModelId(id) => write!(f, "duplicate model id: {id}"),
            Self::InvalidModelDimension { model_id } => {
                write!(f, "invalid dimensions for model id {model_id}")
            }
            Self::InvalidFeatureScale => write!(f, "feature scale must be finite and positive"),
            Self::MatchIndexOutOfBounds => write!(f, "match index out of bounds"),
            Self::ModelIndexOutOfBounds => write!(f, "model index out of bounds"),
            Self::InvalidConfig(message) => write!(f, "invalid Hough configuration: {message}"),
            Self::Geometry(error) => write!(f, "geometry error: {error}"),
        }
    }
}

impl std::error::Error for RecognitionError {}

impl From<GeometryError> for RecognitionError {
    fn from(value: GeometryError) -> Self {
        Self::Geometry(value)
    }
}

/// Predicts the query-image pose of a model from one matched keypoint pair.
pub fn predict_similarity_pose(
    query_feature: &Feature,
    model_feature: &Feature,
) -> Result<SimilarityPose, RecognitionError> {
    let query = &query_feature.keypoint;
    let model = &model_feature.keypoint;
    if !query.scale.is_finite()
        || !model.scale.is_finite()
        || query.scale <= EPSILON
        || model.scale <= EPSILON
    {
        return Err(RecognitionError::InvalidFeatureScale);
    }

    let scale = query.scale / model.scale;
    if !scale.is_finite() || scale <= EPSILON {
        return Err(RecognitionError::InvalidFeatureScale);
    }

    let orientation = wrap_angle(query.angle - model.angle);
    let cos = orientation.cos();
    let sin = orientation.sin();
    let projected_model_x = scale * (cos * model.x - sin * model.y);
    let projected_model_y = scale * (sin * model.x + cos * model.y);

    Ok(SimilarityPose {
        x: query.x - projected_model_x,
        y: query.y - projected_model_y,
        scale,
        orientation,
    })
}

/// Clusters descriptor matches with Lowe's generalized Hough voting scheme.
///
/// Each match votes into the two closest bins in x, y, scale, and orientation,
/// yielding 16 votes per match. Returned clusters are sorted by descending vote count.
pub fn cluster_matches_hough(
    matches: &[DescriptorMatch],
    query_features: &[Feature],
    database: &ModelDatabase,
    config: HoughConfig,
) -> Result<Vec<HoughCluster>, RecognitionError> {
    config.validate()?;

    let orientation_bins = orientation_bin_count(config.orientation_bin_width);
    let mut votes: HashMap<HoughBin, Vec<usize>> = HashMap::new();
    let mut poses_by_match = vec![None; matches.len()];

    for (match_index, descriptor_match) in matches.iter().enumerate() {
        let query_feature = query_features
            .get(descriptor_match.query_index)
            .ok_or(RecognitionError::MatchIndexOutOfBounds)?;
        let record = database.record_for_train_index(descriptor_match.train_index)?;
        let model_feature = database.feature_for_train_index(descriptor_match.train_index)?;
        let model = database.model_for_record(record)?;
        let pose = predict_similarity_pose(query_feature, model_feature)?;
        let projected_dimension = model.max_dimension() * pose.scale;
        if !projected_dimension.is_finite() || projected_dimension <= EPSILON {
            return Err(RecognitionError::InvalidFeatureScale);
        }

        poses_by_match[match_index] = Some(pose);
        let location_bin_size = config.location_bin_factor * projected_dimension;
        let x_bins = two_closest_bins(pose.x / location_bin_size);
        let y_bins = two_closest_bins(pose.y / location_bin_size);
        let scale_bins = two_closest_bins(pose.scale.log2() / config.scale_bin_width);
        let orientation_bins_for_pose = two_closest_orientation_bins(
            pose.orientation,
            config.orientation_bin_width,
            orientation_bins,
        );

        for x_bin in x_bins {
            for y_bin in y_bins {
                for scale_bin in scale_bins {
                    for orientation_bin in orientation_bins_for_pose {
                        let bin = HoughBin {
                            model_id: record.model_id,
                            x_bin,
                            y_bin,
                            scale_bin,
                            orientation_bin,
                        };
                        votes.entry(bin).or_default().push(match_index);
                    }
                }
            }
        }
    }

    let mut clusters = Vec::new();
    let mut seen_match_sets: HashSet<(u32, Vec<usize>)> = HashSet::new();

    for (bin, mut match_indices) in votes {
        match_indices.sort_unstable();
        match_indices.dedup();
        if match_indices.len() < config.min_votes {
            continue;
        }
        if !seen_match_sets.insert((bin.model_id, match_indices.clone())) {
            continue;
        }
        let pose = average_pose(&match_indices, &poses_by_match)?;
        clusters.push(HoughCluster {
            model_id: bin.model_id,
            bin,
            votes: match_indices.len(),
            match_indices,
            pose,
        });
    }

    clusters.sort_by(|a, b| {
        b.votes
            .cmp(&a.votes)
            .then_with(|| a.model_id.cmp(&b.model_id))
            .then_with(|| a.bin.x_bin.cmp(&b.bin.x_bin))
            .then_with(|| a.bin.y_bin.cmp(&b.bin.y_bin))
    });
    Ok(clusters)
}

/// Performs affine least-squares verification and iterative outlier rejection for Hough clusters.
pub fn verify_hough_clusters(
    matches: &[DescriptorMatch],
    query_features: &[Feature],
    database: &ModelDatabase,
    clusters: &[HoughCluster],
    config: HoughConfig,
) -> Result<Vec<ObjectHypothesis>, RecognitionError> {
    config.validate()?;
    let mut hypotheses = Vec::new();

    for (cluster_index, cluster) in clusters.iter().enumerate() {
        let Some(model) = database.model(cluster.model_id) else {
            return Err(RecognitionError::ModelIndexOutOfBounds);
        };
        let residual_threshold =
            config.affine_residual_factor * model.max_dimension() * cluster.pose.scale.max(EPSILON);
        let mut inliers = cluster.match_indices.clone();
        let mut accepted_affine = None;

        for _ in 0..config.max_verification_iterations {
            if inliers.len() < config.min_votes {
                break;
            }
            let pairs = point_pairs_for_matches(&inliers, matches, query_features, database)?;
            let affine = match estimate_affine_from_pairs(&pairs) {
                Ok(affine) => affine,
                Err(GeometryError::NotEnoughPairs | GeometryError::SingularSystem) => break,
                Err(error) => return Err(error.into()),
            };

            let mut next_inliers = Vec::with_capacity(inliers.len());
            for &match_index in &inliers {
                let error =
                    reprojection_error(match_index, matches, query_features, database, affine)?;
                if error <= residual_threshold {
                    next_inliers.push(match_index);
                }
            }

            if next_inliers.len() < config.min_votes {
                break;
            }

            // Lowe's verification stage also performs a top-down pass to add
            // model matches that were missed by the approximate Hough binning.
            for (match_index, descriptor_match) in matches.iter().enumerate() {
                if next_inliers.contains(&match_index) {
                    continue;
                }
                let record = database.record_for_train_index(descriptor_match.train_index)?;
                if record.model_id != cluster.model_id {
                    continue;
                }
                let error =
                    reprojection_error(match_index, matches, query_features, database, affine)?;
                if error <= residual_threshold {
                    next_inliers.push(match_index);
                }
            }
            next_inliers.sort_unstable();
            next_inliers.dedup();

            let converged = next_inliers == inliers;
            inliers = next_inliers;
            accepted_affine = Some(affine);
            if converged {
                break;
            }
        }

        if accepted_affine.is_some() && inliers.len() >= config.min_votes {
            let pairs = point_pairs_for_matches(&inliers, matches, query_features, database)?;
            let affine = match estimate_affine_from_pairs(&pairs) {
                Ok(affine) => affine,
                Err(GeometryError::NotEnoughPairs | GeometryError::SingularSystem) => continue,
                Err(error) => return Err(error.into()),
            };
            let rmse = rmse_for_inliers(&inliers, matches, query_features, database, affine)?;
            hypotheses.push(ObjectHypothesis {
                model_id: cluster.model_id,
                cluster_index,
                affine,
                inlier_match_indices: inliers,
                rmse,
            });
        }
    }

    hypotheses.sort_by(|a, b| {
        b.inlier_match_indices
            .len()
            .cmp(&a.inlier_match_indices.len())
            .then_with(|| a.rmse.total_cmp(&b.rmse))
            .then_with(|| a.model_id.cmp(&b.model_id))
    });
    Ok(hypotheses)
}

fn point_pairs_for_matches(
    match_indices: &[usize],
    matches: &[DescriptorMatch],
    query_features: &[Feature],
    database: &ModelDatabase,
) -> Result<Vec<((f32, f32), (f32, f32))>, RecognitionError> {
    let mut pairs = Vec::with_capacity(match_indices.len());
    for &match_index in match_indices {
        let descriptor_match = matches
            .get(match_index)
            .ok_or(RecognitionError::MatchIndexOutOfBounds)?;
        let query = query_features
            .get(descriptor_match.query_index)
            .ok_or(RecognitionError::MatchIndexOutOfBounds)?;
        let model = database.feature_for_train_index(descriptor_match.train_index)?;
        pairs.push((
            (model.keypoint.x, model.keypoint.y),
            (query.keypoint.x, query.keypoint.y),
        ));
    }
    Ok(pairs)
}

fn reprojection_error(
    match_index: usize,
    matches: &[DescriptorMatch],
    query_features: &[Feature],
    database: &ModelDatabase,
    affine: Affine2,
) -> Result<f32, RecognitionError> {
    let descriptor_match = matches
        .get(match_index)
        .ok_or(RecognitionError::MatchIndexOutOfBounds)?;
    let query = query_features
        .get(descriptor_match.query_index)
        .ok_or(RecognitionError::MatchIndexOutOfBounds)?;
    let model = database.feature_for_train_index(descriptor_match.train_index)?;
    let (x, y) = affine.transform_point(model.keypoint.x, model.keypoint.y);
    let dx = x - query.keypoint.x;
    let dy = y - query.keypoint.y;
    Ok((dx * dx + dy * dy).sqrt())
}

fn rmse_for_inliers(
    inliers: &[usize],
    matches: &[DescriptorMatch],
    query_features: &[Feature],
    database: &ModelDatabase,
    affine: Affine2,
) -> Result<f32, RecognitionError> {
    let mut squared_error_sum = 0.0f32;
    for &match_index in inliers {
        let error = reprojection_error(match_index, matches, query_features, database, affine)?;
        squared_error_sum += error * error;
    }
    Ok((squared_error_sum / inliers.len() as f32).sqrt())
}

fn average_pose(
    match_indices: &[usize],
    poses_by_match: &[Option<SimilarityPose>],
) -> Result<SimilarityPose, RecognitionError> {
    let mut x = 0.0;
    let mut y = 0.0;
    let mut log_scale = 0.0;
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;

    for &match_index in match_indices {
        let pose = poses_by_match
            .get(match_index)
            .and_then(|pose| *pose)
            .ok_or(RecognitionError::MatchIndexOutOfBounds)?;
        x += pose.x;
        y += pose.y;
        log_scale += pose.scale.ln();
        sin_sum += pose.orientation.sin();
        cos_sum += pose.orientation.cos();
    }

    let count = match_indices.len() as f32;
    Ok(SimilarityPose {
        x: x / count,
        y: y / count,
        scale: (log_scale / count).exp(),
        orientation: wrap_angle(sin_sum.atan2(cos_sum)),
    })
}

fn two_closest_bins(value: f32) -> [i32; 2] {
    let nearest = value.round() as i32;
    let second = if value >= nearest as f32 {
        nearest + 1
    } else {
        nearest - 1
    };
    [nearest, second]
}

fn two_closest_orientation_bins(angle: f32, bin_width: f32, bin_count: i32) -> [i32; 2] {
    let raw = wrap_angle(angle) / bin_width;
    let nearest = raw.round() as i32;
    let second = if raw >= nearest as f32 {
        nearest + 1
    } else {
        nearest - 1
    };
    [
        positive_mod(nearest, bin_count),
        positive_mod(second, bin_count),
    ]
}

fn orientation_bin_count(bin_width: f32) -> i32 {
    ((TWO_PI / bin_width).round() as i32).max(1)
}

fn positive_mod(value: i32, modulus: i32) -> i32 {
    ((value % modulus) + modulus) % modulus
}

fn wrap_angle(angle: f32) -> f32 {
    angle.rem_euclid(TWO_PI)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DESCRIPTOR_LEN, Descriptor, Keypoint};

    fn feature(x: f32, y: f32, scale: f32, angle: f32) -> Feature {
        Feature {
            keypoint: Keypoint {
                x,
                y,
                scale,
                size: 2.0 * scale,
                angle,
                response: 1.0,
                octave: 0,
                layer: 0,
            },
            descriptor: Descriptor::new([0.0; DESCRIPTOR_LEN]),
        }
    }

    fn transform(x: f32, y: f32, scale: f32, angle: f32, tx: f32, ty: f32) -> (f32, f32) {
        let cos = angle.cos();
        let sin = angle.sin();
        (
            tx + scale * (cos * x - sin * y),
            ty + scale * (sin * x + cos * y),
        )
    }

    #[test]
    fn hough_clusters_and_affine_verification_recover_similarity() {
        let model_features = vec![
            feature(0.0, 0.0, 1.0, 0.0),
            feature(10.0, 0.0, 1.0, 0.0),
            feature(0.0, 10.0, 1.0, 0.0),
        ];
        let model = ObjectModel::new(7, 10.0, 10.0, model_features).unwrap();
        let database = ModelDatabase::new(vec![model]).unwrap();

        let scale = 2.0;
        let angle = 0.4;
        let tx = 20.0;
        let ty = 30.0;
        let query_features: Vec<_> = database
            .train_features()
            .iter()
            .map(|f| {
                let (x, y) = transform(f.keypoint.x, f.keypoint.y, scale, angle, tx, ty);
                feature(x, y, scale, angle)
            })
            .collect();
        let matches = vec![
            DescriptorMatch {
                query_index: 0,
                train_index: 0,
                distance: 0.1,
                second_distance: 1.0,
                ratio: 0.1,
            },
            DescriptorMatch {
                query_index: 1,
                train_index: 1,
                distance: 0.1,
                second_distance: 1.0,
                ratio: 0.1,
            },
            DescriptorMatch {
                query_index: 2,
                train_index: 2,
                distance: 0.1,
                second_distance: 1.0,
                ratio: 0.1,
            },
        ];

        let clusters =
            cluster_matches_hough(&matches, &query_features, &database, HoughConfig::default())
                .unwrap();
        assert!(!clusters.is_empty());
        assert!(clusters.iter().any(|cluster| cluster.votes >= 3));

        let hypotheses = verify_hough_clusters(
            &matches,
            &query_features,
            &database,
            &clusters,
            HoughConfig::default(),
        )
        .unwrap();
        assert!(!hypotheses.is_empty());
        let best = &hypotheses[0];
        assert_eq!(best.model_id, 7);
        assert_eq!(best.inlier_match_indices.len(), 3);
        assert!(best.rmse < 1.0e-4);
    }

    #[test]
    fn duplicate_model_ids_are_rejected() {
        let model_a = ObjectModel::new(1, 10.0, 10.0, Vec::new()).unwrap();
        let model_b = ObjectModel::new(1, 20.0, 20.0, Vec::new()).unwrap();
        assert_eq!(
            ModelDatabase::new(vec![model_a, model_b]).unwrap_err(),
            RecognitionError::DuplicateModelId(1)
        );
    }
}
