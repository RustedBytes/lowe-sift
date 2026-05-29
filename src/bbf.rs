use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

use crate::{DESCRIPTOR_LEN, Descriptor, Feature, matching::DescriptorMatch};

const DEFAULT_LEAF_SIZE: usize = 16;
const DEFAULT_MAX_CANDIDATES: usize = 200;
const DEFAULT_RATIO_THRESHOLD: f32 = 0.8;
const MIN_SPREAD: f32 = 1.0e-12;

/// Configuration for Best-Bin-First descriptor search.
///
/// Lowe's object-recognition implementation used approximate nearest-neighbor
/// lookup and stopped after checking the first 200 nearest-neighbor candidates.
/// The default values mirror that setting while keeping the leaf size modest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BbfConfig {
    /// Maximum number of descriptor candidates to evaluate for each query.
    pub max_candidates: usize,
    /// Maximum number of descriptors stored in a terminal k-d-tree leaf.
    pub leaf_size: usize,
    /// Lowe distance-ratio threshold applied to the two nearest returned neighbors.
    pub ratio_threshold: f32,
}

impl Default for BbfConfig {
    fn default() -> Self {
        Self {
            max_candidates: DEFAULT_MAX_CANDIDATES,
            leaf_size: DEFAULT_LEAF_SIZE,
            ratio_threshold: DEFAULT_RATIO_THRESHOLD,
        }
    }
}

impl BbfConfig {
    /// Validates that all configuration values are usable.
    pub fn validate(&self) -> Result<(), BbfConfigError> {
        if self.max_candidates == 0 {
            return Err(BbfConfigError::MaxCandidatesZero);
        }
        if self.leaf_size == 0 {
            return Err(BbfConfigError::LeafSizeZero);
        }
        if !self.ratio_threshold.is_finite() || self.ratio_threshold <= 0.0 {
            return Err(BbfConfigError::InvalidRatioThreshold);
        }
        Ok(())
    }
}

/// Errors caused by invalid [`BbfConfig`] values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BbfConfigError {
    /// `max_candidates` must be greater than zero.
    MaxCandidatesZero,
    /// `leaf_size` must be greater than zero.
    LeafSizeZero,
    /// `ratio_threshold` must be finite and greater than zero.
    InvalidRatioThreshold,
}

impl fmt::Display for BbfConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxCandidatesZero => write!(f, "max_candidates must be greater than zero"),
            Self::LeafSizeZero => write!(f, "leaf_size must be greater than zero"),
            Self::InvalidRatioThreshold => {
                write!(f, "ratio_threshold must be finite and greater than zero")
            }
        }
    }
}

impl std::error::Error for BbfConfigError {}

/// A neighbor returned by approximate Best-Bin-First descriptor lookup.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApproxNeighbor {
    /// Index of the neighbor in the indexed descriptor set.
    pub index: usize,
    /// Euclidean descriptor distance.
    pub distance: f32,
    /// Squared Euclidean descriptor distance.
    pub squared_distance: f32,
}

/// Best-Bin-First k-d-tree index for 128-dimensional SIFT descriptors.
///
/// The index copies descriptor values into a balanced k-d tree, then searches tree
/// bins in increasing lower-bound distance order using a heap. Search stops after
/// [`BbfConfig::max_candidates`] descriptor evaluations, which gives the same type
/// of approximate lookup used in Lowe's SIFT recognition pipeline.
#[derive(Clone, Debug)]
pub struct BbfDescriptorIndex {
    descriptors: Vec<Descriptor>,
    indices: Vec<usize>,
    nodes: Vec<Node>,
    root: Option<usize>,
    leaf_size: usize,
}

impl BbfDescriptorIndex {
    /// Builds an index with the default leaf size.
    pub fn new(descriptors: &[Descriptor]) -> Self {
        Self::with_leaf_size(descriptors, DEFAULT_LEAF_SIZE)
            .expect("the default BBF leaf size is valid")
    }

    /// Builds an index with a caller-provided leaf size.
    pub fn with_leaf_size(
        descriptors: &[Descriptor],
        leaf_size: usize,
    ) -> Result<Self, BbfConfigError> {
        if leaf_size == 0 {
            return Err(BbfConfigError::LeafSizeZero);
        }

        let descriptors = descriptors.to_vec();
        let mut index = Self {
            indices: (0..descriptors.len()).collect(),
            descriptors,
            nodes: Vec::new(),
            root: None,
            leaf_size,
        };

        if !index.indices.is_empty() {
            let root = index.build_node(0, index.indices.len());
            index.root = Some(root);
        }

        Ok(index)
    }

    /// Builds an index from feature descriptors with the default leaf size.
    pub fn from_features(features: &[Feature]) -> Self {
        let descriptors: Vec<_> = features.iter().map(|f| f.descriptor.clone()).collect();
        Self::new(&descriptors)
    }

    /// Returns the number of descriptors stored in the index.
    #[inline]
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// Returns true when the index contains no descriptors.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    /// Returns up to `k` approximate nearest neighbors of `query`.
    ///
    /// The result is sorted by increasing distance. `max_candidates` caps the
    /// number of terminal descriptor vectors evaluated during the search.
    fn nearest_into(
        &self,
        query: &Descriptor,
        k: usize,
        max_candidates: usize,
        best: &mut Vec<ApproxNeighbor>,
        queue: &mut BinaryHeap<QueueEntry>,
    ) -> Result<(), BbfConfigError> {
        if max_candidates == 0 {
            return Err(BbfConfigError::MaxCandidatesZero);
        }
        if k == 0 || self.is_empty() {
            return Ok(());
        }

        let Some(root) = self.root else {
            return Ok(());
        };

        queue.push(QueueEntry {
            lower_bound2: 0.0,
            node_index: root as u32,
        });

        let mut candidates = 0usize;

        while let Some(entry) = queue.pop() {
            if candidates >= max_candidates {
                break;
            }
            if best.len() >= k {
                let worst = best.last().map(|n: &ApproxNeighbor| n.squared_distance);
                if let Some(worst) = worst
                    && entry.lower_bound2 > worst {
                        break;
                    }
            }

            let node = &self.nodes[entry.node_index as usize];
            if node.is_leaf() {
                for slot in (node.start as usize)..(node.end as usize) {
                    if candidates >= max_candidates {
                        break;
                    }
                    let descriptor_index = self.indices[slot];
                    let distance2 = query.distance2(&self.descriptors[descriptor_index]);
                    candidates += 1;
                    insert_neighbor(best, k, descriptor_index, distance2);
                }
                continue;
            }

            let q = query.as_slice()[node.split_dim as usize];
            let axis_delta = q - node.split_value;
            let (near, far) = if axis_delta <= 0.0 {
                (node.left, node.right)
            } else {
                (node.right, node.left)
            };

            if near != u32::MAX {
                queue.push(QueueEntry {
                    lower_bound2: entry.lower_bound2,
                    node_index: near,
                });
            }
            if far != u32::MAX {
                queue.push(QueueEntry {
                    lower_bound2: entry.lower_bound2 + axis_delta * axis_delta,
                    node_index: far,
                });
            }
        }

        Ok(())
    }

    /// Returns up to `k` approximate nearest neighbors of `query`.
    ///
    /// The result is sorted by increasing distance. `max_candidates` caps the
    /// number of terminal descriptor vectors evaluated during the search.
    pub fn nearest(
        &self,
        query: &Descriptor,
        k: usize,
        max_candidates: usize,
    ) -> Result<Vec<ApproxNeighbor>, BbfConfigError> {
        let mut best = Vec::with_capacity(k.min(self.len()));
        let mut queue = BinaryHeap::new();
        self.nearest_into(query, k, max_candidates, &mut best, &mut queue)?;
        Ok(best)
    }

    /// Matches query descriptors against this index with Lowe's ratio test.
    pub fn match_descriptors(
        &self,
        query: &[Descriptor],
        config: BbfConfig,
    ) -> Result<Vec<DescriptorMatch>, BbfConfigError> {
        config.validate()?;
        if query.is_empty() || self.len() < 2 {
            return Ok(Vec::new());
        }

        let ratio2 = config.ratio_threshold * config.ratio_threshold;
        let mut matches = Vec::new();

        let mut best = Vec::with_capacity(2);
        let mut queue = BinaryHeap::with_capacity(config.max_candidates);

        for (query_index, descriptor) in query.iter().enumerate() {
            best.clear();
            queue.clear();
            self.nearest_into(descriptor, 2, config.max_candidates, &mut best, &mut queue)?;
            if best.len() < 2 {
                continue;
            }
            let first = best[0];
            let second = best[1];
            if !first.squared_distance.is_finite()
                || !second.squared_distance.is_finite()
                || second.squared_distance <= f32::EPSILON
            {
                continue;
            }
            if first.squared_distance < ratio2 * second.squared_distance {
                matches.push(DescriptorMatch {
                    query_index,
                    train_index: first.index,
                    distance: first.distance,
                    second_distance: second.distance,
                    ratio: first.distance / second.distance,
                });
            }
        }

        Ok(matches)
    }

    fn build_node(&mut self, start: usize, end: usize) -> usize {
        let node_index = self.nodes.len();
        self.nodes.push(Node::leaf(start, end));

        let len = end - start;
        if len <= self.leaf_size {
            return node_index;
        }

        let (split_dim, spread) = self.split_dimension(start, end);
        if spread <= MIN_SPREAD {
            return node_index;
        }

        let mid = start + len / 2;
        let descriptors = &self.descriptors;
        self.indices[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
            descriptors[a].as_slice()[split_dim].total_cmp(&descriptors[b].as_slice()[split_dim])
        });

        let split_value = self.descriptors[self.indices[mid]].as_slice()[split_dim];
        let left = self.build_node(start, mid);
        let right = self.build_node(mid, end);
        self.nodes[node_index] = Node {
            start: start as u32,
            end: end as u32,
            split_dim: split_dim as u32,
            split_value,
            left: left as u32,
            right: right as u32,
        };
        node_index
    }

    fn split_dimension(&self, start: usize, end: usize) -> (usize, f32) {
        let mut min_values = [f32::INFINITY; DESCRIPTOR_LEN];
        let mut max_values = [f32::NEG_INFINITY; DESCRIPTOR_LEN];

        for slot in start..end {
            let descriptor = self.descriptors[self.indices[slot]].as_slice();
            for dim in 0..DESCRIPTOR_LEN {
                let value = descriptor[dim];
                if value < min_values[dim] {
                    min_values[dim] = value;
                }
                if value > max_values[dim] {
                    max_values[dim] = value;
                }
            }
        }

        let mut best_dim = 0usize;
        let mut best_spread = 0.0f32;
        for dim in 0..DESCRIPTOR_LEN {
            let spread = max_values[dim] - min_values[dim];
            if spread > best_spread {
                best_spread = spread;
                best_dim = dim;
            }
        }
        (best_dim, best_spread)
    }
}

/// Matches two descriptor sets with approximate Best-Bin-First nearest-neighbor search.
///
/// This is a drop-in accelerated alternative to [`crate::matching::match_descriptors`].
pub fn match_descriptors_bbf(
    query: &[Descriptor],
    train: &[Descriptor],
    config: BbfConfig,
) -> Result<Vec<DescriptorMatch>, BbfConfigError> {
    config.validate()?;
    if query.is_empty() || train.len() < 2 {
        return Ok(Vec::new());
    }
    let index = BbfDescriptorIndex::with_leaf_size(train, config.leaf_size)?;
    index.match_descriptors(query, config)
}

/// Matches feature descriptors with approximate Best-Bin-First nearest-neighbor search.
pub fn match_features_bbf(
    query: &[Feature],
    train: &[Feature],
    config: BbfConfig,
) -> Result<Vec<DescriptorMatch>, BbfConfigError> {
    config.validate()?;
    if query.is_empty() || train.len() < 2 {
        return Ok(Vec::new());
    }

    let query_descriptors: Vec<_> = query.iter().map(|f| f.descriptor.clone()).collect();
    let train_descriptors: Vec<_> = train.iter().map(|f| f.descriptor.clone()).collect();
    match_descriptors_bbf(&query_descriptors, &train_descriptors, config)
}

#[derive(Clone, Debug)]
struct Node {
    start: u32,
    end: u32,
    split_dim: u32,
    split_value: f32,
    left: u32,
    right: u32,
}

impl Node {
    fn leaf(start: usize, end: usize) -> Self {
        Self {
            start: start as u32,
            end: end as u32,
            split_dim: 0,
            split_value: 0.0,
            left: u32::MAX,
            right: u32::MAX,
        }
    }

    fn is_leaf(&self) -> bool {
        self.left == u32::MAX && self.right == u32::MAX
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct QueueEntry {
    lower_bound2: f32,
    node_index: u32,
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .lower_bound2
            .total_cmp(&self.lower_bound2)
            .then_with(|| other.node_index.cmp(&self.node_index))
    }
}

#[inline]
fn insert_neighbor(best: &mut Vec<ApproxNeighbor>, k: usize, index: usize, squared_distance: f32) {
    if !squared_distance.is_finite() {
        return;
    }
    if best
        .iter()
        .any(|neighbor| neighbor.index == index && neighbor.squared_distance == squared_distance)
    {
        return;
    }
    let neighbor = ApproxNeighbor {
        index,
        distance: squared_distance.sqrt(),
        squared_distance,
    };
    best.push(neighbor);
    best.sort_by(|a, b| a.squared_distance.total_cmp(&b.squared_distance));
    if best.len() > k {
        best.truncate(k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor_with_first(value: f32) -> Descriptor {
        let mut values = [0.0; DESCRIPTOR_LEN];
        values[0] = value;
        Descriptor::new(values)
    }

    #[test]
    fn bbf_matches_simple_ratio_case() {
        let query = [descriptor_with_first(0.0)];
        let train = [descriptor_with_first(0.1), descriptor_with_first(1.0)];
        let matches = match_descriptors_bbf(
            &query,
            &train,
            BbfConfig {
                max_candidates: 8,
                ..BbfConfig::default()
            },
        )
        .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].train_index, 0);
    }

    #[test]
    fn nearest_returns_sorted_neighbors() {
        let descriptors = [
            descriptor_with_first(0.5),
            descriptor_with_first(0.1),
            descriptor_with_first(1.0),
        ];
        let index = BbfDescriptorIndex::with_leaf_size(&descriptors, 1).unwrap();
        let neighbors = index.nearest(&descriptor_with_first(0.0), 2, 10).unwrap();
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].index, 1);
        assert_eq!(neighbors[1].index, 0);
    }

    #[test]
    fn invalid_config_is_reported() {
        let config = BbfConfig {
            max_candidates: 0,
            ..BbfConfig::default()
        };
        assert_eq!(config.validate(), Err(BbfConfigError::MaxCandidatesZero));
    }
}
