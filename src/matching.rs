use crate::{Descriptor, Feature};

/// A descriptor match accepted by Lowe's distance-ratio test.
#[derive(Clone, Debug, PartialEq)]
pub struct DescriptorMatch {
    /// Index in the query descriptor/feature slice.
    pub query_index: usize,
    /// Index in the train descriptor/feature slice.
    pub train_index: usize,
    /// Euclidean distance to the nearest neighbor.
    pub distance: f32,
    /// Euclidean distance to the second-nearest neighbor.
    pub second_distance: f32,
    /// `distance / second_distance`.
    pub ratio: f32,
}

/// Matches two descriptor sets with exact nearest-neighbor search and Lowe's ratio test.
///
/// The usual paper value is `ratio_threshold = 0.8`. The implementation is exhaustive;
/// it is intended as a correctness baseline and for small/medium descriptor sets.
pub fn match_descriptors(
    query: &[Descriptor],
    train: &[Descriptor],
    ratio_threshold: f32,
) -> Vec<DescriptorMatch> {
    if query.is_empty() || train.len() < 2 || !ratio_threshold.is_finite() || ratio_threshold <= 0.0
    {
        return Vec::new();
    }

    let ratio2 = ratio_threshold * ratio_threshold;
    let mut matches = Vec::new();

    for (query_index, descriptor) in query.iter().enumerate() {
        let mut best_index = usize::MAX;
        let mut best = f32::INFINITY;
        let mut second = f32::INFINITY;

        for (train_index, candidate) in train.iter().enumerate() {
            let distance2 = descriptor.distance2(candidate);
            if distance2 < best {
                second = best;
                best = distance2;
                best_index = train_index;
            } else if distance2 < second {
                second = distance2;
            }
        }

        if best_index == usize::MAX
            || !best.is_finite()
            || !second.is_finite()
            || second <= f32::EPSILON
        {
            continue;
        }
        if best < ratio2 * second {
            let distance = best.sqrt();
            let second_distance = second.sqrt();
            matches.push(DescriptorMatch {
                query_index,
                train_index: best_index,
                distance,
                second_distance,
                ratio: distance / second_distance,
            });
        }
    }

    matches
}

/// Matches feature descriptors with exact nearest-neighbor search and Lowe's ratio test.
pub fn match_features(
    query: &[Feature],
    train: &[Feature],
    ratio_threshold: f32,
) -> Vec<DescriptorMatch> {
    if query.is_empty() || train.len() < 2 || !ratio_threshold.is_finite() || ratio_threshold <= 0.0
    {
        return Vec::new();
    }

    let ratio2 = ratio_threshold * ratio_threshold;
    let mut matches = Vec::new();

    for (query_index, feature) in query.iter().enumerate() {
        let mut best_index = usize::MAX;
        let mut best = f32::INFINITY;
        let mut second = f32::INFINITY;

        for (train_index, candidate) in train.iter().enumerate() {
            let distance2 = feature.descriptor.distance2(&candidate.descriptor);
            if distance2 < best {
                second = best;
                best = distance2;
                best_index = train_index;
            } else if distance2 < second {
                second = distance2;
            }
        }

        if best_index == usize::MAX
            || !best.is_finite()
            || !second.is_finite()
            || second <= f32::EPSILON
        {
            continue;
        }
        if best < ratio2 * second {
            let distance = best.sqrt();
            let second_distance = second.sqrt();
            matches.push(DescriptorMatch {
                query_index,
                train_index: best_index,
                distance,
                second_distance,
                ratio: distance / second_distance,
            });
        }
    }

    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor_with_first(value: f32) -> Descriptor {
        let mut values = [0.0; crate::DESCRIPTOR_LEN];
        values[0] = value;
        Descriptor::new(values)
    }

    #[test]
    fn ratio_test_accepts_distinct_best_match() {
        let query = [descriptor_with_first(0.0)];
        let train = [descriptor_with_first(0.1), descriptor_with_first(1.0)];
        let matches = match_descriptors(&query, &train, 0.8);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].train_index, 0);
    }

    #[test]
    fn ratio_test_rejects_ambiguous_match() {
        let query = [descriptor_with_first(0.0)];
        let train = [descriptor_with_first(0.5), descriptor_with_first(0.55)];
        let matches = match_descriptors(&query, &train, 0.8);
        assert!(matches.is_empty());
    }
}
