use std::fmt;

use crate::{Feature, matching::DescriptorMatch};

/// A 2D affine transform.
///
/// The transform maps `(x, y)` to:
///
/// ```text
/// u = m11 * x + m12 * y + tx
/// v = m21 * x + m22 * y + ty
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine2 {
    /// Matrix entry in row 1, column 1.
    pub m11: f32,
    /// Matrix entry in row 1, column 2.
    pub m12: f32,
    /// Matrix entry in row 2, column 1.
    pub m21: f32,
    /// Matrix entry in row 2, column 2.
    pub m22: f32,
    /// X translation.
    pub tx: f32,
    /// Y translation.
    pub ty: f32,
}

impl Affine2 {
    /// Identity transform.
    pub const IDENTITY: Self = Self {
        m11: 1.0,
        m12: 0.0,
        m21: 0.0,
        m22: 1.0,
        tx: 0.0,
        ty: 0.0,
    };

    /// Applies this transform to `(x, y)`.
    #[inline]
    pub fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.m11 * x + self.m12 * y + self.tx,
            self.m21 * x + self.m22 * y + self.ty,
        )
    }
}

/// Errors from affine pose estimation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeometryError {
    /// Fewer than three point pairs were supplied.
    NotEnoughPairs,
    /// A match referenced a missing query or train feature.
    MatchIndexOutOfBounds,
    /// The least-squares normal equations were singular or ill-conditioned.
    SingularSystem,
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotEnoughPairs => write!(f, "at least three point pairs are required"),
            Self::MatchIndexOutOfBounds => write!(f, "match index out of bounds"),
            Self::SingularSystem => write!(f, "singular affine least-squares system"),
        }
    }
}

impl std::error::Error for GeometryError {}

/// Estimates an affine transform from source points to target points by least squares.
///
/// Each pair is `((source_x, source_y), (target_x, target_y))`. At least three
/// non-degenerate pairs are required.
pub fn estimate_affine_from_pairs(
    pairs: &[((f32, f32), (f32, f32))],
) -> Result<Affine2, GeometryError> {
    if pairs.len() < 3 {
        return Err(GeometryError::NotEnoughPairs);
    }

    let mut normal = [[0.0_f64; 6]; 6];
    let mut rhs = [0.0_f64; 6];

    for &((x, y), (u, v)) in pairs {
        if !(x.is_finite() && y.is_finite() && u.is_finite() && v.is_finite()) {
            return Err(GeometryError::SingularSystem);
        }
        let row_u = [x as f64, y as f64, 0.0, 0.0, 1.0, 0.0];
        let row_v = [0.0, 0.0, x as f64, y as f64, 0.0, 1.0];
        accumulate_normal_equations(&mut normal, &mut rhs, row_u, u as f64);
        accumulate_normal_equations(&mut normal, &mut rhs, row_v, v as f64);
    }

    let solution = solve_6x6(normal, rhs).ok_or(GeometryError::SingularSystem)?;
    Ok(Affine2 {
        m11: solution[0] as f32,
        m12: solution[1] as f32,
        m21: solution[2] as f32,
        m22: solution[3] as f32,
        tx: solution[4] as f32,
        ty: solution[5] as f32,
    })
}

/// Estimates the affine transform that maps train-image keypoints to query-image keypoints.
///
/// This matches the model-to-image least-squares verification step described by Lowe:
/// the training features are treated as model points and query features as image points.
pub fn estimate_affine_train_to_query(
    matches: &[DescriptorMatch],
    query: &[Feature],
    train: &[Feature],
) -> Result<Affine2, GeometryError> {
    if matches.len() < 3 {
        return Err(GeometryError::NotEnoughPairs);
    }
    let mut pairs = Vec::with_capacity(matches.len());
    for m in matches {
        let q = query
            .get(m.query_index)
            .ok_or(GeometryError::MatchIndexOutOfBounds)?;
        let t = train
            .get(m.train_index)
            .ok_or(GeometryError::MatchIndexOutOfBounds)?;
        pairs.push(((t.keypoint.x, t.keypoint.y), (q.keypoint.x, q.keypoint.y)));
    }
    estimate_affine_from_pairs(&pairs)
}

fn accumulate_normal_equations(
    normal: &mut [[f64; 6]; 6],
    rhs: &mut [f64; 6],
    row: [f64; 6],
    target: f64,
) {
    for i in 0..6 {
        rhs[i] += row[i] * target;
        for j in 0..6 {
            normal[i][j] += row[i] * row[j];
        }
    }
}

fn solve_6x6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
    const EPS: f64 = 1.0e-12;

    for col in 0..6 {
        let mut pivot = col;
        let mut pivot_abs = a[col][col].abs();
        for row in (col + 1)..6 {
            let value = a[row][col].abs();
            if value > pivot_abs {
                pivot = row;
                pivot_abs = value;
            }
        }
        if pivot_abs <= EPS {
            return None;
        }
        if pivot != col {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }

        let inv = 1.0 / a[col][col];
        for j in col..6 {
            a[col][j] *= inv;
        }
        b[col] *= inv;

        for row in 0..6 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor.abs() <= EPS {
                continue;
            }
            for j in col..6 {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_estimation_recovers_exact_transform() {
        let expected = Affine2 {
            m11: 1.2,
            m12: 0.3,
            m21: -0.2,
            m22: 0.9,
            tx: 4.0,
            ty: -2.0,
        };
        let sources = [(0.0, 0.0), (10.0, 0.0), (0.0, 5.0), (4.0, 6.0)];
        let pairs: Vec<_> = sources
            .iter()
            .map(|&(x, y)| ((x, y), expected.transform_point(x, y)))
            .collect();
        let actual = estimate_affine_from_pairs(&pairs).unwrap();
        assert!((actual.m11 - expected.m11).abs() < 1e-4);
        assert!((actual.m12 - expected.m12).abs() < 1e-4);
        assert!((actual.m21 - expected.m21).abs() < 1e-4);
        assert!((actual.m22 - expected.m22).abs() < 1e-4);
        assert!((actual.tx - expected.tx).abs() < 1e-4);
        assert!((actual.ty - expected.ty).abs() < 1e-4);
    }
}
