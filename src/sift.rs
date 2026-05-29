use std::fmt;

use crate::gray::GrayImage;

/// Length of a Lowe SIFT descriptor: `4 * 4 * 8`.
pub const DESCRIPTOR_LEN: usize = 128;

const DESCRIPTOR_WIDTH: usize = 4;
const DESCRIPTOR_ORIENTATION_BINS: usize = 8;
const TWO_PI: f32 = std::f32::consts::PI * 2.0;
const SQRT_2: f32 = std::f32::consts::SQRT_2;
const EPSILON: f32 = 1.0e-7;

/// A normalized 128-dimensional SIFT descriptor.
#[derive(Clone, Debug, PartialEq)]
#[repr(align(32))]
pub struct Descriptor {
    values: [f32; DESCRIPTOR_LEN],
}

impl Descriptor {
    /// Creates a descriptor from an already normalized 128-element array.
    pub fn new(values: [f32; DESCRIPTOR_LEN]) -> Self {
        Self { values }
    }

    /// Returns the descriptor values.
    #[inline]
    pub fn as_slice(&self) -> &[f32; DESCRIPTOR_LEN] {
        &self.values
    }

    /// Consumes the descriptor and returns the backing array.
    #[inline]
    pub fn into_array(self) -> [f32; DESCRIPTOR_LEN] {
        self.values
    }

    /// Squared Euclidean distance to another descriptor.
    #[inline]
    pub fn distance2(&self, other: &Self) -> f32 {
        #[cfg(feature = "simd")]
        {
            use std::simd::f32x16;
            use std::simd::num::SimdFloat;
            let a = &self.values;
            let b = &other.values;

            let va0 = f32x16::from_slice(&a[0..16]);
            let vb0 = f32x16::from_slice(&b[0..16]);
            let d0 = va0 - vb0;

            let va1 = f32x16::from_slice(&a[16..32]);
            let vb1 = f32x16::from_slice(&b[16..32]);
            let d1 = va1 - vb1;

            let va2 = f32x16::from_slice(&a[32..48]);
            let vb2 = f32x16::from_slice(&b[32..48]);
            let d2 = va2 - vb2;

            let va3 = f32x16::from_slice(&a[48..64]);
            let vb3 = f32x16::from_slice(&b[48..64]);
            let d3 = va3 - vb3;

            let va4 = f32x16::from_slice(&a[64..80]);
            let vb4 = f32x16::from_slice(&b[64..80]);
            let d4 = va4 - vb4;

            let va5 = f32x16::from_slice(&a[80..96]);
            let vb5 = f32x16::from_slice(&b[80..96]);
            let d5 = va5 - vb5;

            let va6 = f32x16::from_slice(&a[96..112]);
            let vb6 = f32x16::from_slice(&b[96..112]);
            let d6 = va6 - vb6;

            let va7 = f32x16::from_slice(&a[112..128]);
            let vb7 = f32x16::from_slice(&b[112..128]);
            let d7 = va7 - vb7;

            let sum0 = d0 * d0 + d1 * d1;
            let sum1 = d2 * d2 + d3 * d3;
            let sum2 = d4 * d4 + d5 * d5;
            let sum3 = d6 * d6 + d7 * d7;

            let final_sum = (sum0 + sum1) + (sum2 + sum3);
            final_sum.reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut sum = 0.0;
            for i in 0..DESCRIPTOR_LEN {
                let d = self.values[i] - other.values[i];
                sum += d * d;
            }
            sum
        }
    }

    pub(crate) fn from_mutated(values: [f32; DESCRIPTOR_LEN]) -> Self {
        Self { values }
    }
}

/// A localized, oriented SIFT keypoint in input-image coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Keypoint {
    /// X coordinate in the original input image.
    pub x: f32,
    /// Y coordinate in the original input image.
    pub y: f32,
    /// Gaussian scale `sigma` in original-image pixels.
    pub scale: f32,
    /// Feature diameter in original-image pixels (`2 * scale`).
    pub size: f32,
    /// Dominant orientation in radians in `[0, 2π)`.
    pub angle: f32,
    /// Absolute value of the interpolated DoG response.
    pub response: f32,
    /// Octave index in the internal pyramid.
    pub octave: i32,
    /// Scale-space layer index in the internal pyramid.
    pub layer: i32,
}

/// A SIFT keypoint and its 128-dimensional descriptor.
#[derive(Clone, Debug, PartialEq)]
pub struct Feature {
    /// Localized keypoint metadata.
    pub keypoint: Keypoint,
    /// Normalized SIFT descriptor.
    pub descriptor: Descriptor,
}

/// Configuration for [`Sift`].
///
/// Defaults follow the parameter choices used in Lowe's paper where the paper
/// gives explicit values: 3 intervals per octave, initial `sigma = 1.6`,
/// contrast rejection at `0.03`, edge-curvature ratio `10`, 36 orientation bins,
/// secondary orientation peaks at 80% of the dominant peak, and descriptor
/// clipping at `0.2`.
#[derive(Clone, Debug, PartialEq)]
pub struct SiftConfig {
    /// Number of sampled scale intervals per octave. Lowe uses 3.
    pub intervals: usize,
    /// Gaussian sigma of the first level in each octave. Lowe uses 1.6.
    pub sigma: f32,
    /// Assumed blur of the input image in input-image pixels. Lowe assumes at least 0.5.
    pub assumed_blur: f32,
    /// Double the input image before building the first octave.
    pub double_image: bool,
    /// Reject keypoints with interpolated absolute DoG response below this value.
    pub contrast_threshold: f32,
    /// Reject keypoints whose principal-curvature ratio is greater than this value.
    pub edge_threshold: f32,
    /// Maximum number of quadratic-localization updates per candidate.
    pub max_interpolation_steps: usize,
    /// Number of bins in the orientation-assignment histogram. Lowe uses 36.
    pub orientation_bins: usize,
    /// Keep secondary orientation peaks at least this fraction of the highest peak.
    pub orientation_peak_ratio: f32,
    /// Gaussian sigma for the orientation window as a multiple of keypoint scale.
    pub orientation_window_factor: f32,
    /// Smoothing passes applied to the orientation histogram before peak selection.
    pub orientation_smooth_passes: usize,
    /// Descriptor spatial-bin size as a multiple of keypoint scale.
    pub descriptor_scale: f32,
    /// Clamp normalized descriptor entries to this value, then renormalize.
    pub descriptor_clipping: f32,
    /// Stop constructing octaves once either image dimension is smaller than this.
    pub min_octave_size: usize,
}

impl Default for SiftConfig {
    fn default() -> Self {
        Self {
            intervals: 3,
            sigma: 1.6,
            assumed_blur: 0.5,
            double_image: true,
            contrast_threshold: 0.03,
            edge_threshold: 10.0,
            max_interpolation_steps: 5,
            orientation_bins: 36,
            orientation_peak_ratio: 0.8,
            orientation_window_factor: 1.5,
            orientation_smooth_passes: 2,
            descriptor_scale: 3.0,
            descriptor_clipping: 0.2,
            min_octave_size: 16,
        }
    }
}

/// Invalid [`SiftConfig`] values.
#[derive(Clone, Debug, PartialEq)]
pub enum SiftConfigError {
    /// `intervals` must be at least 1.
    IntervalsTooSmall,
    /// A floating-point parameter was non-finite or outside its valid range.
    InvalidParameter(&'static str),
    /// `orientation_bins` must be at least 3.
    OrientationBinsTooSmall,
    /// `min_octave_size` must be at least 8.
    MinOctaveSizeTooSmall,
}

impl fmt::Display for SiftConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IntervalsTooSmall => write!(f, "intervals must be at least 1"),
            Self::InvalidParameter(name) => write!(f, "invalid SIFT parameter: {name}"),
            Self::OrientationBinsTooSmall => write!(f, "orientation_bins must be at least 3"),
            Self::MinOctaveSizeTooSmall => write!(f, "min_octave_size must be at least 8"),
        }
    }
}

impl std::error::Error for SiftConfigError {}

impl SiftConfig {
    /// Validates this configuration.
    pub fn validate(&self) -> Result<(), SiftConfigError> {
        if self.intervals == 0 {
            return Err(SiftConfigError::IntervalsTooSmall);
        }
        if self.orientation_bins < 3 {
            return Err(SiftConfigError::OrientationBinsTooSmall);
        }
        if self.min_octave_size < 8 {
            return Err(SiftConfigError::MinOctaveSizeTooSmall);
        }
        validate_positive_finite("sigma", self.sigma)?;
        validate_nonnegative_finite("assumed_blur", self.assumed_blur)?;
        validate_positive_finite("contrast_threshold", self.contrast_threshold)?;
        validate_positive_finite("edge_threshold", self.edge_threshold)?;
        validate_positive_finite("orientation_peak_ratio", self.orientation_peak_ratio)?;
        if self.orientation_peak_ratio > 1.0 {
            return Err(SiftConfigError::InvalidParameter("orientation_peak_ratio"));
        }
        validate_positive_finite("orientation_window_factor", self.orientation_window_factor)?;
        validate_positive_finite("descriptor_scale", self.descriptor_scale)?;
        validate_positive_finite("descriptor_clipping", self.descriptor_clipping)?;
        Ok(())
    }
}

fn validate_positive_finite(name: &'static str, value: f32) -> Result<(), SiftConfigError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(SiftConfigError::InvalidParameter(name))
    }
}

fn validate_nonnegative_finite(name: &'static str, value: f32) -> Result<(), SiftConfigError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(SiftConfigError::InvalidParameter(name))
    }
}

/// SIFT detector and descriptor extractor.
#[derive(Clone, Debug)]
#[derive(Default)]
pub struct Sift {
    config: SiftConfig,
}


impl Sift {
    /// Creates a SIFT extractor after validating `config`.
    pub fn new(config: SiftConfig) -> Result<Self, SiftConfigError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Returns this extractor's configuration.
    #[inline]
    pub fn config(&self) -> &SiftConfig {
        &self.config
    }

    /// Detects oriented keypoints and computes descriptors.
    pub fn detect_and_compute(&self, image: &GrayImage) -> Vec<Feature> {
        if image.width() < 3 || image.height() < 3 {
            return Vec::new();
        }
        let pyramid = self.build_pyramid(image);
        let mut features = Vec::new();
        self.find_oriented_keypoints(&pyramid, |internal| {
            if let Some(descriptor) = self.compute_descriptor(&pyramid, &internal) {
                features.push(Feature {
                    keypoint: internal.keypoint,
                    descriptor,
                });
            }
        });
        features
    }

    /// Detects localized, oriented keypoints without computing descriptors.
    pub fn detect_keypoints(&self, image: &GrayImage) -> Vec<Keypoint> {
        if image.width() < 3 || image.height() < 3 {
            return Vec::new();
        }
        let pyramid = self.build_pyramid(image);
        let mut keypoints = Vec::new();
        self.find_oriented_keypoints(&pyramid, |internal| {
            keypoints.push(internal.keypoint);
        });
        keypoints
    }

    fn build_pyramid(&self, image: &GrayImage) -> Pyramid {
        let mut tmp_buffer = Vec::new();
        let base = self.create_base_image(image, &mut tmp_buffer);
        let mut octaves = Vec::new();
        let mut octave_base = base;
        let intervals = self.config.intervals;
        let gaussian_count = intervals + 3;
        let k = 2.0_f32.powf(1.0 / intervals as f32);

        let mut incremental_sigmas = [0.0; 32];
        incremental_sigmas[0] = self.config.sigma;
        let sigmas_len = gaussian_count.min(32);
        for i in 1..sigmas_len {
            let prev = self.config.sigma * k.powi(i as i32 - 1);
            let total = prev * k;
            incremental_sigmas[i] = (total * total - prev * prev).max(0.0).sqrt();
        }

        while octave_base.width() >= self.config.min_octave_size
            && octave_base.height() >= self.config.min_octave_size
        {
            let mut gaussians = Vec::with_capacity(gaussian_count);
            gaussians.push(octave_base);
            for i in 1..gaussian_count {
                let sigma = if i < sigmas_len { incremental_sigmas[i] } else { 0.0 };
                let next = gaussians
                    .last()
                    .expect("at least one gaussian")
                    .gaussian_blur(sigma, &mut tmp_buffer);
                gaussians.push(next);
            }

            let mut dogs = Vec::with_capacity(gaussian_count - 1);
            for i in 1..gaussians.len() {
                dogs.push(gaussians[i].subtract(&gaussians[i - 1]));
            }

            let next_base = gaussians[intervals].downsample_by_2();
            let prev_width = gaussians[0].width();
            let prev_height = gaussians[0].height();

            octaves.push(Octave { gaussians, dogs });
            if next_base.width() == prev_width
                || next_base.height() == prev_height
            {
                break;
            }
            octave_base = next_base;
        }

        Pyramid { octaves }
    }

    fn create_base_image(&self, image: &GrayImage, tmp_buffer: &mut Vec<f32>) -> GrayImage {
        let base = if self.config.double_image {
            image.double_linear()
        } else {
            image.clone()
        };
        let current_blur = if self.config.double_image {
            self.config.assumed_blur * 2.0
        } else {
            self.config.assumed_blur
        };
        let sigma_diff = (self.config.sigma * self.config.sigma - current_blur * current_blur)
            .max(0.0)
            .sqrt();
        base.gaussian_blur(sigma_diff, tmp_buffer)
    }

    fn find_oriented_keypoints<F>(&self, pyramid: &Pyramid, mut on_keypoint: F)
    where
        F: FnMut(InternalKeypoint),
    {
        for (octave_index, octave) in pyramid.octaves.iter().enumerate() {
            if octave.dogs.len() < 3 {
                continue;
            }
            let width = octave.dogs[0].width();
            let height = octave.dogs[0].height();
            if width <= 2 * self.image_border() || height <= 2 * self.image_border() {
                continue;
            }

            for layer in 1..(octave.dogs.len() - 1) {
                for y in self.image_border()..(height - self.image_border()) {
                    for x in self.image_border()..(width - self.image_border()) {
                        if !self.is_extremum(octave, layer, x, y) {
                            continue;
                        }
                        let Some(localized) = self.localize_extremum(octave, layer, x, y) else {
                            continue;
                        };
                        self.assign_orientations(octave, &localized, |angle| {
                            on_keypoint(self.make_internal_keypoint(
                                octave_index,
                                localized.clone(),
                                angle,
                            ));
                        });
                    }
                }
            }
        }
    }

    fn image_border(&self) -> usize {
        // Enough for derivative tests and most localization updates. Descriptor sampling performs
        // its own bounds checks, so this border can stay small to avoid needlessly dropping points.
        5
    }

    fn is_extremum(&self, octave: &Octave, layer: usize, x: usize, y: usize) -> bool {
        let value = octave.dogs[layer].get(x, y);
        if value.abs() < self.config.contrast_threshold * 0.5 / self.config.intervals as f32 {
            return false;
        }

        if value > 0.0 {
            for s in (layer - 1)..=(layer + 1) {
                for yy in (y - 1)..=(y + 1) {
                    for xx in (x - 1)..=(x + 1) {
                        if s == layer && yy == y && xx == x {
                            continue;
                        }
                        if octave.dogs[s].get(xx, yy) >= value {
                            return false;
                        }
                    }
                }
            }
            true
        } else {
            for s in (layer - 1)..=(layer + 1) {
                for yy in (y - 1)..=(y + 1) {
                    for xx in (x - 1)..=(x + 1) {
                        if s == layer && yy == y && xx == x {
                            continue;
                        }
                        if octave.dogs[s].get(xx, yy) <= value {
                            return false;
                        }
                    }
                }
            }
            true
        }
    }

    fn localize_extremum(
        &self,
        octave: &Octave,
        initial_layer: usize,
        initial_x: usize,
        initial_y: usize,
    ) -> Option<LocalizedKeypoint> {
        let width = octave.dogs[0].width();
        let height = octave.dogs[0].height();
        let mut layer = initial_layer as isize;
        let mut x = initial_x as isize;
        let mut y = initial_y as isize;
        let mut offset = [0.0; 3];
        let mut gradient;

        for _ in 0..self.config.max_interpolation_steps {
            if layer <= 0
                || layer >= octave.dogs.len() as isize - 1
                || x <= self.image_border() as isize
                || x >= width as isize - self.image_border() as isize
                || y <= self.image_border() as isize
                || y >= height as isize - self.image_border() as isize
            {
                return None;
            }

            gradient = dog_gradient(octave, layer as usize, x as usize, y as usize);
            let hessian = dog_hessian(octave, layer as usize, x as usize, y as usize);
            offset = solve_3x3(hessian, [-gradient[0], -gradient[1], -gradient[2]])?;

            if offset.iter().all(|v| v.abs() < 0.5) {
                break;
            }

            x += offset[0].round() as isize;
            y += offset[1].round() as isize;
            layer += offset[2].round() as isize;
        }

        if offset.iter().any(|v| v.abs() >= 1.5) {
            return None;
        }
        if layer <= 0 || layer >= octave.dogs.len() as isize - 1 {
            return None;
        }
        if x <= self.image_border() as isize
            || x >= width as isize - self.image_border() as isize
            || y <= self.image_border() as isize
            || y >= height as isize - self.image_border() as isize
        {
            return None;
        }

        gradient = dog_gradient(octave, layer as usize, x as usize, y as usize);
        let value = octave.dogs[layer as usize].get(x as usize, y as usize);
        let interpolated = value + 0.5 * dot3(gradient, offset);
        if interpolated.abs() < self.config.contrast_threshold {
            return None;
        }

        if !self.passes_edge_response_test(octave, layer as usize, x as usize, y as usize) {
            return None;
        }

        let octave_scale = self.config.sigma
            * 2.0_f32.powf((layer as f32 + offset[2]) / self.config.intervals as f32);
        if !octave_scale.is_finite() || octave_scale <= 0.0 {
            return None;
        }

        Some(LocalizedKeypoint {
            layer: layer as usize,
            octave_x: x as f32 + offset[0],
            octave_y: y as f32 + offset[1],
            octave_scale,
            response: interpolated.abs(),
        })
    }

    fn passes_edge_response_test(&self, octave: &Octave, layer: usize, x: usize, y: usize) -> bool {
        let dog = &octave.dogs[layer];
        let value = dog.get(x, y);
        let dxx = dog.get(x + 1, y) + dog.get(x - 1, y) - 2.0 * value;
        let dyy = dog.get(x, y + 1) + dog.get(x, y - 1) - 2.0 * value;
        let dxy = 0.25
            * (dog.get(x + 1, y + 1) - dog.get(x - 1, y + 1) - dog.get(x + 1, y - 1)
                + dog.get(x - 1, y - 1));

        let trace = dxx + dyy;
        let determinant = dxx * dyy - dxy * dxy;
        if determinant <= 0.0 {
            return false;
        }
        let edge = self.config.edge_threshold;
        trace * trace * edge < (edge + 1.0) * (edge + 1.0) * determinant
    }

    fn assign_orientations<F>(&self, octave: &Octave, keypoint: &LocalizedKeypoint, mut on_angle: F)
    where
        F: FnMut(f32),
    {
        let image = &octave.gaussians[keypoint.layer];
        let bins = self.config.orientation_bins;
        
        let mut stack_hist = [0.0; 128];
        let mut heap_hist;
        let histogram = if bins <= 128 {
            &mut stack_hist[..bins]
        } else {
            heap_hist = vec![0.0; bins];
            &mut heap_hist[..]
        };

        let sigma = self.config.orientation_window_factor * keypoint.octave_scale;
        let radius = (3.0 * sigma).round() as isize;
        let sigma2 = 2.0 * sigma * sigma;
        let center_x = keypoint.octave_x.round() as isize;
        let center_y = keypoint.octave_y.round() as isize;

        let width = image.width();
        let height = image.height();
        let data = image.data();

        for dy in -radius..=radius {
            let y = center_y + dy;
            if y <= 0 || y >= height as isize - 1 {
                continue;
            }
            let y_offset = y as usize * width;
            let y_prev_offset = (y - 1) as usize * width;
            let y_next_offset = (y + 1) as usize * width;

            for dx in -radius..=radius {
                let x = center_x + dx;
                if x <= 0 || x >= width as isize - 1 {
                    continue;
                }
                let rel_x = x as f32 - keypoint.octave_x;
                let rel_y = y as f32 - keypoint.octave_y;
                let weight = (-(rel_x * rel_x + rel_y * rel_y) / sigma2).exp();
                let x_usize = x as usize;
                let gx = data[y_offset + x_usize + 1] - data[y_offset + x_usize - 1];
                let gy = data[y_next_offset + x_usize] - data[y_prev_offset + x_usize];
                let magnitude = (gx * gx + gy * gy).sqrt();
                if magnitude <= EPSILON {
                    continue;
                }
                let angle = normalize_angle(gy.atan2(gx));
                let bin = angle * bins as f32 / TWO_PI;
                let low = bin.floor() as usize % bins;
                let fraction = bin - bin.floor();
                histogram[low] += weight * magnitude * (1.0 - fraction);
                histogram[(low + 1) % bins] += weight * magnitude * fraction;
            }
        }

        for _ in 0..self.config.orientation_smooth_passes {
            smooth_circular_histogram(histogram);
        }

        let max_value = histogram
            .iter()
            .copied()
            .fold(0.0_f32, |acc, v| if v > acc { v } else { acc });
        if max_value <= EPSILON {
            return;
        }

        let threshold = self.config.orientation_peak_ratio * max_value;
        for i in 0..bins {
            let left = histogram[(i + bins - 1) % bins];
            let center = histogram[i];
            let right = histogram[(i + 1) % bins];
            if center < threshold || center <= left || center <= right {
                continue;
            }
            let denom = left - 2.0 * center + right;
            let offset = if denom.abs() > EPSILON {
                0.5 * (left - right) / denom
            } else {
                0.0
            }
            .clamp(-0.5, 0.5);
            let interpolated_bin = i as f32 + offset;
            on_angle(normalize_angle(interpolated_bin * TWO_PI / bins as f32));
        }
    }

    fn make_internal_keypoint(
        &self,
        octave_index: usize,
        localized: LocalizedKeypoint,
        angle: f32,
    ) -> InternalKeypoint {
        let factor = self.octave_to_input_factor(octave_index);
        let scale = localized.octave_scale * factor;
        let keypoint = Keypoint {
            x: localized.octave_x * factor,
            y: localized.octave_y * factor,
            scale,
            size: 2.0 * scale,
            angle,
            response: localized.response,
            octave: octave_index as i32,
            layer: localized.layer as i32,
        };
        InternalKeypoint {
            keypoint,
            octave_index,
            layer: localized.layer,
            octave_x: localized.octave_x,
            octave_y: localized.octave_y,
            octave_scale: localized.octave_scale,
        }
    }

    fn octave_to_input_factor(&self, octave_index: usize) -> f32 {
        let octave_scale = 2.0_f32.powi(octave_index as i32);
        if self.config.double_image {
            octave_scale * 0.5
        } else {
            octave_scale
        }
    }

    fn compute_descriptor(
        &self,
        pyramid: &Pyramid,
        keypoint: &InternalKeypoint,
    ) -> Option<Descriptor> {
        let image = &pyramid.octaves[keypoint.octave_index].gaussians[keypoint.layer];
        let hist_width = self.config.descriptor_scale * keypoint.octave_scale;
        if !hist_width.is_finite() || hist_width <= EPSILON {
            return None;
        }

        let radius = (hist_width * SQRT_2 * (DESCRIPTOR_WIDTH as f32 + 1.0) * 0.5).ceil() as isize;
        let center_x = keypoint.octave_x.round() as isize;
        let center_y = keypoint.octave_y.round() as isize;
        let cos_t = keypoint.keypoint.angle.cos();
        let sin_t = keypoint.keypoint.angle.sin();
        let descriptor_half = DESCRIPTOR_WIDTH as f32 * 0.5;
        let weight_sigma = descriptor_half;
        let weight_denom = 2.0 * weight_sigma * weight_sigma;
        let mut hist = [0.0_f32; DESCRIPTOR_LEN];

        let width = image.width();
        let height = image.height();
        let data = image.data();

        for yy in (center_y - radius)..=(center_y + radius) {
            if yy <= 0 || yy >= height as isize - 1 {
                continue;
            }
            let y_offset = yy as usize * width;
            let y_prev_offset = (yy - 1) as usize * width;
            let y_next_offset = (yy + 1) as usize * width;

            for xx in (center_x - radius)..=(center_x + radius) {
                if xx <= 0 || xx >= width as isize - 1 {
                    continue;
                }

                let rel_x = xx as f32 - keypoint.octave_x;
                let rel_y = yy as f32 - keypoint.octave_y;
                let c_rot = (cos_t * rel_x + sin_t * rel_y) / hist_width;
                let r_rot = (-sin_t * rel_x + cos_t * rel_y) / hist_width;
                let cbin = c_rot + descriptor_half - 0.5;
                let rbin = r_rot + descriptor_half - 0.5;
                if !(rbin > -1.0
                    && rbin < DESCRIPTOR_WIDTH as f32
                    && cbin > -1.0
                    && cbin < DESCRIPTOR_WIDTH as f32)
                {
                    continue;
                }

                let xx_usize = xx as usize;
                let gx = data[y_offset + xx_usize + 1] - data[y_offset + xx_usize - 1];
                let gy = data[y_next_offset + xx_usize] - data[y_prev_offset + xx_usize];
                let magnitude = (gx * gx + gy * gy).sqrt();
                if magnitude <= EPSILON {
                    continue;
                }

                let orientation = normalize_angle(gy.atan2(gx) - keypoint.keypoint.angle);
                let obin = orientation * DESCRIPTOR_ORIENTATION_BINS as f32 / TWO_PI;
                let weight = (-(c_rot * c_rot + r_rot * r_rot) / weight_denom).exp() * magnitude;
                trilinear_accumulate(&mut hist, rbin, cbin, obin, weight);
            }
        }

        normalize_descriptor(&mut hist)?;
        for v in &mut hist {
            if *v > self.config.descriptor_clipping {
                *v = self.config.descriptor_clipping;
            }
        }
        normalize_descriptor(&mut hist)?;
        Some(Descriptor::from_mutated(hist))
    }
}

#[derive(Clone)]
struct Pyramid {
    octaves: Vec<Octave>,
}

#[derive(Clone)]
struct Octave {
    gaussians: Vec<GrayImage>,
    dogs: Vec<GrayImage>,
}

#[derive(Clone, Debug)]
struct LocalizedKeypoint {
    layer: usize,
    octave_x: f32,
    octave_y: f32,
    octave_scale: f32,
    response: f32,
}

#[derive(Clone, Debug)]
struct InternalKeypoint {
    keypoint: Keypoint,
    octave_index: usize,
    layer: usize,
    octave_x: f32,
    octave_y: f32,
    octave_scale: f32,
}

#[inline]
fn dog_gradient(octave: &Octave, layer: usize, x: usize, y: usize) -> [f32; 3] {
    let current = &octave.dogs[layer];
    [
        0.5 * (current.get(x + 1, y) - current.get(x - 1, y)),
        0.5 * (current.get(x, y + 1) - current.get(x, y - 1)),
        0.5 * (octave.dogs[layer + 1].get(x, y) - octave.dogs[layer - 1].get(x, y)),
    ]
}

#[inline]
fn dog_hessian(octave: &Octave, layer: usize, x: usize, y: usize) -> [[f32; 3]; 3] {
    let current = &octave.dogs[layer];
    let previous = &octave.dogs[layer - 1];
    let next = &octave.dogs[layer + 1];
    let value = current.get(x, y);

    let dxx = current.get(x + 1, y) + current.get(x - 1, y) - 2.0 * value;
    let dyy = current.get(x, y + 1) + current.get(x, y - 1) - 2.0 * value;
    let dss = next.get(x, y) + previous.get(x, y) - 2.0 * value;
    let dxy = 0.25
        * (current.get(x + 1, y + 1) - current.get(x - 1, y + 1) - current.get(x + 1, y - 1)
            + current.get(x - 1, y - 1));
    let dxs = 0.25
        * (next.get(x + 1, y) - next.get(x - 1, y) - previous.get(x + 1, y)
            + previous.get(x - 1, y));
    let dys = 0.25
        * (next.get(x, y + 1) - next.get(x, y - 1) - previous.get(x, y + 1)
            + previous.get(x, y - 1));

    [[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]]
}

#[inline]
fn solve_3x3(mut a: [[f32; 3]; 3], mut b: [f32; 3]) -> Option<[f32; 3]> {
    for col in 0..3 {
        let mut pivot = col;
        let mut pivot_abs = a[col][col].abs();
        for row in (col + 1)..3 {
            let value = a[row][col].abs();
            if value > pivot_abs {
                pivot = row;
                pivot_abs = value;
            }
        }
        if pivot_abs <= EPSILON {
            return None;
        }
        if pivot != col {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }

        let inv = 1.0 / a[col][col];
        for j in col..3 {
            a[col][j] *= inv;
        }
        b[col] *= inv;

        for row in 0..3 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor.abs() <= EPSILON {
                continue;
            }
            for j in col..3 {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    Some(b)
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn normalize_angle(mut angle: f32) -> f32 {
    angle %= TWO_PI;
    if angle < 0.0 {
        angle += TWO_PI;
    }
    angle
}

#[inline]
fn smooth_circular_histogram(histogram: &mut [f32]) {
    let n = histogram.len();
    if n < 3 {
        return;
    }
    let mut stack_buf = [0.0; 128];
    let heap_buf;
    let original = if n <= 128 {
        stack_buf[..n].copy_from_slice(histogram);
        &stack_buf[..n]
    } else {
        heap_buf = histogram.to_vec();
        &heap_buf[..]
    };
    histogram[0] = (original[n - 1] + original[0] + original[1]) / 3.0;
    for i in 1..(n - 1) {
        histogram[i] = (original[i - 1] + original[i] + original[i + 1]) / 3.0;
    }
    histogram[n - 1] = (original[n - 2] + original[n - 1] + original[0]) / 3.0;
}

#[inline]
fn trilinear_accumulate(
    hist: &mut [f32; DESCRIPTOR_LEN],
    rbin: f32,
    cbin: f32,
    obin: f32,
    value: f32,
) {
    let r0 = rbin.floor() as isize;
    let c0 = cbin.floor() as isize;
    let o0 = obin.floor() as isize;
    let dr = rbin - r0 as f32;
    let dc = cbin - c0 as f32;
    let do_ = obin - o0 as f32;

    for rr in 0..2 {
        let rb = r0 + rr;
        if rb < 0 || rb >= DESCRIPTOR_WIDTH as isize {
            continue;
        }
        let wr = if rr == 0 { 1.0 - dr } else { dr };
        for cc in 0..2 {
            let cb = c0 + cc;
            if cb < 0 || cb >= DESCRIPTOR_WIDTH as isize {
                continue;
            }
            let wc = if cc == 0 { 1.0 - dc } else { dc };
            for oo in 0..2 {
                let ob = (o0 + oo).rem_euclid(DESCRIPTOR_ORIENTATION_BINS as isize) as usize;
                let wo = if oo == 0 { 1.0 - do_ } else { do_ };
                let idx = ((rb as usize * DESCRIPTOR_WIDTH + cb as usize)
                    * DESCRIPTOR_ORIENTATION_BINS)
                    + ob;
                hist[idx] += value * wr * wc * wo;
            }
        }
    }
}

#[inline]
fn normalize_descriptor(values: &mut [f32; DESCRIPTOR_LEN]) -> Option<()> {
    #[cfg(feature = "simd")]
    {
        use std::simd::f32x16;
        use std::simd::num::SimdFloat;

        let v0 = f32x16::from_slice(&values[0..16]);
        let v1 = f32x16::from_slice(&values[16..32]);
        let v2 = f32x16::from_slice(&values[32..48]);
        let v3 = f32x16::from_slice(&values[48..64]);
        let v4 = f32x16::from_slice(&values[64..80]);
        let v5 = f32x16::from_slice(&values[80..96]);
        let v6 = f32x16::from_slice(&values[96..112]);
        let v7 = f32x16::from_slice(&values[112..128]);

        let sum0 = v0 * v0 + v1 * v1;
        let sum1 = v2 * v2 + v3 * v3;
        let sum2 = v4 * v4 + v5 * v5;
        let sum3 = v6 * v6 + v7 * v7;

        let final_sum = (sum0 + sum1) + (sum2 + sum3);
        let norm2 = final_sum.reduce_sum();
        if norm2 <= EPSILON * EPSILON {
            return None;
        }
        let inv_norm = f32x16::splat(1.0 / norm2.sqrt());

        (v0 * inv_norm).copy_to_slice(&mut values[0..16]);
        (v1 * inv_norm).copy_to_slice(&mut values[16..32]);
        (v2 * inv_norm).copy_to_slice(&mut values[32..48]);
        (v3 * inv_norm).copy_to_slice(&mut values[48..64]);
        (v4 * inv_norm).copy_to_slice(&mut values[64..80]);
        (v5 * inv_norm).copy_to_slice(&mut values[80..96]);
        (v6 * inv_norm).copy_to_slice(&mut values[96..112]);
        (v7 * inv_norm).copy_to_slice(&mut values[112..128]);

        Some(())
    }
    #[cfg(not(feature = "simd"))]
    {
        let norm2: f32 = values.iter().map(|v| v * v).sum();
        if norm2 <= EPSILON * EPSILON {
            return None;
        }
        let inv_norm = 1.0 / norm2.sqrt();
        for v in values {
            *v *= inv_norm;
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GrayImage;

    #[test]
    fn default_config_is_valid() {
        SiftConfig::default().validate().unwrap();
    }

    #[test]
    fn flat_image_has_no_features() {
        let image = GrayImage::new(64, 64, vec![0.5; 64 * 64]).unwrap();
        let features = Sift::default().detect_and_compute(&image);
        assert!(features.is_empty());
    }

    #[test]
    fn descriptor_distance_zero_for_self() {
        let mut values = [0.0; DESCRIPTOR_LEN];
        values[0] = 1.0;
        let descriptor = Descriptor::new(values);
        assert_eq!(descriptor.distance2(&descriptor), 0.0);
    }
}
