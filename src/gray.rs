use std::fmt;

/// A single-channel image with row-major `f32` pixels.
///
/// SIFT assumes pixels are normalized to `[0, 1]`. The type does not clamp
/// values on every access so that intermediate images can contain values
/// outside that range when needed.
#[derive(Clone, Debug, PartialEq)]
pub struct GrayImage {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

/// Errors returned while constructing a [`GrayImage`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GrayImageError {
    /// Width or height was zero.
    EmptyImage,
    /// `width * height` overflowed `usize`.
    DimensionOverflow,
    /// The supplied buffer length did not match `width * height`.
    InvalidBufferLength { expected: usize, actual: usize },
}

impl fmt::Display for GrayImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyImage => write!(f, "image dimensions must be non-zero"),
            Self::DimensionOverflow => write!(f, "image dimensions overflow usize"),
            Self::InvalidBufferLength { expected, actual } => {
                write!(
                    f,
                    "invalid buffer length: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for GrayImageError {}

impl GrayImage {
    /// Creates an image from a row-major buffer.
    pub fn new(width: usize, height: usize, data: Vec<f32>) -> Result<Self, GrayImageError> {
        if width == 0 || height == 0 {
            return Err(GrayImageError::EmptyImage);
        }
        let expected = width
            .checked_mul(height)
            .ok_or(GrayImageError::DimensionOverflow)?;
        if data.len() != expected {
            return Err(GrayImageError::InvalidBufferLength {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Creates an all-zero image.
    pub fn zeros(width: usize, height: usize) -> Result<Self, GrayImageError> {
        if width == 0 || height == 0 {
            return Err(GrayImageError::EmptyImage);
        }
        let len = width
            .checked_mul(height)
            .ok_or(GrayImageError::DimensionOverflow)?;
        Ok(Self {
            width,
            height,
            data: vec![0.0; len],
        })
    }

    /// Creates an image by evaluating `f(x, y)` for each pixel.
    pub fn from_fn<F>(width: usize, height: usize, mut f: F) -> Result<Self, GrayImageError>
    where
        F: FnMut(usize, usize) -> f32,
    {
        if width == 0 || height == 0 {
            return Err(GrayImageError::EmptyImage);
        }
        let len = width
            .checked_mul(height)
            .ok_or(GrayImageError::DimensionOverflow)?;
        let mut data = Vec::with_capacity(len);
        for y in 0..height {
            for x in 0..width {
                data.push(f(x, y));
            }
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Returns the image width in pixels.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the image height in pixels.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the row-major pixel buffer.
    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Returns a mutable row-major pixel buffer.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Returns the pixel at `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics if `x >= width` or `y >= height`.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    /// Sets the pixel at `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics if `x >= width` or `y >= height`.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        self.data[y * self.width + x] = value;
    }

    /// Returns a pixel using edge clamping for out-of-range coordinates.
    #[inline]
    pub fn get_clamped(&self, x: isize, y: isize) -> f32 {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.get(x, y)
    }

    /// Converts from an `image` crate dynamic image by luminance conversion.
    #[cfg(feature = "image")]
    pub fn from_dynamic_image(image: &::image::DynamicImage) -> Self {
        let gray = image.to_luma8();
        Self::from_luma8(&gray)
    }

    /// Converts from an `image` crate `GrayImage` (`Luma<u8>`) into `[0, 1]` pixels.
    #[cfg(feature = "image")]
    pub fn from_luma8(image: &::image::GrayImage) -> Self {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let data = image
            .as_raw()
            .iter()
            .map(|&v| f32::from(v) / 255.0)
            .collect();
        Self {
            width,
            height,
            data,
        }
    }

    #[inline]
    pub(crate) fn subtract(&self, rhs: &Self) -> Self {
        debug_assert_eq!(self.width, rhs.width);
        debug_assert_eq!(self.height, rhs.height);
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Self {
            width: self.width,
            height: self.height,
            data,
        }
    }

    pub(crate) fn gaussian_blur(&self, sigma: f32, tmp: &mut Vec<f32>) -> Self {
        let kernel = gaussian_kernel(sigma);
        if kernel.len() == 1 {
            return self.clone();
        }

        let radius = (kernel.len() / 2) as isize;
        tmp.clear();
        tmp.resize(self.data.len(), 0.0);
        let mut out = vec![0.0; self.data.len()];

        for y in 0..self.height {
            let y_offset = y * self.width;
            
            // Left margin: x in [0, radius)
            for x in 0..radius.min(self.width as isize) {
                let mut acc = 0.0;
                for (i, &w) in kernel.iter().enumerate() {
                    let dx = i as isize - radius;
                    let xx = (x + dx).clamp(0, self.width as isize - 1) as usize;
                    acc += w * self.data[y_offset + xx];
                }
                tmp[y_offset + x as usize] = acc;
            }
            
            // Interior: x in [radius, width - radius)
            let start_x = radius.max(0) as usize;
            let end_x = (self.width as isize - radius).max(0) as usize;
            if start_x < end_x {
                for x in start_x..end_x {
                    let mut acc = 0.0;
                    for (i, &w) in kernel.iter().enumerate() {
                        let dx = i as isize - radius;
                        let xx = (x as isize + dx) as usize;
                        acc += w * self.data[y_offset + xx];
                    }
                    tmp[y_offset + x] = acc;
                }
            }
            
            // Right margin: x in [width - radius, width)
            let start_right = (self.width as isize - radius).max(0);
            for x in start_right..(self.width as isize) {
                let mut acc = 0.0;
                for (i, &w) in kernel.iter().enumerate() {
                    let dx = i as isize - radius;
                    let xx = (x + dx).clamp(0, self.width as isize - 1) as usize;
                    acc += w * self.data[y_offset + xx];
                }
                tmp[y_offset + x as usize] = acc;
            }
        }

        // Top margin: y in [0, radius)
        for y in 0..radius.min(self.height as isize) {
            let y_offset = (y as usize) * self.width;
            for x in 0..self.width {
                let mut acc = 0.0;
                for (i, &w) in kernel.iter().enumerate() {
                    let dy = i as isize - radius;
                    let yy = (y + dy).clamp(0, self.height as isize - 1) as usize;
                    acc += w * tmp[yy * self.width + x];
                }
                out[y_offset + x] = acc;
            }
        }
        
        // Interior: y in [radius, height - radius)
        let start_y = radius.max(0) as usize;
        let end_y = (self.height as isize - radius).max(0) as usize;
        for y in start_y..end_y {
            let y_offset = y * self.width;
            let start_yy = (y as isize - radius) as usize;
            for x in 0..self.width {
                let mut acc = 0.0;
                for (i, &w) in kernel.iter().enumerate() {
                    acc += w * tmp[(start_yy + i) * self.width + x];
                }
                out[y_offset + x] = acc;
            }
        }
        
        // Bottom margin: y in [height - radius, height)
        let start_bottom = (self.height as isize - radius).max(0);
        for y in start_bottom..(self.height as isize) {
            let y_offset = (y as usize) * self.width;
            for x in 0..self.width {
                let mut acc = 0.0;
                for (i, &w) in kernel.iter().enumerate() {
                    let dy = i as isize - radius;
                    let yy = (y + dy).clamp(0, self.height as isize - 1) as usize;
                    acc += w * tmp[yy * self.width + x];
                }
                out[y_offset + x] = acc;
            }
        }

        Self {
            width: self.width,
            height: self.height,
            data: out,
        }
    }

    #[inline]
    pub(crate) fn downsample_by_2(&self) -> Self {
        let width = (self.width / 2).max(1);
        let height = (self.height / 2).max(1);
        let mut data = vec![0.0; width * height];
        for y in 0..height {
            for x in 0..width {
                data[y * width + x] =
                    self.get((x * 2).min(self.width - 1), (y * 2).min(self.height - 1));
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    #[inline]
    pub(crate) fn double_linear(&self) -> Self {
        let width = self.width * 2;
        let height = self.height * 2;
        let mut data = vec![0.0; width * height];
        for y in 0..height {
            let sy = y as f32 * 0.5;
            for x in 0..width {
                let sx = x as f32 * 0.5;
                data[y * width + x] = self.sample_bilinear(sx, sy);
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    #[inline]
    fn sample_bilinear(&self, x: f32, y: f32) -> f32 {
        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let v00 = self.get_clamped(x0, y0);
        let v10 = self.get_clamped(x1, y0);
        let v01 = self.get_clamped(x0, y1);
        let v11 = self.get_clamped(x1, y1);

        let top = v00 * (1.0 - fx) + v10 * fx;
        let bottom = v01 * (1.0 - fx) + v11 * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Kernel {
    pub(crate) data: [f32; 256],
    pub(crate) len: usize,
}

impl Kernel {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, f32> {
        self.data[..self.len].iter()
    }
}

fn gaussian_kernel(sigma: f32) -> Kernel {
    if sigma <= 0.01 || !sigma.is_finite() {
        let mut data = [0.0; 256];
        data[0] = 1.0;
        return Kernel { data, len: 1 };
    }

    let radius = ((3.0 * sigma).ceil() as isize).min(127);
    let len = (2 * radius + 1) as usize;
    let mut data = [0.0; 256];
    let denom = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for i in -radius..=radius {
        let v = (-(i * i) as f32 / denom).exp();
        data[(i + radius) as usize] = v;
        sum += v;
    }
    for i in 0..len {
        data[i] /= sum;
    }
    Kernel { data, len }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_rejects_bad_lengths() {
        let err = GrayImage::new(4, 4, vec![0.0; 15]).unwrap_err();
        assert_eq!(
            err,
            GrayImageError::InvalidBufferLength {
                expected: 16,
                actual: 15
            }
        );
    }

    #[test]
    fn blur_preserves_constant_image() {
        let image = GrayImage::new(11, 7, vec![0.25; 77]).unwrap();
        let mut tmp = Vec::new();
        let blurred = image.gaussian_blur(1.6, &mut tmp);
        for &pixel in blurred.data() {
            assert!((pixel - 0.25).abs() < 1e-5);
        }
    }
}
