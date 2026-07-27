//! Screen capture → downscaled PNG → base64, ready to embed as a Claude image block.

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use image::{ExtendedColorType, ImageEncoder};
use std::io::Cursor;
use xcap::Monitor;

/// Wire format for the screenshot.
///
/// PNG is lossless and reads text crisply. JPEG produces a payload several times smaller
/// for the same pixels, which cuts upload time noticeably on a slow link — at the cost of
/// ringing artifacts around small text, so it trades a little coordinate accuracy for
/// latency.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ImageFormat {
    Png,
    Jpeg,
}

impl ImageFormat {
    pub fn media_type(self) -> &'static str {
        match self {
            ImageFormat::Png => "image/png",
            ImageFormat::Jpeg => "image/jpeg",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "png" => Some(ImageFormat::Png),
            "jpeg" | "jpg" => Some(ImageFormat::Jpeg),
            _ => None,
        }
    }
}

/// JPEG quality when that format is selected. High enough that UI text stays legible.
const JPEG_QUALITY: u8 = 82;

/// Side of the square grayscale thumbnail kept as a frame-change fingerprint.
///
/// 16x16 is deliberately tiny. The question this answers is "did anything happen", not
/// "what changed", and downsampling that hard is what makes the answer robust to JPEG
/// ringing, a moving cursor, and the constant micro-variation of a live game scene.
const FINGERPRINT_DIM: u32 = 16;

/// A screenshot encoded for the model, plus the exact pixel dimensions the model sees
/// (needed to map its coordinates back onto the real screen).
pub struct Screenshot {
    pub png_base64: String,
    pub media_type: &'static str,
    /// Encoded byte count before base64 — what actually goes over the wire, roughly.
    pub bytes: usize,
    pub sent_w: u32,
    pub sent_h: u32,
    /// A small JPEG for a UI to display, base64. Separate from `png_base64` because that
    /// one is sized for the model — several hundred KB of PNG pushed through an IPC channel
    /// every step, to be drawn in a 390pt panel, is waste a resize avoids for a few ms.
    pub preview_base64: String,
    /// Grayscale thumbnail used only by [`Screenshot::difference`].
    fingerprint: Vec<u8>,
}

/// Long edge of the UI preview. Wide enough to stay sharp in the desktop app's panel.
const PREVIEW_DIM: u32 = 480;

impl Screenshot {
    /// Mean per-pixel difference from another frame: 0.0 identical, 1.0 maximally different.
    ///
    /// This is the loop's only way to know whether an action actually did anything. Without
    /// it the model re-issues actions it cannot confirm landed — re-pressing a key that
    /// toggles, re-latching something already held — because one stale low-res frame does
    /// not tell it the previous turn succeeded.
    pub fn difference(&self, other: &Screenshot) -> f64 {
        // Mismatched or missing fingerprints mean we cannot tell; report "changed" so the
        // caller never claims a screen is frozen on the strength of a failed comparison.
        if self.fingerprint.is_empty() || self.fingerprint.len() != other.fingerprint.len() {
            return 1.0;
        }
        let total: u64 = self
            .fingerprint
            .iter()
            .zip(&other.fingerprint)
            .map(|(a, b)| u64::from(a.abs_diff(*b)))
            .sum();
        total as f64 / (self.fingerprint.len() as f64 * 255.0)
    }
}

/// Capture `monitor_index`, downscale so the longest edge is at most `max_dim`, and
/// PNG-encode to base64.
pub fn capture(monitor_index: usize, max_dim: u32, format: ImageFormat) -> Result<Screenshot> {
    let monitors = Monitor::all().map_err(|e| anyhow!("enumerate monitors: {e}"))?;
    if monitors.is_empty() {
        return Err(anyhow!("no monitors found (is a display attached?)"));
    }
    let monitor = monitors
        .get(monitor_index)
        .or_else(|| monitors.first())
        .ok_or_else(|| anyhow!("monitor index {monitor_index} out of range"))?;

    let shot = monitor
        .capture_image()
        .map_err(|e| anyhow!("capture screen: {e}"))?;

    // Decouple from xcap's `image` version: pull raw RGBA bytes + dims (all std types),
    // then rebuild with *our* image crate.
    let (w, h) = (shot.width(), shot.height());
    let raw: Vec<u8> = shot.into_raw();
    let rgba = image::RgbaImage::from_raw(w, h, raw)
        .ok_or_else(|| anyhow!("captured buffer had unexpected size"))?;

    let mut dynimg = image::DynamicImage::ImageRgba8(rgba);
    if w > max_dim || h > max_dim {
        // `resize` preserves aspect ratio, fitting within the max_dim box.
        dynimg = dynimg.resize(max_dim, max_dim, image::imageops::FilterType::Triangle);
    }
    let sent_w = dynimg.width();
    let sent_h = dynimg.height();

    // Taken from the already-resized frame, before encoding, so the fingerprint is not
    // measuring the encoder's own noise.
    let fingerprint = dynimg
        .resize_exact(
            FINGERPRINT_DIM,
            FINGERPRINT_DIM,
            image::imageops::FilterType::Triangle,
        )
        .to_luma8()
        .into_raw();

    let mut encoded = Vec::new();
    match format {
        ImageFormat::Png => dynimg
            .write_to(&mut Cursor::new(&mut encoded), image::ImageFormat::Png)
            .context("PNG encode failed")?,
        ImageFormat::Jpeg => {
            // JPEG has no alpha channel, so drop it first.
            let rgb = dynimg.to_rgb8();
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut encoded, JPEG_QUALITY)
                .write_image(rgb.as_raw(), rgb.width(), rgb.height(), ExtendedColorType::Rgb8)
                .context("JPEG encode failed")?;
        }
    }

    // A small JPEG for any UI that wants to show what the agent is looking at. Always
    // JPEG regardless of the wire format: this one is looked at, never read as text, and a
    // downscaled PNG of a desktop is several times larger for no visible benefit.
    let mut preview = Vec::new();
    let thumb = dynimg
        .resize(PREVIEW_DIM, PREVIEW_DIM, image::imageops::FilterType::Triangle)
        .to_rgb8();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut preview, 70)
        .write_image(
            thumb.as_raw(),
            thumb.width(),
            thumb.height(),
            ExtendedColorType::Rgb8,
        )
        .context("preview encode failed")?;

    Ok(Screenshot {
        png_base64: base64::engine::general_purpose::STANDARD.encode(&encoded),
        media_type: format.media_type(),
        bytes: encoded.len(),
        sent_w,
        sent_h,
        preview_base64: base64::engine::general_purpose::STANDARD.encode(&preview),
        fingerprint,
    })
}
