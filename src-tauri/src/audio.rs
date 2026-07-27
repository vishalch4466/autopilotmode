//! Microphone capture.
//!
//! Capture lives in Rust rather than the webview for one concrete reason: a WebView2 page
//! cannot enumerate host input devices or pin a specific one, and "which microphone" is
//! precisely the setting a user with a headset plus a webcam needs. Doing it here also
//! keeps recorded audio out of the page entirely — it goes straight from the device to the
//! transcription request.

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

/// Speech models want mono at a modest rate; 16 kHz is the usual target and keeps the
/// upload small. Whatever the device gives us is converted to this.
const TARGET_HZ: u32 = 16_000;

/// A capture in progress.
///
/// The `cpal::Stream` is deliberately **not** held here. It is not `Send` — on WASAPI it is
/// bound to the thread that created it — so it cannot live in the app's shared state.
/// Instead a worker thread owns the stream for its whole life and this is a pair of channel
/// ends, which are.
pub struct Recording {
    stop: std::sync::mpsc::Sender<()>,
    result: std::sync::mpsc::Receiver<Result<Vec<u8>>>,
    started: std::time::Instant,
    vad: Arc<Vad>,
}

/// Rolling voice-activity state, written by the audio callback and read by the UI.
///
/// Atomics rather than a lock: this is touched from the real-time audio thread, where
/// blocking on a mutex the UI happens to hold would drop samples.
#[derive(Default)]
pub struct Vad {
    /// Loudest recent chunk, 0..1000 — used to show the user it can hear them.
    level: std::sync::atomic::AtomicU32,
    /// Milliseconds since capture began at the last chunk loud enough to be speech.
    last_voice_ms: std::sync::atomic::AtomicU64,
    /// Total milliseconds of audio above the speech threshold.
    voiced_ms: std::sync::atomic::AtomicU64,
}

/// What the microphone has heard so far.
pub struct Activity {
    /// Cumulative time above the speech threshold — the honest measure of "did they talk".
    /// A boolean "did it ever cross the line" is not: a fan clears the line.
    pub voiced_ms: u64,
    pub silent_ms: u64,
    pub level: u32,
}

/// RMS above which a chunk counts as speech rather than room noise.
///
/// Deliberately low. Failing to notice someone talking stops a sentence mid-word, which is
/// far worse than a little extra silence at the end of a clip that is about to be trimmed
/// by the transcriber anyway.
const SPEECH_RMS: f32 = 0.012;

/// Input device names, most-likely-default first.
pub fn input_devices() -> Vec<String> {
    let host = cpal::default_host();
    let default = host
        .default_input_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_default();

    let mut names: Vec<String> = host
        .input_devices()
        .map(|devices| devices.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default();
    names.dedup();
    // The default first, so an empty setting and the top of the list agree.
    names.sort_by_key(|n| (n != &default, n.to_lowercase()));
    names
}

/// Open `device` (or the system default when `None`/unknown) and start capturing.
///
/// The stream is built and dropped entirely on the worker thread; this call blocks only
/// until the device is open, so a device error still surfaces to the caller rather than
/// being swallowed on a thread nobody is watching.
pub fn start(device: Option<&str>) -> Result<Recording> {
    let device = device.map(str::to_string);
    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
    let (result_tx, result_rx) = std::sync::mpsc::channel::<Result<Vec<u8>>>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<Arc<Vad>, String>>();

    std::thread::spawn(move || match open(device.as_deref()) {
        Err(e) => {
            let _ = ready_tx.send(Err(e.to_string()));
        }
        Ok(open_stream) => {
            let OpenStream {
                stream,
                samples,
                source_hz,
                channels,
                vad,
            } = open_stream;
            let _ = ready_tx.send(Ok(vad));
            // Park until asked to stop, then drop the stream on this same thread.
            let _ = stop_rx.recv();
            drop(stream);
            let _ = result_tx.send(encode(&samples, source_hz, channels));
        }
    });

    match ready_rx.recv() {
        Ok(Ok(vad)) => Ok(Recording {
            stop: stop_tx,
            result: result_rx,
            started: std::time::Instant::now(),
            vad,
        }),
        Ok(Err(e)) => Err(anyhow!(e)),
        Err(_) => Err(anyhow!("microphone thread stopped before it started")),
    }
}

/// A live stream plus what is needed to interpret its samples.
struct OpenStream {
    stream: cpal::Stream,
    samples: Arc<Mutex<Vec<f32>>>,
    source_hz: u32,
    channels: u16,
    vad: Arc<Vad>,
}

fn open(device: Option<&str>) -> Result<OpenStream> {
    let host = cpal::default_host();
    let picked = device
        .filter(|n| !n.trim().is_empty())
        .and_then(|want| {
            host.input_devices()
                .ok()?
                .find(|d| d.name().map(|n| n == want).unwrap_or(false))
        })
        // A device that has been unplugged since it was chosen should not be a hard error;
        // falling back to the default keeps the button working.
        .or_else(|| host.default_input_device())
        .ok_or_else(|| anyhow!("no microphone available"))?;

    let config = picked.default_input_config()?;
    let source_hz = config.sample_rate().0;
    let channels = config.channels();

    let samples = Arc::new(Mutex::new(Vec::<f32>::new()));
    let vad = Arc::new(Vad::default());
    let sink = Arc::clone(&samples);
    let meter = Arc::clone(&vad);
    let clock = std::time::Instant::now();
    let on_error = |e| eprintln!("  ⚠ microphone stream error: {e}");

    // Wraps the per-format closures below so each one both stores samples and updates the
    // activity state, without repeating either three times.
    let frame_divisor = (source_hz as u64).max(1) * (channels as u64).max(1);
    let observe = move |sink: &Arc<Mutex<Vec<f32>>>, chunk: Vec<f32>| {
        use std::sync::atomic::Ordering::Relaxed;
        if !chunk.is_empty() {
            let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
            meter.level.store((rms * 1000.0).min(1000.0) as u32, Relaxed);
            if rms > SPEECH_RMS {
                meter
                    .last_voice_ms
                    .store(clock.elapsed().as_millis() as u64, Relaxed);
                // Accumulate *how much* was voiced, not just that something once crossed
                // the line. A fan or a desk bump clips the threshold for a frame or two;
                // only real speech keeps adding up.
                let ms = (chunk.len() as u64 * 1000) / frame_divisor;
                meter.voiced_ms.fetch_add(ms.max(1), Relaxed);
            }
        }
        if let Ok(mut buf) = sink.lock() {
            buf.extend(chunk);
        }
    };

    // Everything is normalised to f32 here so the resample/downmix below has one input
    // shape to handle rather than three.
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => picked.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &_| observe(&sink, data.to_vec()),
            on_error,
            None,
        )?,
        cpal::SampleFormat::I16 => picked.build_input_stream(
            &config.into(),
            move |data: &[i16], _: &_| {
                observe(
                    &sink,
                    data.iter().map(|s| *s as f32 / i16::MAX as f32).collect(),
                )
            },
            on_error,
            None,
        )?,
        cpal::SampleFormat::U16 => picked.build_input_stream(
            &config.into(),
            move |data: &[u16], _: &_| {
                observe(
                    &sink,
                    data.iter()
                        .map(|s| (*s as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0))
                        .collect(),
                )
            },
            on_error,
            None,
        )?,
        other => return Err(anyhow!("unsupported sample format: {other:?}")),
    };

    stream.play()?;
    Ok(OpenStream {
        stream,
        samples,
        source_hz,
        channels,
        vad,
    })
}

impl Recording {
    /// Stop capturing and hand back a 16 kHz mono WAV.
    pub fn finish(self) -> Result<Vec<u8>> {
        // Send may fail if the worker already died; the recv below reports that properly.
        let _ = self.stop.send(());
        self.result
            .recv()
            .map_err(|_| anyhow!("microphone thread stopped without returning audio"))?
    }

    /// How long capture has been running. Wall clock rather than a sample count: it is used
    /// only to reject an accidental click, and it stays right even if the device stalls.
    pub fn seconds(&self) -> f32 {
        self.started.elapsed().as_secs_f32()
    }

    /// What has been heard so far.
    ///
    /// `silent_ms` is meaningless until speech has actually been heard — before that the
    /// caller must not treat quiet as "finished talking", or the recording ends during the
    /// pause between pressing the button and starting to speak.
    pub fn activity(&self) -> Activity {
        use std::sync::atomic::Ordering::Relaxed;
        let elapsed = self.started.elapsed().as_millis() as u64;
        Activity {
            voiced_ms: self.vad.voiced_ms.load(Relaxed),
            silent_ms: elapsed.saturating_sub(self.vad.last_voice_ms.load(Relaxed)),
            level: self.vad.level.load(Relaxed),
        }
    }
}

/// Downmix, resample to 16 kHz, and encode as WAV.
fn encode(samples: &Arc<Mutex<Vec<f32>>>, source_hz: u32, channels: u16) -> Result<Vec<u8>> {
    let raw = samples
        .lock()
        .map(|b| b.clone())
        .map_err(|_| anyhow!("recording buffer was poisoned"))?;

    {
        // Downmix to mono by averaging frames — taking one channel instead would silently
        // drop the user's voice on a device that puts it only on the right.
        let ch = channels.max(1) as usize;
        let mono: Vec<f32> = raw
            .chunks(ch)
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
            .collect();

        // Linear resample. Nearest-neighbour would alias audibly; a full windowed-sinc is
        // overkill for speech that is about to be transcribed, not listened to.
        let ratio = source_hz as f64 / TARGET_HZ as f64;
        let out_len = (mono.len() as f64 / ratio).floor() as usize;
        let mut pcm = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let pos = i as f64 * ratio;
            let a = pos.floor() as usize;
            let frac = (pos - a as f64) as f32;
            let s = match (mono.get(a), mono.get(a + 1)) {
                (Some(x), Some(y)) => x + (y - x) * frac,
                (Some(x), None) => *x,
                _ => 0.0,
            };
            pcm.push((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16);
        }

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: TARGET_HZ,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
            for s in pcm {
                writer.write_sample(s)?;
            }
            writer.finalize()?;
        }
        Ok(cursor.into_inner())
    }
}
