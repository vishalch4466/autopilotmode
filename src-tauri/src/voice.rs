//! ElevenLabs speech-to-text and text-to-speech.
//!
//! One vendor for both directions on purpose: the alternative is two API keys in a settings
//! pane that already asks for an Anthropic one, and each extra key is a place a new user
//! gives up. Speech *output* still defaults to the webview's built-in voice, so voice mode
//! works with no ElevenLabs key at all — this is the quality upgrade, not the only path.

use anyhow::{anyhow, Result};
use std::time::Duration;

const STT_URL: &str = "https://api.elevenlabs.io/v1/speech-to-text";
const TTS_URL: &str = "https://api.elevenlabs.io/v1/text-to-speech";

/// Scribe. The only STT model ElevenLabs exposes at time of writing.
const STT_MODEL: &str = "scribe_v1";
/// "Rachel" — a stock voice, so a fresh key works without first creating one.
pub const DEFAULT_VOICE_ID: &str = "21m00Tcm4TlvDq8ikWAM";
/// Low-latency synthesis; quality difference is inaudible for one-line status text.
const TTS_MODEL: &str = "eleven_flash_v2_5";

fn client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(Into::into)
}

/// Transcribe a WAV clip. Returns the spoken text, trimmed.
pub fn transcribe(api_key: &str, wav: Vec<u8>) -> Result<String> {
    let part = reqwest::blocking::multipart::Part::bytes(wav)
        .file_name("speech.wav")
        .mime_str("audio/wav")?;
    let form = reqwest::blocking::multipart::Form::new()
        .text("model_id", STT_MODEL)
        .part("file", part);

    let response = client()?
        .post(STT_URL)
        .header("xi-api-key", api_key)
        .multipart(form)
        .send()?;

    let status = response.status();
    let body = response.text()?;
    if !status.is_success() {
        return Err(anyhow!("{}", describe(status, &body)));
    }

    let parsed: serde_json::Value = serde_json::from_str(&body)?;
    Ok(parsed
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or_default()
        .trim()
        .to_string())
}

/// Synthesise `text`, returning MP3 bytes for the webview to play.
pub fn synthesize(api_key: &str, voice_id: &str, text: &str) -> Result<Vec<u8>> {
    let voice = if voice_id.trim().is_empty() {
        DEFAULT_VOICE_ID
    } else {
        voice_id.trim()
    };

    let response = client()?
        .post(format!("{TTS_URL}/{voice}"))
        .header("xi-api-key", api_key)
        // `output_format` is a query parameter, not a body field. Sent in the body it is
        // silently ignored at best and rejected at worst.
        .query(&[("output_format", "mp3_44100_128")])
        .json(&serde_json::json!({
            "text": text,
            "model_id": TTS_MODEL,
        }))
        .send()?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        return Err(anyhow!("{}", describe(status, &body)));
    }
    Ok(response.bytes()?.to_vec())
}

/// The voices this key can use, as `(id, name)`.
///
/// Fetched rather than hardcoded: the interesting voices are the ones on the user's own
/// account, and a built-in list would offer none of them.
pub fn voices(api_key: &str) -> Result<Vec<(String, String)>> {
    let response = client()?
        .get("https://api.elevenlabs.io/v1/voices")
        .header("xi-api-key", api_key)
        .send()?;

    let status = response.status();
    let body = response.text()?;
    if !status.is_success() {
        return Err(anyhow!("{}", describe(status, &body)));
    }

    let parsed: serde_json::Value = serde_json::from_str(&body)?;
    Ok(parsed
        .get("voices")
        .and_then(|v| v.as_array())
        .map(|list| {
            list.iter()
                .filter_map(|v| {
                    Some((
                        v.get("voice_id")?.as_str()?.to_string(),
                        v.get("name")?.as_str()?.to_string(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default())
}

/// Turn an API error into something worth showing a user.
///
/// ElevenLabs returns its reason inside a JSON envelope; surfacing the raw body would put
/// a wall of JSON in a 400px-wide pane, and surfacing only the status code would hide the
/// two failures that actually happen — a bad key and an exhausted quota.
fn describe(status: reqwest::StatusCode, body: &str) -> String {
    let detail = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            let d = v.get("detail")?;
            d.get("message")
                .and_then(|m| m.as_str())
                .or_else(|| d.as_str())
                .map(str::to_string)
        });

    match (status.as_u16(), detail) {
        (401, _) => "ElevenLabs rejected the key.".into(),
        (422, Some(d)) => d,
        (429, _) => "ElevenLabs quota or rate limit reached.".into(),
        (_, Some(d)) => d,
        _ => format!("ElevenLabs error {status}"),
    }
}
