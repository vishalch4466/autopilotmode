//! The desktop front-end — a webview over the same [`autopilotmode::agent`] loop.
//!
//! This crate is deliberately thin. It owns the window and the event bridge; every decision
//! about *what the agent does* lives in the library, exactly as it does for the CLI. The
//! rule that keeps the two front-ends honest: nothing here may duplicate agent logic, only
//! start it and relay what it publishes.

// No console window behind the UI in release. Kept in debug so `println!` from the loop is
// still visible while developing.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod audio;
mod voice;

use autopilotmode::agent::{self, Progress};
use autopilotmode::config::Config;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};

/// Whether a run is in flight, shared with the worker thread.
#[derive(Default)]
struct Running(Arc<Mutex<bool>>);

/// Set to ask the loop to stop at its next step boundary.
#[derive(Default)]
struct Cancel(Arc<std::sync::atomic::AtomicBool>);

/// The `.env` this install reads and writes, resolved once at startup.
struct EnvFile(PathBuf);

/// The capture in progress, if the user is holding the mic open.
#[derive(Default)]
struct Mic(Mutex<Option<audio::Recording>>);

/// Cumulative voiced audio below which a clip is treated as "nothing was said".
///
/// A quarter of a second is under one short word, so it does not clip real speech, but it
/// is far more than the stray frames a noisy room produces.
const MIN_VOICED_MS: u64 = 250;

/// Accepted spellings for the ElevenLabs key, in priority order.
///
/// The settings pane always writes the first, but a hand-written `.env` is just as likely
/// to carry one of the others — and a key that is present but spelled differently looks
/// exactly like no key at all, which is a miserable thing to debug. Same reasoning as the
/// Anthropic loader, which accepts several names for the same reason.
const VOICE_KEY_VARS: [&str; 5] = [
    "AUTOPILOT_ELEVENLABS_KEY",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_KEY",
    "ELEVENLABSAPI",
    "ELEVEN_API_KEY",
];

/// The variable the settings pane writes.
const VOICE_KEY_VAR: &str = VOICE_KEY_VARS[0];

/// The ElevenLabs key under whichever name it was given, or empty if there is none.
fn voice_key() -> String {
    VOICE_KEY_VARS
        .iter()
        .find_map(|name| {
            std::env::var(name)
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        })
        .unwrap_or_default()
}
const VOICE_OUT_VAR: &str = "AUTOPILOT_VOICE";
const VOICE_ENGINE_VAR: &str = "AUTOPILOT_VOICE_ENGINE";
const VOICE_ID_VAR: &str = "AUTOPILOT_VOICE_ID";
const MIC_VAR: &str = "AUTOPILOT_MIC";

/// Microphones the user can choose between.
#[tauri::command]
fn list_microphones() -> Vec<String> {
    audio::input_devices()
}

/// Begin capturing from the selected microphone.
#[tauri::command]
fn start_listening(mic: State<'_, Mic>) -> Result<(), String> {
    let device = std::env::var(MIC_VAR).unwrap_or_default();
    let recording = audio::start(Some(device.as_str())).map_err(|e| e.to_string())?;
    *mic.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(recording);
    Ok(())
}

/// Stop capturing and transcribe what was said.
///
/// Returns the text rather than acting on it: the caller decides whether that becomes a
/// goal, and the user still sees it in the input before anything runs. A voice assistant
/// that silently executes what it *thought* it heard is not one you want driving a mouse.
#[tauri::command]
fn stop_listening(mic: State<'_, Mic>) -> Result<String, String> {
    let recording = mic
        .0
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .ok_or("not recording")?;

    let seconds = recording.seconds();
    let heard = recording.activity();
    let wav = recording.finish().map_err(|e| e.to_string())?;

    // Refuse to transcribe a clip that did not contain enough speech to be a sentence.
    //
    // Not a nicety: a transcriber handed near-silence does not return nothing, it returns a
    // confident guess. Combined with auto-send that turns a stray press into a real agent
    // run — which is exactly what happened twice while testing this. A single threshold
    // crossing is not enough evidence either, because a fan or a desk bump clears it; the
    // test is cumulative voiced time.
    if seconds < 0.35 || heard.voiced_ms < MIN_VOICED_MS {
        return Ok(String::new());
    }

    let key = voice_key();
    if key.is_empty() {
        return Err("Add an ElevenLabs key in settings to use the microphone.".into());
    }
    voice::transcribe(&key, wav).map_err(|e| e.to_string())
}

/// Live capture state, polled by the UI to decide when the user has stopped talking.
///
/// The decision is made in the front-end rather than here because "how long a pause counts
/// as finished" is a feel question that belongs next to the UI that shows it happening.
#[tauri::command]
fn mic_state(mic: State<'_, Mic>) -> serde_json::Value {
    let guard = mic.0.lock().unwrap_or_else(|e| e.into_inner());
    match guard.as_ref() {
        Some(recording) => {
            let a = recording.activity();
            serde_json::json!({
                "recording": true,
                // "spoke" for the UI means *enough* speech to act on, not merely a blip.
                "spoke": a.voiced_ms >= MIN_VOICED_MS,
                "silent_ms": a.silent_ms,
                "level": a.level,
                "seconds": recording.seconds(),
            })
        }
        None => serde_json::json!({ "recording": false }),
    }
}

/// Discard a capture without transcribing it.
#[tauri::command]
fn cancel_listening(mic: State<'_, Mic>) {
    mic.0.lock().unwrap_or_else(|e| e.into_inner()).take();
}

/// The voices available on the configured ElevenLabs account.
#[tauri::command]
fn list_voices() -> Result<Vec<(String, String)>, String> {
    let key = voice_key();
    if key.is_empty() {
        return Ok(Vec::new());
    }
    voice::voices(&key).map_err(|e| e.to_string())
}

/// Synthesise `text` with ElevenLabs, returned base64 for the webview to play.
///
/// Only reached when the engine is set to ElevenLabs; the default engine is the webview's
/// own `speechSynthesis`, which needs no key, no network, and no Rust at all.
#[tauri::command]
fn speak(text: String, voice_id: Option<String>) -> Result<String, String> {
    let key = voice_key();
    if key.is_empty() {
        return Err("no ElevenLabs key".into());
    }
    // An explicit id lets the settings pane audition a voice before it is saved; without
    // one, fall back to whatever is configured.
    let voice_id = voice_id
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| std::env::var(VOICE_ID_VAR).unwrap_or_default());
    let mp3 = voice::synthesize(&key, &voice_id, &text).map_err(|e| e.to_string())?;
    Ok(base64::engine::general_purpose::STANDARD.encode(mp3))
}

/// The project's repository.
const REPO_URL: &str = "https://github.com/vishalch4466/autopilotmode";

/// Open the repository in the user's browser.
///
/// The URL is a compile-time constant. Taking one as a parameter instead would turn a UI
/// button into an arbitrary-launch primitive for anything that can reach the webview, which
/// is also why this is a Rust command rather than the plugin's JS API plus a URL scope.
#[tauri::command]
fn open_repo(app: AppHandle) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    app.opener()
        .open_url(REPO_URL, None::<&str>)
        .map_err(|e| e.to_string())
}

/// Ask the loop to stop. It lands at the next step boundary — up to one model round trip —
/// because a step in flight must finish unwinding rather than be torn out from under the
/// executor.
#[tauri::command]
fn stop_run(cancel: State<'_, Cancel>) {
    cancel.0.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// The credential variable the settings pane writes. The loader accepts several others
/// (see [`Config::from_env`]) so an existing hand-written `.env` keeps working, but there
/// is no reason to offer a user a choice of spellings.
const KEY_VAR: &str = "ANTHROPIC_API_KEY";

/// Settings as the pane displays them. The key itself is deliberately **not** returned —
/// only enough of it to recognise which key is stored. A secret that is never sent to the
/// webview cannot leak from it.
#[derive(Serialize)]
struct SettingsView {
    has_credential: bool,
    key_hint: String,
    model: String,
    max_steps: String,
    effort: String,
    env_path: String,
    // Voice. `has_voice_key` mirrors the credential treatment above — the pane learns that
    // a key exists and what it ends with, never the key.
    has_voice_key: bool,
    voice_key_hint: String,
    voice_out: bool,
    voice_engine: String,
    voice_id: String,
    mic: String,
}

/// Settings as the pane submits them. An empty `api_key` means "keep the stored one",
/// which is what lets the pane show a hint rather than the secret.
#[derive(Deserialize)]
struct SettingsPatch {
    api_key: String,
    model: String,
    max_steps: String,
    effort: String,
    voice_key: String,
    voice_out: bool,
    voice_engine: String,
    voice_id: String,
    mic: String,
}

/// `sk-ant-…7Gbg` — enough to tell two keys apart, not enough to use one.
fn key_hint(key: &str) -> String {
    let chars: Vec<char> = key.trim().chars().collect();
    match chars.len() {
        0 => String::new(),
        1..=12 => "•".repeat(8),
        n => format!(
            "{}…{}",
            chars[..7].iter().collect::<String>(),
            chars[n - 4..].iter().collect::<String>()
        ),
    }
}

#[tauri::command]
fn load_settings(env_file: State<'_, EnvFile>) -> SettingsView {
    let var = |name: &str| std::env::var(name).unwrap_or_default().trim().to_string();
    let voice_key = voice_key();

    // Unset means "not chosen yet", and the sensible default then depends on whether a key
    // exists: someone who added an ElevenLabs key wants the ElevenLabs voice. `"system"` is
    // a real stored value, so choosing the free voice *while* holding a key still sticks.
    let stored_engine = var(VOICE_ENGINE_VAR);
    let voice_engine = match stored_engine.as_str() {
        "" if !voice_key.is_empty() => "elevenlabs".to_string(),
        "" => "system".to_string(),
        chosen => chosen.to_string(),
    };
    SettingsView {
        // Asks the real loader rather than re-testing the variable names here, so the pane
        // cannot disagree with what a run will actually find.
        has_credential: Config::from_env().is_ok(),
        key_hint: key_hint(&var(KEY_VAR)),
        model: var("AUTOPILOT_MODEL"),
        max_steps: var("AUTOPILOT_MAX_STEPS"),
        effort: var("AUTOPILOT_EFFORT"),
        env_path: env_file.0.display().to_string(),
        has_voice_key: !voice_key.is_empty(),
        voice_key_hint: key_hint(&voice_key),
        voice_out: matches!(var(VOICE_OUT_VAR).as_str(), "1" | "true" | "yes" | "on"),
        voice_engine,
        voice_id: var(VOICE_ID_VAR),
        mic: var(MIC_VAR),
    }
}

#[tauri::command]
fn save_settings(patch: SettingsPatch, env_file: State<'_, EnvFile>) -> Result<(), String> {
    let mut updates: Vec<(&str, String)> = Vec::new();
    let key = patch.api_key.trim().to_string();
    if !key.is_empty() {
        updates.push((KEY_VAR, key));
    }
    updates.push(("AUTOPILOT_MODEL", patch.model.trim().to_string()));
    updates.push(("AUTOPILOT_MAX_STEPS", patch.max_steps.trim().to_string()));
    updates.push(("AUTOPILOT_EFFORT", patch.effort.trim().to_string()));

    let voice_key = patch.voice_key.trim().to_string();
    if !voice_key.is_empty() {
        updates.push((VOICE_KEY_VAR, voice_key));
    }
    // Written as "1"/"" rather than "true"/"false" so clearing it removes the line and the
    // documented default (off) applies again.
    updates.push((
        VOICE_OUT_VAR,
        if patch.voice_out { "1".into() } else { String::new() },
    ));
    updates.push((VOICE_ENGINE_VAR, patch.voice_engine.trim().to_string()));
    updates.push((VOICE_ID_VAR, patch.voice_id.trim().to_string()));
    updates.push((MIC_VAR, patch.mic.trim().to_string()));

    // A value carrying a line break would write extra assignments into `.env` — refuse it
    // rather than let a pasted blob smuggle in settings the user did not choose.
    for (name, value) in &updates {
        if value.contains(['\n', '\r']) {
            return Err(format!("{name} cannot contain a line break"));
        }
    }

    write_env(&env_file.0, &updates)
        .map_err(|e| format!("could not write {}: {e}", env_file.0.display()))?;

    // The library reads the process environment at run start, so mirror the write there
    // too — otherwise saved settings would not take effect until the app restarted.
    for (name, value) in updates {
        if value.is_empty() {
            std::env::remove_var(name);
        } else {
            std::env::set_var(name, value);
        }
    }
    Ok(())
}

/// Quote a value if writing it bare would produce an ambiguous `.env` line.
///
/// This is not cosmetic. A bare `AUTOPILOT_MIC=Headset Microphone (USB)` is not reliably
/// parseable, and a single line the loader chokes on can abort the rest of the file —
/// silently taking every setting *after* it with it, which reads as "my API key stopped
/// working" rather than "line 26 is malformed". Device names come from the OS and routinely
/// contain spaces and parentheses, so this is the common case, not an edge one.
fn quote(value: &str) -> String {
    let risky = value.contains(char::is_whitespace)
        || value.contains(['#', '"', '\'', '$', '=', '\\']);
    if !risky {
        return value.to_string();
    }
    let escaped = value.replace('\\', r"\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

/// Upsert `NAME=value` lines in `.env`, leaving every other line untouched.
///
/// The shipped `.env` is mostly comments explaining each setting, and a commented example
/// (`# AUTOPILOT_MODEL=...`) is documentation, not an assignment — so only uncommented
/// lines are candidates for replacement, and rewriting the file wholesale is never right.
fn write_env(path: &Path, updates: &[(&str, String)]) -> std::io::Result<()> {
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    let mut lines: Vec<String> = existing.lines().map(str::to_string).collect();

    for (name, value) in updates {
        let assignment = format!("{name}=");
        let at = lines.iter().position(|line| {
            let l = line.trim_start();
            !l.starts_with('#') && l.starts_with(&assignment)
        });
        match (at, value.is_empty()) {
            // Clearing a setting removes the line so the documented default applies again.
            (Some(i), true) => drop(lines.remove(i)),
            (Some(i), false) => lines[i] = format!("{name}={}", quote(value)),
            (None, true) => {}
            (None, false) => lines.push(format!("{name}={}", quote(value))),
        }
    }

    let mut out = lines.join("\n");
    if !out.ends_with('\n') {
        out.push('\n');
    }
    std::fs::write(path, out)
}

/// Kick off a run on a worker thread.
///
/// The agent loop is blocking and takes seconds per step, so it cannot share the UI thread —
/// the webview would stop painting for the whole run, which is exactly when the user most
/// wants to watch it work and reach the stop corner.
#[tauri::command]
fn start_run(
    app: AppHandle,
    goal: String,
    dry_run: bool,
    running: State<'_, Running>,
    cancel: State<'_, Cancel>,
) -> Result<(), String> {
    {
        let mut flag = running.0.lock().unwrap_or_else(|e| e.into_inner());
        if *flag {
            return Err("a run is already in flight".into());
        }
        *flag = true;
    }
    cancel
        .0
        .store(false, std::sync::atomic::Ordering::Relaxed);

    let flag = Arc::clone(&running.0);
    let cancel = Arc::clone(&cancel.0);
    std::thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel::<Progress>();

        // Relay on its own thread so a slow webview cannot back-pressure the agent loop.
        let relay = {
            let app = app.clone();
            std::thread::spawn(move || {
                while let Ok(event) = rx.recv() {
                    let _ = app.emit("progress", &event);
                }
            })
        };

        // Config is rebuilt per run so `.env` edits take effect without a restart.
        let result = Config::from_env().and_then(|mut cfg| {
            cfg.dry_run = dry_run;
            agent::run_cancellable(&goal, &cfg, Some(&tx), Some(&cancel))
        });
        if let Err(e) = result {
            // A failed run still has to reach the user: without this the bubble would sit
            // on "On it…" forever with no indication anything went wrong.
            let _ = tx.send(Progress::Stopped(format!("{e}")));
        }

        drop(tx);
        let _ = relay.join();
        *flag.lock().unwrap_or_else(|e| e.into_inner()) = false;
        let _ = app.emit("run-ended", ());
    });

    Ok(())
}

fn main() {
    // Same as the CLI: load `.env` (and any parent `.env`) before anything reads config.
    // Without this the desktop app could only see credentials already exported into its
    // environment, which is never true when it is launched from a shortcut.
    //
    // The resolved path is kept so the settings pane writes back to the *same* file the
    // run reads — one source of truth shared with the CLI, rather than a second private
    // config store that would silently disagree with it.
    let env_path = dotenvy::dotenv().unwrap_or_else(|_| {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(".env")
    });

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(Running::default())
        .manage(Cancel::default())
        .manage(Mic::default())
        .manage(EnvFile(env_path))
        .invoke_handler(tauri::generate_handler![
            start_run,
            stop_run,
            load_settings,
            save_settings,
            open_repo,
            list_microphones,
            start_listening,
            stop_listening,
            mic_state,
            cancel_listening,
            speak,
            list_voices
        ])
        // Closing the window must not strand the machine. A run can be several seconds
        // into a blocking step with keys physically down from `keydown`, and the worker
        // thread cannot be interrupted — so release the keys here, on the path that is
        // guaranteed to run, exactly as the CLI's Ctrl-C handler does. Without this,
        // closing mid-run leaves the throttle held at the OS level and the user's keyboard
        // appears broken until they tap the key themselves.
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                if let Some(cancel) = window.try_state::<Cancel>() {
                    cancel.0.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                autopilotmode::executor::release_latched_keys();
                // The agent thread is blocking and holds no OS state we still need, so
                // exit rather than wait out a model round trip the user just cancelled.
                window.app_handle().exit(0);
            }
        })
        .run(tauri::generate_context!())
        .expect("could not start the autopilotmode desktop window");
}
