//! The observe→decide→act loop.

use crate::capture::{self, Screenshot};
use crate::config::Config;
use crate::executor::{Executor, Outcome};
use crate::model;
use anyhow::Result;
use serde_json::{json, Value};

/// How many consecutive unusable actions to tolerate before aborting the run.
const MAX_FAILED_ACTIONS: u32 = 4;

/// What the loop is doing, for any front-end that wants to show it.
///
/// The CLI prints as it goes, but a GUI runs the loop on a worker thread and cannot read
/// stdout, so the interesting moments are published here as well. Kept deliberately coarse:
/// this is for telling a user what is happening, not for reconstructing the run.
/// `Serialize` so a webview front-end can receive these verbatim; the internally-tagged
/// form gives JS a plain `{type: "Acting", action, reason}` to switch on.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "type")]
pub enum Progress {
    /// A new observe→act iteration began.
    Step { n: u32, total: u32 },
    /// The screen as the model is about to see it — a small JPEG, base64.
    ///
    /// Published so a front-end can show what the agent is looking at. Watching the frames
    /// is the fastest way to understand a run that is going wrong, because "clicked at
    /// 840,412" means nothing without the screen it was aimed at.
    Frame { image: String, w: u32, h: u32 },
    /// Waiting on the model.
    Thinking,
    /// The model chose an action; `reason` is its own one-line justification.
    Acting { action: String, reason: Option<String> },
    /// Free-text the model wrote alongside the tool call.
    Note(String),
    /// The run handed this step to the heavier model.
    Escalated { model: String },
    /// The model called `done`.
    Finished { success: bool, message: String },
    /// The run ended without `done` — step cap, failsafe, or error.
    Stopped(String),
}

/// Publish the frame the model is about to see, for a front-end to display.
///
/// Returns early when nobody is listening rather than cloning a preview the CLI would
/// throw away — this runs once per step, on every run.
fn emit_frame(sink: Option<&std::sync::mpsc::Sender<Progress>>, shot: &Screenshot) {
    if sink.is_none() {
        return;
    }
    emit(
        sink,
        Progress::Frame {
            image: shot.preview_base64.clone(),
            w: shot.sent_w,
            h: shot.sent_h,
        },
    );
}

/// Publish a progress event, ignoring a hung-up receiver: a closed UI must not kill the run.
fn emit(sink: Option<&std::sync::mpsc::Sender<Progress>>, event: Progress) {
    if let Some(tx) = sink {
        let _ = tx.send(event);
    }
}

/// Mean per-pixel frame change below which two consecutive screens count as "the same".
///
/// Tuned low on purpose. A live scene always carries some micro-variation — foliage,
/// lighting, an idling engine, a blinking cursor — so a generous threshold would never
/// fire. This is looking for a genuinely frozen screen: a menu that did not open, a click
/// that hit nothing, a car pinned against a wall with the camera settled.
const STASIS_THRESHOLD: f64 = 0.004;

/// Consecutive unchanged frames before the run treats the screen as stuck.
///
/// Two rather than one: a single action can legitimately leave the screen alone (releasing
/// a key, a keystroke the UI swallows), and escalating on that would be as trigger-happy as
/// the action-repetition heuristics this replaces.
const STALE_FRAMES_TO_ESCALATE: u32 = 2;

/// Steps to stay on the heavy model once the run escalates.
///
/// Recovering from a wedged position is a sequence, not one decision — release the throttle,
/// reverse clear, re-latch. Dropping back to the fast model after a single step would hand it
/// over mid-manoeuvre, so the window covers the whole cycle.
const RECOVERY_STEPS: u32 = 4;

/// A fingerprint of an action, used to notice the loop repeating itself.
///
/// Coordinates are quantised to a coarse grid on purpose: a model retrying a click it has
/// already tried rarely reproduces the exact pixel — it drifts a few px each time — so an
/// exact comparison would miss the repetition this exists to catch.
fn signature(a: &crate::action::ActionInput) -> String {
    const GRID: f64 = 32.0;
    let cell = |v: Option<f64>| v.map_or(-1, |n| (n / GRID).round() as i64);
    format!(
        "{}|{:?}|{:?}|{},{}",
        a.action,
        a.keys,
        a.text,
        cell(a.x),
        cell(a.y)
    )
}

/// Run the observe→act loop, printing to stdout as it goes.
pub fn run(goal: &str, cfg: &Config) -> Result<()> {
    run_with_progress(goal, cfg, None)
}

/// As [`run`], but also publishing [`Progress`] events for a front-end to display.
pub fn run_with_progress(
    goal: &str,
    cfg: &Config,
    progress: Option<&std::sync::mpsc::Sender<Progress>>,
) -> Result<()> {
    run_cancellable(goal, cfg, progress, None)
}

/// As [`run_with_progress`], but abortable by setting `cancel`.
///
/// A GUI runs the loop on a worker thread it cannot interrupt, so without this the only
/// ways out are the failsafe corner and the step cap — an unacceptable position for a
/// program that is driving the user's mouse while they watch. The flag is read once per
/// step rather than mid-step, so a stop lands after the action in flight finishes: up to a
/// few seconds, bounded by one model round trip.
///
/// Stopping returns through the normal path on purpose. `Executor`'s `Drop` is what
/// releases keys that `keydown` left physically down, so an abort must unwind rather than
/// exit the process.
pub fn run_cancellable(
    goal: &str,
    cfg: &Config,
    progress: Option<&std::sync::mpsc::Sender<Progress>>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<()> {
    let stopping =
        || cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed));
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()?;
    let mut exec = Executor::new(cfg.dry_run, cfg.human_mouse, cfg.mouse_speed)?;

    let system = model::system_prompt(cfg.terse, cfg.game_mode);
    let tools = model::tools();

    // First observation.
    let mut current = capture::capture(cfg.monitor_index, cfg.max_image_dim, cfg.image_format)?;
    emit_frame(progress, &current);
    let mut messages: Vec<Value> = vec![initial_message(goal, &current)];
    let mut failed_actions = 0u32;
    let mut recovery_left = 0u32;
    let mut last_sig: Option<String> = None;
    let mut stale_frames = 0u32;

    for step in 1..=cfg.max_steps {
        // Failsafe: user slammed the cursor to the top-left corner.
        if exec.panic_corner() {
            println!("\n⛔ Cursor in top-left corner — aborting (failsafe).");
            emit(progress, Progress::Stopped("failsafe — cursor in the corner".into()));
            return Ok(());
        }

        // Asked to stop from a front-end.
        if stopping() {
            println!("\n⛔ Stopped.");
            emit(progress, Progress::Stopped("stopped".into()));
            return Ok(());
        }

        trim_history(&mut messages, cfg.keep_screenshots);

        println!("\n──── step {step}/{} ────", cfg.max_steps);
        emit(progress, Progress::Step { n: step, total: cfg.max_steps });

        // Escalation is decided from the *previous* step's action, so a step that signalled
        // trouble is followed by one the stronger model handles.
        let escalated = recovery_left > 0;
        let active_model = match (&cfg.model_heavy, escalated) {
            (Some(heavy), true) => heavy.as_str(),
            _ => cfg.model.as_str(),
        };
        if escalated && cfg.model_heavy.is_some() {
            println!("  ⤴ escalated to {active_model} ({recovery_left} left)");
            emit(progress, Progress::Escalated { model: active_model.to_string() });
        }
        recovery_left = recovery_left.saturating_sub(1);

        emit(progress, Progress::Thinking);
        let t_think = std::time::Instant::now();
        let resp = model::request_action(&client, cfg, active_model, &system, &messages, &tools)?;
        let think = t_think.elapsed();

        if let Some(note) = &resp.text {
            println!("  🧠 {note}");
            emit(progress, Progress::Note(note.clone()));
        }
        describe(&resp.action);
        emit(
            progress,
            Progress::Acting {
                action: describe_short(&resp.action),
                reason: resp.action.reason.clone(),
            },
        );

        // Repetition only means "that did not work" for discrete, one-shot actions — a click
        // that missed, a key that did not register. Two steering nudges in a row is ordinary
        // closed-loop control, and counting those kept most of a 120-step run pinned to the
        // heavy model for no benefit. Continuous control (`hold`, `move`, `drag`, `scroll`)
        // is covered by the frame comparison instead, which is the better evidence anyway.
        // `wait` and `screenshot` are observation and count for nothing.
        let discrete = matches!(
            resp.action.action.as_str(),
            "click" | "double_click" | "right_click" | "middle_click" | "key" | "keydown" | "keyup" | "type"
        );
        let mut repeated_action = false;
        if discrete {
            let sig = signature(&resp.action);
            repeated_action = last_sig.as_deref() == Some(sig.as_str());
            if cfg.model_heavy.is_some() && repeated_action {
                if recovery_left == 0 {
                    println!("  ⚠ action repeated — escalating for {RECOVERY_STEPS} steps");
                }
                recovery_left = RECOVERY_STEPS;
            }
            last_sig = Some(sig);
        }

        messages.push(json!({ "role": "assistant", "content": resp.assistant_content }));

        let t_act = std::time::Instant::now();
        match exec.execute(&resp.action, current.sent_w, current.sent_h) {
            Ok(Outcome::Done { success, message }) => {
                let mark = if success { "✅" } else { "🛑" };
                println!("\n{mark} agent finished: {}", if message.is_empty() { "(no message)" } else { &message });
                emit(progress, Progress::Finished { success, message });
                return Ok(());
            }
            Ok(Outcome::Continue) => failed_actions = 0,
            // A malformed action (bad name, missing argument) is the model's mistake, not a
            // fatal error: hand it back as the tool result so it can correct itself. Bail out
            // only if it cannot produce a usable action several times running.
            Err(e) => {
                println!("  ⚠ action failed: {e}");
                failed_actions += 1;
                if failed_actions >= MAX_FAILED_ACTIONS {
                    return Err(e.context(format!(
                        "{MAX_FAILED_ACTIONS} unusable actions in a row — giving up"
                    )));
                }
                messages.push(error_result_message(&resp.tool_use_id, &e.to_string()));
                continue;
            }
        }

        let act = t_act.elapsed();

        // Let the UI settle, then observe again.
        std::thread::sleep(std::time::Duration::from_millis(cfg.action_delay_ms));
        let t_look = std::time::Instant::now();
        let next = capture::capture(cfg.monitor_index, cfg.max_image_dim, cfg.image_format)?;
        emit_frame(progress, &next);
        let look = t_look.elapsed();

        // Did that action actually do anything? This is the loop's only ground truth, and
        // the model needs it: without it, it re-issues actions it cannot confirm landed.
        let delta = next.difference(&current);
        let changed = delta >= STASIS_THRESHOLD;
        current = next;
        messages.push(tool_result_message(
            &resp.tool_use_id,
            &current,
            step,
            changed,
            repeated_action,
        ));

        // A frozen screen is what being stuck actually looks like — far better evidence than
        // a repeated action, which is just as often a model steering steadily toward a
        // target. Repetition stays as a second trigger for the case where the world keeps
        // moving but the agent is going nowhere (a car pinned against a wall, engine
        // running, scenery alive).
        stale_frames = if changed { 0 } else { stale_frames + 1 };
        if cfg.model_heavy.is_some() && stale_frames >= STALE_FRAMES_TO_ESCALATE {
            if recovery_left == 0 {
                println!(
                    "  ⚠ screen unchanged for {stale_frames} steps — escalating for {RECOVERY_STEPS} steps"
                );
            }
            recovery_left = RECOVERY_STEPS;
        }

        if cfg.show_timing {
            // `think` is the round trip to the API and normally dominates by an order of
            // magnitude — worth seeing before tuning anything else.
            println!(
                "      ⏱ {:.1}s think · {:.1}s act · {:.2}s look · {} KB · {}{}",
                think.as_secs_f64(),
                act.as_secs_f64(),
                look.as_secs_f64(),
                current.bytes / 1024,
                if changed { resp.usage.label() } else { format!("{} · SCREEN FROZEN", resp.usage.label()) },
                crate::executor::latched_label()
                    .map(|k| format!(" · holding [{k}]"))
                    .unwrap_or_default(),
            );
        }
    }

    println!("\n⏹ Reached step limit ({}) without a 'done'. Stopping.", cfg.max_steps);
    emit(
        progress,
        Progress::Stopped(format!("reached the {}-step limit", cfg.max_steps)),
    );
    Ok(())
}

// ---- message construction ----

fn image_block(shot: &Screenshot) -> Value {
    json!({
        "type": "image",
        "source": { "type": "base64", "media_type": shot.media_type, "data": shot.png_base64 }
    })
}

fn initial_message(goal: &str, shot: &Screenshot) -> Value {
    json!({
        "role": "user",
        "content": [
            image_block(shot),
            { "type": "text", "text": format!(
                "GOAL: {goal}\n\nThis is the current screen ({}×{} px). Decide the single next \
                 action and call computer_action. Coordinates are in this image's pixel space.",
                shot.sent_w, shot.sent_h
            )}
        ]
    })
}

fn tool_result_message(
    tool_use_id: &str,
    shot: &Screenshot,
    step: u32,
    changed: bool,
    repeated: bool,
) -> Value {
    // Latched keys are invisible in a screenshot, and forgetting one is how a run ends up
    // flooring the throttle into a wall — so restate what is still down every turn.
    let held = match crate::executor::latched_label() {
        Some(keys) => format!(
            " Still held down from an earlier keydown: {keys} — release with keyup when done."
        ),
        None => String::new(),
    };
    // The model cannot tell from a single frame whether its last action landed, so it
    // re-issues things — pressing an enter-vehicle key that toggles and stepping straight
    // back out, re-latching a key already down. Comparing frames is something the harness
    // can do and the model cannot, so it gets told the answer outright.
    let effect = if changed {
        ""
    } else {
        " NOTE: the screen is essentially unchanged from before that action, so it most \
         likely had NO effect. Do not simply repeat it — either the target was wrong, or \
         the action was already in force, or it toggled something back. Try a different \
         approach."
    };
    // The frame comparison cannot catch a toggle: entering a vehicle and stepping back out
    // of it both change the screen dramatically, so neither looks frozen. What the harness
    // *can* see is that the same discrete action was issued twice running — which is how the
    // toggle gets hit in the first place.
    let repeat = if repeated {
        " NOTE: you issued this exact action on the previous turn as well. If it did not work \
         the first time, repeating it will not help; and if it DID work, a key that toggles \
         has now undone it. Look at the screen and pick a different action."
    } else {
        ""
    };
    json!({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": [
                image_block(shot),
                { "type": "text", "text": format!(
                    "Step {step} executed. Current screen ({}×{} px).{held}{effect}{repeat} \
                     Continue toward the goal, or call done.", shot.sent_w, shot.sent_h
                )}
            ]
        }]
    })
}

/// Report a rejected action back to the model. No new screenshot: nothing happened, so the
/// last one is still current.
fn error_result_message(tool_use_id: &str, err: &str) -> Value {
    json!({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "is_error": true,
            "content": [{ "type": "text", "text": format!(
                "That action was rejected: {err}. Nothing happened; the screen is unchanged. \
                 Use only these exact action names: move, click, double_click, right_click, \
                 middle_click, drag, scroll, type, key, hold, keydown, keyup, wait, \
                 screenshot, done — with 'button' for the mouse button. Try again."
            )}]
        }]
    })
}

// ---- history trimming (bound token growth) ----

/// Keep only the last `keep` screenshots; replace older image blocks with a placeholder.
fn trim_history(messages: &mut [Value], keep: usize) {
    let img_msgs: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| message_has_image(m))
        .map(|(i, _)| i)
        .collect();
    if img_msgs.len() <= keep {
        return;
    }
    let strip_upto = img_msgs.len() - keep;
    for &i in &img_msgs[..strip_upto] {
        strip_images(&mut messages[i]);
    }
}

fn message_has_image(m: &Value) -> bool {
    let Some(content) = m.get("content").and_then(Value::as_array) else {
        return false;
    };
    content.iter().any(|b| {
        b.get("type").and_then(Value::as_str) == Some("image")
            || b.get("content")
                .and_then(Value::as_array)
                .map(|inner| inner.iter().any(|x| x.get("type").and_then(Value::as_str) == Some("image")))
                .unwrap_or(false)
    })
}

fn placeholder() -> Value {
    json!({ "type": "text", "text": "[earlier screenshot omitted to save context]" })
}

fn strip_images(m: &mut Value) {
    let Some(content) = m.get_mut("content").and_then(Value::as_array_mut) else {
        return;
    };
    for block in content.iter_mut() {
        let is_img = block.get("type").and_then(Value::as_str) == Some("image");
        if is_img {
            *block = placeholder();
            continue;
        }
        // tool_result blocks carry a nested content array with the image.
        if block.get("type").and_then(Value::as_str) == Some("tool_result") {
            if let Some(inner) = block.get_mut("content").and_then(Value::as_array_mut) {
                for b in inner.iter_mut() {
                    if b.get("type").and_then(Value::as_str) == Some("image") {
                        *b = placeholder();
                    }
                }
            }
        }
    }
}

// ---- console output ----

fn describe(a: &crate::action::ActionInput) {
    use std::fmt::Write;
    let mut s = format!("  ▶ {}", a.action);
    let mut extra = String::new();
    if let (Some(x), Some(y)) = (a.x, a.y) {
        let _ = write!(extra, " @({:.0},{:.0})", x, y);
    }
    if let Some(t) = &a.text {
        let _ = write!(extra, " text={:?}", truncate(t, 40));
    }
    if let Some(k) = &a.keys {
        let _ = write!(extra, " keys={k:?}");
    }
    if let Some(ms) = a.ms {
        let _ = write!(extra, " {ms}ms");
    }
    if a.scroll_x.is_some() || a.scroll_y.is_some() {
        let _ = write!(extra, " scroll=({},{})", a.scroll_x.unwrap_or(0), a.scroll_y.unwrap_or(0));
    }
    s.push_str(&extra);
    if let Some(r) = &a.reason {
        let _ = write!(s, "  — {r}");
    }
    println!("{s}");
}

/// The action without its reason or console decoration — e.g. `click @(840,412)`.
///
/// The reason travels separately in [`Progress::Acting`] so a UI can style the two
/// differently instead of parsing them apart again.
fn describe_short(a: &crate::action::ActionInput) -> String {
    use std::fmt::Write;
    let mut s = a.action.clone();
    if let (Some(x), Some(y)) = (a.x, a.y) {
        let _ = write!(s, " @({x:.0},{y:.0})");
    }
    if let Some(k) = &a.keys {
        let _ = write!(s, " {k}");
    }
    if let Some(t) = &a.text {
        let _ = write!(s, " {:?}", truncate(t, 24));
    }
    if let Some(ms) = a.ms {
        let _ = write!(s, " {ms}ms");
    }
    s
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}
