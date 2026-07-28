//! Claude (Anthropic Messages API) client over raw HTTP.
//!
//! There is no official Anthropic Rust SDK, so this speaks the wire protocol directly.
//! Each call sends the running conversation (screenshots + prior actions) and forces the
//! model to answer with exactly one `computer_action` tool call.

use crate::action::ActionInput;
use crate::config::Config;
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

/// Request features the API has told us a given model will not take — `speed` and `effort`
/// are both tier-gated, and which models accept them shifts every release.
///
/// Keyed by model rather than global: with a fast model and a heavy model in the same run,
/// Opus accepts both features and Haiku accepts neither, so one shared flag would either
/// strip them from Opus or re-discover the rejection on Haiku every single step. Learning
/// costs a wasted round trip, so it is paid once per model per process.
static UNSUPPORTED: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(Default::default);

fn unsupported(model: &str, feature: &str) -> bool {
    UNSUPPORTED
        .lock()
        .map(|s| s.contains(&format!("{model}|{feature}")))
        .unwrap_or(false)
}

fn mark_unsupported(model: &str, feature: &str) {
    if let Ok(mut s) = UNSUPPORTED.lock() {
        s.insert(format!("{model}|{feature}"));
    }
}

const API_VERSION: &str = "2023-06-01";
/// The single tool the loop exposes. Shared with [`crate::openrouter`], which has to name
/// the same function when it rebuilds a reply into this module's canonical turn shape.
pub(crate) const TOOL_NAME: &str = "computer_action";
/// Beta flag gating `"speed": "fast"`. Sent only when fast mode is actually requested.
const FAST_MODE_BETA: &str = "fast-mode-2026-02-01";
/// Beta flag required whenever the credential is an OAuth token rather than an API key.
/// Without it `/v1/messages` rejects the Bearer credential.
const OAUTH_BETA: &str = "oauth-2025-04-20";

/// What one model turn produced.
pub struct ModelResponse {
    /// The assistant `content` array, to append verbatim to the conversation.
    pub assistant_content: Value,
    /// The `tool_use` id, echoed back in the next `tool_result`.
    pub tool_use_id: String,
    /// The decoded action.
    pub action: ActionInput,
    /// Any free-text the model wrote alongside the tool call (its reasoning/narration).
    pub text: Option<String>,
    /// Token accounting for the turn. Wall-clock alone is too noisy to tell whether a
    /// latency change actually landed; these numbers say so directly.
    pub usage: Usage,
}

/// The `usage` block of a reply, reduced to what is worth watching per step.
#[derive(Default)]
pub struct Usage {
    /// Prompt tokens served from cache (~0.1x price, and no re-prefill).
    pub cache_read: u64,
    /// Prompt tokens written to cache this turn (~1.25x price, first turn only).
    pub cache_write: u64,
    /// Prompt tokens processed at full price — the uncached remainder.
    pub input: u64,
    /// Generated tokens. Produced serially, so this is wall-clock, not just cost.
    pub output: u64,
    /// Which speed actually served the turn — fast mode can silently fall back.
    pub speed: Option<String>,
}

impl Usage {
    fn from_response(v: &Value) -> Self {
        let Some(u) = v.get("usage") else {
            return Self::default();
        };
        let n = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
        Self {
            cache_read: n("cache_read_input_tokens"),
            cache_write: n("cache_creation_input_tokens"),
            input: n("input_tokens"),
            output: n("output_tokens"),
            speed: u.get("speed").and_then(Value::as_str).map(str::to_string),
        }
    }

    /// The same accounting from an OpenAI-shaped `usage` block.
    ///
    /// The two count differently: OpenAI's `prompt_tokens` is the *whole* prompt including
    /// anything served from cache, where Anthropic's `input_tokens` is the uncached
    /// remainder. Subtracting here means the per-step readout says the same thing on both
    /// providers instead of appearing to double-count a cache hit.
    pub(crate) fn from_openai(v: &Value) -> Self {
        let Some(u) = v.get("usage") else {
            return Self::default();
        };
        let n = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
        let cached = u
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        Self {
            cache_read: cached,
            cache_write: 0,
            input: n("prompt_tokens").saturating_sub(cached),
            output: n("completion_tokens"),
            speed: None,
        }
    }

    /// Compact one-line summary for the per-step timing readout.
    pub fn label(&self) -> String {
        format!(
            "{} in ({} cached) · {} out{}",
            self.input + self.cache_read + self.cache_write,
            self.cache_read,
            self.output,
            match self.speed.as_deref() {
                Some("fast") => " · fast",
                _ => "",
            }
        )
    }
}

/// The `computer_action` tool definition.
pub fn tools() -> Value {
    json!([{
        "name": TOOL_NAME,
        "description": "Perform exactly one desktop input action to make progress toward the user's goal. \
                        All coordinates are in the pixel space of the screenshot you were just shown \
                        (origin at top-left). Call this once per turn. Use action \"done\" when the goal \
                        is complete or cannot be completed.",
        "input_schema": crate::action::tool_schema(),
    }])
}

pub fn system_prompt(terse: bool, game: bool) -> String {
    // The general rules below explain latching, but a live test showed a small model reading
    // past it and driving with `key "w"` then `hold "w" 2000ms` — a tap, then a two-second
    // lurch, then `done`. The mechanism was there; the instruction was not emphatic enough.
    // In real-time mode it gets its own block, stated as rules rather than prose.
    let realtime = if game {
        "\n\nREAL-TIME MODE — the world does not pause while you think:\n\
         - For ANY sustained movement (driving, running, flying), call `keydown` ONCE to latch \
           the key down, then leave it latched across turns. Never use `hold` for forward \
           motion and never re-tap it — both produce stop-start lurching, because the throttle \
           is off while you look at the next screenshot.\n\
         - Movement keys are relative to the vehicle or character, NOT to the terrain or the \
           compass. Forward (`w`) means the direction it is currently facing. Reverse (`s`) \
           is backwards — it is not \"downhill\", \"south\", or \"back the way you came\". To \
           travel down a slope, face down the slope and go FORWARD.\n\
         - Steer with SHORT `hold` nudges (100-400ms) on the steering key while the throttle \
           stays latched. Correct in small increments; you are always looking at a stale frame.\n\
         - DO NOT USE `wait` OR `screenshot`. You are given a fresh screenshot every single \
           turn automatically, and the 1-3 second gap between turns is already real time in \
           which your latched keys keep acting — the observing and the waiting are both free. \
           Spending a turn on `wait` or `screenshot` buys you nothing and doubles how long you \
           are blind, so you walk or drive straight past what you were aiming for. Every turn, \
           take a real action: steer, latch, release, press a key, or `done`.\n\
         - The tool_result tells you what is still held down each turn. If the throttle is \
           already latched, do not latch it again — steer, or make a small correction.\n\
         - MANY GAME KEYS TOGGLE. The enter-vehicle key is the same key as exit-vehicle, so \
           pressing it again after you are already inside puts you back on the street. If a \
           key appears not to have worked, check the screen before repeating it: if the view \
           changed, it DID work and pressing again will undo it. The tool_result tells you \
           when the screen did not change — trust that over your own guess.\n\
         - RELEASE BEFORE REVERSING. A latched key stays physically down, so pressing the \
           reverse/brake key while the throttle is still latched holds both at once and the \
           vehicle fights itself and goes nowhere. To reverse or brake: `keyup` the throttle \
           FIRST, then `hold` the reverse key. Re-latch the throttle with `keydown` when you \
           are clear and want to move forward again. The same applies to any opposed pair.\n\
         - If you are stuck or colliding, the first thing to check is whether something is \
           still latched that should not be. Repeating an action that did not work last turn \
           will not work this turn — change the approach.\n\
         - Otherwise use `keyup` to stop, slow down, or when the goal is finished.\n\
         - An open-ended goal (\"drive along the road\") is NOT complete after one action. Keep \
           driving and correcting for many turns. Only call `done` when the goal is genuinely \
           achieved or you are truly stuck."
    } else {
        ""
    };
    // Output tokens are generated one at a time, so a paragraph of narration costs real
    // wall-clock on every single step. In fast mode we buy that time back.
    let brevity = if terse {
        "\n\nSPEED MODE: reply with the tool call and nothing else. No narration, no \
         explanation. Keep `reason` under six words. Latency matters more than detail here."
    } else {
        ""
    };
    format!(
        "{}{realtime}{brevity}",
        format!(
        "You are Autopilot Mode, an autonomous agent that operates a real computer by looking at \
         screenshots and issuing mouse/keyboard actions.\n\n\
         ENVIRONMENT: {platform}\n\n\
         Each turn you are shown the current screen and must call the `computer_action` tool \
         exactly once with the single best next action toward the user's goal. Work in small, \
         verifiable steps: look at the latest screenshot, decide one action, then observe the \
         result on the next turn before continuing.\n\n\
         Rules:\n\
         - YOU are the one doing the work. The terminal on screen running `autopilotmode` is your \
           own harness — it is not a separate program that will act for you. Never wait for it \
           to do something, and do not type into it unless the goal is about that terminal.\n\
         - Coordinates are pixels in the screenshot you were just shown (top-left is 0,0). Aim \
           for the center of the target (button, field, icon).\n\
         - Prefer clicking a field before typing into it. Use `key` for shortcuts and Enter \
           (e.g. keys \"ctrl+l\", \"enter\"); use the shortcut conventions of the operating system above.\n\
         - `key` taps instantly. For a press of a definite length — a short steering nudge, \
           a timed sprint — use `hold` with `keys` and `ms` (e.g. keys \"a\", ms 300). All \
           keys listed in `hold` are held down together.\n\
         - `hold` blocks until it finishes, so nothing is pressed while you look at the next \
           screenshot and decide (several seconds). For CONTINUOUS control — accelerating a \
           car, walking a long distance — use `keydown` instead: it presses a key and leaves \
           it down across turns, so the vehicle keeps moving while you think. Steer with \
           short `hold` nudges on top of it, and `keyup` to let go (`keyup` with no `keys` \
           releases everything). Prefer `keydown` \"w\" over repeated `hold` \"w\" — repeating \
           `hold` produces stop-start lurching, because the throttle is off between steps.\n\
         - Whatever you latch stays down while you observe, so the screen you are looking at \
           is already a few seconds old by the time your next action lands. Keep speed \
           moderate, correct in small increments rather than large ones, and release early \
           when you need to stop.\n\
         - After an action that opens/loads something, expect the next screenshot to reflect it; \
           if it did not change, try a different approach rather than repeating.\n\
         - Keep `reason` to one short sentence describing what the action does.\n\
         - Call action \"done\" with success=true when the goal is achieved, or success=false with \
           a reason if you are stuck. Do not keep acting after the goal is met.\n\
         - Never take destructive or irreversible actions (deleting files, purchases, sending \
           messages) unless the goal explicitly asks for them.",
            platform = platform_hint()
        )
    )
}

/// A one-line description of the OS the binary was built for, with example shortcuts, so
/// the model uses the right conventions (Start menu vs Spotlight, ctrl vs cmd, ...).
fn platform_hint() -> String {
    match std::env::consts::OS {
        "windows" => "You are operating Windows. Open apps from the Start menu (press the `win` key \
                      or click Start in the taskbar, then type the app name). Typical shortcuts: \
                      `ctrl+c`/`ctrl+v` (copy/paste), `alt+tab` (switch window), `ctrl+l` \
                      (browser/Explorer address bar), `win` (Start), `win+r` (Run)."
            .to_string(),
        "macos" => "You are operating macOS. Open apps via Spotlight (`cmd+space`, then type the \
                    name) or the Dock. Typical shortcuts: `cmd+c`/`cmd+v` (copy/paste), `cmd+tab` \
                    (switch app), `cmd+l` (browser address bar), `cmd+space` (Spotlight)."
            .to_string(),
        "linux" => "You are operating Linux (desktop). Open apps from the applications launcher \
                    (often the `super`/`win` key) or a terminal. Typical shortcuts: `ctrl+c`/`ctrl+v` \
                    in apps, `alt+tab` (switch window), `ctrl+l` (browser address bar). In a terminal, \
                    copy/paste is usually `ctrl+shift+c`/`ctrl+shift+v`."
            .to_string(),
        other => format!(
            "You are operating {other}. Use that platform's standard mouse and keyboard conventions."
        ),
    }
}

/// Send one request and return the chosen action. Retries transient errors (429/5xx).
///
/// Dispatches on the configured provider. `messages` is always in Anthropic's shape — that
/// is the loop's canonical history format — and [`crate::openrouter`] translates it for the
/// OpenAI-compatible path rather than the agent knowing there is a choice.
pub fn request_action(
    client: &reqwest::blocking::Client,
    cfg: &Config,
    model: &str,
    system: &str,
    messages: &[Value],
    tools: &Value,
) -> Result<ModelResponse> {
    match cfg.provider {
        crate::config::Provider::Anthropic => {
            request_anthropic(client, cfg, model, system, messages, tools)
        }
        crate::config::Provider::OpenRouter => {
            crate::openrouter::request_action(client, cfg, model, system, messages, tools)
        }
    }
}

/// The native path: Anthropic's Messages API.
fn request_anthropic(
    client: &reqwest::blocking::Client,
    cfg: &Config,
    model: &str,
    system: &str,
    messages: &[Value],
    tools: &Value,
) -> Result<ModelResponse> {
    let url = format!("{}/v1/messages", cfg.base_url.trim_end_matches('/'));
    let mut body = json!({
        "model": model,
        // Generous headroom: the reply must fit any narration *plus* the tool call. If the
        // narration alone exhausts the budget the turn arrives with no action at all.
        "max_tokens": 4096,
        // The system prompt and tool schema are byte-identical on every turn of a run, and
        // together they are the whole cacheable prefix (render order is tools → system →
        // messages). One breakpoint on the last system block therefore caches both, so every
        // step after the first skips re-prefilling them. Requires `system` to be a block
        // array rather than a bare string — that is the only reason for the shape here.
        "system": [{
            "type": "text",
            "text": system,
            "cache_control": { "type": "ephemeral" }
        }],
        "tools": tools,
        // Force exactly one tool call per turn.
        "tool_choice": { "type": "any", "disable_parallel_tool_use": true },
        // Opus 5 thinks by default, at `high` effort unless told otherwise — seconds of
        // deliberation before every click. Thinking itself stays ON: disabling it on Opus 5
        // can make the model write the tool call into its visible text instead of emitting a
        // tool_use block, and that turn "succeeds" with the action silently never running.
        // Dialling the depth back is the safe way to buy the latency.
        "messages": messages,
    });

    // Only Opus/Sonnet-tier models accept `effort`; sending it to Haiku is a 400.
    if !cfg.effort.is_empty() && !unsupported(model, "effort") {
        body["output_config"] = json!({ "effort": cfg.effort });
    }

    // Fast mode is a beta, so it needs both the request field and the header.
    let mut fast = cfg.speed_fast && !unsupported(model, "speed");
    if fast {
        body["speed"] = json!("fast");
    }

    let mut last_err = None;
    for attempt in 0..ATTEMPTS {
        let mut req = client
            .post(&url)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json");

        // An API key and an OAuth token are not interchangeable headers: sending a token as
        // `x-api-key` is a 401. The OAuth path also needs its own beta flag, and beta flags
        // combine comma-separated rather than overwriting each other.
        let mut betas: Vec<&str> = Vec::new();
        if cfg.oauth {
            req = req.header("authorization", format!("Bearer {}", cfg.api_key));
            betas.push(OAUTH_BETA);
        } else {
            req = req.header("x-api-key", &cfg.api_key);
        }
        if fast {
            betas.push(FAST_MODE_BETA);
        }
        if !betas.is_empty() {
            req = req.header("anthropic-beta", betas.join(","));
        }
        let resp = req.json(&body).send();

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                last_err = Some(anyhow!("network error: {e}"));
                backoff(attempt);
                continue;
            }
        };

        let status = resp.status();
        let text = resp.text().unwrap_or_default();

        if status.is_success() {
            match parse_success(&text)? {
                Parsed::Action(r) => return Ok(*r),
                // `tool_choice: any` forces a tool call, so this is almost always the reply
                // being truncated mid-narration. Resampling usually fixes it; a hard error
                // here would throw away an otherwise healthy run.
                Parsed::NoAction { stop_reason, said } => {
                    eprintln!(
                        "  ⚠ no action in reply (stop_reason: {stop_reason}) — retrying{}",
                        if attempt + 1 < ATTEMPTS { "" } else { " (last attempt)" }
                    );
                    last_err = Some(anyhow!(
                        "model did not call {TOOL_NAME} after {ATTEMPTS} attempts \
                         (stop_reason: {stop_reason}){}",
                        said.map(|t| format!(" — it said: {}", truncate(&t, 200))).unwrap_or_default()
                    ));
                    backoff(attempt);
                    continue;
                }
            }
        }

        // Fast mode draws on its own rate-limit pool, separate from standard Opus. Running
        // it dry should cost the run some speed, not end it — drop back to standard and keep
        // driving rather than burning the remaining attempts on a bucket that is empty.
        if status.as_u16() == 429 && fast {
            eprintln!("  ⚠ fast-mode rate limit — falling back to standard speed");
            fast = false;
            mark_unsupported(model, "speed");
            if let Some(obj) = body.as_object_mut() {
                obj.remove("speed");
            }
            continue;
        }

        // `speed` and `effort` are tier-gated: Haiku rejects both, and which models accept
        // them shifts every release. The whole point of AUTOPILOT_MODEL is that swapping
        // models needs no code change, so carrying a hardcoded allowlist here would defeat
        // it — let the API say what it will not take, drop that field, and carry on.
        if status.as_u16() == 400 {
            if fast && text.contains("speed") {
                eprintln!("  ⚠ {model} does not support fast mode — using standard speed");
                fast = false;
                mark_unsupported(model, "speed");
                if let Some(obj) = body.as_object_mut() {
                    obj.remove("speed");
                }
                continue;
            }
            if body.get("output_config").is_some() && text.contains("effort") {
                eprintln!("  ⚠ {model} does not support effort — omitting it");
                mark_unsupported(model, "effort");
                if let Some(obj) = body.as_object_mut() {
                    obj.remove("output_config");
                }
                continue;
            }
        }

        // Retry rate limits / overload / server errors.
        if status.as_u16() == 429 || status.as_u16() == 529 || status.is_server_error() {
            last_err = Some(anyhow!("API {status}: {}", truncate(&text, 300)));
            backoff(attempt);
            continue;
        }

        // Non-retryable (400/401/403/404 ...).
        return Err(anyhow!("API error {status}: {}", truncate(&text, 600)));
    }

    Err(last_err.unwrap_or_else(|| anyhow!("request failed after retries")))
}

/// How many times to (re)issue a turn before giving up.
pub(crate) const ATTEMPTS: u32 = 3;

/// Outcome of reading a 200 response. Shared by both provider paths.
pub(crate) enum Parsed {
    /// Boxed because a `ModelResponse` carries the whole assistant turn — enough larger
    /// than the other variant that every `Parsed` would otherwise be sized for it.
    Action(Box<ModelResponse>),
    /// Well-formed reply that carried no tool call — worth resampling rather than failing.
    NoAction {
        stop_reason: String,
        said: Option<String>,
    },
}

/// Coerce loosely-typed tool input into the shape the schema asks for.
///
/// Smaller models fill numeric fields with strings, and sometimes pack both coordinates into
/// one (`"x": "168, 152"`). Resampling does not help — the same prompt reproduces the same
/// mistake, which burns every retry and kills the run. The intent is unambiguous, so repair
/// it here instead.
pub(crate) fn normalize_action_input(input: &mut Value) {
    let Some(obj) = input.as_object_mut() else {
        return;
    };

    // Both coordinates packed into `x`.
    let packed = obj.get("x").and_then(Value::as_str).and_then(|s| {
        let (a, b) = s.split_once(',')?;
        Some((a.trim().parse::<f64>().ok()?, b.trim().parse::<f64>().ok()?))
    });
    if let Some((x, y)) = packed {
        obj.insert("x".to_string(), json!(x));
        if obj.get("y").is_none_or(Value::is_null) {
            obj.insert("y".to_string(), json!(y));
        }
    }

    // A number sent as text.
    for key in ["x", "y", "scroll_x", "scroll_y", "ms"] {
        let parsed = obj
            .get(key)
            .and_then(Value::as_str)
            .and_then(|s| s.trim().parse::<f64>().ok());
        if let Some(n) = parsed {
            obj.insert(key.to_string(), json!(n));
        }
    }
}

fn parse_success(text: &str) -> Result<Parsed> {
    let v: Value = serde_json::from_str(text)
        .map_err(|e| anyhow!("could not parse API response: {e}\nbody: {}", truncate(text, 400)))?;

    if v.get("stop_reason").and_then(Value::as_str) == Some("refusal") {
        let detail = v
            .get("stop_details")
            .and_then(|d| d.get("explanation"))
            .and_then(Value::as_str)
            .unwrap_or("model refused the request");
        return Err(anyhow!("model refusal: {detail}"));
    }

    let content = v
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("response had no content array"))?;

    let mut narration: Option<String> = None;
    let mut tool_use: Option<(String, ActionInput)> = None;

    for block in content {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(t) = block.get("text").and_then(Value::as_str) {
                    if !t.trim().is_empty() {
                        narration = Some(t.trim().to_string());
                    }
                }
            }
            Some("tool_use") if block.get("name").and_then(Value::as_str) == Some(TOOL_NAME) => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("tool_use block missing id"))?
                    .to_string();
                let mut input = block
                    .get("input")
                    .cloned()
                    .ok_or_else(|| anyhow!("tool_use block missing input"))?;
                normalize_action_input(&mut input);
                // Smaller models sometimes fill the schema loosely — packing both
                // coordinates into one string ("385, 87"), or a number as text. That is a
                // bad sample, not a broken run: resampling almost always yields clean JSON,
                // so hand it back to the retry loop instead of aborting mid-drive.
                let action: ActionInput = match serde_json::from_value(input) {
                    Ok(a) => a,
                    Err(e) => {
                        return Ok(Parsed::NoAction {
                            stop_reason: format!("malformed action input ({e})"),
                            said: narration,
                        })
                    }
                };
                tool_use = Some((id, action));
            }
            _ => {}
        }
    }

    let Some((tool_use_id, action)) = tool_use else {
        return Ok(Parsed::NoAction {
            stop_reason: v
                .get("stop_reason")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
            said: narration,
        });
    };

    Ok(Parsed::Action(Box::new(ModelResponse {
        assistant_content: Value::Array(content.clone()),
        tool_use_id,
        action,
        text: narration,
        usage: Usage::from_response(&v),
    })))
}

pub(crate) fn backoff(attempt: u32) {
    let secs = 1u64 << attempt; // 1s, 2s, 4s
    std::thread::sleep(std::time::Duration::from_secs(secs));
}

pub(crate) fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}
