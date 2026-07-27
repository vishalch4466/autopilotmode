//! Runtime configuration, read from environment / `.env`.

use anyhow::{anyhow, Result};

/// All tunables for a run. Built once from the environment in [`Config::from_env`].
pub struct Config {
    /// Anthropic credential — either an API key (`sk-ant-api…`) or an OAuth access token
    /// (`sk-ant-oat…`) minted by `ant auth login`.
    pub api_key: String,
    /// True when [`Config::api_key`] is an OAuth token rather than an API key.
    ///
    /// The two are sent completely differently: an API key goes in `x-api-key`, while an
    /// OAuth token goes in `Authorization: Bearer` *and* additionally requires the
    /// `oauth-2025-04-20` beta header. Sending an OAuth token as `x-api-key` is a 401, so
    /// this is detected from the prefix rather than left to the user to configure.
    pub oauth: bool,
    /// API base URL (override for proxies / gateways).
    pub base_url: String,
    /// Model id driving the loop, e.g. `claude-opus-5`. Every step uses this unless the run
    /// escalates (see [`Config::model_heavy`]).
    pub model: String,
    /// Optional stronger model, used only for the steps immediately after the loop stops
    /// making progress.
    ///
    /// A faster model handles ordinary steering well and halves the round trip, but reasoning
    /// out of a wedged position is where the extra capability actually pays. Escalating on
    /// demand keeps the fast path fast and spends the slower model only where it earns its
    /// latency. Unset (the default) means every step uses `model`.
    pub model_heavy: Option<String>,
    /// Hard cap on the number of observe→act iterations.
    pub max_steps: u32,
    /// Longest edge (px) of the screenshot sent to the model. Smaller = cheaper/faster,
    /// larger = more precise coordinates. The model is told the exact dimensions it sees.
    pub max_image_dim: u32,
    /// Pause after executing an action before grabbing the next screenshot, so the UI
    /// has time to react (menus open, pages load, etc.).
    pub action_delay_ms: u64,
    /// How many of the most-recent screenshots to keep in the conversation. Older ones
    /// are replaced with a placeholder to bound token growth.
    pub keep_screenshots: usize,
    /// Which monitor to capture/drive (0 = first).
    pub monitor_index: usize,
    /// Print intended actions without actually moving the mouse/keyboard.
    pub dry_run: bool,
    /// Glide the pointer along a curved, eased path rather than teleporting to the
    /// target. Also makes hover/drag behave, since intermediate motion events are what
    /// most UIs actually listen for.
    pub human_mouse: bool,
    /// Multiplier on pointer travel time (higher = faster, 1.0 = human-ish).
    pub mouse_speed: f64,
    /// Wire format for screenshots. JPEG uploads several times faster than PNG.
    pub image_format: crate::capture::ImageFormat,
    /// Ask the model to skip narration and keep `reason` to a few words. Output tokens are
    /// produced serially, so this is a direct latency saving, not just a cosmetic one.
    pub terse: bool,
    /// Reasoning depth: `low` | `medium` | `high` | `xhigh` | `max`.
    ///
    /// Claude Opus 5 thinks by default and defaults to `high`, i.e. seconds of deliberation
    /// before every click. This loop takes one small action per screenshot and re-checks its
    /// work on the next frame, which is exactly the shape low effort handles well — so `low`
    /// is the right default here rather than the API's. Raise it via AUTOPILOT_EFFORT if the
    /// model starts misreading the screen or picking poor targets.
    pub effort: String,
    /// Request Anthropic's fast mode: the same model, generating output up to ~2.5x faster.
    ///
    /// Measurement on this loop says the round trip is dominated by model time, not payload
    /// size — shrinking the screenshot 6x changed nothing — so this is the lever that
    /// actually moves it. It bills at premium rates ($10/$50 per MTok vs $5/$25), so it
    /// stays off unless asked for, and `--game` turns it on because that is the mode where
    /// latency IS the product.
    pub speed_fast: bool,
    /// Real-time control mode. Changes what the model is told to optimise for: latching the
    /// throttle and steering in nudges, rather than one discrete action per screenshot.
    pub game_mode: bool,
    /// Print a per-step breakdown of where the wall-clock went.
    pub show_timing: bool,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        // Accept several key names so an existing `.env` keeps working. `CLAUDELLM`
        // matches the variable this project's `.env` was seeded with.
        let api_key = first_env(&[
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "AUTOPILOT_API_KEY",
            "CLAUDELLM",
            "ClAUDELLM",
            "CLAUDE_API_KEY",
        ])
        .ok_or_else(|| {
            anyhow!(
                "No Anthropic credential found. Put one in autopilotmode/.env, e.g.\n  \
                 ANTHROPIC_API_KEY=sk-ant-api...\nor, to use a subscription credential from \
                 `ant auth login`:\n  \
                 ANTHROPIC_AUTH_TOKEN=$(ant auth print-credentials --access-token)\n\
                 (the loader also accepts CLAUDELLM / AUTOPILOT_API_KEY)."
            )
        })?;

        Ok(Self {
            // OAuth access tokens carry an `oat` prefix; API keys carry `api`.
            oauth: api_key.starts_with("sk-ant-oat"),
            api_key,
            base_url: env_str("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            // Claude Opus 5 is the current top Opus. Override anytime via AUTOPILOT_MODEL
            // when a newer model ships.
            model: env_str("AUTOPILOT_MODEL", "claude-opus-5"),
            model_heavy: env_var("AUTOPILOT_MODEL_HEAVY").map(|v| v.trim().to_string()),
            max_steps: env_num("AUTOPILOT_MAX_STEPS", 25),
            max_image_dim: env_num("AUTOPILOT_MAX_IMAGE_DIM", 1280),
            action_delay_ms: env_num("AUTOPILOT_ACTION_DELAY_MS", 600),
            keep_screenshots: env_num::<usize>("AUTOPILOT_KEEP_SCREENSHOTS", 3).max(1),
            monitor_index: env_num("AUTOPILOT_MONITOR", 0),
            dry_run: false,
            // On by default: teleporting the cursor breaks hover menus and drag-and-drop.
            human_mouse: !env_is_off("AUTOPILOT_HUMAN_MOUSE"),
            mouse_speed: env_num::<f64>("AUTOPILOT_MOUSE_SPEED", 1.0).clamp(0.25, 8.0),
            image_format: env_var("AUTOPILOT_IMAGE_FORMAT")
                .and_then(|v| crate::capture::ImageFormat::parse(&v))
                .unwrap_or(crate::capture::ImageFormat::Png),
            terse: env_flag_on("AUTOPILOT_TERSE"),
            effort: env_effort("AUTOPILOT_EFFORT", "low"),
            speed_fast: env_flag_on("AUTOPILOT_FAST_MODE"),
            game_mode: false,
            show_timing: !env_is_off("AUTOPILOT_SHOW_TIMING"),
        })
    }

    /// Trade fidelity for round-trip latency.
    ///
    /// Smaller images mean less to upload *and* fewer vision tokens to process; one kept
    /// screenshot instead of three cuts the request size again; terse replies shorten the
    /// serially-generated output. Explicit env vars still win — this only lowers values
    /// the user has not set themselves.
    pub fn apply_fast_preset(&mut self) {
        if env_var("AUTOPILOT_MAX_IMAGE_DIM").is_none() {
            self.max_image_dim = 768;
        }
        if env_var("AUTOPILOT_KEEP_SCREENSHOTS").is_none() {
            self.keep_screenshots = 1;
        }
        if env_var("AUTOPILOT_ACTION_DELAY_MS").is_none() {
            self.action_delay_ms = 150;
        }
        if env_var("AUTOPILOT_IMAGE_FORMAT").is_none() {
            self.image_format = crate::capture::ImageFormat::Jpeg;
        }
        self.terse = true;
    }

    /// Tuning for real-time control (games), on top of the fast preset.
    ///
    /// The world does not pause while you think, so round-trip latency matters more than
    /// coordinate precision, and there is no menu to "settle" — the sooner the next frame
    /// is grabbed, the less stale the picture the model steers on.
    pub fn apply_game_preset(&mut self) {
        self.apply_fast_preset();
        self.game_mode = true;
        if env_var("AUTOPILOT_ACTION_DELAY_MS").is_none() {
            self.action_delay_ms = 60;
        }
        // The round trip is model-bound, not payload-bound, so this is the only preset knob
        // that meaningfully shortens it. Opt out with AUTOPILOT_FAST_MODE=0 if the premium
        // token rate matters more than the frames.
        if !env_is_off("AUTOPILOT_FAST_MODE") {
            self.speed_fast = true;
        }
        // Pointer gliding exists for hover menus and drag-and-drop. A game captures the
        // mouse and reads raw deltas, so tracing a curved path buys nothing and costs
        // real time on every move.
        self.human_mouse = false;
    }
}

/// Read a configuration variable, falling back to the pre-rename `ROBOTMOUSE_` spelling.
///
/// This project was renamed from `robotmouse`, and `.env` is the one file a rename cannot
/// carry with it: it is git-ignored, so it is precisely the file that does not get updated.
/// Without this fallback an out-of-date `.env` would not fail loudly — it would silently
/// revert every tunable to its default, which is much harder to notice than an error.
fn env_var(name: &str) -> Option<String> {
    fn read(name: &str) -> Option<String> {
        std::env::var(name).ok().filter(|v| !v.trim().is_empty())
    }
    read(name).or_else(|| {
        name.strip_prefix("AUTOPILOT_")
            .and_then(|suffix| read(&format!("ROBOTMOUSE_{suffix}")))
    })
}

fn first_env(names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|n| env_var(n).map(|v| v.trim().to_string()))
}

fn env_str(name: &str, default: &str) -> String {
    env_var(name).unwrap_or_else(|| default.to_string())
}

/// True when the variable is explicitly set to a falsey value — used for flags that
/// default to on.
fn env_is_off(name: &str) -> bool {
    matches!(
        env_var(name).as_deref().map(str::trim),
        Some("0") | Some("false") | Some("no") | Some("off")
    )
}

/// True when the variable is explicitly set to a truthy value — for flags defaulting off.
pub fn env_flag_on(name: &str) -> bool {
    matches!(
        env_var(name).as_deref().map(str::trim),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

/// Read an effort level, rejecting anything the API would not accept.
///
/// An unknown level is a 400 on the very first turn, which would look like a broken build
/// rather than a typo in `.env` — so an invalid override warns and falls back instead.
fn env_effort(name: &str, default: &str) -> String {
    const LEVELS: [&str; 5] = ["low", "medium", "high", "xhigh", "max"];
    match env_var(name).map(|v| v.trim().to_ascii_lowercase()) {
        // Effort is an Opus/Sonnet-tier parameter; Haiku 4.5 rejects it outright. An escape
        // hatch beats hardcoding a model list that goes stale on the next release.
        Some(v) if matches!(v.as_str(), "none" | "off") => String::new(),
        Some(v) if LEVELS.contains(&v.as_str()) => v,
        Some(v) if !v.is_empty() => {
            eprintln!(
                "  ⚠ ignoring {name}={v:?} — expected one of {}. Using {default:?}.",
                LEVELS.join(", ")
            );
            default.to_string()
        }
        _ => default.to_string(),
    }
}

fn env_num<T: std::str::FromStr>(name: &str, default: T) -> T {
    env_var(name)
        .and_then(|v| v.trim().parse::<T>().ok())
        .unwrap_or(default)
}
