//! autopilotmode — drive the real mouse/keyboard with an AI vision loop.
//!
//! Usage:  autopilotmode "open a browser and search for the weather"
//! See README.md for setup and safety notes.

// The loop itself lives in the library (src/lib.rs); this binary is the console front-end.
// `buddy` is the egui window and stays bin-local so the library carries no UI toolkit.
mod buddy;

use autopilotmode::{agent, config, executor};

use anyhow::Result;
use clap::Parser;
use std::io::Write;

#[derive(Parser)]
#[command(
    name = "autopilotmode",
    about = "AI agent that controls your mouse/keyboard from screenshots via Claude.",
    long_about = None
)]
struct Cli {
    /// The goal to accomplish, in natural language (quote it, or pass as trailing words).
    #[arg(trailing_var_arg = true, value_name = "GOAL")]
    goal: Vec<String>,

    /// Override the model id (default: claude-opus-5, or $AUTOPILOT_MODEL).
    #[arg(long)]
    model: Option<String>,

    /// Which API to use: `anthropic` or `openrouter`. Defaults to whichever key is present
    /// (see $AUTOPILOT_PROVIDER), preferring Anthropic when both are.
    #[arg(long, value_name = "NAME")]
    provider: Option<String>,

    /// Cap on observe→act iterations (default: 25, or $AUTOPILOT_MAX_STEPS).
    #[arg(long)]
    max_steps: Option<u32>,

    /// Plan and print actions without actually moving the mouse/keyboard.
    #[arg(long)]
    dry_run: bool,

    /// Open the floating desktop buddy instead of running a goal from the command line.
    #[arg(long)]
    buddy: bool,

    /// Seconds to wait before the first screenshot, so you can focus the target window.
    /// Defaults to 5 with --game, 0 otherwise.
    #[arg(long, value_name = "SECS")]
    wait: Option<u64>,

    /// Skip the confirmation prompt.
    #[arg(long)]
    yes: bool,

    /// Move the real cursor in a square (no clicks, no API call) to prove input works.
    #[arg(long)]
    selftest: bool,

    /// Jump the cursor straight to coordinates instead of gliding along a human-like path.
    #[arg(long)]
    teleport: bool,

    /// Pointer travel speed multiplier (default: 1.0, or $AUTOPILOT_MOUSE_SPEED).
    #[arg(long)]
    mouse_speed: Option<f64>,

    /// Shorten the observe→act loop: smaller JPEG screenshots, one kept frame, terse
    /// replies, minimal settle delay. Trades some coordinate precision for speed.
    #[arg(long)]
    fast: bool,

    /// Real-time control preset (games): --fast, plus almost no settle delay and no
    /// pointer gliding. Use with `keydown` for smooth continuous movement.
    #[arg(long)]
    game: bool,
}

fn main() -> Result<()> {
    // Load autopilotmode/.env (and any parent .env). Non-fatal if absent.
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    // --selftest: prove the real cursor moves, with no API key and no AI loop.
    if cli.selftest {
        println!("autopilotmode self-test — moving the REAL cursor in a square.");
        println!("No clicks, no typing, no API call. Watch your pointer.");
        for n in (1..=3).rev() {
            println!("  starting in {n}...  (Ctrl+C to cancel)");
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        // Uses the CLI flags directly — --selftest runs before any config/API key load,
        // so you can compare --teleport against the default glide with no key present.
        let mut exec = executor::Executor::new(
            false,
            !cli.teleport,
            cli.mouse_speed.unwrap_or(1.0),
        )?;
        exec.self_test()?;
        println!("Done — if the pointer moved, input injection works on this machine.");
        return Ok(());
    }

    // The buddy supplies its own goal from its text box, so it runs before the goal check.
    if cli.buddy {
        return buddy::run();
    }

    let goal = cli.goal.join(" ").trim().to_string();
    if goal.is_empty() {
        eprintln!("error: a goal is required, e.g.  autopilotmode \"click the Start button\"  (or use --selftest)");
        std::process::exit(2);
    }

    // Resolved before the config is built, since the provider decides which credential is
    // required and which model default applies. An unknown name is rejected rather than
    // quietly falling back — silently running against the wrong API would be billed to the
    // wrong account and reported as "my flag did nothing".
    let mut cfg = match &cli.provider {
        Some(name) => {
            let p = config::Provider::parse(name).ok_or_else(|| {
                anyhow::anyhow!("unknown --provider {name:?} — expected `anthropic` or `openrouter`")
            })?;
            config::Config::for_provider(p)?
        }
        None => config::Config::from_env()?,
    };
    // Before the CLI overrides, so an explicit --max-steps etc. still wins.
    if cli.game {
        cfg.apply_game_preset();
    } else if cli.fast {
        cfg.apply_fast_preset();
    }
    if let Some(m) = cli.model {
        cfg.model = m;
    }
    if let Some(s) = cli.max_steps {
        cfg.max_steps = s;
    }
    cfg.dry_run = cli.dry_run;
    if cli.teleport {
        cfg.human_mouse = false;
    }
    if let Some(s) = cli.mouse_speed {
        cfg.mouse_speed = s.clamp(0.25, 8.0);
    }

    println!("autopilotmode");
    println!("  provider   : {}", cfg.provider.label());
    println!(
        "  model      : {}{}{}",
        cfg.model,
        if cfg.effort.is_empty() { String::new() } else { format!(" (effort {})", cfg.effort) },
        if cfg.speed_fast { "  [fast mode — premium rate]" } else { "" }
    );
    if let Some(heavy) = &cfg.model_heavy {
        println!("  escalate to: {heavy} (when the loop stops making progress)");
    }
    println!("  goal       : {goal}");
    println!("  max steps  : {}", cfg.max_steps);
    println!(
        "  vision     : {}px {:?}, keep {}, settle {}ms{}",
        cfg.max_image_dim,
        cfg.image_format,
        cfg.keep_screenshots,
        cfg.action_delay_ms,
        if cli.game { "  [game]" } else if cli.fast { "  [fast]" } else { "" }
    );
    println!("  mode       : {}", if cfg.dry_run { "dry-run (no input sent)" } else { "LIVE — controls real input" });
    println!("  pointer    : {}", if cfg.human_mouse {
        format!("human glide (speed {:.2})", cfg.mouse_speed)
    } else {
        "teleport".to_string()
    });
    println!("  failsafe   : throw the mouse into the top-left corner to abort");

    if !cfg.dry_run && !cli.yes && !config::env_flag_on("AUTOPILOT_YES") {
        print!("\nThis will control your real mouse and keyboard. Type 'go' to continue: ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).ok();
        if line.trim() != "go" {
            println!("Aborted.");
            return Ok(());
        }
    }

    // A key latched by `keydown` is physically down at the OS level. Ctrl-C kills the
    // process without unwinding, so `Drop` never runs — without this the throttle would
    // stay pressed after the run "stopped".
    if !cfg.dry_run {
        let _ = ctrlc::set_handler(|| {
            executor::release_latched_keys();
            eprintln!("\n⛔ Interrupted — released any held keys.");
            std::process::exit(130);
        });
    }

    // Whatever is focused when the first screenshot is taken is what the model will try to
    // operate — and starting from a terminal means the terminal is focused. For a game that
    // is fatal twice over: the model sees a desktop instead of the road, and a latched `w`
    // types into the terminal rather than driving. Hold the loop so the window can be
    // brought up first. `--game` assumes it, since a game is never the launching window.
    let wait = cli.wait.unwrap_or(if cli.game { 5 } else { 0 });
    if wait > 0 {
        println!("\n▶ Focus the window you want driven — starting in:");
        for n in (1..=wait).rev() {
            println!("   {n}...");
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        println!("   go\n");
    }

    agent::run(&goal, &cfg)
}
