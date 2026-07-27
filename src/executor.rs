//! Turn an [`ActionInput`] into real OS input via enigo.
//!
//! Coordinates arrive in *screenshot pixel space* (whatever size we downscaled to and
//! showed the model). We convert to a fraction of that image, then multiply by enigo's
//! own display size — so it's correct regardless of HiDPI scaling or capture resolution.

use crate::action::ActionInput;
use crate::human;
use anyhow::{anyhow, Result};
use enigo::{
    Axis, Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings,
};
use std::sync::{Mutex, MutexGuard};

/// Result of executing one action.
pub enum Outcome {
    /// Keep looping.
    Continue,
    /// The agent declared it finished.
    Done { success: bool, message: String },
}

/// Keys latched down by `keydown`, paired with the name they were requested under, in
/// press order.
///
/// Global rather than a field on [`Executor`] because a latched key is physically down at
/// the OS level: the Ctrl-C handler has to be able to release it without access to the
/// running executor, or the process dies with the throttle stuck on.
static LATCHED: Mutex<Vec<(String, Key)>> = Mutex::new(Vec::new());

/// A poisoned lock still holds a valid key list, and dropping keys on the floor is the one
/// outcome we cannot accept — recover rather than panic.
fn latched() -> MutexGuard<'static, Vec<(String, Key)>> {
    LATCHED.lock().unwrap_or_else(|e| e.into_inner())
}

fn is_latched(k: Key) -> bool {
    latched().iter().any(|(_, held)| *held == k)
}

/// Human-readable list of what is currently held down, e.g. `w+shift`.
pub fn latched_label() -> Option<String> {
    let held = latched();
    if held.is_empty() {
        return None;
    }
    Some(held.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join("+"))
}

/// Release every latched key without an [`Executor`] — for the Ctrl-C handler, which runs
/// on its own thread after the main loop has lost its chance to clean up.
pub fn release_latched_keys() {
    let held: Vec<(String, Key)> = latched().drain(..).collect();
    if held.is_empty() {
        return;
    }
    if let Ok(mut enigo) = Enigo::new(&Settings::default()) {
        for (_, k) in held.iter().rev() {
            let _ = enigo.key(*k, Direction::Release);
        }
    }
}

/// Split a `+`-separated combo into (name, key) pairs. Unlike [`Executor::key_combo`]
/// there is no modifier/main split — every part is an equal participant.
fn parse_combo(combo: &str) -> Result<Vec<(String, Key)>> {
    let parts: Vec<&str> = combo.split('+').map(str::trim).filter(|s| !s.is_empty()).collect();
    if parts.is_empty() {
        return Err(anyhow!("empty key combo"));
    }
    parts
        .iter()
        .map(|p| Ok((p.to_ascii_lowercase(), parse_key(p)?)))
        .collect()
}

pub struct Executor {
    enigo: Enigo,
    screen_w: i32,
    screen_h: i32,
    dry_run: bool,
    /// Glide the pointer along a curved path instead of jumping straight to coordinates.
    human_mouse: bool,
    /// Multiplier on glide duration (higher = faster).
    mouse_speed: f64,
    rng: human::Rng,
}

impl Executor {
    pub fn new(dry_run: bool, human_mouse: bool, mouse_speed: f64) -> Result<Self> {
        let enigo = Enigo::new(&Settings::default()).map_err(|e| {
            anyhow!("could not initialise input backend ({e}). A graphical session (X11/Wayland, macOS, or Windows) is required.")
        })?;
        let (screen_w, screen_h) = enigo
            .main_display()
            .map_err(|e| anyhow!("query display size: {e}"))?;
        Ok(Self {
            enigo,
            screen_w,
            screen_h,
            dry_run,
            human_mouse,
            mouse_speed,
            rng: human::Rng::from_entropy(),
        })
    }

    /// Map a point from screenshot space (`sw`×`sh`) to absolute screen pixels.
    fn to_screen(&self, x: f64, y: f64, sw: u32, sh: u32) -> (i32, i32) {
        let fx = (x / sw.max(1) as f64).clamp(0.0, 1.0);
        let fy = (y / sh.max(1) as f64).clamp(0.0, 1.0);
        (
            (fx * self.screen_w as f64).round() as i32,
            (fy * self.screen_h as f64).round() as i32,
        )
    }

    /// True if the cursor is parked in the very top-left corner — the abort failsafe.
    pub fn panic_corner(&mut self) -> bool {
        matches!(self.enigo.location(), Ok((x, y)) if x <= 1 && y <= 1)
    }

    /// Move the real cursor around a visible square (no clicks / no typing) so you can
    /// confirm input injection works on this machine, without any API call.
    pub fn self_test(&mut self) -> Result<()> {
        let (w, h) = (self.screen_w, self.screen_h);
        let points = [
            (w / 2, h / 2),
            (w / 5, h / 5),
            (w * 4 / 5, h / 5),
            (w * 4 / 5, h * 4 / 5),
            (w / 5, h * 4 / 5),
            (w / 2, h / 2),
        ];
        for (x, y) in points {
            self.mouse_move(x, y)?;
            let at = self.enigo.location().ok();
            println!("  → asked ({x}, {y}); cursor now at {at:?}");
            std::thread::sleep(std::time::Duration::from_millis(450));
        }
        Ok(())
    }

    pub fn execute(&mut self, a: &ActionInput, sw: u32, sh: u32) -> Result<Outcome> {
        let action = normalize_action(&a.action);
        match action.as_str() {
            "done" => {
                return Ok(Outcome::Done {
                    success: a.success.unwrap_or(true),
                    message: a.reason.clone().unwrap_or_default(),
                });
            }
            "wait" => {
                let ms = a.ms.unwrap_or(500).min(30_000);
                if !self.dry_run {
                    std::thread::sleep(std::time::Duration::from_millis(ms));
                }
            }
            "screenshot" => { /* observe only — next loop grabs a fresh frame */ }

            "move" => {
                let (x, y) = self.point(a, sw, sh)?;
                self.mouse_move(x, y)?;
            }

            "click" | "double_click" | "right_click" | "middle_click" => {
                // Move first if a target was given.
                if a.x.is_some() && a.y.is_some() {
                    let (x, y) = self.point(a, sw, sh)?;
                    self.mouse_move(x, y)?;
                }
                let button = match action.as_str() {
                    "right_click" => Button::Right,
                    "middle_click" => Button::Middle,
                    _ => parse_button(a.button.as_deref()),
                };
                let count = if action == "double_click" {
                    2
                } else {
                    a.clicks.unwrap_or(1).clamp(1, 3)
                };
                for _ in 0..count {
                    self.click(button)?;
                }
            }

            "drag" => {
                let (x1, y1) = self.point(a, sw, sh)?;
                let x2 = a.x2.ok_or_else(|| anyhow!("drag needs x2"))?;
                let y2 = a.y2.ok_or_else(|| anyhow!("drag needs y2"))?;
                let (x2, y2) = self.to_screen(x2, y2, sw, sh);
                let button = parse_button(a.button.as_deref());
                self.mouse_move(x1, y1)?;
                self.button(button, Direction::Press)?;
                // Apps commonly need the press to land before motion counts as a drag.
                self.pause(70, 130);
                self.mouse_move(x2, y2)?;
                self.pause(60, 110);
                self.button(button, Direction::Release)?;
            }

            "scroll" => {
                if a.x.is_some() && a.y.is_some() {
                    let (x, y) = self.point(a, sw, sh)?;
                    self.mouse_move(x, y)?;
                }
                let dy = a.scroll_y.unwrap_or(0);
                let dx = a.scroll_x.unwrap_or(0);
                if dy != 0 {
                    self.scroll(dy, Axis::Vertical)?;
                }
                if dx != 0 {
                    self.scroll(dx, Axis::Horizontal)?;
                }
            }

            "type" => {
                let text = a.text.as_deref().unwrap_or("");
                self.log(&format!("type {text:?}"));
                if !self.dry_run {
                    self.enigo
                        .text(text)
                        .map_err(|e| anyhow!("type text: {e}"))?;
                }
            }

            "key" => {
                let combo = a.keys.as_deref().unwrap_or("");
                self.key_combo(combo)?;
            }

            // Games and other real-time UIs need a key held down, not tapped: `w` for
            // 2000ms to drive forward, `shift` to sprint. `key` cannot express this
            // because it presses and releases in the same instant.
            "hold" => {
                let combo = a.keys.as_deref().unwrap_or("");
                let ms = a.ms.unwrap_or(800).clamp(1, 15_000);
                self.key_hold(combo, ms)?;
            }

            // `hold` blocks the loop for its whole duration, so the key is always up again
            // before the next screenshot and model call — several seconds during which a
            // car coasts. Latching keeps a key down *across* steps instead, which is the
            // difference between stop-start lurching and continuous driving.
            "keydown" => {
                let combo = a.keys.as_deref().unwrap_or("");
                self.key_latch(combo)?;
            }
            "keyup" => {
                self.key_unlatch(a.keys.as_deref().unwrap_or(""))?;
            }

            other => return Err(anyhow!("unknown action {other:?}")),
        }
        Ok(Outcome::Continue)
    }

    // ---- primitives (each honours dry_run) ----

    fn point(&self, a: &ActionInput, sw: u32, sh: u32) -> Result<(i32, i32)> {
        let x = a.x.ok_or_else(|| anyhow!("{} needs x", a.action))?;
        let y = a.y.ok_or_else(|| anyhow!("{} needs y", a.action))?;
        Ok(self.to_screen(x, y, sw, sh))
    }

    fn mouse_move(&mut self, x: i32, y: i32) -> Result<()> {
        self.log(&format!("move -> ({x}, {y})"));
        if self.dry_run {
            return Ok(());
        }
        // Without a known starting point there is no path to trace, so fall back to a jump.
        let from = if self.human_mouse { self.enigo.location().ok() } else { None };
        let Some(from) = from else {
            return self
                .enigo
                .move_mouse(x, y, Coordinate::Abs)
                .map_err(|e| anyhow!("move mouse: {e}"));
        };

        for step in human::path(from, (x, y), self.mouse_speed, &mut self.rng) {
            self.enigo
                .move_mouse(step.x, step.y, Coordinate::Abs)
                .map_err(|e| anyhow!("move mouse: {e}"))?;
            if !step.delay.is_zero() {
                std::thread::sleep(step.delay);
            }
        }
        Ok(())
    }

    fn click(&mut self, button: Button) -> Result<()> {
        self.log(&format!("click {button:?}"));
        if !self.dry_run {
            self.enigo
                .button(button, Direction::Click)
                .map_err(|e| anyhow!("click: {e}"))?;
        }
        Ok(())
    }

    fn button(&mut self, button: Button, dir: Direction) -> Result<()> {
        if !self.dry_run {
            self.enigo
                .button(button, dir)
                .map_err(|e| anyhow!("button {dir:?}: {e}"))?;
        }
        Ok(())
    }

    fn scroll(&mut self, length: i32, axis: Axis) -> Result<()> {
        self.log(&format!("scroll {length} {axis:?}"));
        if !self.dry_run {
            self.enigo
                .scroll(length, axis)
                .map_err(|e| anyhow!("scroll: {e}"))?;
        }
        Ok(())
    }

    /// Hold every key in `combo` down together for `ms`, then release in reverse order.
    ///
    /// Unlike [`Self::key_combo`] there is no modifier/main split: `w+shift` means both
    /// are held simultaneously (run forward), which is what continuous controls need.
    /// Keys are always released, even if a press part-way through fails, so a panic here
    /// can never leave the keyboard stuck down.
    fn key_hold(&mut self, combo: &str, ms: u64) -> Result<()> {
        self.log(&format!("hold {combo:?} for {ms}ms"));
        let keys = parse_combo(combo)?;
        if self.dry_run {
            return Ok(());
        }

        let mut held: Vec<Key> = Vec::with_capacity(keys.len());
        let mut res = Ok(());
        for (_, k) in keys {
            // Already latched by `keydown`: it is down, and releasing it below would
            // silently break the latch while the registry still believed it was held.
            if is_latched(k) {
                continue;
            }
            match self.enigo.key(k, Direction::Press) {
                Ok(()) => held.push(k),
                Err(e) => {
                    res = Err(anyhow!("key down: {e}"));
                    break;
                }
            }
        }
        if res.is_ok() {
            std::thread::sleep(std::time::Duration::from_millis(ms));
        }
        for k in held.iter().rev() {
            let _ = self.enigo.key(*k, Direction::Release);
        }
        res
    }

    /// Press keys and leave them down until `keyup`, the end of the run, or Ctrl-C.
    ///
    /// Re-latching an already-held key is a no-op rather than a second press: repeating
    /// the press would emit key-repeat events some games read as a fresh input.
    fn key_latch(&mut self, combo: &str) -> Result<()> {
        let keys = parse_combo(combo)?;
        self.log(&format!("keydown {combo:?} (stays down)"));
        if self.dry_run {
            return Ok(());
        }
        for (name, k) in keys {
            if is_latched(k) {
                continue;
            }
            // Register *before* pressing: if Ctrl-C lands between the two, the handler
            // must already know the key needs releasing. A spurious release is harmless;
            // a missed one leaves the key stuck down.
            latched().push((name, k));
            if let Err(e) = self.enigo.key(k, Direction::Press) {
                latched().retain(|(_, held)| *held != k);
                return Err(anyhow!("key down: {e}"));
            }
        }
        Ok(())
    }

    /// Release latched keys — the ones named, or all of them when `combo` is empty or
    /// "all". Naming a key that is not held is a no-op, not an error.
    fn key_unlatch(&mut self, combo: &str) -> Result<()> {
        let trimmed = combo.trim();
        let all = trimmed.is_empty() || trimmed.eq_ignore_ascii_case("all");
        self.log(&format!("keyup {}", if all { "all" } else { trimmed }));
        if self.dry_run {
            return Ok(());
        }
        let wanted = if all { None } else { Some(parse_combo(trimmed)?) };
        let release: Vec<(String, Key)> = {
            let mut held = latched();
            let (release, keep): (Vec<_>, Vec<_>) = held.drain(..).partition(|(_, k)| {
                wanted.as_ref().is_none_or(|w| w.iter().any(|(_, want)| want == k))
            });
            *held = keep;
            release
        };
        for (_, k) in release.iter().rev() {
            let _ = self.enigo.key(*k, Direction::Release);
        }
        Ok(())
    }

    /// Release everything latched. Called on every exit path from the loop via `Drop`.
    fn release_all(&mut self) {
        let held: Vec<(String, Key)> = latched().drain(..).collect();
        for (_, k) in held.iter().rev() {
            let _ = self.enigo.key(*k, Direction::Release);
        }
    }

    fn key_combo(&mut self, combo: &str) -> Result<()> {
        self.log(&format!("key {combo:?}"));
        let parts: Vec<&str> = combo.split('+').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        if parts.is_empty() {
            return Err(anyhow!("empty key combo"));
        }
        let (main, mods) = parts.split_last().unwrap();
        let main_key = parse_key(main)?;
        let mod_keys: Vec<Key> = mods.iter().map(|m| parse_modifier(m)).collect::<Result<_>>()?;
        if self.dry_run {
            return Ok(());
        }
        for k in &mod_keys {
            self.enigo.key(*k, Direction::Press).map_err(|e| anyhow!("modifier down: {e}"))?;
        }
        let res = self.enigo.key(main_key, Direction::Click).map_err(|e| anyhow!("key: {e}"));
        // Always release modifiers, even if the main key failed.
        for k in mod_keys.iter().rev() {
            let _ = self.enigo.key(*k, Direction::Release);
        }
        res
    }

    /// Sleep for a randomised interval (ms), skipped entirely in dry-run.
    fn pause(&mut self, lo: u64, hi: u64) {
        if self.dry_run || !self.human_mouse {
            return;
        }
        let ms = human::jitter_ms(&mut self.rng, lo, hi);
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }

    fn log(&self, what: &str) {
        if self.dry_run {
            println!("      [dry-run] {what}");
        }
    }
}

/// Backstop for every ordinary exit path — goal reached, failsafe corner, step cap, or an
/// error bubbling out of the loop. Ctrl-C bypasses `Drop` entirely, which is why
/// [`release_latched_keys`] exists separately.
impl Drop for Executor {
    fn drop(&mut self) {
        self.release_all();
    }
}

/// Map the model's action name onto one of our canonical actions.
///
/// The schema pins an enum, but models still paraphrase (`left_click`, `mouse_move`,
/// `type_text`). Rather than abort the run over a synonym, fold punctuation/case away
/// and accept the common variants. Unknown names pass through so `execute` still errors.
fn normalize_action(name: &str) -> String {
    let flat: String = name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    match flat.as_str() {
        "move" | "mousemove" | "movemouse" | "moveto" | "cursor" | "hover" => "move",
        "click" | "leftclick" | "mouseclick" | "singleclick" | "tap" => "click",
        "doubleclick" | "dblclick" | "doubleleftclick" => "double_click",
        "rightclick" | "contextclick" | "secondaryclick" => "right_click",
        "middleclick" | "wheelclick" => "middle_click",
        "drag" | "dragto" | "clickdrag" | "draganddrop" => "drag",
        "scroll" | "mousescroll" | "wheel" => "scroll",
        "type" | "typetext" | "text" | "write" | "input" => "type",
        "key" | "keys" | "keypress" | "presskey" | "hotkey" | "keyboard" | "shortcut" => "key",
        "hold" | "keyhold" | "holdkey" | "presshold" | "holdkeys" => "hold",
        "keydown" | "latch" | "latchkey" | "holddown" | "pressandhold" => "keydown",
        "keyup" | "keyrelease" | "releasekey" | "release" | "unlatch" => "keyup",
        "wait" | "sleep" | "pause" => "wait",
        "screenshot" | "screencapture" | "observe" | "look" | "capture" => "screenshot",
        "done" | "finish" | "finished" | "complete" | "stop" => "done",
        _ => return name.to_string(),
    }
    .to_string()
}

fn parse_button(name: Option<&str>) -> Button {
    match name.map(|s| s.to_ascii_lowercase()).as_deref() {
        Some("right") => Button::Right,
        Some("middle") => Button::Middle,
        _ => Button::Left,
    }
}

fn parse_modifier(name: &str) -> Result<Key> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "ctrl" | "control" => Key::Control,
        "shift" => Key::Shift,
        "alt" | "option" | "opt" => Key::Alt,
        "cmd" | "command" | "super" | "win" | "windows" | "meta" => Key::Meta,
        other => return Err(anyhow!("unknown modifier {other:?}")),
    })
}

fn parse_key(name: &str) -> Result<Key> {
    let n = name.to_ascii_lowercase();
    Ok(match n.as_str() {
        "enter" | "return" => Key::Return,
        "tab" => Key::Tab,
        "esc" | "escape" => Key::Escape,
        "space" | "spacebar" => Key::Space,
        "backspace" | "bksp" => Key::Backspace,
        "delete" | "del" => Key::Delete,
        "up" | "uparrow" => Key::UpArrow,
        "down" | "downarrow" => Key::DownArrow,
        "left" | "leftarrow" => Key::LeftArrow,
        "right" | "rightarrow" => Key::RightArrow,
        "home" => Key::Home,
        "end" => Key::End,
        "pageup" | "pgup" => Key::PageUp,
        "pagedown" | "pgdn" => Key::PageDown,
        "f1" => Key::F1,
        "f2" => Key::F2,
        "f3" => Key::F3,
        "f4" => Key::F4,
        "f5" => Key::F5,
        "f6" => Key::F6,
        "f7" => Key::F7,
        "f8" => Key::F8,
        "f9" => Key::F9,
        "f10" => Key::F10,
        "f11" => Key::F11,
        "f12" => Key::F12,
        // Single character -> that literal key.
        _ if n.chars().count() == 1 => Key::Unicode(n.chars().next().unwrap()),
        // A modifier pressed on its own, e.g. `win` to open the Start menu.
        _ => parse_modifier(&n).map_err(|_| anyhow!("unknown key {n:?}"))?,
    })
}
