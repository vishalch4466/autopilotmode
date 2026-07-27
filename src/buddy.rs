//! The floating desktop buddy — a small always-on-top window with the autopilotmode mark as
//! a character, a speech bubble, a text box and a mic button.
//!
//! Deliberately a *separate* front-end over the same [`crate::agent`] loop rather than a
//! reimplementation of it: the loop runs on a worker thread and reports back through a
//! [`Progress`] channel, so what the character says is what the agent is actually doing.
//!
//! Window shape matters here. It is borderless, transparent and always-on-top because the
//! agent drives the *real* cursor and keyboard: a normal window would either be covered by
//! the app being driven, or would itself steal the clicks meant for that app. Dragging is
//! handled manually for the same reason — there is no title bar to grab.

use autopilotmode::agent::{self, Progress};
use autopilotmode::config::Config;
use eframe::egui;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

/// The mark, baked into the binary so the buddy has no runtime asset path to get wrong.
///
/// The transparent horizontal mark rather than the square tile: the tile is opaque ink, and
/// the mood halo behind the mark is the whole point of the character — a solid background
/// would cover it. The tile is used for the window icon instead, which is what it is for.
const MARK_PNG: &[u8] = include_bytes!("../Logos/logo-mark.png");

/// The square dark tile, used for the taskbar / alt-tab icon.
const ICON_PNG: &[u8] = include_bytes!("../Logos/logo-square-mark.png");

/// Fallback aspect ratio of the mark (336x208), used only if the PNG fails to decode.
const MARK_ASPECT: f32 = 336.0 / 208.0;

/// Brand green, sampled from the mark.
const GREEN: egui::Color32 = egui::Color32::from_rgb(0x0F, 0x9D, 0x6E);
/// The one "attention" accent, for the LIVE label. Deliberately *outside* the brand palette:
/// "this is driving your real mouse" must not read as just another green status.
const AMBER: egui::Color32 = egui::Color32::from_rgb(0xF5, 0xA0, 0x3C);
/// Failure, and the mic's recording state.
const RED: egui::Color32 = egui::Color32::from_rgb(0xD9, 0x53, 0x4F);
/// Card background — brand ink.
const INK: egui::Color32 = egui::Color32::from_rgb(0x10, 0x13, 0x18);
/// Card border and dividers.
const LINE: egui::Color32 = egui::Color32::from_rgb(0x26, 0x2E, 0x38);
/// Speech-bubble fill, and the resting fill of secondary controls.
const BUBBLE: egui::Color32 = egui::Color32::from_rgb(0x1A, 0x1F, 0x26);
/// Text-field well — sunk slightly below the card so it reads as an input.
const FIELD: egui::Color32 = egui::Color32::from_rgb(0x15, 0x1A, 0x21);
/// Hover fill for secondary controls.
const HOVER: egui::Color32 = egui::Color32::from_rgb(0x23, 0x2A, 0x34);
/// Primary text.
const TEXT: egui::Color32 = egui::Color32::from_rgb(0xE8, 0xEE, 0xF4);
/// Secondary text.
const MUTED: egui::Color32 = egui::Color32::from_rgb(0x8A, 0x97, 0xA6);

/// Uniform height for every control on the bottom two rows.
const CTRL_H: f32 = 30.0;

/// Blend two colours. Used for the switch track, which animates between states.
fn mix(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let c = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    egui::Color32::from_rgb(
        c(a.r(), b.r()),
        c(a.g(), b.g()),
        c(a.b(), b.b()),
    )
}

/// Install the app's own widget styling.
///
/// egui's stock theme is a mid-grey desktop look: a near-black text well, a blue focus ring
/// and grey buttons. Dropped onto this card they read as OS chrome sitting on top of a
/// designed surface rather than part of it — which is exactly how the window looked before
/// this existed. Everything interactive is restated in the brand palette here, once, so the
/// widget code below can stay about layout.
fn configure_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    style.animation_time = 0.12;
    // Horizontal spacing is generous, vertical is not: the card is only ~210pt tall and
    // every extra point between rows comes straight off the speech bubble.
    style.spacing.item_spacing = egui::vec2(6.0, 3.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    style.spacing.interact_size.y = CTRL_H;

    let v = &mut style.visuals;
    v.panel_fill = INK;
    v.window_fill = INK;
    // The well behind text edits. egui's default here is near-black, which on a dark card
    // reads as a hole punched through it.
    v.extreme_bg_color = FIELD;
    // Focus and text selection in brand green rather than egui's default blue — the blue was
    // the single loudest off-palette colour in the window.
    v.selection = egui::style::Selection {
        bg_fill: GREEN.gamma_multiply(0.35),
        stroke: egui::Stroke::new(1.0_f32,GREEN),
    };

    let w = &mut v.widgets;
    w.noninteractive.bg_fill = INK;
    w.noninteractive.weak_bg_fill = INK;
    w.noninteractive.bg_stroke = egui::Stroke::new(1.0_f32,LINE);
    w.noninteractive.fg_stroke = egui::Stroke::new(1.0_f32,TEXT);
    w.noninteractive.corner_radius = egui::CornerRadius::same(9);

    w.inactive.bg_fill = BUBBLE;
    w.inactive.weak_bg_fill = BUBBLE;
    w.inactive.bg_stroke = egui::Stroke::new(1.0_f32,LINE);
    w.inactive.fg_stroke = egui::Stroke::new(1.0_f32,TEXT);
    w.inactive.corner_radius = egui::CornerRadius::same(9);

    w.hovered.bg_fill = HOVER;
    w.hovered.weak_bg_fill = HOVER;
    w.hovered.bg_stroke = egui::Stroke::new(1.0_f32,GREEN.gamma_multiply(0.55));
    w.hovered.fg_stroke = egui::Stroke::new(1.0_f32,TEXT);
    w.hovered.corner_radius = egui::CornerRadius::same(9);
    // egui's default grows hovered widgets by a pixel. On controls this small that is a
    // visible twitch, not feedback — the colour change carries it.
    w.hovered.expansion = 0.0;

    w.active.bg_fill = GREEN.gamma_multiply(0.30);
    w.active.weak_bg_fill = GREEN.gamma_multiply(0.30);
    w.active.bg_stroke = egui::Stroke::new(1.0_f32,GREEN);
    w.active.fg_stroke = egui::Stroke::new(1.0_f32,TEXT);
    w.active.corner_radius = egui::CornerRadius::same(9);
    w.active.expansion = 0.0;

    w.open = w.hovered;

    ctx.set_style(style);
}

/// How the character is feeling, which drives the face and the bubble colour.
#[derive(Clone, Copy, PartialEq)]
enum Mood {
    Idle,
    Thinking,
    Acting,
    Happy,
    Sad,
}

impl Mood {
    fn tint(self) -> egui::Color32 {
        match self {
            Mood::Idle => GREEN,
            Mood::Thinking => egui::Color32::from_rgb(0x3B, 0x82, 0xF6),
            Mood::Acting => GREEN,
            // Success is a *brighter* green than the brand's, not a near-neighbour of it —
            // the halo is read at a glance from across the room, and two deep greens one
            // shade apart is the same signal twice.
            Mood::Happy => egui::Color32::from_rgb(0x34, 0xD3, 0x99),
            Mood::Sad => RED,
        }
    }
}

/// Shared handle to the worker, so the UI thread can tell whether a run is in flight.
#[derive(Default)]
struct Running(Arc<Mutex<bool>>);

impl Running {
    fn get(&self) -> bool {
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn set(&self, v: bool) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = v;
    }
    fn clone_handle(&self) -> Arc<Mutex<bool>> {
        Arc::clone(&self.0)
    }
}

pub struct Buddy {
    mark: Option<egui::TextureHandle>,
    input: String,
    bubble: String,
    detail: Option<String>,
    mood: Mood,
    step: Option<(u32, u32)>,
    listening: bool,
    running: Running,
    rx: Option<Receiver<Progress>>,
    /// Config is rebuilt per run so `.env` edits take effect without a restart.
    dry_run: bool,
    log: Vec<String>,
}

impl Buddy {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_style(&cc.egui_ctx);
        let mark = load_mark(&cc.egui_ctx);
        Self {
            mark,
            input: String::new(),
            bubble: "Hi! Tell me what to do and I'll drive.".to_string(),
            detail: None,
            mood: Mood::Idle,
            step: None,
            listening: false,
            running: Running::default(),
            rx: None,
            dry_run: true,
            log: Vec::new(),
        }
    }

    /// Kick off a run on a worker thread.
    ///
    /// The agent loop is blocking and takes seconds per step, so it cannot share the UI
    /// thread — egui would stop repainting and the window would appear hung for the whole
    /// run, which is exactly when the user most wants to see it working (and to reach the
    /// stop button).
    fn start(&mut self, goal: String) {
        if self.running.get() {
            return;
        }
        let (tx, rx): (Sender<Progress>, Receiver<Progress>) = channel();
        self.rx = Some(rx);
        self.log.clear();
        self.mood = Mood::Thinking;
        self.bubble = "On it…".to_string();
        self.detail = None;
        self.step = None;
        self.running.set(true);

        let flag = self.running.clone_handle();
        let dry_run = self.dry_run;
        std::thread::spawn(move || {
            let result = Config::from_env().and_then(|mut cfg| {
                cfg.dry_run = dry_run;
                agent::run_with_progress(&goal, &cfg, Some(&tx))
            });
            if let Err(e) = result {
                // A failed run still has to reach the user: without this the bubble would sit
                // on "On it…" forever with no indication anything went wrong.
                let _ = tx.send(Progress::Stopped(format!("{e}")));
            }
            *flag.lock().unwrap_or_else(|e| e.into_inner()) = false;
        });
    }

    /// Drain everything the worker has published since the last frame.
    fn pump(&mut self) {
        let Some(rx) = &self.rx else { return };
        while let Ok(ev) = rx.try_recv() {
            match ev {
                Progress::Step { n, total } => self.step = Some((n, total)),
                Progress::Thinking => {
                    self.mood = Mood::Thinking;
                    self.bubble = "Thinking…".into();
                }
                Progress::Acting { action, reason } => {
                    self.mood = Mood::Acting;
                    self.bubble = reason.clone().unwrap_or_else(|| action.clone());
                    self.detail = Some(action.clone());
                    self.log.push(action);
                    if self.log.len() > 40 {
                        self.log.remove(0);
                    }
                }
                // The webview app shows these; this one has no surface for an image, and
                // it is superseded anyway.
                Progress::Frame { .. } => {}
                Progress::Note(n) => self.detail = Some(n),
                Progress::Escalated { model } => {
                    self.detail = Some(format!("thinking harder ({model})"))
                }
                Progress::Finished { success, message } => {
                    self.mood = if success { Mood::Happy } else { Mood::Sad };
                    self.bubble = if message.is_empty() {
                        "Done.".into()
                    } else {
                        message
                    };
                    self.step = None;
                }
                Progress::Stopped(why) => {
                    self.mood = Mood::Sad;
                    self.bubble = why;
                    self.step = None;
                }
            }
        }
    }
}

impl eframe::App for Buddy {
    /// Transparent so the rounded card can float over the desktop without a grey box.
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.pump();
        // Repaint continuously while working: the worker publishes between frames, and the
        // idle bob / thinking dots are time-driven.
        if self.running.get() {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(80));
        }

        let panel = egui::Frame::new()
            .fill(INK)
            .stroke(egui::Stroke::new(1.0_f32, LINE))
            .corner_radius(16)
            .shadow(egui::epaint::Shadow {
                offset: [0, 6],
                blur: 24,
                spread: 0,
                color: egui::Color32::from_black_alpha(120),
            })
            .inner_margin(14);

        egui::CentralPanel::default().frame(panel).show(ctx, |ui| {
            // Whole-card drag handle, registered FIRST so every widget added afterwards sits
            // on top of it and still receives its own clicks. Without this the only draggable
            // spot is the logo, which is a 48px target on a borderless window — the user has
            // no title bar to fall back on.
            let bg = ui.interact(
                ui.max_rect(),
                ui.id().with("drag-surface"),
                egui::Sense::click_and_drag(),
            );
            if bg.drag_started_by(egui::PointerButton::Primary) {
                ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
            }

            self.header(ui, ctx);
            ui.add_space(10.0);
            self.speech(ui);
            ui.add_space(10.0);
            self.controls(ui, ctx);
        });
    }
}

impl Buddy {
    /// Character, name/status pill, and the close button.
    fn header(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let t = ui.input(|i| i.time) as f32;
        ui.horizontal(|ui| {
            // The mark is a wide switch, not a round face, so the slot is sized from the
            // texture's own aspect rather than assumed square — dropping a 1.6:1 logo into a
            // square box squashes it, and a squashed logo looks like a bug, not a style.
            const MARK_H: f32 = 42.0;
            let aspect = self.mark.as_ref().map_or(MARK_ASPECT, |tex| {
                let size = tex.size_vec2();
                if size.y > 0.0 { size.x / size.y } else { MARK_ASPECT }
            });
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(MARK_H * aspect, MARK_H), egui::Sense::hover());

            // A slow bob while idle, quicker while working: enough motion to read as alive
            // without becoming a distraction on a window that is always on top.
            let speed = if self.running.get() { 3.0 } else { 1.2 };
            let bob = (t * speed).sin() * if self.running.get() { 2.0 } else { 1.0 };
            let face = rect.translate(egui::vec2(0.0, bob));
            let c = face.center();
            let tint = self.mood.tint();

            // Mood glow, plus an orbiting comet while thinking, so state is legible at a
            // glance without reading the bubble. Both trace the mark's pill silhouette: a
            // circle drawn around a wide switch reads as a misplaced shape.
            //
            // Three stacked rings rather than one band: a single flat 20%-alpha ring over
            // a near-black card lands on a muddy dark green barely lighter than the card,
            // which reads as a dirty rim around the logo — a rendering artifact, not a
            // light source. Stacking equal low-alpha layers accumulates toward the mark
            // and gives the falloff a glow actually needs.
            let halo = face.expand(9.0);
            for i in 0..3 {
                let ring = face.expand(9.0 - i as f32 * 3.0);
                ui.painter()
                    .rect_filled(ring, ring.height() * 0.5, tint.gamma_multiply(0.10));
            }
            if self.mood == Mood::Thinking {
                let n = 26;
                let (rx, ry) = (halo.width() * 0.5 + 1.5, halo.height() * 0.5 + 1.5);
                for i in 0..n {
                    let a = t * 2.4 + i as f32 * std::f32::consts::TAU / n as f32;
                    let fade = i as f32 / n as f32;
                    let p = c + egui::vec2(a.cos() * rx, a.sin() * ry);
                    ui.painter().circle_filled(p, 1.7, tint.gamma_multiply(fade));
                }
            }

            if let Some(tex) = &self.mark {
                ui.painter().image(
                    tex.id(),
                    face,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            } else {
                ui.painter().rect_filled(face, face.height() * 0.5, GREEN);
            }

            ui.add_space(4.0);
            ui.vertical(|ui| {
                ui.add_space(3.0);
                // Two-tone wordmark, matching the lockup in `Logos/` — ink "autopilot",
                // green "mode". Item spacing is zeroed so it reads as one word.
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.label(egui::RichText::new("autopilot").strong().size(15.0).color(TEXT));
                    ui.label(egui::RichText::new("mode").strong().size(15.0).color(GREEN));
                });
                ui.add_space(2.0);
                // Run status only. Dry-run/LIVE used to live here too, but it now sits next
                // to the switch that sets it — a mode readout two rows away from its own
                // control is the sort of thing you end up reading twice to be sure.
                let (label, colour) = match (self.running.get(), self.step) {
                    (true, Some((n, total))) => (format!("step {n}/{total}"), tint),
                    (true, None) => ("starting".to_string(), tint),
                    (false, _) => ("ready".to_string(), MUTED),
                };
                pill(ui, &label, colour);
            });

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                // `×` (U+00D7) rather than a dingbat like `✕`: egui's default font subset does
                // not carry the dingbat block, and a missing glyph draws as a blank box.
                let close = egui::Button::new(egui::RichText::new("×").size(17.0).color(MUTED))
                    .frame(false)
                    .min_size(egui::vec2(24.0, 24.0));
                let resp = ui.add(close);
                if resp.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                if resp.on_hover_text("Close").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
        });
    }

    /// Speech bubble, with a small tail pointing back up at the character.
    fn speech(&mut self, ui: &mut egui::Ui) {
        ui.add_space(7.0);

        let frame = egui::Frame::new()
            .fill(BUBBLE)
            .corner_radius(12)
            .inner_margin(egui::Margin::symmetric(12, 10))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                let mut text = self.bubble.clone();
                if self.mood == Mood::Thinking {
                    // Cycling ellipsis, so a slow model still looks alive rather than hung.
                    let dots = (ui.input(|i| i.time) * 2.0) as usize % 4;
                    text = format!("{}{}", text.trim_end_matches(['…', '.']), ".".repeat(dots));
                }
                ui.label(egui::RichText::new(text).size(13.5).color(TEXT));
                if let Some(d) = &self.detail {
                    ui.add_space(3.0);
                    ui.label(
                        egui::RichText::new(d)
                            .size(11.5)
                            .color(MUTED)
                            .family(egui::FontFamily::Monospace),
                    );
                }
            });

        // Painted *after* the bubble, from the bubble's real rect. Drawing it first meant
        // guessing where the bubble would land, and the guess was wrong — the tail rendered
        // as a triangle floating detached above the bubble. Same fill, and it overlaps the
        // top edge by a pixel, so the join is seamless.
        let top = frame.response.rect.top();
        let x = frame.response.rect.left() + 20.0;
        ui.painter().add(egui::Shape::convex_polygon(
            vec![
                egui::pos2(x, top - 7.0),
                egui::pos2(x + 14.0, top + 1.0),
                egui::pos2(x, top + 1.0),
            ],
            BUBBLE,
            egui::Stroke::NONE,
        ));
    }

    /// Text box, mic, send, and the live/dry toggle.
    fn controls(&mut self, ui: &mut egui::Ui, _ctx: &egui::Context) {
        let busy = self.running.get();

        ui.horizontal(|ui| {
            ui.set_min_height(CTRL_H);

            // Mic is a stub: it toggles state and says so, rather than pretending to listen.
            // Wiring real capture means an audio device plus a speech-to-text backend, which
            // is a separate decision from the UI.
            if mic_button(ui, self.listening, !busy)
                .on_hover_text("Voice input (not wired up yet)")
                .clicked()
            {
                self.listening = !self.listening;
                self.bubble = if self.listening {
                    "Listening… (placeholder — no audio backend yet)".into()
                } else {
                    "Stopped listening.".into()
                };
            }

            // Green only when it will actually do something. Setting the fill explicitly for
            // both states beats letting egui fade a green button, which lands on a murky
            // half-green that reads as neither enabled nor disabled.
            let ready = !busy && !self.input.trim().is_empty();
            let hint = if busy { "running…" } else { "Tell me what to do" };

            // Go is placed from the right and the field takes whatever is left. Subtracting a
            // guessed button width from `available_width` instead left the row exactly as
            // wide as the card, so a pixel of rounding overflowed it — and an overflowed row
            // widens the Ui, which dragged the *next* row's right-aligned note off the card
            // edge too. Letting the layout do the arithmetic cannot drift.
            let (resp, send) = ui
                .with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let go = egui::Button::new(
                        egui::RichText::new("Go")
                            .size(13.0)
                            .strong()
                            .color(if ready { INK } else { MUTED }),
                    )
                    .fill(if ready { GREEN } else { BUBBLE })
                    .stroke(egui::Stroke::new(1.0_f32, if ready { GREEN } else { LINE }))
                    .corner_radius(9)
                    .min_size(egui::vec2(48.0, CTRL_H));
                    let send = ui.add_enabled(ready, go);

                    let edit = egui::TextEdit::singleline(&mut self.input)
                        .hint_text(hint)
                        .margin(egui::Margin::symmetric(10, 7))
                        .desired_width(ui.available_width())
                        .vertical_align(egui::Align::Center);
                    let resp = ui.add_enabled(!busy, edit);
                    (resp, send)
                })
                .inner;

            let submitted =
                resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) && !busy;
            // `ready` was computed before the field consumed this frame's keystroke, so the
            // first character typed would otherwise leave Go grey until the next repaint —
            // which, idle, is up to 80ms away and reads as the button being slow.
            if resp.changed() {
                ui.ctx().request_repaint();
            }
            if send.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }

            if (submitted || send.clicked()) && !self.input.trim().is_empty() {
                let goal = self.input.trim().to_string();
                self.input.clear();
                self.start(goal);
            }
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            // The product's own mark is a toggle switch, so the one mode control in the UI is
            // drawn as that same object instead of an OS checkbox.
            if switch(ui, &mut self.dry_run, !busy).changed() {
                self.detail = None;
            }
            ui.add_space(2.0);
            let (label, colour) = if self.dry_run {
                ("Dry run", MUTED)
            } else {
                ("LIVE", AMBER)
            };
            ui.label(egui::RichText::new(label).size(12.0).strong().color(colour));

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Only surfaced when it matters — in dry run nothing is being driven, so a
                // failsafe reminder would just be noise.
                let note = if self.dry_run {
                    "won't touch your mouse"
                } else {
                    "failsafe: cursor to top-left"
                };
                ui.label(egui::RichText::new(note).size(11.0).color(MUTED));
            });
        });
    }
}

/// The mic button, with its glyph drawn rather than typed.
///
/// `🎤` and `🔴` are not in egui's default font subset, so they rendered as a mangled
/// fallback glyph — the same trap the close button's `×` already documents. Painting the
/// icon sidesteps the font question entirely and scales cleanly at any DPI.
fn mic_button(ui: &mut egui::Ui, listening: bool, enabled: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(CTRL_H + 4.0, CTRL_H),
        if enabled {
            egui::Sense::click()
        } else {
            egui::Sense::hover()
        },
    );
    let p = ui.painter();
    let (fill, stroke, ink) = match (enabled, listening, resp.hovered()) {
        (false, _, _) => (BUBBLE, LINE, MUTED.gamma_multiply(0.5)),
        (_, true, _) => (RED.gamma_multiply(0.18), RED, RED),
        (_, false, true) => (HOVER, GREEN.gamma_multiply(0.55), TEXT),
        _ => (BUBBLE, LINE, MUTED),
    };
    p.rect_filled(rect, 9.0, fill);
    p.rect_stroke(
        rect,
        9.0,
        egui::Stroke::new(1.0_f32,stroke),
        egui::StrokeKind::Inside,
    );

    let c = rect.center();
    let h = rect.height();
    let line = egui::Stroke::new(1.6_f32, ink);
    // Capsule body.
    let bw = h * 0.26;
    p.rect_filled(
        egui::Rect::from_center_size(egui::pos2(c.x, c.y - h * 0.12), egui::vec2(bw, h * 0.38)),
        bw * 0.5,
        ink,
    );
    // Cradle: a U opening upward, under the body.
    let r = h * 0.24;
    let cradle = egui::pos2(c.x, c.y - h * 0.06);
    let arc: Vec<egui::Pos2> = (0..=12)
        .map(|i| {
            let a = std::f32::consts::PI * (i as f32 / 12.0);
            cradle + egui::vec2(-a.cos() * r, a.sin() * r)
        })
        .collect();
    p.add(egui::Shape::line(arc, line));
    // Stem and base.
    let base_y = c.y + h * 0.30;
    p.line_segment([egui::pos2(c.x, cradle.y + r), egui::pos2(c.x, base_y)], line);
    p.line_segment(
        [
            egui::pos2(c.x - h * 0.14, base_y),
            egui::pos2(c.x + h * 0.14, base_y),
        ],
        line,
    );

    if enabled && resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    resp
}

/// A pill switch, drawn to echo the mark.
///
/// The product's logo *is* a toggle, so the window's one mode control is the same object
/// rather than an OS checkbox — and a checkbox was the last piece of stock chrome left in
/// the card. The knob animates, which is also what tells you the click registered on a
/// control with no text of its own.
fn switch(ui: &mut egui::Ui, on: &mut bool, enabled: bool) -> egui::Response {
    let size = egui::vec2(36.0, 20.0);
    let (rect, mut resp) = ui.allocate_exact_size(
        size,
        if enabled {
            egui::Sense::click()
        } else {
            egui::Sense::hover()
        },
    );
    if enabled && resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }

    let how_on = ui.ctx().animate_bool_responsive(resp.id, *on);
    let track = mix(LINE, GREEN, how_on);
    let track = if enabled {
        track
    } else {
        track.gamma_multiply(0.45)
    };

    let p = ui.painter();
    let r = rect.height() * 0.5;
    p.rect_filled(rect, r, track);
    if enabled && resp.hovered() {
        p.rect_stroke(
            rect,
            r,
            egui::Stroke::new(1.0_f32,TEXT.gamma_multiply(0.35)),
            egui::StrokeKind::Inside,
        );
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    let pad = 3.0;
    let knob_r = r - pad;
    let x = egui::lerp((rect.left() + r)..=(rect.right() - r), how_on);
    p.circle_filled(
        egui::pos2(x, rect.center().y),
        knob_r,
        if enabled { TEXT } else { MUTED },
    );
    resp
}

/// A small rounded status chip — `step 4/25`, `LIVE`, `dry run`.
fn pill(ui: &mut egui::Ui, text: &str, colour: egui::Color32) {
    egui::Frame::new()
        .fill(colour.gamma_multiply(0.16))
        .corner_radius(7)
        .inner_margin(egui::Margin::symmetric(7, 2))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).size(11.0).color(colour));
        });
}

/// Decode the embedded mark into a texture, resampled for the size it is drawn at.
///
/// Two things make a 1344px source look broken at ~85px, and the naive
/// decode-and-upload hits both:
///
/// - **egui generates no mipmaps.** Handing the GPU the full-size texture makes
///   every frame a ~16x bilinear minification, which samples 4 texels out of a
///   ~250-texel footprint — the edges alias and crawl. Resampling once on the CPU
///   with a real filter, down to a size the GPU only has to shrink ~2x, fixes it.
/// - **Straight-alpha filtering pulls black into the edges.** The source is
///   transparent *black* outside the mark, so averaging unpremultiplied pixels
///   blends that black into every edge — a dark fringe that reads as dirt.
///   Premultiplying first makes transparent pixels contribute nothing but alpha.
fn load_mark(ctx: &egui::Context) -> Option<egui::TextureHandle> {
    /// Wide enough to stay sharp well past 2x DPI at the ~68pt draw width.
    const TARGET_W: u32 = 320;

    let mut src = image::load_from_memory(MARK_PNG).ok()?.to_rgba8();
    for px in src.pixels_mut() {
        let a = px.0[3] as u32;
        for c in 0..3 {
            px.0[c] = ((px.0[c] as u32 * a + 127) / 255) as u8;
        }
    }

    let height = ((TARGET_W as f32 / src.width() as f32) * src.height() as f32).round() as u32;
    let mut small = image::imageops::resize(
        &src,
        TARGET_W,
        height.max(1),
        image::imageops::FilterType::Lanczos3,
    );
    // Lanczos overshoots on high-contrast edges, which can leave a channel above
    // its own alpha — an invalid premultiplied pixel that renders as a bright
    // speckle. Clamp back into range.
    for px in small.pixels_mut() {
        let a = px.0[3];
        for c in 0..3 {
            px.0[c] = px.0[c].min(a);
        }
    }

    let (w, h) = small.dimensions();
    let colour =
        egui::ColorImage::from_rgba_premultiplied([w as usize, h as usize], small.as_raw());
    Some(ctx.load_texture("mark", colour, egui::TextureOptions::LINEAR))
}

/// Decode the square tile into a window icon.
///
/// Downscaled to 256px: the source is 1024x1024 so it can be reused at any size, but the
/// icon is drawn at ~32px, and egui asks for a square whose width is a multiple of 4.
fn load_icon() -> Option<egui::IconData> {
    let img = image::load_from_memory(ICON_PNG)
        .ok()?
        .resize_exact(256, 256, image::imageops::FilterType::Lanczos3)
        .to_rgba8();
    let (width, height) = img.dimensions();
    Some(egui::IconData {
        rgba: img.into_raw(),
        width,
        height,
    })
}

/// Show the buddy. Blocks until the window is closed.
pub fn run() -> anyhow::Result<()> {
    // Height fits the idle card with just enough slack for the monospace detail line that
    // appears under the bubble during a run — sizing it to the idle state exactly would make
    // the window grow a scrollbar the moment it starts working.
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([400.0, 228.0])
        .with_min_inner_size([360.0, 216.0])
        .with_decorations(false)
        .with_transparent(true)
        .with_always_on_top()
        .with_resizable(true);
    // The window is borderless, so there is no title bar naming it — the taskbar and
    // alt-tab entry are the only places the user sees what it is. Without this they see
    // eframe's default `e`.
    if let Some(icon) = load_icon() {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "autopilotmode",
        options,
        Box::new(|cc| Ok(Box::new(Buddy::new(cc)))),
    )
    .map_err(|e| anyhow::anyhow!("could not open the buddy window: {e}"))
}
