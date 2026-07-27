//! Human-like pointer motion — curved, eased paths instead of coordinate teleports.
//!
//! This is not (only) cosmetic. `move_mouse` to an absolute coordinate emits a single
//! position update, so nothing downstream ever sees the pointer *travel*:
//!
//! - Hover states, tooltips and menus that open on `mouseover` never fire.
//! - HTML5 drag-and-drop and canvas freehand tools track `mousemove` deltas; with a
//!   teleport they receive a press and a release at two unrelated points.
//! - Instant jumps are the classic signature of synthetic input.
//!
//! [`path`] samples a curved trajectory at roughly mouse-polling rate, so the OS emits a
//! realistic stream of motion events on the way to the target.

use std::time::Duration;

/// xorshift64* — we need jitter, not cryptography, and this keeps the dependency list
/// unchanged. Seeded from the clock so runs don't trace identical arcs.
pub struct Rng(u64);

impl Rng {
    pub fn from_entropy() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x2545_F491_4F6C_DD1D);
        // A zero seed is a fixed point of xorshift; anything else is fine.
        Rng(nanos | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[-1, 1)`.
    fn signed(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }

    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.unit() * (hi - lo)
    }

    /// True with probability `p`.
    fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }
}

/// A randomised delay in `[lo, hi)` milliseconds, for pauses between discrete inputs.
pub fn jitter_ms(rng: &mut Rng, lo: u64, hi: u64) -> u64 {
    rng.range(lo as f64, hi.max(lo + 1) as f64) as u64
}

/// One sampled point on the way to the target, and how long to rest before the next.
pub struct Step {
    pub x: i32,
    pub y: i32,
    pub delay: Duration,
}

/// Interval between samples. ~125 Hz, matching a common mouse polling rate.
const SAMPLE_MS: f64 = 8.0;
/// Upper bound on a single glide, so a cross-screen move never feels stuck.
const MAX_DURATION_MS: f64 = 850.0;

/// Sample a curved, eased path from `from` to `to`.
///
/// `speed` scales the duration (2.0 = twice as fast). The final step always lands
/// exactly on `to` — jitter is applied to intermediate points only, because the click
/// that usually follows has to hit what the model aimed at.
pub fn path(from: (i32, i32), to: (i32, i32), speed: f64, rng: &mut Rng) -> Vec<Step> {
    let dist = (((to.0 - from.0) as f64).powi(2) + ((to.1 - from.1) as f64).powi(2)).sqrt();

    // Sub-pixel move: nothing to animate.
    if dist < 2.0 {
        return vec![Step { x: to.0, y: to.1, delay: Duration::ZERO }];
    }

    // Humans reach for distant targets sublinearly (Fitts' law): doubling the distance
    // costs far less than double the time.
    let speed = speed.clamp(0.25, 8.0);
    let base_ms = (90.0 + 200.0 * (1.0 + dist / 100.0).ln()) * rng.range(0.85, 1.15) / speed;
    let duration_ms = base_ms.min(MAX_DURATION_MS);

    // A long reach often lands slightly past the target and is corrected — one of the
    // most recognisable features of real pointer traces.
    let overshoots = dist > 250.0 && rng.chance(0.35);
    let aim = if overshoots {
        let k = rng.range(0.02, 0.06);
        (
            to.0 + ((to.0 - from.0) as f64 * k) as i32,
            to.1 + ((to.1 - from.1) as f64 * k) as i32,
        )
    } else {
        to
    };

    let mut steps = arc(from, aim, duration_ms, dist, rng);
    if overshoots {
        // The corrective move back: short, slower, and it does land on the target.
        let correct_ms = rng.range(90.0, 170.0) / speed;
        steps.extend(arc(aim, to, correct_ms, dist * 0.05, rng));
    }

    // Settle before whatever comes next (usually a click) — nobody clicks the instant
    // the pointer stops.
    if let Some(last) = steps.last_mut() {
        last.x = to.0;
        last.y = to.1;
        last.delay = Duration::from_millis(rng.range(18.0, 55.0) as u64);
    }
    steps
}

/// One eased, slightly bowed leg of a movement.
fn arc(from: (i32, i32), to: (i32, i32), duration_ms: f64, dist: f64, rng: &mut Rng) -> Vec<Step> {
    let n = ((duration_ms / SAMPLE_MS).round() as usize).clamp(2, 240);

    // Quadratic Bézier control point: the midpoint pushed perpendicular to the straight
    // line, so the path bows the way a wrist-and-arm movement does rather than tracking
    // a ruler. Sign is random, magnitude scales with distance but is capped.
    let (dx, dy) = ((to.0 - from.0) as f64, (to.1 - from.1) as f64);
    let bow = (dist * rng.range(0.04, 0.13)).min(110.0) * if rng.chance(0.5) { 1.0 } else { -1.0 };
    let len = dist.max(1.0);
    let (px, py) = (-dy / len, dx / len); // unit normal
    let ctrl = (
        from.0 as f64 + dx * 0.5 + px * bow,
        from.1 as f64 + dy * 0.5 + py * bow,
    );

    let mut out = Vec::with_capacity(n);
    for i in 1..=n {
        let t = i as f64 / n as f64;
        // Smootherstep: accelerate out of rest, decelerate into the target.
        let e = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let inv = 1.0 - e;

        let bx = inv * inv * from.0 as f64 + 2.0 * inv * e * ctrl.0 + e * e * to.0 as f64;
        let by = inv * inv * from.1 as f64 + 2.0 * inv * e * ctrl.1 + e * e * to.1 as f64;

        // Hand tremor, on everything but the final sample.
        let (jx, jy) = if i == n { (0.0, 0.0) } else { (rng.signed() * 0.9, rng.signed() * 0.9) };

        out.push(Step {
            x: (bx + jx).round() as i32,
            y: (by + jy).round() as i32,
            delay: Duration::from_micros((SAMPLE_MS * 1000.0 * rng.range(0.75, 1.25)) as u64),
        });
    }
    out
}
