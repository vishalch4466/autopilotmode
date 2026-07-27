//! The action vocabulary the model chooses from, and the JSON schema exposed to it.
//!
//! The model always calls the single `computer_action` tool. Its input deserializes
//! into [`ActionInput`]; [`tool_schema`] is the `input_schema` we advertise.

use serde::Deserialize;
use serde_json::{json, Value};

/// One action requested by the model. `action` is the discriminator; the remaining
/// fields are interpreted per-action and validated at execution time.
#[derive(Debug, Clone, Deserialize)]
pub struct ActionInput {
    /// One of: move, click, double_click, right_click, middle_click, drag, scroll,
    /// type, key, hold, keydown, keyup, wait, screenshot, done.
    pub action: String,

    /// Target X in screenshot pixel space (origin top-left).
    #[serde(default)]
    pub x: Option<f64>,
    /// Target Y in screenshot pixel space.
    #[serde(default)]
    pub y: Option<f64>,
    /// Second point X, for `drag`.
    #[serde(default)]
    pub x2: Option<f64>,
    /// Second point Y, for `drag`.
    #[serde(default)]
    pub y2: Option<f64>,

    /// Mouse button: left (default), right, middle.
    #[serde(default)]
    pub button: Option<String>,
    /// Number of clicks for `click` (1 or 2).
    #[serde(default)]
    pub clicks: Option<u32>,

    /// Text to type verbatim, for `type`.
    #[serde(default)]
    pub text: Option<String>,
    /// Key or chord for `key`, e.g. "enter", "ctrl+c", "cmd+shift+t".
    #[serde(default)]
    pub keys: Option<String>,

    /// Horizontal scroll amount (clicks), for `scroll`. Positive = right.
    #[serde(default)]
    pub scroll_x: Option<i32>,
    /// Vertical scroll amount (clicks), for `scroll`. Positive = down.
    #[serde(default)]
    pub scroll_y: Option<i32>,

    /// Milliseconds to wait, for `wait`.
    #[serde(default)]
    pub ms: Option<u64>,

    /// Whether the goal was achieved, for `done`.
    #[serde(default)]
    pub success: Option<bool>,
    /// Short human-readable rationale / status for this action (or the final message).
    #[serde(default)]
    pub reason: Option<String>,
}

/// The `input_schema` advertised to the model for the `computer_action` tool.
pub fn tool_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["action"],
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "move", "click", "double_click", "right_click", "middle_click",
                    "drag", "scroll", "type", "key", "hold", "keydown", "keyup",
                    "wait", "screenshot", "done"
                ],
                "description": "The single action to perform this step."
            },
            "x": { "type": "number", "description": "Target X in screenshot pixels (top-left origin). For move/click/scroll/drag start." },
            "y": { "type": "number", "description": "Target Y in screenshot pixels." },
            "x2": { "type": "number", "description": "Drag end X in screenshot pixels." },
            "y2": { "type": "number", "description": "Drag end Y in screenshot pixels." },
            "button": { "type": "string", "enum": ["left", "right", "middle"], "description": "Mouse button (default left)." },
            "clicks": { "type": "integer", "description": "Click count for 'click' (1 or 2)." },
            "text": { "type": "string", "description": "Text to type verbatim for 'type'." },
            "keys": { "type": "string", "description": "Key or chord for 'key', e.g. 'enter', 'ctrl+c', 'cmd+shift+t', 'tab', 'escape', 'f5'. For 'hold'/'keydown'/'keyup', every key listed is acted on together, e.g. 'w' or 'w+shift'. For 'keyup', omit to release everything currently held." },
            "scroll_x": { "type": "integer", "description": "Horizontal scroll clicks (positive = right)." },
            "scroll_y": { "type": "integer", "description": "Vertical scroll clicks (positive = down)." },
            "ms": { "type": "integer", "description": "Milliseconds — how long to wait for 'wait', or how long to hold the keys down for 'hold' (default 800, max 15000)." },
            "success": { "type": "boolean", "description": "For 'done': whether the goal was achieved." },
            "reason": { "type": "string", "description": "One short sentence: what this action does, or the final status for 'done'." }
        }
    })
}
