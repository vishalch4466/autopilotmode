//! autopilotmode — drive the real mouse/keyboard with an AI vision loop.
//!
//! The loop lives here rather than in the binary so that every front-end runs the *same*
//! agent: the CLI ([`main.rs`](../src/main.rs)), the egui buddy, and the Tauri desktop app
//! are each a thin shell over [`agent::run_with_progress`]. A front-end that reimplemented
//! any of this would drift from it, and the one thing every front-end must agree on is
//! exactly which keys are currently held down.

pub mod action;
pub mod agent;
pub mod capture;
pub mod config;
pub mod executor;
pub mod human;
pub mod model;
