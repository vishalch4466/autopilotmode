//! OpenRouter client — the same agent loop, spoken in OpenAI's Chat Completions dialect.
//!
//! # Why this is a translator and not a second loop
//!
//! The loop in [`crate::agent`] builds its conversation in Anthropic's shape: content
//! arrays, `image` blocks with a base64 `source`, `tool_use`/`tool_result` pairs. That is
//! the *canonical* history for both providers — [`request_action`] converts it to OpenAI
//! messages on the way out and converts the reply back into Anthropic-shaped
//! `assistant_content` on the way in, so what lands back in the history is indistinguishable
//! from an Anthropic turn.
//!
//! Doing it here rather than in the agent is what keeps the rule the rest of the codebase
//! follows — one loop, one history format — from quietly becoming two. Every screenshot,
//! trim, and stasis check keeps working without knowing which provider is live.
//!
//! # The one thing OpenAI's format cannot express
//!
//! Anthropic lets a `tool_result` carry an image, which is exactly what this loop does every
//! step: here is the screen after your action. OpenAI's `tool` role takes a plain string and
//! nothing else. So one Anthropic turn becomes two here — a `tool` message with the text,
//! then a `user` message carrying the screenshot. See [`push_user_blocks`].

use crate::action::ActionInput;
use crate::config::Config;
use crate::model::{
    backoff, normalize_action_input, truncate, ModelResponse, Parsed, Usage, ATTEMPTS, TOOL_NAME,
};
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

/// Sent as attribution so a run is identifiable on the OpenRouter dashboard.
const REFERER: &str = "https://autopilotmode.ai";
const TITLE: &str = "autopilotmode";

/// Optional request fields a given model has already told us it will not take.
///
/// Same reasoning as [`crate::model`]'s `UNSUPPORTED` table, and the same shape. Without
/// it the rejection is re-learned on every single step: Qwen and Z.AI each refuse two
/// fields, so a run would spend *three* round trips per step to send one usable request.
/// Keyed by model because a run can involve two of them (see `model_heavy`), and what one
/// accepts says nothing about the other.
static RELAXED: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(Default::default);

fn is_relaxed(model: &str, field: &str) -> bool {
    RELAXED
        .lock()
        .map(|s| s.contains(&format!("{model}|{field}")))
        .unwrap_or(false)
}

fn remember_relaxed(model: &str, field: &str) {
    if let Ok(mut s) = RELAXED.lock() {
        s.insert(format!("{model}|{field}"));
    }
}

/// Vendors always kept in the picker, even with no published benchmark score.
///
/// Deliberately vendors and not model ids. Ids churn constantly — this catalogue carries
/// several `gpt-5.x` and `gemini-3.x` entries that did not exist a few months ago — so a
/// hardcoded list of "popular models" is stale the week after it is written. Vendor
/// prefixes are stable, and the catalogue's own benchmark and `created` fields decide
/// *which* of a vendor's models is the current best without this file needing to know.
const MAJOR_VENDORS: [&str; 8] = [
    "anthropic",
    "openai",
    "google",
    "x-ai",
    "qwen",
    "moonshotai",
    "z-ai",
    "mistralai",
];

/// One model as the picker needs it.
#[derive(serde::Serialize, Clone, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    /// Vendor prefix (`anthropic`, `openai`, …), shown as the group label.
    pub vendor: String,
    pub context_length: u64,
    /// USD per million input tokens, formatted for display. Screenshots are the bulk of
    /// what this loop sends, so input price is the number that predicts a run's cost.
    pub prompt_price: String,
    /// True when the model takes a `reasoning.effort` hint.
    pub reasoning: bool,
    /// Artificial Analysis intelligence index, when the catalogue publishes one. This is
    /// what "top" is decided by; `None` means unranked, not bad.
    pub index: Option<f64>,
}

/// The best model each company currently offers, for the picker.
///
/// Two filters, for different reasons:
///
/// **Vision + tool calling** is a hard requirement, not a preference. Every step sends a
/// screenshot and demands a tool call back, so a model missing either does not work at all —
/// it fails on the first turn with something unhelpful. That cuts the catalogue from ~340 to
/// ~157.
///
/// **One model per company** is the useful list. A dropdown of 157 entries, most of them
/// older revisions of each other (`gpt-4o`, `gpt-4o-2024-05-13`, `gpt-4o-2024-08-06` …), is
/// a worse way to choose than a dozen named "the best each company has". Which one that is
/// comes from the catalogue's published intelligence index rather than a list maintained
/// here, so it stays correct as models ship. Any id remains reachable by typing it into the
/// model field in settings — this narrows the menu, not what the loop can run.
pub fn models(client: &reqwest::blocking::Client, cfg: &Config) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/models", cfg.base_url.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .header("authorization", format!("Bearer {}", cfg.api_key))
        .header("http-referer", REFERER)
        .header("x-title", TITLE)
        .send()
        .map_err(|e| anyhow!("could not reach OpenRouter: {e}"))?;

    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!(
            "OpenRouter model list failed ({status}): {}",
            truncate(&text, 200)
        ));
    }

    let v: Value = serde_json::from_str(&text)
        .map_err(|e| anyhow!("could not parse the OpenRouter model list: {e}"))?;
    let data = v
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("OpenRouter model list had no `data` array"))?;

    // Best-so-far per vendor, replaced as better entries appear.
    let mut best: std::collections::HashMap<String, (f64, i64, ModelInfo)> =
        std::collections::HashMap::new();

    for m in data {
        let Some(id) = m.get("id").and_then(Value::as_str) else {
            continue;
        };

        let has = |field: &str, want: &str| {
            m.get("architecture")
                .and_then(|a| a.get(field))
                .and_then(Value::as_array)
                .map(|xs| xs.iter().any(|x| x.as_str() == Some(want)))
                .unwrap_or(false)
        };
        let param = |want: &str| {
            m.get("supported_parameters")
                .and_then(Value::as_array)
                .map(|xs| xs.iter().any(|x| x.as_str() == Some(want)))
                .unwrap_or(false)
        };
        if !has("input_modalities", "image") || !param("tools") {
            continue;
        }

        let vendor = id.split('/').next().unwrap_or_default().to_string();
        // `~vendor` is OpenRouter's namespace for variants and stealth listings rather than
        // a company's own catalogue entry — not something to offer as "the best from X".
        if vendor.starts_with('~') || vendor.is_empty() {
            continue;
        }

        let index = m
            .get("benchmarks")
            .and_then(|b| b.get("artificial_analysis"))
            .and_then(|a| a.get("intelligence_index"))
            .and_then(Value::as_f64);
        // Unranked models still compete, just below every ranked one — otherwise a vendor
        // whose catalogue carries no benchmark would drop out entirely.
        let score = index.unwrap_or(f64::MIN);
        let created = m.get("created").and_then(Value::as_i64).unwrap_or(0);

        let candidate = ModelInfo {
            id: id.to_string(),
            name: m
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(id)
                .to_string(),
            vendor: vendor.clone(),
            context_length: m.get("context_length").and_then(Value::as_u64).unwrap_or(0),
            prompt_price: price_per_mtok(m),
            reasoning: param("reasoning") || param("reasoning_effort"),
            index,
        };

        match best.get(&vendor) {
            // Higher benchmark wins; between two unranked models, the newer one does.
            Some((s, c, _)) if (*s, *c) >= (score, created) => {}
            _ => {
                best.insert(vendor, (score, created, candidate));
            }
        }
    }

    let mut out: Vec<(f64, i64, ModelInfo)> = best
        .into_values()
        // A company with no benchmarked model at all is only worth listing if it is one
        // people actually reach for; otherwise the menu fills with names nobody recognises.
        .filter(|(_, _, m)| m.index.is_some() || MAJOR_VENDORS.contains(&m.vendor.as_str()))
        .collect();

    // Strongest first. `total_cmp` rather than `partial_cmp` — these are catalogue-supplied
    // floats, and an unexpected NaN should order consistently instead of panicking.
    out.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)));
    Ok(out.into_iter().map(|(_, _, m)| m).collect())
}

/// `pricing.prompt` is USD per token as a string; per-million is the readable unit.
fn price_per_mtok(m: &Value) -> String {
    let per_token = m
        .get("pricing")
        .and_then(|p| p.get("prompt"))
        .and_then(Value::as_str)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let per_mtok = per_token * 1_000_000.0;
    match per_mtok {
        p if p <= 0.0 => "free".to_string(),
        // Cheap models cluster well under a dollar, where two decimals rounds most of them
        // to the same "$0.05" and the comparison the price is there to support disappears.
        p if p < 1.0 => format!("${p:.3}"),
        p => format!("${p:.2}"),
    }
}

/// Send one turn and return the chosen action. Retries transient errors, exactly as the
/// Anthropic path does.
pub fn request_action(
    client: &reqwest::blocking::Client,
    cfg: &Config,
    model: &str,
    system: &str,
    messages: &[Value],
    tools: &Value,
) -> Result<ModelResponse> {
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let mut body = json!({
        "model": model,
        "max_tokens": 4096,
        "messages": to_chat_messages(system, messages),
        "tools": to_chat_tools(tools),
        // One tool is defined, so "required" is the same instruction as Anthropic's
        // `tool_choice: any` — and it is far more widely supported across OpenRouter's
        // providers than naming the function explicitly.
        "tool_choice": if is_relaxed(model, "tool_choice") { "auto" } else { "required" },
    });

    // Each of these is added only if this model has not already refused it earlier in the
    // process, so the cost of discovering a provider's limits is paid once rather than on
    // every step of the run.
    if !is_relaxed(model, "parallel_tool_calls") {
        body["parallel_tool_calls"] = json!(false);
    }
    // Several models on OpenRouter — Claude and the GPT-5 line among them — think by
    // default, at high effort. That is seconds of deliberation before every click on a loop
    // that takes one small step per screenshot, so the effort setting matters here for the
    // same reason it does on the native path.
    if !is_relaxed(model, "reasoning") {
        if let Some(effort) = openrouter_effort(&cfg.effort) {
            body["reasoning"] = json!({ "effort": effort });
        }
    }

    let mut last_err = None;
    // Counted separately from `attempt` on purpose. Dropping a parameter the provider will
    // not take is not a failed try — the conversation was fine — so it must not spend one of
    // the three real attempts. Qwen's endpoints reject two different optional fields in
    // sequence, which would otherwise exhaust the retries before the request had been sent
    // even once in a form the provider accepts.
    let mut relaxations = 0u32;
    const MAX_RELAXATIONS: u32 = 3;

    let mut attempt = 0u32;
    while attempt < ATTEMPTS {
        let resp = client
            .post(&url)
            .header("authorization", format!("Bearer {}", cfg.api_key))
            .header("content-type", "application/json")
            .header("http-referer", REFERER)
            .header("x-title", TITLE)
            .json(&body)
            .send();

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                last_err = Some(anyhow!("network error: {e}"));
                backoff(attempt);
                attempt += 1;
                continue;
            }
        };

        let status = resp.status();
        let text = resp.text().unwrap_or_default();

        if status.is_success() {
            match parse_success(&text)? {
                Parsed::Action(r) => return Ok(*r),
                Parsed::NoAction { stop_reason, said } => {
                    eprintln!(
                        "  ⚠ no action in reply (stop_reason: {stop_reason}) — retrying{}",
                        if attempt + 1 < ATTEMPTS { "" } else { " (last attempt)" }
                    );
                    last_err = Some(anyhow!(
                        "model did not call {TOOL_NAME} after {ATTEMPTS} attempts \
                         (stop_reason: {stop_reason}){}",
                        said.map(|t| format!(" — it said: {}", truncate(&t, 200)))
                            .unwrap_or_default()
                    ));
                    backoff(attempt);
                    attempt += 1;
                    continue;
                }
            }
        }

        // OpenRouter fronts many providers and they do not accept the same optional
        // parameters. Rather than carry a per-model allowlist that goes stale — the same
        // argument the Anthropic path makes — let the API say what it will not take, drop
        // that field, and carry on.
        if status.as_u16() == 400 || status.as_u16() == 422 {
            if relaxations < MAX_RELAXATIONS {
                if let Some(r) = relax(&mut body, &text) {
                    eprintln!("  ⚠ {model} rejected {}, omitting it for this run", r.why);
                    // Remembered per model, so the next step sends an acceptable request
                    // first time instead of rediscovering this.
                    remember_relaxed(model, r.field);
                    relaxations += 1;
                    continue;
                }
            }
            return Err(anyhow!(
                "OpenRouter error {status}: {}",
                truncate(&text, 600)
            ));
        }

        if status.as_u16() == 429 || status.as_u16() == 529 || status.is_server_error() {
            last_err = Some(anyhow!("OpenRouter {status}: {}", truncate(&text, 300)));
            backoff(attempt);
            attempt += 1;
            continue;
        }

        return Err(anyhow!(
            "OpenRouter error {status}: {}",
            truncate(&text, 600)
        ));
    }

    Err(last_err.unwrap_or_else(|| anyhow!("request failed after retries")))
}

/// Drop whichever optional parameter the provider objected to, naming what went.
///
/// Ordered by how much the loop would miss the field. A forced tool call is the thing this
/// agent genuinely depends on — every turn must produce an action — so reasoning depth is
/// given up first and `tool_choice` is only relaxed when nothing else is left.
/// What [`relax`] gave up: the request field, and how to describe it in the log.
struct Relaxation {
    field: &'static str,
    why: &'static str,
}

fn relax(body: &mut Value, err: &str) -> Option<Relaxation> {
    // Matched case-insensitively and against more than one spelling, because these strings
    // are written by whichever upstream provider rejected the call and they do not agree:
    // Alibaba says `tool_choice`, Z.AI says "Tool choice must be auto". Matching the exact
    // JSON field name would silently miss half of them and surface as a hard 400.
    let lower = err.to_ascii_lowercase();
    let mentions = |k: &str| lower.contains(k);
    let tool_choice_named = mentions("tool_choice") || mentions("tool choice");
    let obj = body.as_object_mut()?;

    // Alibaba's Qwen endpoints reject `tool_choice: required` *while thinking is on*, and
    // name tool_choice in the error rather than the reasoning that actually conflicts — so
    // matching on the named field alone would drop the wrong one and fail again.
    if (tool_choice_named || mentions("thinking")) && obj.contains_key("reasoning") {
        obj.remove("reasoning");
        return Some(Relaxation {
            field: "reasoning",
            why: "`reasoning` (it conflicts with a forced tool call here)",
        });
    }
    if (mentions("parallel_tool_calls") || mentions("parallel tool"))
        && obj.remove("parallel_tool_calls").is_some()
    {
        return Some(Relaxation {
            field: "parallel_tool_calls",
            why: "`parallel_tool_calls`",
        });
    }
    if mentions("reasoning") && obj.remove("reasoning").is_some() {
        return Some(Relaxation {
            field: "reasoning",
            why: "`reasoning`",
        });
    }
    // Last resort. The loop already resamples a turn that arrives without a tool call, so
    // this costs turns rather than the run.
    if tool_choice_named && obj.get("tool_choice") != Some(&json!("auto")) {
        obj.insert("tool_choice".to_string(), json!("auto"));
        return Some(Relaxation {
            field: "tool_choice",
            why: "a forced `tool_choice` (falling back to auto)",
        });
    }
    None
}

/// Map this project's effort levels onto the three OpenRouter accepts.
///
/// `xhigh` and `max` are Anthropic-only rungs; folding them into `high` keeps the setting
/// meaningful across providers rather than silently dropping it when someone picks a level
/// OpenRouter has never heard of.
fn openrouter_effort(effort: &str) -> Option<&'static str> {
    match effort {
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" | "xhigh" | "max" => Some("high"),
        _ => None,
    }
}

// ---- outbound translation: Anthropic history → OpenAI messages ----

fn to_chat_messages(system: &str, messages: &[Value]) -> Vec<Value> {
    let mut out = vec![json!({ "role": "system", "content": system })];
    for m in messages {
        let blocks = m
            .get("content")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        match m.get("role").and_then(Value::as_str) {
            Some("assistant") => out.push(assistant_message(&blocks)),
            _ => push_user_blocks(&mut out, &blocks),
        }
    }
    out
}

/// An assistant turn: narration in `content`, the action as a `tool_calls` entry.
fn assistant_message(blocks: &[Value]) -> Value {
    let mut text = String::new();
    let mut calls: Vec<Value> = Vec::new();

    for b in blocks {
        match b.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(t) = b.get("text").and_then(Value::as_str) {
                    text.push_str(t);
                }
            }
            Some("tool_use") => calls.push(json!({
                "id": b.get("id").and_then(Value::as_str).unwrap_or_default(),
                "type": "function",
                "function": {
                    "name": b.get("name").and_then(Value::as_str).unwrap_or(TOOL_NAME),
                    // OpenAI carries arguments as a JSON *string*, not an object.
                    "arguments": b.get("input").map(Value::to_string).unwrap_or_else(|| "{}".into()),
                }
            })),
            _ => {}
        }
    }

    let mut m = json!({ "role": "assistant" });
    // Null rather than "" when there is only a tool call: some providers reject an empty
    // string here, and every one of them accepts null.
    m["content"] = if text.trim().is_empty() {
        Value::Null
    } else {
        json!(text)
    };
    if !calls.is_empty() {
        m["tool_calls"] = Value::Array(calls);
    }
    m
}

/// A user turn, splitting `tool_result` blocks out into their own `tool` messages.
///
/// This is where the formats genuinely disagree. Anthropic puts the screenshot *inside* the
/// tool result; OpenAI's `tool` role takes a string. So the text becomes the tool message —
/// which must directly follow the assistant turn that requested it, or providers reject the
/// conversation — and the screenshot follows as a separate user turn.
fn push_user_blocks(out: &mut Vec<Value>, blocks: &[Value]) {
    let mut parts: Vec<Value> = Vec::new();

    for b in blocks {
        if b.get("type").and_then(Value::as_str) == Some("tool_result") {
            let id = b
                .get("tool_use_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let inner = b
                .get("content")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();

            let mut text = String::new();
            let mut images: Vec<Value> = Vec::new();
            for x in &inner {
                match x.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = x.get("text").and_then(Value::as_str) {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(t);
                        }
                    }
                    Some("image") => {
                        if let Some(p) = image_part(x) {
                            images.push(p);
                        }
                    }
                    _ => {}
                }
            }

            out.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "content": if text.is_empty() { "(no output)".to_string() } else { text },
            }));

            if !images.is_empty() {
                images.push(json!({
                    "type": "text",
                    "text": "This is the screen as it looks now. Decide the next action.",
                }));
                out.push(json!({ "role": "user", "content": images }));
            }
            continue;
        }

        if let Some(p) = to_part(b) {
            parts.push(p);
        }
    }

    if !parts.is_empty() {
        out.push(json!({ "role": "user", "content": parts }));
    }
}

fn to_part(b: &Value) -> Option<Value> {
    match b.get("type").and_then(Value::as_str)? {
        "text" => Some(json!({
            "type": "text",
            "text": b.get("text").and_then(Value::as_str).unwrap_or_default(),
        })),
        "image" => image_part(b),
        _ => None,
    }
}

/// An Anthropic base64 image block as an OpenAI `image_url` data URI.
fn image_part(b: &Value) -> Option<Value> {
    let src = b.get("source")?;
    let media = src
        .get("media_type")
        .and_then(Value::as_str)
        .unwrap_or("image/png");
    let data = src.get("data").and_then(Value::as_str)?;
    Some(json!({
        "type": "image_url",
        "image_url": { "url": format!("data:{media};base64,{data}") },
    }))
}

/// Anthropic tool definitions (`input_schema`) as OpenAI functions (`parameters`).
fn to_chat_tools(tools: &Value) -> Value {
    let empty = Vec::new();
    let arr = tools.as_array().unwrap_or(&empty);
    Value::Array(
        arr.iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.get("name").and_then(Value::as_str).unwrap_or(TOOL_NAME),
                        "description": t.get("description").and_then(Value::as_str).unwrap_or_default(),
                        "parameters": t.get("input_schema").cloned().unwrap_or_else(|| json!({})),
                    }
                })
            })
            .collect(),
    )
}

// ---- inbound translation: OpenAI reply → canonical Anthropic-shaped turn ----

fn parse_success(text: &str) -> Result<Parsed> {
    let v: Value = serde_json::from_str(text).map_err(|e| {
        anyhow!(
            "could not parse the OpenRouter response: {e}\nbody: {}",
            truncate(text, 400)
        )
    })?;

    // OpenRouter reports an upstream provider failure as a 200 carrying an `error` object,
    // so a status check alone is not enough to know the turn succeeded.
    if let Some(err) = v.get("error") {
        let msg = err
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown error");
        return Err(anyhow!("OpenRouter upstream error: {msg}"));
    }

    let choice = v
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .ok_or_else(|| anyhow!("response had no choices"))?;
    let message = choice
        .get("message")
        .ok_or_else(|| anyhow!("choice had no message"))?;

    let narration = message
        .get("content")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_string);

    let finish = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    let call = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .and_then(|c| c.first());
    let Some(call) = call else {
        return Ok(Parsed::NoAction {
            stop_reason: finish,
            said: narration,
        });
    };

    let id = call
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("call_0")
        .to_string();

    // Spec says `arguments` is a JSON string. A few providers send an object instead, and
    // rejecting those would be pedantry — the intent is identical either way.
    let raw = call.get("function").and_then(|f| f.get("arguments"));
    let mut input = match raw {
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(parsed) => parsed,
            // Truncated or malformed JSON is a bad sample, not a broken run — the same
            // judgement the Anthropic path makes about a malformed tool input.
            Err(e) => {
                return Ok(Parsed::NoAction {
                    stop_reason: format!("tool arguments were not valid JSON ({e})"),
                    said: narration,
                })
            }
        },
        Some(other) => other.clone(),
        None => {
            return Ok(Parsed::NoAction {
                stop_reason: "tool call had no arguments".to_string(),
                said: narration,
            })
        }
    };
    normalize_action_input(&mut input);

    let action: ActionInput = match serde_json::from_value(input.clone()) {
        Ok(a) => a,
        Err(e) => {
            return Ok(Parsed::NoAction {
                stop_reason: format!("malformed action input ({e})"),
                said: narration,
            })
        }
    };

    // Rebuild the turn in Anthropic's shape before it goes back into the history. This is
    // what lets `agent.rs` — and the trimming, and the next translation pass — stay
    // provider-blind.
    let mut content: Vec<Value> = Vec::new();
    if let Some(t) = &narration {
        content.push(json!({ "type": "text", "text": t }));
    }
    content.push(json!({
        "type": "tool_use",
        "id": id.clone(),
        "name": TOOL_NAME,
        "input": input,
    }));

    Ok(Parsed::Action(Box::new(ModelResponse {
        assistant_content: Value::Array(content),
        // The provider's own call id, carried through so the `tool_call_id` on the next
        // turn's tool message matches what it issued.
        tool_use_id: id,
        action,
        text: narration,
        usage: Usage::from_openai(&v),
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(data: &str) -> Value {
        json!({
            "type": "image",
            "source": { "type": "base64", "media_type": "image/png", "data": data }
        })
    }

    fn roles(msgs: &[Value]) -> Vec<&str> {
        msgs.iter()
            .map(|m| m.get("role").and_then(Value::as_str).unwrap_or("?"))
            .collect()
    }

    #[test]
    fn first_turn_becomes_system_plus_user_with_image() {
        let history = vec![json!({
            "role": "user",
            "content": [image("AAAA"), { "type": "text", "text": "GOAL: open Start" }]
        })];
        let out = to_chat_messages("be an agent", &history);

        assert_eq!(roles(&out), ["system", "user"]);
        let parts = out[1]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "image_url");
        assert_eq!(
            parts[0]["image_url"]["url"],
            "data:image/png;base64,AAAA",
            "image blocks must become data-URI image_url parts"
        );
        assert_eq!(parts[1]["text"], "GOAL: open Start");
    }

    #[test]
    fn assistant_tool_use_becomes_a_tool_call_with_string_arguments() {
        let history = vec![json!({
            "role": "assistant",
            "content": [
                { "type": "text", "text": "clicking Start" },
                { "type": "tool_use", "id": "call_7", "name": TOOL_NAME,
                  "input": { "action": "click", "x": 12.0, "y": 34.0 } }
            ]
        })];
        let out = to_chat_messages("sys", &history);

        let m = &out[1];
        assert_eq!(m["role"], "assistant");
        assert_eq!(m["content"], "clicking Start");
        let call = &m["tool_calls"][0];
        assert_eq!(call["id"], "call_7");
        assert_eq!(call["function"]["name"], TOOL_NAME);

        // The distinguishing detail of the OpenAI format: arguments is a JSON *string*.
        let args = call["function"]["arguments"].as_str().expect("a string");
        let parsed: Value = serde_json::from_str(args).unwrap();
        assert_eq!(parsed["action"], "click");
        assert_eq!(parsed["x"], 12.0);
    }

    #[test]
    fn assistant_with_only_a_tool_call_sends_null_content() {
        let history = vec![json!({
            "role": "assistant",
            "content": [{ "type": "tool_use", "id": "c1", "name": TOOL_NAME, "input": {} }]
        })];
        let out = to_chat_messages("sys", &history);
        assert!(
            out[1]["content"].is_null(),
            "empty narration must be null, not \"\" — some providers reject the empty string"
        );
    }

    /// The core incompatibility: Anthropic puts the screenshot inside the tool result,
    /// OpenAI's `tool` role takes a string. One turn in must become two turns out.
    #[test]
    fn tool_result_splits_into_a_tool_message_and_a_user_image() {
        let history = vec![json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "call_7",
                "content": [image("BBBB"), { "type": "text", "text": "Step 1 executed." }]
            }]
        })];
        let out = to_chat_messages("sys", &history);

        assert_eq!(roles(&out), ["system", "tool", "user"]);
        assert_eq!(out[1]["tool_call_id"], "call_7");
        assert_eq!(
            out[1]["content"], "Step 1 executed.",
            "the tool message must carry plain text, never an array"
        );
        assert!(out[1]["content"].is_string());
        assert_eq!(out[2]["content"][0]["type"], "image_url");
    }

    /// A tool message must directly follow the assistant turn that requested it, or
    /// providers reject the whole conversation.
    #[test]
    fn tool_message_directly_follows_its_assistant_turn() {
        let history = vec![
            json!({ "role": "user", "content": [image("A"), { "type": "text", "text": "go" }] }),
            json!({ "role": "assistant", "content": [
                { "type": "tool_use", "id": "c1", "name": TOOL_NAME, "input": { "action": "click" } }
            ]}),
            json!({ "role": "user", "content": [{
                "type": "tool_result", "tool_use_id": "c1",
                "content": [image("B"), { "type": "text", "text": "done" }]
            }]}),
        ];
        let out = to_chat_messages("sys", &history);
        assert_eq!(roles(&out), ["system", "user", "assistant", "tool", "user"]);
    }

    /// Trimming replaces old screenshots with a text placeholder; that must not leave an
    /// empty user turn behind.
    #[test]
    fn trimmed_tool_result_emits_no_trailing_user_turn() {
        let history = vec![json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "c1",
                "content": [
                    { "type": "text", "text": "[earlier screenshot omitted to save context]" },
                    { "type": "text", "text": "Step 2 executed." }
                ]
            }]
        })];
        let out = to_chat_messages("sys", &history);
        assert_eq!(roles(&out), ["system", "tool"]);
    }

    #[test]
    fn tools_become_openai_functions() {
        let anthropic = json!([{
            "name": TOOL_NAME,
            "description": "do one thing",
            "input_schema": { "type": "object", "properties": { "action": { "type": "string" } } }
        }]);
        let out = to_chat_tools(&anthropic);
        let f = &out[0]["function"];
        assert_eq!(out[0]["type"], "function");
        assert_eq!(f["name"], TOOL_NAME);
        assert_eq!(
            f["parameters"]["properties"]["action"]["type"], "string",
            "input_schema must be carried over as `parameters`"
        );
    }

    /// A reply must come back as an Anthropic-shaped turn, or the next translation pass —
    /// and everything in `agent.rs` — would be looking at a format it does not know.
    #[test]
    fn reply_is_rebuilt_into_a_canonical_anthropic_turn() {
        let body = json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": "Clicking the button.",
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": TOOL_NAME,
                            "arguments": "{\"action\":\"click\",\"x\":100,\"y\":200}"
                        }
                    }]
                }
            }],
            "usage": { "prompt_tokens": 1500, "completion_tokens": 20,
                       "prompt_tokens_details": { "cached_tokens": 500 } }
        });

        let Parsed::Action(r) = parse_success(&body.to_string()).unwrap() else {
            panic!("expected an action");
        };
        assert_eq!(r.action.action, "click");
        assert_eq!(r.tool_use_id, "call_abc");
        assert_eq!(r.text.as_deref(), Some("Clicking the button."));

        let blocks = r.assistant_content.as_array().unwrap();
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["id"], "call_abc");

        // Anthropic's `input_tokens` excludes cache reads; OpenAI's `prompt_tokens`
        // includes them. Without the subtraction the readout double-counts a cache hit.
        assert_eq!(r.usage.cache_read, 500);
        assert_eq!(r.usage.input, 1000);
        assert_eq!(r.usage.output, 20);
    }

    /// Feeding a reply straight back through the outbound translation is the real
    /// round-trip: the id the provider issued has to survive to the `tool_call_id`.
    #[test]
    fn call_id_survives_a_full_round_trip() {
        let body = json!({
            "choices": [{ "message": { "content": null, "tool_calls": [{
                "id": "call_xyz", "type": "function",
                "function": { "name": TOOL_NAME, "arguments": "{\"action\":\"done\"}" }
            }]}}]
        });
        let Parsed::Action(r) = parse_success(&body.to_string()).unwrap() else {
            panic!("expected an action");
        };

        let history = vec![
            json!({ "role": "assistant", "content": r.assistant_content }),
            json!({ "role": "user", "content": [{
                "type": "tool_result", "tool_use_id": r.tool_use_id,
                "content": [{ "type": "text", "text": "ok" }]
            }]}),
        ];
        let out = to_chat_messages("sys", &history);
        assert_eq!(out[1]["tool_calls"][0]["id"], "call_xyz");
        assert_eq!(out[2]["tool_call_id"], "call_xyz");
    }

    /// Non-compliant providers sometimes send `arguments` as an object rather than a
    /// string. The intent is identical, so it is accepted rather than failed.
    #[test]
    fn object_arguments_are_accepted() {
        let body = json!({
            "choices": [{ "message": { "tool_calls": [{
                "id": "c", "function": { "name": TOOL_NAME, "arguments": { "action": "wait" } }
            }]}}]
        });
        let Parsed::Action(r) = parse_success(&body.to_string()).unwrap() else {
            panic!("expected an action");
        };
        assert_eq!(r.action.action, "wait");
    }

    /// Loosely-typed coordinates are repaired, not rejected — smaller models reproduce the
    /// same mistake on a resample, so retrying would burn the whole run.
    #[test]
    fn string_coordinates_are_repaired() {
        let body = json!({
            "choices": [{ "message": { "tool_calls": [{
                "id": "c", "function": { "name": TOOL_NAME,
                    "arguments": "{\"action\":\"click\",\"x\":\"168, 152\"}" }
            }]}}]
        });
        let Parsed::Action(r) = parse_success(&body.to_string()).unwrap() else {
            panic!("expected an action");
        };
        assert_eq!(r.action.x, Some(168.0));
        assert_eq!(r.action.y, Some(152.0));
    }

    #[test]
    fn malformed_arguments_ask_for_a_resample_rather_than_failing() {
        let body = json!({
            "choices": [{ "finish_reason": "length", "message": { "tool_calls": [{
                "id": "c", "function": { "name": TOOL_NAME, "arguments": "{\"action\":\"cli" }
            }]}}]
        });
        assert!(matches!(
            parse_success(&body.to_string()).unwrap(),
            Parsed::NoAction { .. }
        ));
    }

    #[test]
    fn a_reply_with_no_tool_call_asks_for_a_resample() {
        let body = json!({
            "choices": [{ "finish_reason": "stop",
                          "message": { "content": "I think you should click Start." } }]
        });
        let Parsed::NoAction { stop_reason, said } = parse_success(&body.to_string()).unwrap()
        else {
            panic!("expected NoAction");
        };
        assert_eq!(stop_reason, "stop");
        assert!(said.unwrap().contains("click Start"));
    }

    /// OpenRouter reports upstream provider failures as a 200 carrying an `error` object,
    /// so a status check alone would treat this as a successful turn.
    #[test]
    fn upstream_error_in_a_200_is_an_error() {
        let body = json!({ "error": { "message": "provider returned 502", "code": 502 } });
        // Matched rather than `unwrap_err`, which would need `Parsed: Debug` and so a
        // Debug bound on the whole response type for one assertion.
        let Err(e) = parse_success(&body.to_string()) else {
            panic!("a 200 carrying an error object must not read as a successful turn");
        };
        let err = e.to_string();
        assert!(err.contains("provider returned 502"), "got: {err}");
    }

    /// A synthetic screen: light background, one dark "button" around (200, 145).
    ///
    /// Generated rather than a real screenshot so the live test below can run anywhere and
    /// uploads nothing from the machine it runs on.
    #[cfg(test)]
    fn fake_screen() -> String {
        use base64::Engine as _;
        let mut img = image::ImageBuffer::from_pixel(400, 300, image::Rgba([245u8, 245, 248, 255]));
        for y in 120..170u32 {
            for x in 150..250u32 {
                img.put_pixel(x, y, image::Rgba([30u8, 90, 200, 255]));
            }
        }
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut png, image::ImageFormat::Png)
            .expect("encode png");
        base64::engine::general_purpose::STANDARD.encode(png.into_inner())
    }

    /// End-to-end against the real API: does a live model, reached through the translation,
    /// come back with an action this loop can execute?
    ///
    /// Ignored by default — it needs a key and spends real tokens. Run it after touching
    /// anything in the translation:
    ///
    /// ```text
    /// cargo test -p autopilotmode --lib -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "hits the live OpenRouter API and spends tokens"]
    fn live_round_trip_returns_an_executable_action() {
        dotenvy::dotenv().ok();
        let cfg = match Config::for_provider(crate::config::Provider::OpenRouter) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping — {e}");
                return;
            }
        };
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(180))
            .build()
            .unwrap();

        let history = vec![json!({
            "role": "user",
            "content": [
                image(&fake_screen()),
                { "type": "text", "text":
                  "GOAL: click the blue button.\n\nThis is the current screen (400×300 px). \
                   Decide the single next action and call computer_action. Coordinates are \
                   in this image's pixel space." }
            ]
        })];

        let resp = request_action(
            &client,
            &cfg,
            &cfg.model,
            &crate::model::system_prompt(true, false),
            &history,
            &crate::model::tools(),
        )
        .expect("live request failed");

        eprintln!(
            "  turn 1: {} action={} x={:?} y={:?} · {}",
            cfg.model,
            resp.action.action,
            resp.action.x,
            resp.action.y,
            resp.usage.label()
        );
        // What this test is actually for is the *translation*, so the assertion is that a
        // usable action came back — not that the model aimed well. Aim is a property of the
        // model, and folding the two together would report "OpenRouter is broken" for what
        // is really "this model is bad at coordinates".
        assert_eq!(
            resp.action.action, "click",
            "a model shown one button on an otherwise empty screen should click it; \
             anything else suggests it did not receive the image"
        );
        let (x, y) = (
            resp.action.x.expect("a click needs an x"),
            resp.action.y.expect("a click needs a y"),
        );

        // Reported, not asserted. The button spans x 150-250, y 120-170; landing outside it
        // — or outside the image entirely, which some models do — says the model misread
        // the coordinate space, and that is worth seeing without failing the integration.
        let on_target = (150.0..=250.0).contains(&x) && (120.0..=170.0).contains(&y);
        let in_frame = (0.0..=400.0).contains(&x) && (0.0..=300.0).contains(&y);
        eprintln!(
            "  aim: ({x}, {y}) — {}",
            match (on_target, in_frame) {
                (true, _) => "on the button",
                (false, true) => "missed the button but stayed in frame",
                (false, false) => "OUTSIDE the 400x300 image — poor fit for this loop",
            }
        );

        // Second turn, which is the part worth testing. One Anthropic tool_result becomes a
        // `tool` message plus a `user` message here, and providers validate that ordering
        // strictly — a conversation that survives turn one can still be rejected at turn
        // two. Nothing but a live call proves this shape is accepted.
        let mut history = history;
        history.push(json!({ "role": "assistant", "content": resp.assistant_content }));
        history.push(json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": resp.tool_use_id,
                "content": [
                    image(&fake_screen()),
                    { "type": "text", "text":
                      "Step 1 executed. Current screen (400×300 px). NOTE: the screen is \
                       essentially unchanged from before that action, so it most likely had \
                       NO effect. Continue toward the goal, or call done." }
                ]
            }]
        }));

        let second = request_action(
            &client,
            &cfg,
            &cfg.model,
            &crate::model::system_prompt(true, false),
            &history,
            &crate::model::tools(),
        )
        .expect("second turn failed — the tool/user message split was likely rejected");

        eprintln!(
            "  turn 2: action={} · {}",
            second.action.action,
            second.usage.label()
        );
    }

    /// Qwen's real 400: it names `tool_choice`, but the field that actually has to go is
    /// `reasoning`. Dropping the named field would fail again on the retry.
    #[test]
    fn thinking_mode_conflict_drops_reasoning_not_tool_choice() {
        let mut body = json!({
            "tool_choice": "required",
            "parallel_tool_calls": false,
            "reasoning": { "effort": "low" },
        });
        let err = "The tool_choice parameter does not support being set to required or \
                   object in thinking mode";

        assert!(relax(&mut body, err).is_some());
        assert!(body.get("reasoning").is_none(), "reasoning should have gone");
        assert_eq!(
            body["tool_choice"], "required",
            "the forced tool call is what the loop depends on and must survive"
        );
    }

    #[test]
    fn tool_choice_is_only_relaxed_once_nothing_else_is_left() {
        let mut body = json!({ "tool_choice": "required" });
        assert!(relax(&mut body, "tool_choice is not supported").is_some());
        assert_eq!(body["tool_choice"], "auto");
        // Nothing further to give up.
        assert!(relax(&mut body, "tool_choice is not supported").is_none());
    }

    /// Providers write these messages themselves and do not agree on spelling. Z.AI's real
    /// 400 is "Tool choice must be auto" — capitalised, spaced, no underscore — which an
    /// exact match on the JSON field name misses entirely.
    #[test]
    fn provider_wordings_for_tool_choice_are_all_recognised() {
        for wording in [
            "Tool choice must be auto",
            "tool_choice is not supported",
            "Invalid tool choice parameter",
            "TOOL_CHOICE unsupported",
        ] {
            let mut body = json!({ "tool_choice": "required" });
            assert!(
                relax(&mut body, wording).is_some(),
                "did not recognise: {wording}"
            );
            assert_eq!(body["tool_choice"], "auto", "for wording: {wording}");
        }
    }

    #[test]
    fn an_unrelated_400_relaxes_nothing() {
        let mut body = json!({ "tool_choice": "required", "reasoning": { "effort": "low" } });
        assert!(relax(&mut body, "context length exceeded").is_none());
        assert!(body.get("reasoning").is_some());
    }

    /// Prints the picker exactly as the chat bar will render it, and checks the two
    /// promises the list makes: one entry per company, and every entry usable by this loop.
    ///
    /// Ignored by default — it needs a key and hits the network.
    #[test]
    #[ignore = "hits the live OpenRouter catalogue"]
    fn live_model_list_is_one_per_company() {
        dotenvy::dotenv().ok();
        let cfg = match Config::for_provider(crate::config::Provider::OpenRouter) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping — {e}");
                return;
            }
        };
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();
        let list = models(&client, &cfg).expect("model list failed");

        eprintln!("\n  {} models in the picker:", list.len());
        for m in &list {
            eprintln!(
                "    {:<44} {:>8}/M  {:>7}k  {}",
                m.id,
                m.prompt_price,
                m.context_length / 1000,
                m.index.map(|i| format!("idx {i}")).unwrap_or_default()
            );
        }

        assert!(!list.is_empty(), "the catalogue should not come back empty");

        let mut seen = HashSet::new();
        for m in &list {
            assert!(
                seen.insert(m.vendor.clone()),
                "{} is a second entry for {} — the list is meant to be one per company",
                m.id,
                m.vendor
            );
            assert!(
                !m.vendor.starts_with('~'),
                "{} is a variant namespace, not a company",
                m.id
            );
        }
        // The vendors people actually reach for should survive both filters.
        for want in ["anthropic", "openai", "google"] {
            assert!(
                list.iter().any(|m| m.vendor == want),
                "{want} missing from the picker"
            );
        }
    }

    #[test]
    fn effort_levels_map_onto_the_three_openrouter_accepts() {
        assert_eq!(openrouter_effort("low"), Some("low"));
        assert_eq!(openrouter_effort("medium"), Some("medium"));
        // xhigh/max are Anthropic-only rungs; folding them into high keeps the setting
        // meaningful rather than silently dropping it.
        assert_eq!(openrouter_effort("xhigh"), Some("high"));
        assert_eq!(openrouter_effort("max"), Some("high"));
        assert_eq!(openrouter_effort(""), None);
    }
}
