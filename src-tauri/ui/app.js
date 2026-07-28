// Front-end wiring. Everything here is presentation: the agent publishes what it is doing
// and this file decides how to say it. No agent logic lives on this side.

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const { getCurrentWindow } = window.__TAURI__.window;
// Tauri v2 moved the size types into their own `dpi` module; fall back to the window
// module so this keeps working either way.
const { LogicalSize } = window.__TAURI__.dpi ?? window.__TAURI__.window;

const el = (id) => document.getElementById(id);
const card = el("card");
// `statusPill`, not `status` — a bare `status` shadows the deprecated `window.status`.
const statusPill = el("status");
const log = el("log");
const [goal, go, stop, mic] = [el("goal"), el("go"), el("stop"), el("mic")];
const [dry, mode, note] = [el("dry"), el("mode"), el("note")];
const panel = el("panel");

let running = false;
let dryRun = true;
let listening = false;

/** The card is sized for the compact view; settings need room to breathe. */
const SIZE = {
  compact: [420, 208],
  // Once there is a transcript to read, the window earns the extra height.
  chat: [420, 520],
  settings: [420, 504],
  settingsVoice: [420, 544],
};

/** Height the model picker row adds, so the window grows only when it is shown. */
const MODEL_ROW_H = 36;

let grown = false;
async function growForChat() {
  if (grown) return;
  grown = true;
  await resizeMain();
}

/** Size the main view for its current contents — transcript open, picker shown, or not. */
async function resizeMain() {
  if (!panel.hidden) return; // settings is over the card; it owns the size while open
  const [w, h] = grown ? SIZE.chat : SIZE.compact;
  await resize([w, h + (el("model-row").hidden ? 0 : MODEL_ROW_H)]);
}
// Surfaced rather than swallowed: a silently-failed resize leaves the pane scrolling in a
// window too short for it, which is confusing in a way a console line is not.
async function resize([w, h]) {
  try {
    await getCurrentWindow().setSize(new LogicalSize(w, h));
  } catch (e) {
    console.error("resize failed", e);
  }
}

/** Mood drives the glow colour; the card class is the single source for it. */
function setMood(mood) {
  card.classList.remove("mood-idle", "mood-thinking", "mood-acting", "mood-happy", "mood-sad");
  card.classList.add(`mood-${mood}`);
}

/** Oldest entries are dropped rather than kept forever; a long run is otherwise unbounded. */
const MAX_LOG = 80;

function trimLog() {
  while (log.children.length > MAX_LOG) log.firstElementChild.remove();
}

/** True when the user is already scrolled to the bottom — only then should we follow. */
function atBottom() {
  return log.scrollHeight - log.scrollTop - log.clientHeight < 40;
}

function append(node) {
  // Don't yank the view down if they have scrolled up to read something.
  const follow = atBottom();
  log.append(node);
  trimLog();
  if (follow) log.scrollTop = log.scrollHeight;
  return node;
}

/** Add a transcript entry. `kind` is `bot`, `you`, or `err`. */
function addMsg(kind, text, code) {
  const el = document.createElement("div");
  el.className = `msg ${kind}`;
  const p = document.createElement("p");
  p.textContent = text;
  el.append(p);
  if (code) {
    const c = document.createElement("code");
    c.textContent = code;
    el.append(c);
  }
  return append(el);
}

/** The screen as the agent saw it. Starts small; click to enlarge. */
function addShot(b64) {
  const box = document.createElement("div");
  box.className = "shot small";
  const img = document.createElement("img");
  img.src = `data:image/jpeg;base64,${b64}`;
  img.alt = "what the agent sees";
  box.append(img);
  box.addEventListener("click", () => box.classList.toggle("small"));
  return append(box);
}

function addStep(n, total) {
  const el = document.createElement("div");
  el.className = "step";
  el.textContent = `step ${n}/${total}`;
  return append(el);
}

/** Replace the last entry's text — used so "Thinking…" does not stack up. */
let transient = null;
function setTransient(text) {
  if (!transient || !transient.isConnected) transient = addMsg("bot", text);
  else transient.querySelector("p").textContent = text;
}

/* ── model picker ─────────────────────────────────────────────────────
   Only meaningful on OpenRouter: that provider's whole point is the choice, and the list
   comes from its live catalogue rather than anything hardcoded here. Rust filters it to
   models this loop can actually drive and then to the best from each company — see
   `openrouter::models`. */

const modelSel = el("modelsel");
const modelRow = el("model-row");

let provider = "anthropic";
let models = [];
let currentModel = "";

/** `Claude Opus 5 · $5.00/M · 1000k` — name, what it costs, how much it can hold. */
function modelLabel(m) {
  const ctx =
    m.context_length >= 1000
      ? `${Math.round(m.context_length / 1000)}k`
      : String(m.context_length);
  return `${m.name} · ${m.prompt_price}/M · ${ctx}`;
}

function fillModels() {
  modelSel.innerHTML = "";
  for (const m of models) modelSel.append(new Option(modelLabel(m), m.id));

  // A model set by hand in `.env`, or one that has since dropped off the per-company list,
  // is kept rather than silently swapped for something else. Same reasoning as the
  // microphone picker keeping a device that is currently unplugged: quietly changing a
  // saved choice is worse than showing one that looks unusual.
  if (currentModel && !models.some((m) => m.id === currentModel)) {
    modelSel.append(new Option(`${currentModel} — set by hand`, currentModel));
  }
  modelSel.value = currentModel || models[0]?.id || "";
  currentModel = modelSel.value;
}

async function loadModels() {
  modelSel.innerHTML = "";
  modelSel.append(new Option("Loading models…", ""));
  modelSel.disabled = true;
  try {
    models = await invoke("list_models");
    fillModels();
  } catch (e) {
    // Said out loud rather than left as an empty dropdown, which looks identical to
    // "OpenRouter has no models" when the real answer is almost always a bad key.
    models = [];
    modelSel.innerHTML = "";
    modelSel.append(new Option(currentModel || "Couldn't load models", currentModel || ""));
    addMsg("err", `Couldn't load the model list — ${e}`);
  } finally {
    modelSel.disabled = running;
  }
}

/** Show the picker for OpenRouter, hide it otherwise, and resize to match. */
async function syncModelRow({ reload = false } = {}) {
  const on = provider === "openrouter";
  modelRow.hidden = !on;
  await resizeMain();
  if (on && (reload || !models.length)) await loadModels();
}

modelSel.addEventListener("change", async () => {
  if (!modelSel.value) return;
  currentModel = modelSel.value;
  // Persisted immediately — there is no Save button out here, and a choice that reverted
  // on restart would read as the picker not working.
  try {
    await invoke("select_model", { model: currentModel });
  } catch (e) {
    addMsg("err", String(e));
  }
});

/** Disable what cannot be used mid-run, and reflect it in the pill. */
function syncBusy() {
  card.classList.toggle("busy", running);
  goal.disabled = running;
  mic.disabled = running;
  dry.disabled = running;
  // Swapping the model mid-run would not take effect until the next one, so the control
  // says so by being unavailable rather than appearing to work.
  modelSel.disabled = running;
  goal.placeholder = running ? "running…" : "Tell me what to do";
  // Stop replaces Go rather than sitting beside it: only one of them is ever the
  // action you want, and the row has no space to spare.
  go.hidden = running;
  stop.hidden = !running;
  go.disabled = running || goal.value.trim() === "";
}

function syncMode() {
  dry.classList.toggle("is-on", dryRun);
  dry.setAttribute("aria-checked", String(dryRun));
  mode.textContent = dryRun ? "Dry run" : "LIVE";
  mode.classList.toggle("live", !dryRun);
  // Only surfaced when it matters — in dry run nothing is being driven, so a failsafe
  // reminder would just be noise.
  note.textContent = dryRun ? "won't touch your mouse" : "failsafe: cursor to top-left";
}

async function start() {
  const text = goal.value.trim();
  if (!text || running) return;

  running = true;
  goal.value = "";
  addMsg("you", text);
  setMood("thinking");
  transient = null;
  setTransient("On it…");
  statusPill.textContent = "starting";
  syncBusy();
  await growForChat();

  try {
    // The model travels with the request so the run uses exactly what the picker shows,
    // rather than whatever the last save happened to land in `.env`.
    await invoke("start_run", {
      goal: text,
      dryRun,
      model: provider === "openrouter" ? currentModel || null : null,
    });
  } catch (e) {
    running = false;
    setMood("sad");
    addMsg("err", String(e));
    statusPill.textContent = "ready";
    syncBusy();
  }
}

listen("progress", ({ payload }) => {
  switch (payload.type) {
    case "Step":
      statusPill.textContent = `step ${payload.n}/${payload.total}`;
      addStep(payload.n, payload.total);
      transient = null;
      break;
    case "Frame":
      addShot(payload.image);
      break;
    // Thinking replaces itself instead of appending: it happens every step and would
    // otherwise be most of the transcript.
    case "Thinking":
      setMood("thinking");
      setTransient("Thinking…");
      break;
    case "Acting":
      setMood("acting");
      transient = null;
      // `done` is immediately followed by a Finished carrying the same message, so showing
      // the action too prints the outcome twice.
      if (!payload.action.startsWith("done")) {
        addMsg("bot", payload.reason ?? payload.action, payload.action);
      }
      break;
    case "Note":
      addMsg("bot", payload[0]);
      break;
    case "Escalated":
      addMsg("bot", `Thinking harder (${payload.model}).`);
      break;
    // Only the outcomes are spoken. Narrating every step would talk over whatever the
    // agent is driving, which is the one thing a voice assistant here must not do.
    case "Finished": {
      setMood(payload.success ? "happy" : "sad");
      transient = null;
      const text = payload.message || "Done.";
      addMsg(payload.success ? "bot" : "err", text);
      speakOut(text);
      break;
    }
    case "Stopped":
      setMood("sad");
      transient = null;
      addMsg("err", payload[0]);
      speakOut(payload[0]);
      break;
  }
});

listen("run-ended", () => {
  running = false;
  statusPill.textContent = "ready";
  syncBusy();
});

goal.addEventListener("input", syncBusy);
goal.addEventListener("keydown", (e) => {
  if (e.key === "Enter") start();
});
go.addEventListener("click", start);

dry.addEventListener("click", () => {
  if (running) return;
  dryRun = !dryRun;
  syncMode();
});

/* ── voice ────────────────────────────────────────────────────────── */

let voiceOut = false;
let voiceEngine = "";
let voiceId = "";
let hasVoiceKey = false;
let player = null;

/**
 * Say `text` with `engine`.
 *
 * ElevenLabs is strictly opt-in: it runs only when it was chosen *and* a key exists.
 * Every other path — including "chose ElevenLabs, no key" and "ElevenLabs errored" — ends
 * at the webview's own `speechSynthesis`, which costs nothing and needs no network. Losing
 * the answer because a quota ran out would be worse than losing the nicer voice.
 */
async function playVoice(text, engine, { strict = false, voice } = {}) {
  if (!text) return "none";
  if (engine === "elevenlabs" && hasVoiceKey) {
    try {
      const b64 = await invoke("speak", { text, voiceId: voice ?? voiceId ?? null });
      player?.pause();
      player = new Audio(`data:audio/mpeg;base64,${b64}`);
      await player.play();
      return "elevenlabs";
    } catch (e) {
      // Mid-run, fall through quietly — but `strict` (the Test button) rethrows, because
      // a silent fallback is exactly why "I set ElevenLabs and still hear the system
      // voice" is so hard to diagnose.
      if (strict) throw e;
      console.error("elevenlabs tts failed, using system voice", e);
    }
  }
  speechSynthesis.cancel();
  speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  return "system";
}

/** Speak `text`, if the user asked for spoken replies. */
const speakOut = (text) => (voiceOut ? playVoice(text, voiceEngine) : undefined);

/** Pause that ends a phrase in hands-free mode. Long enough to survive a mid-sentence beat. */
const SILENCE_MS = 1400;
/** Hard ceiling, so a stuck-open mic cannot record forever. */
const MAX_LISTEN_S = 30;
/** Press longer than this and it is a hold; shorter and it is a tap. */
const HOLD_MS = 350;

let micPoll = null;
let pressAt = 0;
let starting = null;
/** True when the mic was tapped rather than held, so silence should end the phrase. */
let handsFree = false;

function stopPolling() {
  clearInterval(micPoll);
  micPoll = null;
}

/** Stop capture, transcribe, and run it. */
async function finishListening() {
  stopPolling();
  listening = false;
  mic.classList.remove("listening");
  setTransient("Transcribing…");
  try {
    const heard = await invoke("stop_listening");
    if (!heard) {
      setTransient("Didn't catch that — try again.");
      return;
    }
    goal.value = heard;
    syncBusy();
    // Sent automatically, as asked. The transcript stays visible in the field and Stop is
    // one click away — which matters, because a misheard goal here drives the real mouse.
    await start();
  } catch (e) {
    addMsg("err", String(e));
  }
}

async function beginListening() {
  try {
    await invoke("start_listening");
    listening = true;
    mic.classList.add("listening");
    setTransient("Listening…");

    micPoll = setInterval(async () => {
      const s = await invoke("mic_state").catch(() => null);
      if (!s?.recording) return stopPolling();

      // Silence ends the phrase only when hands-free, and only once something has actually
      // been said — otherwise the gap between tapping and starting to speak ends it at once.
      if (handsFree && s.spoke && s.silent_ms > SILENCE_MS) return finishListening();
      if (s.seconds > MAX_LISTEN_S) return finishListening();

      // A live level is the one bit of feedback that matters while recording: whether it
      // can hear you at all.
      const bar = "▁▂▃▄▅▆▇█"[Math.min(7, Math.floor(s.level / 12))];
      setTransient(
        s.spoke
          ? `Listening… ${bar}`
          : handsFree
            ? "Listening… say what you want done."
            : "Listening… release to send.",
      );
    }, 200);
  } catch (e) {
    addMsg("err", String(e));
  }
}

// Two ways in, because neither alone fits every room. **Hold** to talk and release to send
// — deterministic, and immune to a silence threshold that does not suit your microphone.
// **Tap** to go hands-free, where a pause ends the phrase. Both send automatically.
mic.addEventListener("pointerdown", async (e) => {
  if (listening) {
    // Tapping again is how you stop a hands-free session early. Mid-hold it must do
    // nothing: a synthetic or repeated pointerdown would otherwise end the recording
    // before the user has released the button, sending a half-caught phrase.
    if (handsFree) await finishListening();
    return;
  }

  // Check before recording, not after. Letting someone speak a whole sentence and *then*
  // telling them transcription is unavailable wastes the one thing they can't redo.
  if (!hasVoiceKey) {
    addMsg("err", "Add an ElevenLabs key to use the microphone.");
    // Point at the *voice* key. `openSettings(reason)` annotates the Anthropic field, which
    // is the wrong one to highlight when it is the microphone that is unavailable.
    await openSettings();
    vkeyState.classList.add("warn");
    vkeyInput.focus();
    return;
  }

  // Capture the pointer so releasing off the button still ends the hold, rather than
  // stranding the recording open because the cursor drifted a few pixels.
  mic.setPointerCapture?.(e.pointerId);
  pressAt = performance.now();
  handsFree = false;
  starting = beginListening();
  await starting;
});

mic.addEventListener("pointerup", async () => {
  if (!pressAt) return;
  const held = performance.now() - pressAt;
  pressAt = 0;
  // The press may have been released before capture finished opening the device.
  await starting;
  if (!listening) return;

  if (held >= HOLD_MS) {
    await finishListening();
  } else {
    handsFree = true;
    setTransient("Listening… say what you want done.");
  }
});

stop.addEventListener("click", async () => {
  await invoke("stop_run");
  // The loop checks between steps, so honesty about the delay beats a UI that looks hung.
  setTransient("Stopping… finishing the current step.");
});

// Closing is handled Rust-side (it releases any keys the run left held down), so this
// only has to ask.
el("close").addEventListener("click", () => getCurrentWindow().close());

/* ── settings ─────────────────────────────────────────────────────── */

const [keyInput, keyState, reveal] = [el("key"), el("key-state"), el("reveal")];
const [modelInput, stepsInput, effortInput] = [el("model"), el("steps"), el("effort")];
const [providerSel, orkeyInput, orreveal] = [el("provider"), el("orkey"), el("orreveal")];
/** Last-loaded settings, so switching provider in the pane can restore the other's model. */
let loaded = {};
const [saveBtn, savedNote, envPath] = [el("save"), el("saved"), el("env-path")];
const [vkeyInput, vkeyState, vreveal] = [el("vkey"), el("vkey-state"), el("vreveal")];
const [micSel, voutBtn, vengineSel] = [el("micsel"), el("vout"), el("vengine")];
const vvoiceSel = el("vvoice");

function syncVoiceOut() {
  voutBtn.classList.toggle("is-on", voiceOut);
  voutBtn.setAttribute("aria-checked", String(voiceOut));
}

async function fillMicrophones(selected) {
  const devices = await invoke("list_microphones").catch(() => []);
  micSel.innerHTML = "";
  const def = document.createElement("option");
  def.value = "";
  def.textContent = "System default";
  micSel.append(def);
  for (const name of devices) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    micSel.append(opt);
  }
  // A device saved earlier may be unplugged now; keep showing it so saving does not
  // silently reset the choice just because the headset is in a drawer.
  if (selected && !devices.includes(selected)) {
    const gone = document.createElement("option");
    gone.value = selected;
    gone.textContent = `${selected} (not connected)`;
    micSel.append(gone);
  }
  micSel.value = selected ?? "";
}

/**
 * Show the fields belonging to the selected provider.
 *
 * The Model box edits whichever provider is active, because the two id namespaces do not
 * overlap — `claude-opus-5` is a 404 on OpenRouter and `anthropic/claude-sonnet-4.5` is a
 * 404 on Anthropic. One shared box would break the model every time the provider changed.
 */
function syncProviderFields() {
  const or = providerSel.value === "openrouter";
  el("key-anthropic").hidden = or;
  el("key-openrouter").hidden = !or;

  modelInput.value = or ? (loaded.openrouter_model ?? "") : (loaded.model ?? "");
  modelInput.placeholder = or ? "anthropic/claude-sonnet-4.5" : "claude-opus-5";

  const has = or ? loaded.has_openrouter_key : loaded.has_credential;
  const hint = or ? loaded.openrouter_key_hint : loaded.key_hint;
  keyState.textContent = has
    ? `A key is set (${hint}). Leave blank to keep it.`
    : or
      ? "No OpenRouter key yet — get one at openrouter.ai/keys."
      : "No key set yet — paste one to get started.";
  keyState.classList.toggle("warn", !has);
}

async function openSettings(reason) {
  const s = await invoke("load_settings");
  loaded = s;
  keyInput.value = "";
  orkeyInput.value = "";
  providerSel.value = s.provider;
  stepsInput.value = s.max_steps;
  effortInput.value = s.effort;
  envPath.textContent = `Saved to ${s.env_path}`;
  savedNote.textContent = "";

  // Fills the model box and the key hint for whichever provider is selected.
  syncProviderFields();
  if (reason) {
    keyState.textContent = reason;
    keyState.classList.add("warn");
  }

  vkeyInput.value = "";
  hasVoiceKey = s.has_voice_key;
  voiceOut = s.voice_out;
  voiceEngine = s.voice_engine;
  voiceId = s.voice_id;
  vengineSel.value = s.voice_engine;
  vvoiceSel.value = s.voice_id;
  await syncVoicePicker();
  vkeyState.textContent = s.has_voice_key
    ? `Set (${s.voice_key_hint}). Leave blank to keep it.`
    : "No key = microphone off. Spoken replies work without one.";
  // Cleared on every open; the mic handler re-adds it when it is the reason we are here.
  vkeyState.classList.remove("warn");
  syncVoiceOut();
  await fillMicrophones(s.mic);

  panel.hidden = false;
  // The voice picker is only present for ElevenLabs, so the pane has two heights rather
  // than one with a hole in it.
  await resize(el("voice-pick").hidden ? SIZE.settings : SIZE.settingsVoice);
}

async function closeSettings() {
  panel.hidden = true;
  await resizeMain();
}

providerSel.addEventListener("change", syncProviderFields);

saveBtn.addEventListener("click", async () => {
  const or = providerSel.value === "openrouter";
  try {
    await invoke("save_settings", {
      patch: {
        api_key: keyInput.value,
        // The Model box edits the active provider; the other one's model is written back
        // unchanged so switching provider and saving does not wipe it.
        model: or ? (loaded.model ?? "") : modelInput.value,
        openrouter_model: or ? modelInput.value : (loaded.openrouter_model ?? ""),
        provider: providerSel.value,
        openrouter_key: orkeyInput.value,
        max_steps: stepsInput.value,
        effort: effortInput.value,
        voice_key: vkeyInput.value,
        voice_out: voiceOut,
        voice_engine: vengineSel.value,
        voice_id: vvoiceSel.value,
        mic: micSel.value,
      },
    });
    voiceEngine = vengineSel.value;
    voiceId = vvoiceSel.value;
    if (vkeyInput.value.trim()) hasVoiceKey = true;

    // The chat bar reflects the saved provider straight away. A new key means the model
    // list has to be fetched again — the old one was fetched with a key that may not have
    // worked, or with no key at all.
    const switched = provider !== providerSel.value;
    provider = providerSel.value;
    if (or) currentModel = modelInput.value.trim() || currentModel;
    await syncModelRow({ reload: switched || orkeyInput.value.trim() !== "" });

    savedNote.textContent = "Saved.";
    keyInput.value = "";
    orkeyInput.value = "";
    vkeyInput.value = "";
    setTimeout(closeSettings, 450);
  } catch (e) {
    savedNote.textContent = String(e);
  }
});

const toggleReveal = (input) => () => {
  input.type = input.type === "password" ? "text" : "password";
};
reveal.addEventListener("click", toggleReveal(keyInput));
orreveal.addEventListener("click", toggleReveal(orkeyInput));
vreveal.addEventListener("click", toggleReveal(vkeyInput));

voutBtn.addEventListener("click", () => {
  voiceOut = !voiceOut;
  syncVoiceOut();
});

// Hearing it is the only way to judge it. Uses the dropdown's current values rather than
// the saved ones, so you can audition a voice before committing to it — and reports which
// engine actually spoke, so a silent fallback cannot masquerade as success.
el("vtest").addEventListener("click", async () => {
  savedNote.textContent = "Speaking…";
  try {
    const used = await playVoice("Autopilot mode ready.", vengineSel.value, {
      strict: true,
      voice: vvoiceSel.value,
    });
    savedNote.textContent =
      used === "elevenlabs" ? "Played with ElevenLabs." : "Played with the system voice.";
  } catch (e) {
    savedNote.textContent = String(e);
  }
});

/** Show the voice picker only for ElevenLabs, and fill it from the account. */
async function syncVoicePicker() {
  const on = vengineSel.value === "elevenlabs" && hasVoiceKey;
  el("voice-pick").hidden = !on;
  if (!on || vvoiceSel.dataset.filled === "yes") return;

  const voices = await invoke("list_voices").catch(() => []);
  const chosen = vvoiceSel.value || voiceId;
  vvoiceSel.innerHTML = "";
  const def = document.createElement("option");
  def.value = "";
  def.textContent = voices.length ? "Default" : "Default (couldn't load voices)";
  vvoiceSel.append(def);
  for (const [id, name] of voices) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = name;
    vvoiceSel.append(opt);
  }
  vvoiceSel.value = chosen;
  vvoiceSel.dataset.filled = "yes";
}

vengineSel.addEventListener("change", syncVoicePicker);

// Rust-side so the URL stays a compile-time constant; a plain <a> would navigate the app's
// own webview away from the UI. Confirm either way — this window is always-on-top, so the
// browser often opens *behind* it and a silent success looks like a dead button.
el("repo").addEventListener("click", () => {
  invoke("open_repo")
    .then(() => {
      savedNote.textContent = "Opened in your browser.";
    })
    .catch((e) => {
      savedNote.textContent = String(e);
    });
});

el("settings").addEventListener("click", () => openSettings());
el("panel-close").addEventListener("click", closeSettings);

syncMode();
syncBusy();

// Voice preferences have to be known before the first run finishes, not only after the
// pane has been opened once — otherwise the first reply is silent whatever the setting says.
// The provider matters just as early: the model picker is part of the main view, so it has
// to be right before the first goal is typed, not after a trip through settings.
invoke("load_settings").then(async (s) => {
  loaded = s;
  voiceOut = s.voice_out;
  voiceEngine = s.voice_engine;
  voiceId = s.voice_id;
  hasVoiceKey = s.has_voice_key;

  provider = s.provider;
  currentModel = s.openrouter_model;
  await syncModelRow();

  // Surface a missing credential up front rather than letting the first run fail with it.
  if (!s.has_credential) {
    const which = provider === "openrouter" ? "OpenRouter" : "Anthropic";
    addMsg("err", `Add your ${which} API key to get started.`);
    openSettings(`No ${which} key set yet — paste one to get started.`);
  }
});
