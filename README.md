<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="Logos/logo-lockup-dark.png">
    <img src="Logos/logo-lockup.png" alt="autopilotmode" width="460">
  </picture>
</p>

<p align="center">
  <em>Tell your computer what to do. Watch it do it.</em>
</p>

<p align="center">
  <a href="https://autopilotmode.ai/"><strong>autopilotmode.ai</strong></a>
</p>

---

**autopilotmode** hands your mouse and keyboard to an AI that can see your screen.

You type or say a goal — *"open the Start menu"*, *"find the cheapest flight to Lisbon"*,
*"get in a car and drive along the road"* — and it works through it the way a person would:
look at the screen, decide the next single action, do it, look again.

There is no scripting, no selectors, no integration to write. If you can see it, it can
use it: any app, any game, any website, including ones that have no API at all.

> ⚠️ **This is unaudited software that controls your real mouse and keyboard.**
> Run it on a machine you're willing to have driven, watch it while it works, and start in
> **Dry run**. See [Safety](#safety).

---

## How it works

```
        ┌──────────────┐        ┌──────────────────┐        ┌──────────────┐
        │  look at the │───────▶│  decide the next │───────▶│   do it      │
        │    screen    │        │   single action  │        │              │
        └──────▲───────┘        └──────────────────┘        └──────┬───────┘
               │                                                   │
               └───────────────  look again  ──────────────────────┘
```

One screenshot, one decision, one action — then it checks its own work on the next frame.
That loop is the whole product. It keeps going until the goal is done, you stop it, or it
hits its step limit.

Because every step is grounded in a fresh screenshot, it recovers from things a script
can't: a dialog that appeared, a page that loaded slowly, a button that moved.

---

## What you get

### It shows you its work

The app is a running transcript — your goal, what it decided, the exact action it took,
and **the screenshot it was looking at when it decided**. Click any screenshot to enlarge
it.

That last part matters more than it sounds. `click @(840,412)` tells you nothing on its
own. Next to the screen it was aimed at, you can see instantly whether it understood the
page or misread it.

### You can talk to it

Hold the mic and speak, release to send. Or tap it once for hands-free, where a natural
pause ends the phrase. Either way what it heard lands in the box **for you to confirm**
before anything moves — a misheard word should be a typo, not a click somewhere you didn't
intend.

It can talk back too, if you want it to. Off by default.

### It stops when you say so

`Stop` while it's running. Close the window. Or throw your cursor into the top-left corner
of the screen and it aborts before its next action. Any of the three releases whatever it
was holding down.

---

## Getting started

**Build it yourself. It's the recommended way to run this** — and for software that takes
over your mouse and keyboard, you should want a build you made from source you can read.
That's the whole point of it being open.

```bash
git clone https://github.com/vishalch4466/autopilotmode
cd autopilotmode
cargo build --release
cargo run --release -p autopilotmode-desktop
```

You'll need [Rust](https://rustup.rs), a machine with a display (it drives the real cursor,
so it can't run headless), and an [Anthropic API key](https://console.anthropic.com). The
first build takes a few minutes; after that it's seconds.

On first launch it'll ask for your API key and save it. That's the whole setup.

<details>
<summary><b>Prebuilt Windows binary</b> — if you'd rather not build</summary>

<br>

[**Download the latest release →**](https://github.com/vishalch4466/autopilotmode/releases/latest)

Windows 10/11, 64-bit. No installer — download and run.

The binaries are **unsigned**, so SmartScreen will warn you ("Windows protected your PC").
That warning is correct to show for any unsigned binary off the internet. Each release
publishes SHA256 checksums so you can verify what you downloaded:

```powershell
Get-FileHash .\autopilotmode-desktop-v0.1.0-windows-x64.exe -Algorithm SHA256
```

If clicking through a security warning to run something that controls your keyboard feels
wrong — good instinct. Build from source instead.

</details>

Then type a goal, leave **Dry run** on for your first few, and watch what it decides to do
before letting it touch anything.

> **A Claude Pro/Max subscription does not pay for this.** That covers claude.ai and Claude
> Code. A program calling the API is separate, metered usage billed to your API key — the
> two run out independently.

---

## Command line

There's a terminal version too, for scripting and headless-ish use:

```bash
# Watch it plan without touching your input
cargo run --release -- --dry-run "search google for the weather"

# For real (asks you to confirm first)
cargo run --release -- "open a terminal and run 'ls'"
```

| Flag | What it does |
|------|--------------|
| `--dry-run` | Decide and print actions without sending any input |
| `--yes` | Skip the confirmation prompt |
| `--max-steps <N>` | Cap the number of actions (default 25) |
| `--model <ID>` | Use a different model for this run |
| `--wait <SECS>` | Pause before starting so you can focus the right window |
| `--fast` | Trade a little accuracy for a shorter loop |
| `--game` | Real-time preset for games — see below |
| `--selftest` | Move the cursor in a square to prove input works. No API key needed |

---

## Settings

Everything you'd want to change is in the gear menu — no config files to hand-edit.

| | |
|---|---|
| **API key** | Required. Stored locally, never leaves your machine except to Anthropic |
| **Model** | Swap in a faster or cheaper one whenever you like |
| **Max steps** | How long a single run may go before it gives up |
| **Effort** | How hard it thinks before each action |
| **Microphone** | Which input device to listen on |
| **Voice** | Whether it speaks its results, and in whose voice |

Voice input needs an [ElevenLabs](https://elevenlabs.io) key. Spoken *replies* don't —
those use your system voice for free, and ElevenLabs is an optional upgrade if you want it
to sound better.

---

## Games and real-time control

```bash
cargo run --release -- --game "get in a car and drive along the road"
```

A menu waits for you. A game does not, and that changes the rules:

- **It holds keys down across steps** rather than tapping them, so a car keeps moving while
  it looks and thinks. Tapping produces stop-start lurching.
- **It steers on a slightly stale picture.** The screen it's reacting to is a moment old by
  the time the action lands, so it corrects in small increments.
- **Keep the game focused and full-screen.** Input goes wherever focus is — click another
  window mid-run and a held key starts typing into *that* one instead.

**There's a floor to this.** Roughly one decision per second means a vehicle covers real
ground between corrections. Anything with a fail-state inside that window — a wall, a
cliff, a race you can lose — isn't reachable by an AI in the reflex loop. Cruising,
navigating, and exploring work well; twitch gameplay doesn't.

---

## Safety

A program that moves your mouse and types for you is genuinely powerful and can genuinely
misfire. What protects you:

- **Dry run is on by default.** Nothing that can seize your mouse the instant you press
  Enter should start out able to.
- **Failsafe corner.** Slam the cursor into the top-left of the screen and it aborts.
- **Stop button**, and closing the window — both release any keys still held down.
- **Step cap** bounds a runaway loop.
- **`Ctrl-C`** in the terminal version.
- It's told to avoid destructive and irreversible actions.

> ⚠️ **Treat that last one as a nudge, not a guarantee.** In testing, a model reached for
> "New Game" in a pause menu — one click from wiping a save. Keep destructive UI out of
> reach rather than trusting the instruction.

The **`unaudited`** tag in the app's header is there deliberately and stays there. This has
had no security review.

---

## Honest limits

- **It can't tell you what it doesn't know.** It reads the screen; it has no idea what's
  behind a menu it hasn't opened.
- **Small targets are hard.** It sees a downscaled screenshot, so fine detail costs
  accuracy. Bigger windows work better than dense ones.
- **Speed is bounded by thinking, not by anything you can tune.** Screenshot size and
  quality barely move it. Choosing a faster model is the lever that works.
- **It doesn't remember across runs.** Each goal starts fresh.

---

<p align="center">
  <a href="https://autopilotmode.ai/"><strong>autopilotmode.ai</strong></a>
</p>

<p align="center">
  <sub>Built by Vishal. If this is useful to you, a ⭐ on the repo is appreciated.</sub>
</p>
