"""Watch a race in the terminal: the live F1 feed, or an archived session replayed as if live.

Usage:
  python -m src.live --live                 # follow the live feed (and record it)
  python -m src.live --year 2026 --meeting Azerbaijan --speed 30
  python -m src.live --year 2024 --meeting Miami --speed 20 --from-lap 26
"""

import argparse
import asyncio

from src.livetiming.archive import ArchiveSession, find_session
from src.livetiming.client import LiveTimingClient, Recorder, recording_folder
from src.livetiming.monitor import RaceMonitor, pump, replay
from src.llm.transcribe import RadioTranscriber

COLOURS = {"SAFETY_CAR": "\033[33m", "VSC": "\033[33m", "RED_FLAG": "\033[31m", "GREEN": "\033[32m"}
RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"
# Routine marshalling messages that drown out the ones that matter.
NOISE = ("WAVED BLUE FLAG", "CLEAR IN TRACK SECTOR", "YELLOW IN TRACK SECTOR", "TRACK LIMITS")
TYRE = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}


def _lap_line(event: dict) -> str:
    colour = COLOURS.get(event["status"], "")
    top = []
    for d in event["drivers"][:6]:
        gap = (
            "LEAD" if d["position"] == 1 else (f"+{d['gap']:.1f}" if d["gap"] is not None else "--")
        )
        top.append(f"{d['tla']} {gap} {TYRE.get(d['compound'] or '', '?')}{d['tyre_age'] or 0}")
    return (
        f"{BOLD}Lap {event['lap']}/{event['total_laps']}{RESET} {colour}{event['status']}{RESET}  "
        + "  ".join(top)
    )


def _print_pit_calls(report: dict) -> None:
    print(
        f"\n{BOLD}\033[33m=== {report['track_status']} on lap {report['lap']} — PIT WALL CALLS "
        f"({report['laps_remaining']} laps left, stop costs ~{report['pit_loss_now_s']}s "
        f"vs ~{report['green_pit_loss_s']}s green) ==={RESET}"
    )
    for c in report["calls"]:
        colour = "\033[35m" if c["call"] == "PIT" else DIM if c["call"] != "STAY OUT" else ""
        tyre = f"{c['compound'] or '?'}/{c['tyre_age_laps']}"
        print(
            f"  P{c['position'] or '-':<2} {c['tla']:3} {colour}{c['call']:8}{RESET} {tyre:10} {c['reason']}"
        )
    print(f"{DIM}  Assumptions: " + " ".join(report["assumptions"]) + f"{RESET}\n")


async def _live_events(transcriber, record: bool):
    client = LiveTimingClient()
    recorder = Recorder(recording_folder()) if record else None
    print(
        f"{DIM}Following the live F1 feed ({len(client.topics)} topics)"
        + (f", recording to {recorder.folder}" if recorder else "")
        + ". Between sessions this shows the last session's final state.{RESET}".replace(
            "{RESET}", RESET
        )
    )

    async def source():
        async for message in client.messages():
            if recorder:
                recorder.write(message)
            yield message, False

    try:
        async for event in pump(
            source(), RaceMonitor(), transcriber=transcriber, load_circuit_live=True
        ):
            yield event
    finally:
        if recorder:
            recorder.close()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="follow the live F1 timing feed")
    ap.add_argument("--no-record", action="store_true", help="--live: don't record to disk")
    ap.add_argument("--year", type=int, help="replay: season")
    ap.add_argument("--meeting", help='replay: e.g. "Azerbaijan", "Miami"')
    ap.add_argument("--session", default="Race")
    ap.add_argument("--speed", type=float, default=20.0, help="replay speed multiplier")
    ap.add_argument("--from-lap", type=int, default=None, help="fast-forward to this lap")
    ap.add_argument("--no-transcribe", action="store_true", help="skip team radio transcription")
    args = ap.parse_args()
    transcriber = (
        None if args.no_transcribe or not RadioTranscriber.available() else RadioTranscriber()
    )

    if transcriber is None:
        print(f"{DIM}Team radio transcription is off (--no-transcribe or no GROQ_API_KEY).{RESET}")
    if args.live:
        events = _live_events(transcriber, record=not args.no_record)
    else:
        if not (args.year and args.meeting):
            ap.error("--year and --meeting are required for a replay (or use --live)")
        session = ArchiveSession(find_session(args.year, args.meeting, args.session))
        print(f"{DIM}Replaying {session.path} at {args.speed}x (downloads on first run)...{RESET}")
        events = replay(session, speed=args.speed, from_lap=args.from_lap, transcriber=transcriber)
    last_lap = None
    async for event in events:
        kind = event["type"]
        if kind == "session":
            print(
                f"{BOLD}{event['meeting']} — {event['session']}, {event['total_laps']} laps{RESET}"
            )
        elif kind == "snapshot" and event["lap"] != last_lap:
            last_lap = event["lap"]
            print(_lap_line(event))
        elif kind == "track_status":
            colour = COLOURS.get(event["status"], "")
            print(f"  {colour}▶ TRACK STATUS: {event['status']}{RESET}")
        elif kind == "race_control":
            if any(noise in event["message"] for noise in NOISE):
                continue
            print(f"  {DIM}[RC lap {event['lap']}] {event['message']}{RESET}")
        elif kind == "radio":
            if event.get("text") is not None:
                text = f'"{event["text"] or "(no speech)"}"'
            else:
                text = "(transcribing...)" if transcriber else event["url"]
            print(f"  📻 {BOLD}{event['tla']}{RESET} team radio: {text}")
        elif kind == "radio_transcript":
            print(f'  📻 {BOLD}{event["tla"]}{RESET}: "{event["text"] or "(no speech)"}"')
        elif kind == "pit_calls":
            _print_pit_calls(event["report"])
        elif kind == "end":
            print(f"{BOLD}Chequered flag.{RESET}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
