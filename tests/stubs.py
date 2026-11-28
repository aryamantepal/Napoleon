"""Test doubles for LLM-backed agents. Nothing here touches the network."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

END_TURN = '{"type":"end_turn"}'


def legal_lines(messages: list[dict]) -> list[str]:
    """The legal-action lines of the newest prompt."""
    text = next(m["content"] for m in reversed(messages) if "Legal actions" in m["content"])
    return [ln for ln in text.splitlines() if ln.startswith('{"type"')]


class PickLegal:
    """A stand-in model: end the turn when allowed, else try each legal action once a turn."""

    def __init__(self) -> None:
        self.turn = None
        self.tried: set[str] = set()

    def __call__(self, messages: list[dict]) -> str:
        text = next(m["content"] for m in reversed(messages) if "Legal actions" in m["content"])
        turn = next(ln for ln in text.splitlines() if ln.startswith("Turn "))
        if turn != self.turn:
            self.turn, self.tried = turn, set()
        lines = legal_lines(messages)
        fresh = [ln for ln in lines if ln not in self.tried] or lines
        choice = END_TURN if END_TURN in lines else fresh[0]
        self.tried.add(choice)
        return json.dumps({"reasoning": "stub pick", "action": json.loads(choice)})


def pick_legal(messages: list[dict]) -> str:
    return PickLegal()(messages)


@contextmanager
def chat_server(reply: Callable[[list[dict]], str], status: int = 200):
    """A local OpenAI-compatible server. Yields (base_url, recorded requests)."""
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "auth": self.headers.get("Authorization"), **body})
            if status != 200:
                self.send_response(status)
                self.end_headers()
                self.wfile.write(b'{"error": "boom"}')
                return
            text = reply(body["messages"])
            message = text if isinstance(text, dict) else {"role": "assistant", "content": text}
            payload = {
                "choices": [{"message": message}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        server.shutdown()
        server.server_close()


class ReactPlayer:
    """A stand-in ReAct model: list the options once, then commit like ``PickLegal``."""

    def __init__(self) -> None:
        self.turn = None
        self.tried: set[str] = set()

    def __call__(self, messages: list[dict]) -> str:
        last = messages[-1]["content"]
        if not last.startswith("Observation"):
            turn = next(ln for ln in messages[1]["content"].splitlines() if ln.startswith("Turn "))
            if turn != self.turn:
                self.turn, self.tried = turn, set()
            return json.dumps({"thought": "look", "tool": "list_options", "args": {}})
        lines = [ln for ln in last.splitlines() if ln.startswith('{"type"')]
        fresh = [ln for ln in lines if ln not in self.tried] or lines
        choice = END_TURN if END_TURN in lines else fresh[0]
        self.tried.add(choice)
        return json.dumps({"thought": "pick", "action": json.loads(choice)})
