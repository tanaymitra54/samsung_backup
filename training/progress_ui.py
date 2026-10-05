import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Pipeline progress</title>
<style>
  :root { color-scheme: light; }
  body { margin: 0; font-family: "Segoe UI", sans-serif; background: #f4f6f8; color: #1c2430; }
  header { background: #0f2744; color: white; padding: 20px 28px; }
  header h1 { margin: 0 0 6px; font-size: 22px; font-weight: 650; }
  header p { margin: 0; opacity: 0.85; }
  main { padding: 22px 28px 40px; }
  .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
  .card { background: white; border-radius: 12px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(16,24,40,.08); }
  .card span { display: block; color: #5c6b7a; font-size: 13px; }
  .card strong { display: block; margin-top: 4px; font-size: 28px; }
  .bar-wrap { margin: 18px 0; background: white; border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(16,24,40,.08); }
  .bar { height: 22px; background: #e6ebf1; border-radius: 999px; overflow: hidden; }
  .bar > div { height: 100%; width: 0; background: #1f7a4d; transition: width .3s ease; }
  .meta { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; }
  table { width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #e8edf2; font-size: 13px; vertical-align: top; }
  th { background: #eef3f8; }
  .ok { color: #1f7a4d; font-weight: 650; }
  .bad { color: #a33b3b; font-weight: 650; }
  .wait { color: #8a6a12; font-weight: 650; }
</style>
</head>
<body>
<header>
  <h1>QUBO pipeline on NVIDIA GPU</h1>
  <p id="gpu">Waiting for the run to start</p>
</header>
<main>
  <div class="cards">
    <div class="card"><span>Stage</span><strong id="stage">—</strong></div>
    <div class="card"><span>Finished</span><strong id="done">0</strong></div>
    <div class="card"><span>Remaining</span><strong id="left">0</strong></div>
    <div class="card"><span>Accuracy</span><strong id="acc">—</strong></div>
  </div>
  <div class="bar-wrap">
    <div class="meta"><span id="count">0 / 0</span><span id="eta">ETA —</span></div>
    <div class="bar"><div id="fill"></div></div>
    <p id="message"></p>
  </div>
  <table>
    <thead><tr><th>#</th><th>Result</th><th>Gold</th><th>Prediction</th><th>Question</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
</main>
<script>
async function refresh() {
  const data = await (await fetch("/status")).json();
  document.getElementById("gpu").textContent = data.gpu || "";
  document.getElementById("stage").textContent = data.stage || "—";
  document.getElementById("done").textContent = data.done ?? 0;
  document.getElementById("left").textContent = data.remaining ?? 0;
  document.getElementById("acc").textContent = data.accuracy == null ? "—" : (data.accuracy * 100).toFixed(1) + "%";
  const total = data.total || 0;
  const done = data.done || 0;
  document.getElementById("count").textContent = done + " / " + total;
  document.getElementById("eta").textContent = data.eta || "ETA —";
  document.getElementById("fill").style.width = (total ? Math.min(100, 100 * done / total) : 0) + "%";
  document.getElementById("message").textContent = data.message || "";
  const esc = (value) => String(value ?? "").replace(/[&<>"]/g, (ch) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[ch]));
  const body = document.getElementById("rows");
  body.innerHTML = (data.rows || []).map(row => {
    const mark = row.result || "—";
    const cls = mark === "CORRECT" ? "ok" : mark === "WRONG" ? "bad" : "wait";
    return `<tr><td>${esc(row.index)}</td><td class="${cls}">${esc(mark)}</td><td>${esc(row.gold)}</td><td>${esc(row.prediction)}</td><td>${esc(row.question)}</td></tr>`;
  }).join("");
}
setInterval(() => refresh().catch(() => {}), 1000);
refresh().catch(() => {});
</script>
</body>
</html>
"""


class PipelineProgress:
    def __init__(self, port: int = 8765):
        self.lock = threading.Lock()
        self.started = time.time()
        self.stage_started = time.time()
        self.state = {
            "gpu": "",
            "stage": "Starting",
            "done": 0,
            "total": 0,
            "remaining": 0,
            "correct": 0,
            "graded": 0,
            "accuracy": None,
            "eta": "ETA —",
            "message": "",
            "rows": [],
        }
        self.port = self._serve(port)
        print(f"Progress GUI: http://127.0.0.1:{self.port}", flush=True)

    def _serve(self, port: int) -> int:
        progress = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                if self.path.startswith("/status"):
                    body = json.dumps(progress.snapshot()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                page = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(page)

        last_error = None
        for candidate in range(port, port + 20):
            try:
                server = ThreadingHTTPServer(("0.0.0.0", candidate), Handler)
            except OSError as exc:
                last_error = exc
                continue
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self._server = server
            return candidate
        raise RuntimeError(f"Could not start the progress GUI: {last_error}")

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state))

    def set_gpu(self, description: str):
        with self.lock:
            self.state["gpu"] = description
        print(description, flush=True)

    def start_stage(self, stage: str, total: int, message: str = ""):
        with self.lock:
            self.state["stage"] = stage
            self.state["done"] = 0
            self.state["total"] = total
            self.state["remaining"] = total
            self.state["correct"] = 0
            self.state["graded"] = 0
            self.state["accuracy"] = None
            self.state["rows"] = []
            self.state["message"] = message
            self.state["eta"] = "ETA —"
            self.stage_started = time.time()
        print(f"\n{stage}: 0/{total} finished, {total} remaining", flush=True)
        if message:
            print(message, flush=True)

    def note(self, message: str):
        with self.lock:
            self.state["message"] = message
        print(message, flush=True)

    def _eta(self, done: int, total: int) -> str:
        elapsed = time.time() - self.stage_started
        if done <= 0 or total <= done:
            return "ETA —" if done < total else "done"
        seconds = int(elapsed / done * (total - done))
        minutes, sec = divmod(seconds, 60)
        return f"ETA {minutes}m {sec}s"

    def question_done(self, row: dict):
        with self.lock:
            self.state["done"] += 1
            done = self.state["done"]
            total = self.state["total"]
            self.state["remaining"] = max(0, total - done)
            result = row.get("result") or "SELECTED"
            if result in {"CORRECT", "WRONG"}:
                self.state["graded"] += 1
                self.state["correct"] += int(result == "CORRECT")
                graded = self.state["graded"]
                self.state["accuracy"] = self.state["correct"] / graded if graded else None
            self.state["eta"] = self._eta(done, total)
            self.state["rows"].append({
                "index": row.get("index", done),
                "result": result,
                "gold": row.get("gold", ""),
                "prediction": row.get("prediction", ""),
                "question": row.get("question", ""),
            })
            remaining = self.state["remaining"]
            eta = self.state["eta"]
        filled = int(24 * done / total) if total else 0
        bar = "#" * filled + "-" * (24 - filled)
        print(
            f"[{done}/{total}] {bar} remaining {remaining} | {eta} | {result} | "
            f"gold={row.get('gold', '')} pred={row.get('prediction', '') or '-'} | "
            f"{row.get('question', '')}",
            flush=True,
        )

    def train_step(self, step: int, total: int, loss: float | None = None):
        with self.lock:
            self.state["done"] = step
            self.state["total"] = total
            self.state["remaining"] = max(0, total - step)
            self.state["eta"] = self._eta(step, total)
            if loss is not None:
                self.state["message"] = f"QLoRA loss {loss:.4f}"
            done = self.state["done"]
            remaining = self.state["remaining"]
            eta = self.state["eta"]
            message = self.state["message"]
        if step == 1 or step == total or step % 5 == 0:
            print(
                f"[train {done}/{total}] remaining {remaining} | {eta} | {message}",
                flush=True,
            )
