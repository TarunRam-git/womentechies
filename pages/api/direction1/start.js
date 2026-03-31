import { spawn } from "child_process";
import fs from "fs";
import path from "path";

const STATE_FILE = path.join(process.cwd(), "data", "processed", "direction1_live.json");
const LOG_FILE = path.join(process.cwd(), "data", "processed", "direction1_live.log");

function isProcAlive(proc) {
  return !!proc && !proc.killed && proc.exitCode === null;
}

export default function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const g = globalThis;
  if (isProcAlive(g.__direction1Proc)) {
    res.status(200).json({ ok: true, running: true, pid: g.__direction1Proc.pid, stateFile: STATE_FILE, logFile: LOG_FILE });
    return;
  }

  const model = req.body?.model || "models/dynamic_lstm_10class_combined.onnx";
  const labels = req.body?.labels || "models/labels_10class_combined.json";
  const modelAbs = path.join(process.cwd(), model);
  const labelsAbs = path.join(process.cwd(), labels);
  if (!fs.existsSync(modelAbs)) {
    res.status(400).json({ ok: false, running: false, error: `Missing model file: ${model}` });
    return;
  }
  if (!fs.existsSync(labelsAbs)) {
    res.status(400).json({ ok: false, running: false, error: `Missing labels file: ${labels}` });
    return;
  }

  const args = [
    "-3.12",
    "app.py",
    "--model",
    model,
    "--labels",
    labels,
    "--out-json",
    STATE_FILE,
    "--headless",
    "--stable-frames",
    "6",
    "--stability-window",
    "8",
    "--consensus",
    "0.78",
    "--ema-alpha",
    "0.58",
    "--cooldown",
    "1.4",
    "--repeat-cooldown",
    "2.5"
  ];

  fs.mkdirSync(path.dirname(LOG_FILE), { recursive: true });
  fs.writeFileSync(LOG_FILE, "", "utf-8");

  const proc = spawn("py", args, {
    cwd: process.cwd(),
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true
  });

  const writeLog = (chunk) => {
    const text = String(chunk || "");
    if (!text) return;
    try {
      fs.appendFileSync(LOG_FILE, text, "utf-8");
    } catch {
    }
  };

  proc.stdout?.on("data", writeLog);
  proc.stderr?.on("data", writeLog);
  proc.on("error", (err) => {
    writeLog(`\n[spawn_error] ${err?.message || String(err)}\n`);
  });

  g.__direction1Proc = proc;
  proc.on("exit", (code, signal) => {
    g.__direction1LastExit = { code, signal, at: Date.now() };
    if (g.__direction1Proc === proc) {
      g.__direction1Proc = null;
    }
  });

  setTimeout(() => {
    if (!isProcAlive(proc)) {
      let logTail = "";
      try {
        const raw = fs.readFileSync(LOG_FILE, "utf-8");
        logTail = raw.length > 1200 ? raw.slice(-1200) : raw;
      } catch {
        logTail = "";
      }
      res.status(500).json({
        ok: false,
        running: false,
        error: logTail || "Direction-1 failed to start",
        stateFile: STATE_FILE,
        logFile: LOG_FILE,
      });
      return;
    }
    res.status(200).json({ ok: true, running: true, pid: proc.pid, stateFile: STATE_FILE, logFile: LOG_FILE });
  }, 800);
}
