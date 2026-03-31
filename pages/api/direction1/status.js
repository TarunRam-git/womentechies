import fs from "fs";
import path from "path";

const STATE_FILE = path.join(process.cwd(), "data", "processed", "direction1_live.json");
const LOG_FILE = path.join(process.cwd(), "data", "processed", "direction1_live.log");

function isProcAlive(proc) {
  return !!proc && !proc.killed && proc.exitCode === null;
}

export default function handler(req, res) {
  const proc = globalThis.__direction1Proc;
  const running = isProcAlive(proc);
  const lastExit = globalThis.__direction1LastExit || null;

  let state = null;
  if (fs.existsSync(STATE_FILE)) {
    try {
      const raw = fs.readFileSync(STATE_FILE, "utf-8");
      state = JSON.parse(raw);
    } catch {
      state = null;
    }
  }

  let logTail = "";
  if (fs.existsSync(LOG_FILE)) {
    try {
      const raw = fs.readFileSync(LOG_FILE, "utf-8");
      logTail = raw.length > 2000 ? raw.slice(-2000) : raw;
    } catch {
      logTail = "";
    }
  }

  res.status(200).json({
    running,
    pid: running ? proc.pid : null,
    stateFile: STATE_FILE,
    state,
    logFile: LOG_FILE,
    logTail,
    lastExit
  });
}
