import fs from "fs";
import path from "path";
import { execFile, execFileSync } from "child_process";
import { randomUUID } from "crypto";

const POSE_DIR = path.join(process.cwd(), "data", "processed", "pose");
const PYTHON_CANDIDATES = [
  { cmd: "py", args: ["-3.12"] },
  { cmd: "python3", args: [] },
  { cmd: "python", args: [] }
];

function pickPython() {
  for (const candidate of PYTHON_CANDIDATES) {
    try {
      execFileSync(candidate.cmd, [...candidate.args, "--version"], { windowsHide: true, stdio: "pipe" });
      return candidate;
    } catch {
      // try next
    }
  }
  throw new Error("Python 3.12+ is required to render avatars. Install Python and ensure it is on PATH.");
}

function runPython(text, outPath) {
  return new Promise((resolve, reject) => {
    let python;
    try {
      python = pickPython();
    } catch (err) {
      reject(err);
      return;
    }

    const args = [
      ...python.args,
      "-m",
      "src.avatar_frontend_bridge",
      "--text",
      text,
      "--out",
      outPath,
    ];

    execFile(python.cmd, args, { cwd: process.cwd(), windowsHide: true }, (error, stdout, stderr) => {
      if (error) {
        reject(new Error((stderr || stdout || error.message || "python failed").trim()));
        return;
      }
      resolve(stdout.trim());
    });
  });
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const text = String(req.body?.text || "").trim();
  if (!text) {
    res.status(400).json({ error: "Missing text" });
    return;
  }

  if (!fs.existsSync(POSE_DIR)) {
    res.status(400).json({ ok: false, error: "Missing processed pose directory for src avatar pipeline." });
    return;
  }

  const outMotion = path.join(process.cwd(), "data", "processed", `frontend_avatar_motion_${Date.now()}_${randomUUID()}.json`);

  try {
    await runPython(text, outMotion);
    if (!fs.existsSync(outMotion)) {
      throw new Error("Avatar motion output not found");
    }
    const motion = JSON.parse(fs.readFileSync(outMotion, "utf-8"));
    res.status(200).json({ ok: true, motion });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message || "Avatar render failed" });
  } finally {
    try {
      if (fs.existsSync(outMotion)) fs.unlinkSync(outMotion);
    } catch {
    }
  }
}
