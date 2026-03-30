import fs from "fs";
import path from "path";
import { execFile } from "child_process";

const OUT_MOTION = path.join(process.cwd(), "data", "processed", "frontend_avatar_motion.json");

function runPython(text) {
  return new Promise((resolve, reject) => {
    const args = [
      "-3.12",
      "avatar_frontend_bridge.py",
      "--text",
      text,
      "--out",
      OUT_MOTION,
      "--dataset",
      path.join("data", "processed", "avatar_direction2_templates.npz"),
      "--labels",
      path.join("models", "avatar_direction2_labels.json"),
      "--blend-frames",
      "12"
    ];

    execFile("py", args, { cwd: process.cwd(), windowsHide: true }, (error, stdout, stderr) => {
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

  try {
    await runPython(text);
    if (!fs.existsSync(OUT_MOTION)) {
      throw new Error("Avatar motion output not found");
    }
    const motion = JSON.parse(fs.readFileSync(OUT_MOTION, "utf-8"));
    res.status(200).json({ ok: true, motion });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message || "Avatar render failed" });
  }
}
