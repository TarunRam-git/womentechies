function isProcAlive(proc) {
  return !!proc && !proc.killed && proc.exitCode === null;
}

export default function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const g = globalThis;
  const proc = g.__direction1Proc;
  if (!isProcAlive(proc)) {
    g.__direction1Proc = null;
    g.__direction1LastExit = g.__direction1LastExit || null;
    res.status(200).json({ ok: true, running: false });
    return;
  }

  try {
    proc.kill("SIGTERM");
  } catch {
  }

  g.__direction1Proc = null;
  g.__direction1LastExit = { code: null, signal: "SIGTERM", at: Date.now() };
  res.status(200).json({ ok: true, running: false });
}
