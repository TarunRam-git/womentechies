"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import io, { Socket } from "socket.io-client";

type Message = { from: string; text: string; ts: string };
type MotionFrame = {
  pose: number[][];
  left_hand: number[][];
  right_hand: number[][];
};
type MotionPayload = {
  fps: number;
  text: string;
  total_frames: number;
  frames: MotionFrame[];
};

const POSE_CONNECTIONS: Array<[number, number]> = [
  [0, 1],
  [0, 2], [2, 4],
  [1, 3], [3, 5],
  [0, 6], [1, 7],
  [6, 7]
];
const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];

export default function CommPage() {
  return (
    <Suspense fallback={<main className="comm-surface"><div className="page comm-page">Loading...</div></main>}>
      <CommPageContent />
    </Suspense>
  );
}

function CommPageContent() {
  const params = useSearchParams();
  const room = params?.get("room") || "room-001";
  const role = params?.get("role") || "sign";
  const name = params?.get("name") || (role === "speech" ? "Speech User" : "Sign User");
  const [messages, setMessages] = useState<Message[]>([
    { from: "System", text: "Welcome to Sign-Sync!", ts: new Date().toLocaleTimeString() }
  ]);
  const [input, setInput] = useState("");
  const socketRef = useRef<Socket | null>(null);
  const [direction1Running, setDirection1Running] = useState(false);
  const [direction1State, setDirection1State] = useState<any>(null);
  const [direction1Busy, setDirection1Busy] = useState(false);
  const [direction1Error, setDirection1Error] = useState("");
  const [avatarMotion, setAvatarMotion] = useState<MotionPayload | null>(null);
  const [avatarLoading, setAvatarLoading] = useState(false);
  const [avatarError, setAvatarError] = useState("");

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);
  const [micLevel, setMicLevel] = useState(0);
  const [mediaError, setMediaError] = useState("");
  const [camOn, setCamOn] = useState(true);
  const [micOn, setMicOn] = useState(true);
  const [mediaReady, setMediaReady] = useState(false);

  useEffect(() => {
    fetch("/api/socket");
    const socket = io({ path: "/api/socketio" });
    socketRef.current = socket;
    socket.emit("join-room", { room, name, role });

    socket.on("new-message", (payload: { from: string; text: string; ts: number }) => {
      setMessages((prev) => [...prev, { from: payload.from, text: payload.text, ts: new Date(payload.ts).toLocaleTimeString() }]);
    });
    socket.on("system", (msg: { text: string }) => {
      setMessages((prev) => [...prev, { from: "System", text: msg.text, ts: new Date().toLocaleTimeString() }]);
    });

    return () => {
      socket.disconnect();
    };
  }, [room, name, role]);

  const stopMedia = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    analyserRef.current?.disconnect();
    analyserRef.current = null;
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setMicLevel(0);
    if (videoRef.current) videoRef.current.srcObject = null;
  };

  const refreshMedia = async (useCam: boolean, useMic: boolean) => {
    if (!useCam && !useMic) {
      stopMedia();
      return;
    }
    try {
      const constraints: MediaStreamConstraints = {
        video: useCam
          ? {
              width: { ideal: 1280 },
              height: { ideal: 720 },
              facingMode: "user"
            }
          : false,
        audio: useMic
          ? {
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true
            }
          : false
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      stopMedia();
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      if (useMic) {
        const audioCtx = new AudioContext();
        audioCtxRef.current = audioCtx;
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyserRef.current = analyser;
        analyser.fftSize = 256;
        source.connect(analyser);
        const data = new Uint8Array(analyser.frequencyBinCount);
        const tick = () => {
          analyser.getByteFrequencyData(data);
          const avg = data.reduce((a, b) => a + b, 0) / data.length;
          setMicLevel(avg / 255);
          rafRef.current = requestAnimationFrame(tick);
        };
        tick();
      }
      setMediaError("");
      setMediaReady(true);
    } catch (err: any) {
      const msg =
        err?.name === "NotAllowedError"
          ? "Permission denied. Click the padlock in your browser and allow Camera + Microphone."
          : err?.message || "Could not access camera/mic";
      setMediaError(msg);
      setCamOn(false);
      setMicOn(false);
      stopMedia();
    }
  };

  useEffect(() => {
    refreshMedia(camOn, micOn);
    return () => stopMedia();
  }, [camOn, micOn]);

  useEffect(() => {
    if (role !== "sign") return;
    let active = true;

    const tick = async () => {
      try {
        const res = await fetch("/api/direction1/status");
        const data = await res.json();
        if (!active) return;
        setDirection1Running(Boolean(data.running));
        setDirection1State(data.state || null);
        if (data.running) {
          setDirection1Error("");
        }
        if (!data.running && data.lastExit && data.lastExit.code !== null && data.lastExit.code !== 0) {
          const tail = String(data.logTail || "").trim();
          setDirection1Error(tail ? tail : `Direction-1 exited with code ${data.lastExit.code}`);
        } else if (!data.running && data.lastExit && data.lastExit.signal) {
          setDirection1Error("");
        }
      } catch {
      }
    };

    tick();
    const id = setInterval(tick, 1200);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [role]);

  useEffect(() => {
    if (role !== "speech") return;
    const latest = [...messages].reverse().find((m) => m.from.toLowerCase() !== "system");
    if (!latest || !latest.text.trim()) return;

    let cancelled = false;
    const run = async () => {
      try {
        setAvatarLoading(true);
        setAvatarError("");
        const res = await fetch("/api/avatar/render", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: latest.text })
        });
        const data = await res.json();
        if (cancelled) return;
        if (!res.ok || !data?.ok) {
          setAvatarError(data?.error || "Avatar render failed");
          return;
        }
        setAvatarMotion(data.motion || null);
      } catch (err: any) {
        if (!cancelled) setAvatarError(err?.message || "Avatar render failed");
      } finally {
        if (!cancelled) setAvatarLoading(false);
      }
    };

    run();
    return () => {
      cancelled = true;
    };
  }, [role, messages]);

  const startDirection1 = async () => {
    try {
      setDirection1Busy(true);
      setDirection1Error("");
      if (!camOn) {
        setCamOn(true);
      }
      const res = await fetch("/api/direction1/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
      });
      const data = await res.json();
      setDirection1Running(Boolean(data.running));
      if (!data.running && data.error) {
        setDirection1Error(String(data.error));
      }
    } finally {
      setDirection1Busy(false);
    }
  };

  const stopDirection1 = async () => {
    try {
      setDirection1Busy(true);
      const res = await fetch("/api/direction1/stop", { method: "POST" });
      const data = await res.json();
      setDirection1Running(Boolean(data.running));
      setDirection1Error("");
    } finally {
      setDirection1Busy(false);
    }
  };

  const rightTitle = useMemo(() => (role === "speech" ? "Live Sign View" : "Incoming Speech"), [role]);

  return (
    <main className="comm-surface">
      <div className="page comm-page">
        <TopBar room={room} name={name} role={role} />

        <div
          className="glass"
          style={{
            display: "grid",
            gridTemplateColumns: "1.3fr 0.9fr",
            gap: 24,
            padding: 26,
            borderRadius: 22,
            background: "#f9f4ec",
            boxShadow: "0 20px 60px rgba(0,0,0,0.16), 0 1px 0 rgba(255,255,255,0.8) inset"
          }}
        >
          <Panel title={`You — ${name}`}>
            <div
              style={{
                height: 420,
                borderRadius: 20,
                border: "1px solid var(--border)",
                background: "#050505",
                boxShadow: "0 18px 36px rgba(0,0,0,0.32)",
                position: "relative",
                overflow: "hidden"
              }}
            >
              {!camOn && (
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    display: "grid",
                    placeItems: "center",
                    color: "#e5e7eb",
                    background: "rgba(0,0,0,0.5)"
                  }}
                >
                  Camera off
                </div>
              )}
              {!camOn && (
                <div
                  style={{
                    position: "absolute",
                    inset: "30% 35%",
                    color: "rgba(255,255,255,0.8)",
                    fontSize: 46
                  }}
                >
                  📷
                </div>
              )}
              {!mediaReady && mediaError === "" && camOn && (
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    display: "grid",
                    placeItems: "center",
                    color: "#e5e7eb",
                    background: "rgba(0,0,0,0.4)"
                  }}
                >
                  Waiting for permission...
                </div>
              )}
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                style={{ width: "100%", height: "100%", objectFit: "cover", borderRadius: "inherit" }}
              />
              {!mediaReady && (
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    display: "grid",
                    placeItems: "center",
                    background: "rgba(0,0,0,0.4)"
                  }}
                >
                  <div
                    style={{
                      width: 70,
                      height: 70,
                      borderRadius: 20,
                      background: "#201c19",
                      display: "grid",
                      placeItems: "center",
                      color: "#f3e9dc",
                      fontSize: 28,
                      boxShadow: "0 10px 24px rgba(0,0,0,0.35)"
                    }}
                  >
                    ▢▢
                  </div>
                </div>
              )}
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  borderRadius: "inherit",
                  border: "1px solid rgba(139,92,246,0.35)",
                  boxShadow: "0 0 16px rgba(139,92,246,0.35)",
                  animation: "pulse 2.4s infinite"
                }}
              />
              <div style={{ position: "absolute", top: 12, left: 12, padding: "4px 10px", fontSize: 13 }} className="pill">
                Camera + Mic
              </div>
              <div style={{ position: "absolute", bottom: 12, left: 12, display: "flex", gap: 8 }}>
                <ToggleButton
                  active={camOn}
                  label="Cam"
                  onClick={() => setCamOn((v) => !v)}
                  icon={camOn ? "📷" : "🚫"}
                />
                <ToggleButton
                  active={micOn}
                  label="Mic"
                  onClick={() => setMicOn((v) => !v)}
                  icon={micOn ? "🎙️" : "🔇"}
                />
              </div>
              {role === "sign" && direction1Running && !camOn && (
                <div
                  className="pill"
                  style={{
                    position: "absolute",
                    top: 12,
                    right: 12,
                    background: "rgba(191,219,254,0.25)",
                    borderColor: "rgba(96,165,250,0.6)",
                    color: "#dbeafe"
                  }}
                >
                  Direction-1 running (turn Cam on for frontend preview)
                </div>
              )}
              {mediaError && (
                <div
                  className="pill"
                  style={{
                    position: "absolute",
                    bottom: 12,
                    right: 12,
                    background: "rgba(248,113,113,0.12)",
                    borderColor: "rgba(248,113,113,0.5)",
                    color: "#fca5a5"
                  }}
                >
                  {mediaError || "Allow camera/mic in your browser bar"}
                </div>
              )}
            </div>
          </Panel>

          <Panel title={rightTitle}>
            <div style={{ display: "grid", gap: 12 }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: 14,
                  borderRadius: 16,
                  border: "1px solid var(--border)",
                  background: "#fefcf8"
                }}
              >
                <AudioBars level={micLevel} />
                <div>
                  <div style={{ fontWeight: 700, color: "#312118" }}>
                    {role === "speech" ? "Sign stream active" : "Listening..."}
                  </div>
                  <div className="muted" style={{ fontSize: 14, color: "#7a6856" }}>
                    {role === "speech" ? "Your speech shows as sign visuals." : "Speech transcribed to text below."}
                  </div>
                </div>
              </div>
              {role === "sign" ? (
                <Direction1Panel
                  running={direction1Running}
                  busy={direction1Busy}
                  state={direction1State}
                  error={direction1Error}
                  onStart={startDirection1}
                  onStop={stopDirection1}
                  onSend={() => {
                    const sentence = String(direction1State?.sentence || "").trim();
                    if (!sentence) return;
                    socketRef.current?.emit("send-message", { room, text: sentence, name });
                  }}
                />
              ) : (
                <AvatarPanel motion={avatarMotion} loading={avatarLoading} error={avatarError} />
              )}
              <MessageList messages={messages} />
            </div>
          </Panel>
        </div>

        <div
          className="glass"
          style={{
            padding: 16,
            borderRadius: 18,
            display: "grid",
            gridTemplateColumns: "1fr auto",
            gap: 12,
            boxShadow: "0 18px 36px rgba(0,0,0,0.18)"
          }}
        >
          <input
            className="input"
            placeholder={role === "speech" ? "Type to send as speech text..." : "Type to send as sign cue..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button
            className="btn btn-primary"
            onClick={() => {
              if (!input) return;
              socketRef.current?.emit("send-message", { room, text: input, name });
              setInput("");
            }}
          >
            Send
          </button>
        </div>
      </div>
    </main>
  );
}

function Direction1Panel({
  running,
  busy,
  state,
  error,
  onStart,
  onStop,
  onSend
}: {
  running: boolean;
  busy: boolean;
  state: any;
  error: string;
  onStart: () => void;
  onStop: () => void;
  onSend: () => void;
}) {
  const sentence = String(state?.sentence || "").trim();
  return (
    <div
      style={{
        display: "grid",
        gap: 10,
        padding: 14,
        borderRadius: 16,
        border: "1px solid var(--border)",
        background: "#fefcf8",
        boxShadow: "0 10px 20px rgba(0,0,0,0.06)"
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ fontWeight: 700, color: "#312118" }}>Direction-1 Live Recognition</div>
        <div className="pill" style={{ background: running ? "#dcfce7" : "#fee2e2" }}>
          {running ? "Running" : "Stopped"}
        </div>
      </div>
      <div className="muted" style={{ fontSize: 14, color: "#7a6856" }}>
        Uses Python MediaPipe realtime recognizer. Browser cam is turned off while Direction-1 runs so webcam is free.
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <button className="btn btn-primary" disabled={busy || running} onClick={onStart}>Start</button>
        <button className="btn btn-secondary" disabled={busy || !running} onClick={onStop}>Stop</button>
        <button className="btn btn-primary" disabled={!sentence} onClick={onSend}>Send Sentence</button>
      </div>
      <div style={{ fontSize: 14, color: "#6b5b49" }}>
        <strong>Latest:</strong> {state?.text || "-"} · conf {Number(state?.confidence || 0).toFixed(2)}
      </div>
      <div style={{ fontSize: 13, color: "#7a6856" }}>
        <strong>Model:</strong> {state?.model || "models/dynamic_lstm_10class_combined.onnx"}
      </div>
      {!!error && (
        <div style={{ fontSize: 13, color: "#7f1d1d", background: "#fee2e2", border: "1px solid #fca5a5", padding: "10px 12px", borderRadius: 12, whiteSpace: "pre-wrap" }}>
          {error}
        </div>
      )}
      <div style={{ fontSize: 14, color: "#2f2416", background: "#f0e3d5", padding: "10px 12px", borderRadius: 12 }}>
        {sentence || "No sentence yet"}
      </div>
    </div>
  );
}

function AvatarPanel({ motion, loading, error }: { motion: MotionPayload | null; loading: boolean; error: string }) {
  return (
    <div
      style={{
        display: "grid",
        gap: 10,
        padding: 14,
        borderRadius: 16,
        border: "1px solid var(--border)",
        background: "#fefcf8",
        boxShadow: "0 10px 20px rgba(0,0,0,0.06)"
      }}
    >
      <div style={{ fontWeight: 700, color: "#312118" }}>Avatar Playback</div>
      <div className="muted" style={{ fontSize: 14, color: "#7a6856" }}>
        Uses the improved avatar pipeline and renders landmark motion directly in this page.
      </div>
      {loading && <div className="pill">Rendering avatar...</div>}
      {error && <div className="pill" style={{ background: "#fee2e2", borderColor: "#fca5a5" }}>{error}</div>}
      <AvatarMotionCanvas motion={motion} />
    </div>
  );
}

function AvatarMotionCanvas({ motion }: { motion: MotionPayload | null }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !motion || !motion.frames?.length) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let frameIdx = 0;
    let raf = 0;
    let last = performance.now();
    const fps = Math.max(1, Number(motion.fps || 15));
    const frameMs = 1000 / fps;

    const toPx = (pt: number[]) => {
      const x = Math.min(1, Math.max(0, Number(pt?.[0] || 0)));
      const y = Math.min(1, Math.max(0, Number(pt?.[1] || 0)));
      return [Math.round(x * canvas.width), Math.round(y * canvas.height)] as const;
    };
    const visible = (pt: number[]) => Number(pt?.[3] ?? 1) > 0.05;

    const drawGraph = (pts: number[][], edges: Array<[number, number]>, color: string, radius: number, lineW: number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineW;
      ctx.lineCap = "round";
      for (const [a, b] of edges) {
        if (!pts[a] || !pts[b] || !visible(pts[a]) || !visible(pts[b])) continue;
        const p1 = toPx(pts[a]);
        const p2 = toPx(pts[b]);
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.stroke();
      }
      ctx.fillStyle = color;
      for (const p of pts) {
        if (!p || !visible(p)) continue;
        const [x, y] = toPx(p);
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    const paint = (ts: number) => {
      if (ts - last >= frameMs) {
        frameIdx = (frameIdx + 1) % motion.frames.length;
        last = ts;
      }

      const f = motion.frames[frameIdx];
      const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
      grad.addColorStop(0, "#0e1420");
      grad.addColorStop(1, "#222936");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      drawGraph(f.pose || [], POSE_CONNECTIONS, "#50d2ff", 3, 2);
      drawGraph(f.left_hand || [], HAND_CONNECTIONS, "#50ff78", 2, 2);
      drawGraph(f.right_hand || [], HAND_CONNECTIONS, "#ffb446", 2, 2);

      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.font = "13px sans-serif";
      ctx.fillText(`Text: ${motion.text || ""}`, 10, canvas.height - 12);

      raf = requestAnimationFrame(paint);
    };

    raf = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(raf);
  }, [motion]);

  return <canvas ref={canvasRef} width={560} height={320} style={{ width: "100%", borderRadius: 12, border: "1px solid var(--border)" }} />;
}

function TopBar({ room, name, role }: { room: string; name: string; role: string }) {
  return (
    <div
      className="glass"
      style={{
        padding: 14,
        borderRadius: "18px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 12
      }}
    >
      <div className="row" style={{ gap: 12 }}>
        <div className="pill">
          <span className="dot" />
          Connected
        </div>
        <div className="pill">
          Room: <strong>{room}</strong>
        </div>
        <div className="pill">
          You: <strong>{name}</strong> ({role})
        </div>
      </div>
      <Link className="btn btn-secondary" href="/join">
        Leave
      </Link>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={{ display: "grid", gap: 12, animation: "slideUp 240ms ease" }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
      </div>
      {children}
    </section>
  );
}

function MessageList({ messages }: { messages: Message[] }) {
  const params = useSearchParams();
  const selfName = params?.get("name") || "";
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages.length]);

  return (
    <div style={{ maxHeight: 360, overflowY: "auto", display: "grid", gap: 10 }} ref={listRef}>
      {messages.map((m, idx) => {
        const mine = selfName && m.from.toLowerCase() === selfName.toLowerCase();
        const system = m.from.toLowerCase() === "system";
        return (
          <div key={idx} style={{ display: "grid", gap: 4, justifyItems: mine ? "end" : "start" }}>
            <div style={{ fontSize: 12, color: "#8c7c68" }}>
              {m.from} · {m.ts}
            </div>
            <div
              style={{
                maxWidth: "92%",
                background: system ? "#d9ede7" : "#d9cabc",
                color: "#2f2416",
                padding: "10px 12px",
                borderRadius: 14,
                boxShadow: "0 8px 16px rgba(0,0,0,0.08)"
              }}
            >
              {m.text}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function AudioBars({ level = 0 }: { level?: number }) {
  const bars = [6, 12, 18, 10, 16, 8].map((h) => Math.max(6, h * (0.6 + level)));
  return (
    <div style={{ display: "flex", gap: 4, alignItems: "flex-end", height: 28 }}>
      {bars.map((h, i) => (
        <div
          key={i}
          style={{
            width: 4,
            height: h,
            borderRadius: 2,
            background: "var(--cyan)",
            animation: `bar 1.2s ease-in-out ${i * 0.08}s infinite`
          }}
        />
      ))}
      <style jsx>{`
        @keyframes bar {
          0% {
            height: 6px;
          }
          50% {
            height: 24px;
          }
          100% {
            height: 8px;
          }
        }
      `}</style>
    </div>
  );
}

function SignAvatar({ cue, message }: { cue: number; message: string }) {
  const [stamp, setStamp] = useState(cue);
  const [label, setLabel] = useState("Welcome");
  const [gestureKey, setGestureKey] = useState("welcome");

  useEffect(() => {
    setStamp(cue);
    const g = getGestureLabel(message);
    setLabel(g.label);
    setGestureKey(g.key);
  }, [cue, message]);

  return (
    <div className="sign-avatar" key={stamp}>
      <StickSigner gesture={gestureKey} label={label} />
      <div className="text">
        <div style={{ fontWeight: 700 }}>Signing: {label}</div>
      </div>
    </div>
  );
}

function StatusTile({ icon, label }: { icon: string; label: string }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: 14,
        borderRadius: 16,
        border: "1px solid var(--border)",
        background: "#fefcf8",
        boxShadow: "0 10px 20px rgba(0,0,0,0.06)"
      }}
    >
      <div
        style={{
          width: 42,
          height: 42,
          borderRadius: 12,
          background: "#f0e3d5",
          display: "grid",
          placeItems: "center",
          fontSize: 20
        }}
      >
        {icon}
      </div>
      <div style={{ fontWeight: 700, color: "#312118" }}>{label}</div>
    </div>
  );
}

function StickSigner({ gesture, label }: { gesture: string; label: string }) {
  return (
    <div className={`stick-signer gesture-${gesture}`} aria-label={`signing ${label}`}>
      <svg viewBox="0 0 120 140" width="64" height="64">
        <g fill="none" stroke="#22d3ee" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round">
          <circle className="head" cx="60" cy="24" r="14" />
          <path className="torso" d="M60 38v28" />
          <path className="arm-left" d="M60 46l-18 12" />
          <path className="arm-wave" d="M60 46l22 -12" />
          <path className="hip" d="M60 66l0 32" />
          <path className="leg-left" d="M60 98l-14 24" />
          <path className="leg-right" d="M60 98l14 24" />
        </g>
        <g className="hand" fill="#8b5cf6">
          <circle cx="84" cy="32" r="5" />
        </g>
        <text x="60" y="128" textAnchor="middle" fontSize="10" fill="#9ca3af">
          {label}
        </text>
      </svg>
    </div>
  );
}

function getGestureLabel(text: string): { label: string; key: string } {
  const t = text.toLowerCase();
  if (!t.trim()) return { label: "Welcome", key: "welcome" };
  if (t.includes("hello") || t.includes("hi")) return { label: "Hello", key: "hello" };
  if (t.includes("thanks") || t.includes("thank")) return { label: "Thank you", key: "thankyou" };
  if (t.includes("yes") || t.includes("yeah") || t.includes("sure")) return { label: "Yes", key: "yes" };
  if (t.includes("no") || t.includes("nah") || t.includes("nope")) return { label: "No", key: "no" };
  if (t.includes("help")) return { label: "Help", key: "help" };
  if (t.includes("love")) return { label: "I love you", key: "love" };
  if (t.includes("sorry")) return { label: "Sorry", key: "sorry" };
  if (t.includes("bye")) return { label: "Goodbye", key: "goodbye" };
  return { label: "Message", key: "message" };
}

function ToggleButton({
  active,
  label,
  icon,
  onClick,
  disabled = false
}: {
  active: boolean;
  label: string;
  icon: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      type="button"
      disabled={disabled}
      className="pill"
      style={{
        borderColor: active ? "var(--cyan)" : "var(--border)",
        color: disabled ? "#9ca3af" : active ? "#0c0f16" : "var(--text)",
        background: disabled ? "rgba(255,255,255,0.05)" : active ? "var(--cyan)" : "rgba(255,255,255,0.06)",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.6 : 1
      }}
    >
      <span style={{ marginRight: 6 }}>{icon}</span>
      {label}
    </button>
  );
}
