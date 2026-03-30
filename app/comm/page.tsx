"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import io, { Socket } from "socket.io-client";

type Message = { from: string; text: string; ts: string };

export default function CommPage() {
  const params = useSearchParams();
  const room = params.get("room") || "room-001";
  const role = params.get("role") || "sign";
  const name = params.get("name") || (role === "speech" ? "Speech User" : "Sign User");
  const [messages, setMessages] = useState<Message[]>([
    { from: "System", text: "Welcome to Sign-Sync!", ts: new Date().toLocaleTimeString() }
  ]);
  const [input, setInput] = useState("");
  const socketRef = useRef<Socket | null>(null);
  const [avatarCue, setAvatarCue] = useState<number>(Date.now());

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number>();
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
      setAvatarCue(payload.ts);
    });
    socket.on("system", (msg: { text: string }) => {
      setMessages((prev) => [...prev, { from: "System", text: msg.text, ts: new Date().toLocaleTimeString() }]);
    });

    return () => socket.disconnect();
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
              <StatusTile icon="🤟" label="Signing: Message" />
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
  const selfName = params.get("name") || "";
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
  onClick
}: {
  active: boolean;
  label: string;
  icon: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      type="button"
      className="pill"
      style={{
        borderColor: active ? "var(--cyan)" : "var(--border)",
        color: active ? "#0c0f16" : "var(--text)",
        background: active ? "var(--cyan)" : "rgba(255,255,255,0.06)",
        cursor: "pointer"
      }}
    >
      <span style={{ marginRight: 6 }}>{icon}</span>
      {label}
    </button>
  );
}
