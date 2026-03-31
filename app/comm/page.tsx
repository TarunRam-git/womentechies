"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import io, { Socket } from "socket.io-client";

declare global {
  interface Window {
    ort?: any;
  }
}

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
  [0, 1], [1, 2],
  [0, 5], [5, 6],
  [5, 9], [9, 10],
  [9, 13], [13, 14],
  [13, 17], [17, 18],
  [0, 17]
];
const BODY_CONNECTIONS_33: Array<[number, number]> = [
  [11, 12],
  [11, 13], [13, 15],
  [12, 14], [14, 16],
  [11, 23], [12, 24],
  [23, 24],
];
const BODY_CONNECTIONS_8: Array<[number, number]> = [
  [0, 1],
  [0, 2], [2, 4],
  [1, 3], [3, 5],
  [0, 6], [1, 7],
  [6, 7],
];

const SEQUENCE_LENGTH = 30;
const FEATURE_DIM = 197;
const BASE_FEATURE_DIM = 180;
const POSE_IDX = [11, 12, 13, 14, 15, 16, 23, 24];
const FACE_IDX = [61, 291, 13, 14, 70, 300, 159, 145, 386, 374];

function normalizeHand(landmarks: any[] | undefined): Float32Array {
  const out = new Float32Array(63);
  if (!landmarks || landmarks.length < 21) return out;

  const wx = Number(landmarks[0].x || 0);
  const wy = Number(landmarks[0].y || 0);
  const wz = Number(landmarks[0].z || 0);
  let maxDist = 0;
  const temp = new Float32Array(63);

  for (let i = 0; i < 21; i++) {
    const x = Number(landmarks[i].x || 0) - wx;
    const y = Number(landmarks[i].y || 0) - wy;
    const z = Number(landmarks[i].z || 0) - wz;
    temp[i * 3] = x;
    temp[i * 3 + 1] = y;
    temp[i * 3 + 2] = z;
    const d = Math.hypot(x, y, z);
    if (d > maxDist) maxDist = d;
  }

  if (maxDist > 1e-6) {
    for (let i = 0; i < 63; i++) out[i] = temp[i] / maxDist;
  }
  return out;
}

function extractPose(landmarks: any[] | undefined): Float32Array {
  const out = new Float32Array(24);
  if (!landmarks || landmarks.length < 33) return out;
  for (let i = 0; i < POSE_IDX.length; i++) {
    const lm = landmarks[POSE_IDX[i]];
    out[i * 3] = Number(lm?.x || 0);
    out[i * 3 + 1] = Number(lm?.y || 0);
    out[i * 3 + 2] = Number(lm?.z || 0);
  }
  return out;
}

function extractFace(landmarks: any[] | undefined): Float32Array {
  const out = new Float32Array(30);
  if (!landmarks || landmarks.length < 400) return out;
  for (let i = 0; i < FACE_IDX.length; i++) {
    const lm = landmarks[FACE_IDX[i]];
    out[i * 3] = Number(lm?.x || 0);
    out[i * 3 + 1] = Number(lm?.y || 0);
    out[i * 3 + 2] = Number(lm?.z || 0);
  }
  return out;
}

function jointAngle(a: number[], b: number[], c: number[]): number {
  const ba = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
  const n1 = Math.hypot(ba[0], ba[1], ba[2]);
  const n2 = Math.hypot(bc[0], bc[1], bc[2]);
  if (n1 * n2 < 1e-6) return 0;
  const cos = Math.max(-1, Math.min(1, (ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]) / (n1 * n2)));
  return Math.acos(cos) / Math.PI;
}

function computeRelational(base: Float32Array): Float32Array {
  const rel = new Float32Array(17);
  const pose = new Array(8).fill(0).map((_, i) => [base[126 + i * 3], base[126 + i * 3 + 1], base[126 + i * 3 + 2]]);
  const face = new Array(10).fill(0).map((_, i) => [base[150 + i * 3], base[150 + i * 3 + 1], base[150 + i * 3 + 2]]);

  const leftShoulder = pose[0];
  const rightShoulder = pose[1];
  const leftElbow = pose[2];
  const rightElbow = pose[3];
  const leftWrist = pose[4];
  const rightWrist = pose[5];

  const faceCenter = [
    face.reduce((s, v) => s + v[0], 0) / 10,
    face.reduce((s, v) => s + v[1], 0) / 10,
    face.reduce((s, v) => s + v[2], 0) / 10,
  ];
  const mouthCenter = [
    (face[2][0] + face[3][0]) * 0.5,
    (face[2][1] + face[3][1]) * 0.5,
    (face[2][2] + face[3][2]) * 0.5,
  ];

  const shoulderWidth = Math.hypot(
    leftShoulder[0] - rightShoulder[0],
    leftShoulder[1] - rightShoulder[1],
    leftShoulder[2] - rightShoulder[2],
  );
  const scale = shoulderWidth > 1e-4 ? shoulderWidth : 1;

  const lwf = [(leftWrist[0] - faceCenter[0]) / scale, (leftWrist[1] - faceCenter[1]) / scale, (leftWrist[2] - faceCenter[2]) / scale];
  const rwf = [(rightWrist[0] - faceCenter[0]) / scale, (rightWrist[1] - faceCenter[1]) / scale, (rightWrist[2] - faceCenter[2]) / scale];
  const lwm = [(leftWrist[0] - mouthCenter[0]) / scale, (leftWrist[1] - mouthCenter[1]) / scale, (leftWrist[2] - mouthCenter[2]) / scale];
  const rwm = [(rightWrist[0] - mouthCenter[0]) / scale, (rightWrist[1] - mouthCenter[1]) / scale, (rightWrist[2] - mouthCenter[2]) / scale];

  const wd = Math.hypot(leftWrist[0] - rightWrist[0], leftWrist[1] - rightWrist[1], leftWrist[2] - rightWrist[2]) / scale;
  const lws = Math.hypot(leftWrist[0] - leftShoulder[0], leftWrist[1] - leftShoulder[1], leftWrist[2] - leftShoulder[2]) / scale;
  const rws = Math.hypot(rightWrist[0] - rightShoulder[0], rightWrist[1] - rightShoulder[1], rightWrist[2] - rightShoulder[2]) / scale;
  const la = jointAngle(leftShoulder, leftElbow, leftWrist);
  const ra = jointAngle(rightShoulder, rightElbow, rightWrist);

  const all = [...lwf, ...rwf, ...lwm, ...rwm, wd, lws, rws, la, ra];
  for (let i = 0; i < rel.length; i++) rel[i] = all[i] || 0;
  return rel;
}

function composeFeature(results: any): { feature: Float32Array; handCount: number } {
  const rightHand = normalizeHand(results?.rightHandLandmarks);
  const leftHand = normalizeHand(results?.leftHandLandmarks);
  const pose = extractPose(results?.poseLandmarks);
  const face = extractFace(results?.faceLandmarks);

  if (results?.leftHandLandmarks?.length) {
    const lw = results.leftHandLandmarks[0];
    pose[12] = Number(lw?.x || 0);
    pose[13] = Number(lw?.y || 0);
    pose[14] = Number(lw?.z || 0);
  }
  if (results?.rightHandLandmarks?.length) {
    const rw = results.rightHandLandmarks[0];
    pose[15] = Number(rw?.x || 0);
    pose[16] = Number(rw?.y || 0);
    pose[17] = Number(rw?.z || 0);
  }

  const base = new Float32Array(BASE_FEATURE_DIM);
  base.set(rightHand, 0);
  base.set(leftHand, 63);
  base.set(pose, 126);
  base.set(face, 150);

  const rel = computeRelational(base);
  const feature = new Float32Array(FEATURE_DIM);
  feature.set(base, 0);
  feature.set(rel, BASE_FEATURE_DIM);

  const handCount = (results?.leftHandLandmarks?.length ? 1 : 0) + (results?.rightHandLandmarks?.length ? 1 : 0);
  return { feature, handCount };
}

function softmax(logits: Float32Array): number[] {
  const max = Math.max(...Array.from(logits));
  const exp = Array.from(logits).map((v) => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / (sum || 1));
}

function hasLiveVideoTrack(stream: MediaStream | null): boolean {
  if (!stream) return false;
  const tracks = stream.getVideoTracks();
  return tracks.some((t) => t.readyState === "live" && t.enabled);
}

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
  const [socketConnected, setSocketConnected] = useState(false);
  const socketRef = useRef<Socket | null>(null);
  const [direction1Running, setDirection1Running] = useState(false);
  const [direction1State, setDirection1State] = useState<any>(null);
  const direction1StateRef = useRef<any>(null);
  const [direction1Busy, setDirection1Busy] = useState(false);
  const [direction1Error, setDirection1Error] = useState("");
  const direction1RunningRef = useRef(false);
  const ortRef = useRef<any>(null);
  const ortLoadPromiseRef = useRef<Promise<any> | null>(null);
  const direction1SessionRef = useRef<any>(null);
  const direction1LabelsRef = useRef<Record<string, string>>({});
  const direction1HolisticRef = useRef<any>(null);
  const direction1LoopRef = useRef<number | null>(null);
  const inferBusyRef = useRef(false);
  const seqRef = useRef<Float32Array[]>([]);
  const handRef = useRef<number[]>([]);
  const topHistoryRef = useRef<number[]>([]);
  const probEmaRef = useRef<number[] | null>(null);
  const candidateIdxRef = useRef<number | null>(null);
  const candidateStreakRef = useRef(0);
  const lastAcceptRef = useRef(0);
  const lastTokenRef = useRef(0);
  const [avatarMotion, setAvatarMotion] = useState<MotionPayload | null>(null);
  const [remoteAvatarMotion, setRemoteAvatarMotion] = useState<MotionPayload | null>(null);
  const [avatarLoading, setAvatarLoading] = useState(false);
  const [avatarError, setAvatarError] = useState("");
  const [remoteFrame, setRemoteFrame] = useState("");
  const [peers, setPeers] = useState<Array<{ id: string; name: string; role: string }>>([]);

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
  const lastSystemRef = useRef<{ text: string; at: number }>({ text: "", at: 0 });

  useEffect(() => {
    fetch("/api/socket");
    const socket = io({ path: "/api/socketio", transports: ["websocket", "polling"] });
    socketRef.current = socket;
    const joinRoom = () => socket.emit("join-room", { room, name, role });

    socket.on("connect", () => {
      setSocketConnected(true);
      joinRoom();
    });
    socket.on("reconnect", joinRoom);
    socket.on("disconnect", () => setSocketConnected(false));
    socket.on("connect_error", () => setSocketConnected(false));

    joinRoom();

    socket.on("new-message", (payload: { from: string; text: string; ts: number }) => {
      setMessages((prev) => [...prev, { from: payload.from, text: payload.text, ts: new Date(payload.ts).toLocaleTimeString() }]);
    });
    socket.on("system", (msg: { text: string }) => {
      const now = Date.now();
      const clean = String(msg?.text || "");
      if (clean && lastSystemRef.current.text === clean && now - lastSystemRef.current.at < 8000) return;
      lastSystemRef.current = { text: clean, at: now };
      setMessages((prev) => [...prev, { from: "System", text: clean, ts: new Date().toLocaleTimeString() }]);
    });
    socket.on("room-peers", (list: Array<{ id: string; name: string; role: string }>) => {
      setPeers(Array.isArray(list) ? list : []);
    });
    socket.on("avatar-motion", (payload: { motion?: MotionPayload }) => {
      if (payload?.motion) setRemoteAvatarMotion(payload.motion);
    });
    socket.on("video-frame", (payload: { frame?: string }) => {
      if (payload?.frame) setRemoteFrame(payload.frame);
    });

    return () => {
      socket.removeAllListeners();
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
    setMediaReady(false);
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
    direction1StateRef.current = direction1State;
  }, [direction1State]);

  useEffect(() => {
    return () => {
      direction1RunningRef.current = false;
      if (direction1LoopRef.current) {
        cancelAnimationFrame(direction1LoopRef.current);
        direction1LoopRef.current = null;
      }
      if (direction1HolisticRef.current?.close) {
        try {
          direction1HolisticRef.current.close();
        } catch {
        }
      }
    };
  }, []);

  useEffect(() => {
    if (role === "sign") return;
    direction1RunningRef.current = false;
    setDirection1Running(false);
    if (direction1LoopRef.current) {
      cancelAnimationFrame(direction1LoopRef.current);
      direction1LoopRef.current = null;
    }
  }, [role]);

  useEffect(() => {
    if (role !== "sign" || !direction1Running) return;
    if (camOn && hasLiveVideoTrack(streamRef.current)) return;

    direction1RunningRef.current = false;
    setDirection1Running(false);
    if (direction1LoopRef.current) {
      cancelAnimationFrame(direction1LoopRef.current);
      direction1LoopRef.current = null;
    }
    setDirection1Error("Camera turned off or unavailable. Turn camera on, then press Start again.");
  }, [role, direction1Running, camOn, mediaReady]);

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
        const motion = data.motion || null;
        setAvatarMotion(motion);
        socketRef.current?.emit("avatar-motion", { room, motion });
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
  }, [messages, room, role]);

  useEffect(() => {
    if (!camOn || !mediaReady) return;
    let timer: any = null;
    const video = videoRef.current;
    if (!video) return;

    const c = document.createElement("canvas");
    const x = c.getContext("2d");
    if (!x) return;

    timer = setInterval(() => {
      if (!socketRef.current || !videoRef.current || videoRef.current.readyState < 2) return;
      const vw = videoRef.current.videoWidth || 320;
      const vh = videoRef.current.videoHeight || 180;
      c.width = 320;
      c.height = Math.max(180, Math.round((vh / Math.max(vw, 1)) * 320));
      x.drawImage(videoRef.current, 0, 0, c.width, c.height);
      const frame = c.toDataURL("image/jpeg", 0.5);
      socketRef.current.emit("video-frame", { room, frame });
    }, 220);

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [camOn, mediaReady, room]);

  const startDirection1 = async () => {
    try {
      if (role !== "sign") return;
      setDirection1Busy(true);
      setDirection1Error("");
      if (!camOn || !hasLiveVideoTrack(streamRef.current)) {
        await refreshMedia(true, micOn);
        setCamOn(true);
      }

      if (!streamRef.current || !videoRef.current || !hasLiveVideoTrack(streamRef.current)) {
        setDirection1Error("Turn camera on and allow permission first.");
        return;
      }

      if (videoRef.current.readyState < 2) {
        await new Promise<void>((resolve) => {
          let done = false;
          const finish = () => {
            if (done) return;
            done = true;
            resolve();
          };
          videoRef.current?.addEventListener("loadeddata", finish, { once: true });
          setTimeout(finish, 1200);
        });
      }

      const ensureLocalOrt = async () => {
        if (window.ort) return window.ort;
        if (!ortLoadPromiseRef.current) {
          ortLoadPromiseRef.current = new Promise((resolve, reject) => {
            const existing = document.querySelector('script[data-ort-local="1"]') as HTMLScriptElement | null;
            if (existing) {
              if (window.ort) {
                resolve(window.ort);
                return;
              }
              existing.addEventListener("load", () => {
                if (window.ort) resolve(window.ort);
                else reject(new Error("ONNX runtime loaded but global object missing"));
              }, { once: true });
              existing.addEventListener("error", () => reject(new Error("Failed to load local ONNX runtime")), { once: true });
              return;
            }

            const script = document.createElement("script");
            script.src = "/ort/ort.min.js";
            script.async = true;
            script.dataset.ortLocal = "1";
            script.onload = () => {
              if (window.ort) resolve(window.ort);
              else reject(new Error("ONNX runtime loaded but global object missing"));
            };
            script.onerror = () => reject(new Error("Failed to load local ONNX runtime"));
            document.head.appendChild(script);
          }).catch((err) => {
            ortLoadPromiseRef.current = null;
            throw err;
          });
        }
        return ortLoadPromiseRef.current;
      };

      if (!direction1SessionRef.current) {
        if (!ortRef.current) {
          ortRef.current = await ensureLocalOrt();
          ortRef.current.env.wasm.wasmPaths = "/ort/";
          ortRef.current.env.wasm.numThreads = 1;
        }
        const modelRes = await fetch("/models/dynamic_lstm_10class_combined.onnx", { cache: "no-store" });
        if (!modelRes.ok) throw new Error("Failed to load local 10-class model");
        const modelBytes = new Uint8Array(await modelRes.arrayBuffer());
        direction1SessionRef.current = await ortRef.current.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
      }

      if (!Object.keys(direction1LabelsRef.current).length) {
        const labelsRes = await fetch("/models/labels_10class_combined.json", { cache: "no-store" });
        if (!labelsRes.ok) throw new Error("Failed to load local labels");
        direction1LabelsRef.current = await labelsRes.json();
      }

      if (!direction1HolisticRef.current) {
        const mp = await import("@mediapipe/holistic");
        const holistic = new mp.Holistic({ locateFile: (file: string) => `/mediapipe/holistic/${file}` });
        holistic.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          refineFaceLandmarks: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        holistic.onResults((results: any) => {
          if (!direction1RunningRef.current) return;
          const { feature, handCount } = composeFeature(results);
          seqRef.current.push(feature);
          handRef.current.push(handCount > 0 ? 1 : 0);
          if (seqRef.current.length > SEQUENCE_LENGTH) seqRef.current.shift();
          if (handRef.current.length > SEQUENCE_LENGTH) handRef.current.shift();

          const baseState: any = {
            text: "-",
            status: "collecting",
            confidence: 0,
            margin: 0,
            consensus: 0,
            sentence_tokens: direction1StateRef.current?.sentence_tokens || [],
            sentence: direction1StateRef.current?.sentence || "",
            model: "models/dynamic_lstm_10class_combined.onnx",
            labels: "models/labels_10class_combined.json",
            updated_at: Date.now() / 1000,
          };

          if (seqRef.current.length < SEQUENCE_LENGTH) {
            setDirection1State(baseState);
            return;
          }

          const handRatio = handRef.current.reduce((a, b) => a + b, 0) / Math.max(1, handRef.current.length);
          if (handRatio < 0.55) {
            baseState.status = "low_hands";
            setDirection1State(baseState);
            return;
          }

          const session = direction1SessionRef.current;
          if (!session || inferBusyRef.current) {
            setDirection1State(baseState);
            return;
          }

          inferBusyRef.current = true;
          void (async () => {
            try {
              const flat = new Float32Array(SEQUENCE_LENGTH * FEATURE_DIM);
              for (let t = 0; t < SEQUENCE_LENGTH; t++) flat.set(seqRef.current[t], t * FEATURE_DIM);

              const input = new ortRef.current.Tensor("float32", flat, [1, SEQUENCE_LENGTH, FEATURE_DIM]);
              const output = await session.run({ [session.inputNames[0]]: input });
              const logits = output[session.outputNames[0]].data as Float32Array;
              const probsRaw = softmax(logits);

              const alpha = 0.58;
              const ema = probEmaRef.current;
              const probs = ema ? probsRaw.map((p, i) => (alpha * p) + ((1 - alpha) * (ema[i] || 0))) : probsRaw;
              probEmaRef.current = probs;

              let idx = 0;
              for (let i = 1; i < probs.length; i++) if (probs[i] > probs[idx]) idx = i;
              const sorted = [...probs].sort((a, b) => a - b);
              const conf = probs[idx] || 0;
              const margin = conf - (sorted.length > 1 ? sorted[sorted.length - 2] : 0);

              topHistoryRef.current.push(idx);
              if (topHistoryRef.current.length > 8) topHistoryRef.current.shift();
              if (candidateIdxRef.current === idx) candidateStreakRef.current += 1;
              else {
                candidateIdxRef.current = idx;
                candidateStreakRef.current = 1;
              }

              const consensus = topHistoryRef.current.filter((v) => v === idx).length / Math.max(1, topHistoryRef.current.length);
              const now = performance.now() / 1000;

              const state: any = {
                ...baseState,
                confidence: conf,
                margin,
                consensus,
                status: "low_conf",
              };

              if ((now - lastAcceptRef.current) < 1.4) state.status = "cooldown";
              else if (conf < 0.75) state.status = "low_conf";
              else if (margin < 0.07) state.status = "low_margin";
              else if (candidateStreakRef.current < 6) state.status = "unstable";
              else if (consensus < 0.78) state.status = "no_consensus";
              else {
                const label = direction1LabelsRef.current[String(idx)] || String(idx);
                state.text = label;
                state.status = "accepted";
                const prevTokens = Array.isArray(direction1StateRef.current?.sentence_tokens) ? [...direction1StateRef.current.sentence_tokens] : [];
                if (!(prevTokens.length && prevTokens[prevTokens.length - 1] === label && (now - lastTokenRef.current) < 2.5)) {
                  if (!prevTokens.length || prevTokens[prevTokens.length - 1] !== label || (now - lastTokenRef.current) > 1.2) {
                    prevTokens.push(label);
                    if (prevTokens.length > 18) prevTokens.splice(0, prevTokens.length - 18);
                    lastTokenRef.current = now;
                  }
                }
                state.sentence_tokens = prevTokens;
                state.sentence = prevTokens.join(" ").replaceAll("_", " ");
                lastAcceptRef.current = now;

              }

              direction1StateRef.current = state;
              setDirection1State(state);
            } catch (err: any) {
              setDirection1Error(err?.message || "Local inference error");
            } finally {
              inferBusyRef.current = false;
            }
          })();
        });

        direction1HolisticRef.current = holistic;
      }

      seqRef.current = [];
      handRef.current = [];
      topHistoryRef.current = [];
      probEmaRef.current = null;
      candidateIdxRef.current = null;
      candidateStreakRef.current = 0;
      lastAcceptRef.current = 0;
      lastTokenRef.current = 0;
      direction1RunningRef.current = true;
      setDirection1Running(true);
      setDirection1State({
        text: "-",
        status: "collecting",
        confidence: 0,
        margin: 0,
        consensus: 0,
        sentence_tokens: [],
        sentence: "",
        model: "models/dynamic_lstm_10class_combined.onnx",
        labels: "models/labels_10class_combined.json",
        updated_at: Date.now() / 1000,
      });

      const loop = async () => {
        if (!direction1RunningRef.current) return;
        try {
          const video = videoRef.current;
          const holistic = direction1HolisticRef.current;
          if (video && holistic && video.readyState >= 2) {
            await holistic.send({ image: video });
          }
        } catch (err: any) {
          setDirection1Error(err?.message || "Local camera inference failed");
        } finally {
          if (direction1RunningRef.current) {
            direction1LoopRef.current = requestAnimationFrame(() => {
              void loop();
            });
          }
        }
      };
      direction1LoopRef.current = requestAnimationFrame(() => {
        void loop();
      });
    } catch (err: any) {
      setDirection1Error(err?.message || "Failed to start local direction-1");
      direction1RunningRef.current = false;
      setDirection1Running(false);
    } finally {
      setDirection1Busy(false);
    }
  };

  const stopDirection1 = async () => {
    setDirection1Busy(true);
    direction1RunningRef.current = false;
    setDirection1Running(false);
    if (direction1LoopRef.current) {
      cancelAnimationFrame(direction1LoopRef.current);
      direction1LoopRef.current = null;
    }
    seqRef.current = [];
    handRef.current = [];
    topHistoryRef.current = [];
    probEmaRef.current = null;
    candidateIdxRef.current = null;
    candidateStreakRef.current = 0;
    setDirection1Error("");
    setDirection1State((prev: any) => ({
      ...(prev || {}),
      status: "stopped",
      text: prev?.text || "-",
      confidence: 0,
      margin: 0,
      consensus: 0,
      updated_at: Date.now() / 1000,
    }));
    setDirection1Busy(false);
  };

  const rightTitle = useMemo(() => (role === "speech" ? "Live Sign View" : "Incoming Speech"), [role]);
  const latestSentence = String(direction1State?.sentence || "").trim() || "No sentence yet";

  return (
    <main className="comm-surface">
      <div className="page comm-page">
        <TopBar room={room} name={name} role={role} connected={socketConnected} />

        <div
          className="glass"
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(520px, 1.15fr) minmax(420px, 0.95fr)",
            gap: 20,
            padding: 26,
            borderRadius: 22,
            background: "#f9f4ec",
            boxShadow: "0 20px 60px rgba(0,0,0,0.16), 0 1px 0 rgba(255,255,255,0.8) inset"
          }}
        >
          <Panel title={`You — ${name}`}>
            <div style={{ display: "grid", gap: 12 }}>
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

              {role === "speech" && (
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
                      Microphone Active
                    </div>
                    <div className="muted" style={{ fontSize: 14, color: "#7a6856" }}>
                      Your speech shows as sign visuals to your partner.
                    </div>
                  </div>
                </div>
              )}

              {role === "speech" && (
                <AvatarPanel title="Avatar Generated (Preview)" motion={avatarMotion} loading={avatarLoading} error={avatarError} />
              )}

              {role === "sign" && (
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
              )}
              {role === "sign" && (
                <div
                  className="glass"
                  style={{
                    padding: 14,
                    borderRadius: 16,
                    display: "grid",
                    gap: 10,
                    border: "1px solid var(--border)"
                  }}
                >
                  <div style={{ fontWeight: 700, color: "#312118" }}>Live Text (shared with partner)</div>
                  <div className="muted" style={{ fontSize: 13, color: "#7a6856" }}>
                    Your recognized sentence only shows as text; no avatar is generated for you.
                  </div>
                  <div
                    style={{
                      padding: "10px 12px",
                      borderRadius: 12,
                      background: "#f0e3d5",
                      border: "1px solid #e3d6c8",
                      color: "#2f2416",
                      minHeight: 44
                    }}
                  >
                    {latestSentence}
                  </div>
                </div>
              )}
            </div>
          </Panel>

          <Panel title={rightTitle}>
            <div style={{ display: "grid", gap: 12 }}>
              {role === "sign" && (
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
                  <div style={{ width: 8, height: 8, borderRadius: 4, background: "var(--green)", animation: "pulse 2s infinite" }} />
                  <div>
                    <div style={{ fontWeight: 700, color: "#312118" }}>
                      Listening to Speech...
                    </div>
                    <div className="muted" style={{ fontSize: 14, color: "#7a6856" }}>
                      Partner's speech transcribed and shown as signs below.
                    </div>
                  </div>
                </div>
              )}
              <RemoteVideoPanel frame={remoteFrame} peers={peers} />
              {role === "sign" && (
                <AvatarPanel title="Partner's Sign Avatar" motion={remoteAvatarMotion} loading={false} error="" />
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
        Uses local browser MediaPipe + ONNX inference (offline in-room). Webcam preview and detection run in this same tile.
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

function RemoteVideoPanel({ frame, peers }: { frame: string; peers: Array<{ id: string; name: string; role: string }> }) {
  return (
    <div
      style={{
        display: "grid",
        gap: 8,
        padding: 12,
        borderRadius: 16,
        border: "1px solid var(--border)",
        background: "#fefcf8",
      }}
    >
      <div style={{ fontWeight: 700, color: "#312118" }}>Partner Webcam</div>
      <div className="muted" style={{ fontSize: 13, color: "#7a6856" }}>
        {peers.filter((p) => p.name).map((p) => `${p.name} (${p.role})`).join(" · ") || "Waiting for partner..."}
      </div>
      <div style={{ borderRadius: 12, overflow: "hidden", border: "1px solid var(--border)", background: "#111827", minHeight: 180 }}>
        {frame ? (
          <img src={frame} alt="Partner webcam" style={{ width: "100%", height: 220, objectFit: "cover", display: "block" }} />
        ) : (
          <div style={{ height: 220, display: "grid", placeItems: "center", color: "#cbd5e1", fontSize: 14 }}>No remote video yet</div>
        )}
      </div>
    </div>
  );
}

function AvatarPanel({ title, motion, loading, error }: { title: string; motion: MotionPayload | null; loading: boolean; error: string }) {
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
      <div style={{ fontWeight: 700, color: "#312118" }}>{title}</div>
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

    const frames = motion.frames;
    const fps = Math.max(1, Number(motion.fps || 15));
    const frameMs = 1000 / fps;
    let raf = 0;
    let frameIdx = 0;
    let last = performance.now();
    let prevDrawFrame: MotionFrame | null = null;

    const poseEdges: Array<[number, number]> = [
      [0, 1],
      [1, 2], [2, 3], [3, 7],
      [0, 4], [4, 5], [5, 6], [6, 8],
      [9, 10],
      [11, 12],
      [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
      [12, 14], [14, 16], [16, 18], [16, 20], [16, 22],
      [11, 23], [12, 24], [23, 24],
      [23, 25], [25, 27], [27, 29], [27, 31],
      [24, 26], [26, 28], [28, 30], [28, 32],
    ];

    const isVisible = (pt: number[] | undefined, threshold = 0.2) => {
      if (!pt || pt.length < 2) return false;
      if (!Number.isFinite(pt[0]) || !Number.isFinite(pt[1])) return false;
      if (pt.length < 4) return true;
      return Number(pt[3]) >= threshold;
    };

    const project = (pt: number[] | undefined) => {
      if (!isVisible(pt)) return null;
      const x = Number(pt![0]) * canvas.width;
      const y = Number(pt![1]) * canvas.height;
      return [x, y] as const;
    };

    const drawGraph = (pts: number[][], edges: Array<[number, number]>, color: string, lineW: number, radius: number) => {
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      for (const [a, b] of edges) {
        if (a >= pts.length || b >= pts.length) continue;
        if (!isVisible(pts[a]) || !isVisible(pts[b])) continue;
        const p1 = project(pts[a]);
        const p2 = project(pts[b]);
        if (!p1 || !p2) continue;

        ctx.strokeStyle = "rgb(24,24,24)";
        ctx.lineWidth = lineW + 2;
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.stroke();

        ctx.strokeStyle = color;
        ctx.lineWidth = lineW;
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.stroke();
      }

      for (const pt of pts) {
        if (!isVisible(pt)) continue;
        const p = project(pt);
        if (!p) continue;

        ctx.fillStyle = "rgb(20,20,20)";
        ctx.beginPath();
        ctx.arc(p[0], p[1], radius + 1, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    const blendLandmarks = (prev: number[][] = [], curr: number[][] = [], alpha = 0.7) => {
      if (!prev || !curr || prev.length !== curr.length) return curr;
      return curr.map((pt, i) => {
        const pPrev = prev[i] || [];
        const n = Math.min(pPrev.length, pt.length);
        const merged: number[] = [];
        for (let k = 0; k < n; k++) {
          merged.push(alpha * Number(pPrev[k] || 0) + (1 - alpha) * Number(pt[k] || 0));
        }
        for (let k = n; k < pt.length; k++) merged.push(Number(pt[k] || 0));
        return merged;
      });
    };

    const smoothFrame = (prev: MotionFrame | null, curr: MotionFrame): MotionFrame => {
      if (!prev) return curr;
      return {
        pose: blendLandmarks(prev.pose || [], curr.pose || [], 0.7),
        left_hand: blendLandmarks(prev.left_hand || [], curr.left_hand || [], 0.76),
        right_hand: blendLandmarks(prev.right_hand || [], curr.right_hand || [], 0.76),
      };
    };

    const drawAvatarBase = () => {
      const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
      grad.addColorStop(0, "#1a120e");
      grad.addColorStop(1, "#362822");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    };

    const bgLayer = document.createElement("canvas");
    bgLayer.width = canvas.width;
    bgLayer.height = canvas.height;
    const bgCtx = bgLayer.getContext("2d");
    if (bgCtx) {
      const grad = bgCtx.createLinearGradient(0, 0, 0, bgLayer.height);
      grad.addColorStop(0, "#1a120e");
      grad.addColorStop(1, "#362822");
      bgCtx.fillStyle = grad;
      bgCtx.fillRect(0, 0, bgLayer.width, bgLayer.height);
    }

    const smoothedFrames: MotionFrame[] = [];
    let tmpPrev: MotionFrame | null = null;
    for (const raw of frames) {
      const f = raw || { pose: [], left_hand: [], right_hand: [] };
      const sf = smoothFrame(tmpPrev, f);
      smoothedFrames.push(sf);
      tmpPrev = sf;
    }

    const drawFrame = (curr: MotionFrame) => {
      if (bgCtx) ctx.drawImage(bgLayer, 0, 0);
      else drawAvatarBase();

      drawGraph(curr.pose || [], poseEdges, "rgb(80,210,255)", 2, 3);
      drawGraph(curr.left_hand || [], HAND_CONNECTIONS, "rgb(80,255,120)", 2, 2);
      drawGraph(curr.right_hand || [], HAND_CONNECTIONS, "rgb(255,180,70)", 2, 2);

      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.font = "600 13px Inter, sans-serif";
      ctx.fillText(`Text: ${motion.text || ""}`, 12, canvas.height - 12);
    };

    const paint = (ts: number) => {
      if (ts - last >= frameMs) {
        frameIdx = (frameIdx + 1) % frames.length;
        last = ts;
      }

      const curr = smoothedFrames[frameIdx] || { pose: [], left_hand: [], right_hand: [] };
      drawFrame(curr);
      prevDrawFrame = curr;

      raf = requestAnimationFrame(paint);
    };

    raf = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(raf);
  }, [motion]);

  return <canvas ref={canvasRef} width={560} height={320} style={{ width: "100%", borderRadius: 12, border: "1px solid var(--border)" }} />;
}

function TopBar({ room, name, role, connected }: { room: string; name: string; role: string; connected: boolean }) {
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
        <div
          className="pill"
          style={{
            background: connected ? undefined : "#fee2e2",
            borderColor: connected ? undefined : "#fca5a5",
            color: connected ? undefined : "#7f1d1d"
          }}
        >
          <span className="dot" style={{ background: connected ? "var(--green)" : "#dc2626" }} />
          {connected ? "Connected" : "Offline"}
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
