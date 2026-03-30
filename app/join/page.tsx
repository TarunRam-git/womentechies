"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";

export default function JoinPage() {
  const router = useRouter();
  const params = useSearchParams();
  const role = params.get("role") || "sign";
  const [name, setName] = useState("");
  const [room, setRoom] = useState("");

  const go = () => {
    if (!room) return;
    const url = `/comm?room=${encodeURIComponent(room)}&role=${role}${
      name ? `&name=${encodeURIComponent(name)}` : ""
    }`;
    router.push(url);
  };

  return (
    <main className="landing">
      <div className="page join-page">
        <header className="topbar">
          <div className="topbar__title">Room Join</div>
          <div className="pill-btn">Focused Entry</div>
        </header>

        <section className="hero-card join-card">
          <div className="hero__rail">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>

          <div className="join-window">
            <div className="join-header">
              <h1 className="join-title">Join your conversation</h1>
              <p className="join-sub">
                Enter your details to move into the shared communication room.
              </p>
            </div>

            <div className="join-form">
              <input
                className="join-input"
                placeholder="Your name"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
              <input
                className="join-input"
                placeholder="Enter room ID"
                value={room}
                onChange={(e) => setRoom(e.target.value)}
              />
              <button className="join-button" disabled={!room} onClick={go}>
                Join Room
              </button>
              <Link className="join-switch" href="/role">
                Switch Roles
              </Link>
              <Link className="join-switch" href="/">
                Back to Home
              </Link>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
