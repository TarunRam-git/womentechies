"use client";

import Link from "next/link";

const features = [
  { iconSrc: "/background-1.svg", title: "Accessibility", desc: "Designed to reduce friction in conversation." },
  { iconSrc: "/background-2.svg", title: "Inclusivity", desc: "Built for different communication styles." },
  { iconSrc: "/background-1.svg", title: "Interactive", desc: "Designed to communicate effortlessly." }
];

export default function LandingPage() {
  return (
    <main className="landing">
      <div className="page landing-page">
        <header className="topbar">
          <div className="topbar__left">
            <div className="topbar__title">Home Experience</div>
          </div>
          <div className="pill-btn">SIGN-SYNC</div>
        </header>

        <section className="hero-card">
          <div className="hero__rail">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>

          <div className="hero__body">
            <div className="hero__copy">
              <div className="hero__eyebrow">Sign-Sync</div>
              <h1 className="hero__title">Bridging communication between sign and speech</h1>

              <div className="hero__actions">
                <Link className="btn btn-solid" href="/role">
                  Start Connecting
                </Link>
                <Link className="btn btn-ghost" href="#learn">
                  Learn More
                </Link>
              </div>

              <div className="hero__tagline">
                <strong>What is Sign Language?</strong> A visual language using movement, hand shape, and expression to
                communicate clearly and naturally.
              </div>

              <div className="feature-grid" id="learn">
                {features.map((item) => (
                  <article key={item.title} className="feature-card">
                    <div className="feature-icon" aria-hidden>
                      <img src={item.iconSrc} alt="" className="feature-icon__img" />
                    </div>
                    <div className="feature-title">{item.title}</div>
                    <div className="feature-desc">{item.desc}</div>
                  </article>
                ))}
              </div>
            </div>

            <div className="hero__image">
              <img src="/rectangle-16.png" alt="Hands forming a sign" />
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
