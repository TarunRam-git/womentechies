import Link from "next/link";

const roles = [
  {
    key: "sign",
    title: "Sign Language User",
    desc: "Send and receive communication through clear visual support.",
    Icon: () => (
      <div
        style={{
          width: "100%",
          height: "160px",
          borderRadius: 14,
          background: "#f4ecdf",
          backgroundImage: "url('/background-3.svg')",
          backgroundRepeat: "no-repeat",
          backgroundPosition: "center",
          backgroundSize: "180px auto"
        }}
      />
    )
  },
  {
    key: "speech",
    title: "Speech User",
    desc: "Speak naturally while the interface helps share meaning clearly.",
    Icon: () => (
      <svg width="140" height="140" viewBox="0 0 140 140" role="presentation" aria-hidden>
        <rect x="56" y="26" width="48" height="82" rx="24" fill="#c7a782" />
        <rect x="64" y="104" width="32" height="10" rx="5" fill="#b08e68" />
        <rect x="52" y="112" width="56" height="12" rx="6" fill="#8b6b4d" />
      </svg>
    )
  }
];

export default function RolePage() {
  return (
    <main className="landing">
      <div className="page role-page">
        <header className="topbar">
          <div className="topbar__left">
            <div className="topbar__title">Choose Your Role</div>
          </div>
          <div className="pill-btn">Simple Choice</div>
        </header>

        <section className="hero-card role-card">
          <div className="hero__rail">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>

          <div className="role-body">
            <div className="role-header">
              <h1 className="hero__title role-title">How would you like to connect?</h1>
              <p className="role-sub">Pick the experience that matches how you want to communicate in the room.</p>
            </div>

            <div className="role-options">
              {roles.map((role) => (
                <Link key={role.key} href={`/join?role=${role.key}`} className="role-card__option">
                  <div className="role-illustration">
                    <role.Icon />
                  </div>
                  <div className="role-text">
                    <div className="role-name">{role.title}</div>
                    <div className="role-desc">{role.desc}</div>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
