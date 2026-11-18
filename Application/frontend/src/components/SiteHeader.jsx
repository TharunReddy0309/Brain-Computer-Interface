const links = [
  { label: "Overview", href: "/" },
  { label: "Research", href: "/research" },
  { label: "Upload & Analysis", href: "/evaluate" },
];

export default function SiteHeader({ activePage, online, apiAddress }) {
  const showStatus = typeof online === "boolean";

  return (
    <header className={`header${showStatus ? " eval-header" : ""}`}>
      <a className="logo" href="/" aria-label="Neuro overview">
        NEURO <i /> <span>Brain computer interface</span>
      </a>
      <nav aria-label="Main navigation">
        {links.map(({ label, href }, index) => (
          <a
            className={index === activePage ? "active" : ""}
            href={href}
            key={label}
          >
            {label}
          </a>
        ))}
      </nav>
      {showStatus && (
        <div className={`engine-status ${online ? "online" : "offline"}`} role="status">
          <i />
          <span>{online ? "Local engine online" : "Local engine offline"}</span>
          <span className="api-address">{apiAddress}</span>
        </div>
      )}
      {!showStatus && (
        <div className="campus-status">
          <i />
          <span>IIIT SRI CITY</span>
        </div>
      )}
    </header>
  );
}
