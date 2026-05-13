interface SwiftParkMarkProps {
  size?: "sm" | "md" | "lg";
  variant?: "default" | "on-dark";
  withWordmark?: boolean;
  tagline?: boolean;
  className?: string;
}

/**
 * Reusable SwiftPark brand mark. The pin (a navy teardrop with a white
 * "P" inside) matches the frontend driver app, so the mobile-web feels
 * like the same product family. Tagline is opt-in so the mark can sit
 * in tight spots (top app strip, parked hero, footer) without weight.
 */
export function SwiftParkMark({
  size = "md",
  variant = "default",
  withWordmark = true,
  tagline = false,
  className = "",
}: SwiftParkMarkProps) {
  const cls = ["brand-mark", `brand-mark--${size}`, `brand-mark--${variant}`, className]
    .filter(Boolean)
    .join(" ");
  return (
    <span className={cls}>
      <span className="brand-mark__pin" aria-hidden="true">
        <svg viewBox="0 0 44 52" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="brand-pin-gradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#4FB3FF" />
              <stop offset="100%" stopColor="#1547c4" />
            </linearGradient>
          </defs>
          <path
            d="M22 0 C9 0 0 10 0 22 C0 36 22 52 22 52 C22 52 44 36 44 22 C44 10 35 0 22 0 Z"
            fill="url(#brand-pin-gradient)"
          />
          <text
            x="22"
            y="28"
            textAnchor="middle"
            fontFamily="Inter, system-ui, sans-serif"
            fontSize="22"
            fontWeight="700"
            fill="white"
            letterSpacing="-0.02em"
          >
            P
          </text>
        </svg>
      </span>
      {withWordmark ? (
        <span className="brand-mark__word">
          <span className="brand-mark__word-row">
            <span className="brand-mark__swift">Swift</span>
            <span className="brand-mark__park">Park</span>
          </span>
          {tagline ? (
            <span className="brand-mark__tagline">Stress less. Park better.</span>
          ) : null}
        </span>
      ) : null}
    </span>
  );
}
