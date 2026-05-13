import { Link } from "react-router-dom";
import { buildGoogleMapsDirectionsUrl } from "../lib/navigation";
import type { FacilitySlug } from "../lib/types";

interface NavigationHandoffPreviewProps {
  facilitySlug: FacilitySlug;
}

/**
 * Compact preview card pointing at /f/:slug/navigate. Renders a small
 * stylized map (light gray map base, a few thin "blocks", two
 * intersecting roads, a brand-blue route from a current-location dot
 * to a navy destination pin) so the card visually communicates
 * "Maps gets you to the entrance — SwiftPark guides the final step"
 * without depending on a real map tile.
 *
 * The card itself is an <article> (not a single <Link>) because we
 * need two independent inner links — Open Maps (external) and See
 * handoff (internal route) — and HTML disallows nested anchors.
 */
export function NavigationHandoffPreview({
  facilitySlug,
}: NavigationHandoffPreviewProps) {
  const directionsUrl = buildGoogleMapsDirectionsUrl(facilitySlug);
  return (
    <article
      className="preview-card preview-card--nav"
      aria-label="Navigation handoff preview"
    >
      <div className="preview-card__visual preview-card__visual--nav">
        <svg
          viewBox="0 0 220 110"
          className="preview-card__route-svg"
          aria-hidden="true"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <linearGradient id="preview-route-gradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#3b82f6" />
              <stop offset="100%" stopColor="#1547c4" />
            </linearGradient>
            <linearGradient id="preview-map-bg" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#eef3fa" />
              <stop offset="100%" stopColor="#dde6f3" />
            </linearGradient>
          </defs>

          <rect width="220" height="110" fill="url(#preview-map-bg)" />

          {/* Block tints — quiet "city blocks" */}
          <g opacity="0.55">
            <rect x="0"   y="0"  width="88"  height="46" fill="#e3eaf5" />
            <rect x="88"  y="0"  width="62"  height="46" fill="#dce4f0" />
            <rect x="150" y="0"  width="70"  height="46" fill="#e6eef8" />
            <rect x="0"   y="64" width="72"  height="46" fill="#dce4f0" />
            <rect x="72"  y="64" width="78"  height="46" fill="#e3eaf5" />
            <rect x="150" y="64" width="70"  height="46" fill="#dae3f0" />
          </g>

          {/* Park-like green block (subtle landmark) */}
          <rect x="10" y="6" width="38" height="34" rx="4" fill="#d5e8dd" opacity="0.85" />
          <g fill="#a3d9b6" opacity="0.7">
            <circle cx="20" cy="20" r="3" />
            <circle cx="32" cy="26" r="2.5" />
            <circle cx="38" cy="16" r="2.5" />
          </g>

          {/* Major roads (white) + thin road shoulders for depth */}
          <g stroke="white" strokeLinecap="butt">
            <line x1="0"  y1="55"  x2="220" y2="55" strokeWidth="9" />
            <line x1="88" y1="0"   x2="88"  y2="110" strokeWidth="9" />
            <line x1="150" y1="0"  x2="150" y2="110" strokeWidth="9" />
          </g>
          <g stroke="#c8d2e2" strokeWidth="0.5" opacity="0.7" fill="none">
            <line x1="0" y1="50.5" x2="220" y2="50.5" />
            <line x1="0" y1="59.5" x2="220" y2="59.5" />
            <line x1="83.5" y1="0" x2="83.5" y2="110" />
            <line x1="92.5" y1="0" x2="92.5" y2="110" />
            <line x1="145.5" y1="0" x2="145.5" y2="110" />
            <line x1="154.5" y1="0" x2="154.5" y2="110" />
          </g>

          {/* Centerline dashes on the main horizontal road */}
          <line
            x1="0"
            y1="55"
            x2="220"
            y2="55"
            stroke="#dee3ec"
            strokeWidth="0.8"
            strokeDasharray="6 6"
          />

          {/* Route shadow (under route, brand-blue at low opacity) */}
          <path
            d="M 22 94 L 22 55 L 88 55 L 88 20 L 195 20"
            stroke="rgba(25, 118, 255, 0.18)"
            strokeWidth="10"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
          />
          {/* Route */}
          <path
            d="M 22 94 L 22 55 L 88 55 L 88 20 L 195 20"
            stroke="url(#preview-route-gradient)"
            strokeWidth="3.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
          />

          {/* Current location dot (bottom-left) */}
          <g transform="translate(22, 94)">
            <circle r="9" fill="rgba(25, 118, 255, 0.18)" />
            <circle r="5.5" fill="#1976ff" />
            <circle r="2" fill="white" />
          </g>

          {/* Destination pin (top-right, near the brand-blue P) */}
          <g transform="translate(195, 16)">
            <ellipse cx="0" cy="20" rx="7" ry="2" fill="rgba(10,31,68,0.18)" />
            <path
              d="M 0 -12 C -8 -12 -13 -6 -13 1 c 0 8 13 19 13 19 s 13 -11 13 -19 C 13 -6 8 -12 0 -12 Z"
              fill="#1547c4"
            />
            <circle cx="0" cy="-2" r="5" fill="white" />
            <text
              x="0"
              y="1.5"
              textAnchor="middle"
              fontFamily="Inter, system-ui, sans-serif"
              fontSize="7"
              fontWeight="700"
              fill="#1547c4"
            >
              P
            </text>
          </g>
        </svg>
        <span className="preview-card__badge preview-card__badge--soft">
          Final-step guidance
        </span>
      </div>
      <div className="preview-card__body">
        <h3>Maps + SwiftPark</h3>
        <p>
          Maps gets you to the entrance. SwiftPark guides the final parking
          step.
        </p>
        <span className="preview-card__cta-row">
          <a
            className="preview-card__cta-link"
            href={directionsUrl}
            target="_blank"
            rel="noreferrer"
            onClick={(event) => event.stopPropagation()}
          >
            Open Maps
          </a>
          <Link
            to={`/f/${facilitySlug}/navigate`}
            className="preview-card__cta"
          >
            See handoff →
          </Link>
        </span>
      </div>
    </article>
  );
}
