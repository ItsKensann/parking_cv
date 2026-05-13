import type { FacilityOccupancy } from "../lib/types";

interface MapPreviewProps {
  occupancy: FacilityOccupancy;
}

/**
 * Stylised map placeholder for the navigation handoff page. Uses inline
 * SVG so there is no map-tile dependency in this pass — Phase 7B can
 * swap this for a real interactive map without touching the surrounding
 * page layout.
 */
export function MapPreview({ occupancy }: MapPreviewProps) {
  const label =
    occupancy.facility.type === "garage"
      ? "Garage entrance"
      : "Facility entrance";

  return (
    <section className="map-preview-card" aria-label="Route preview placeholder">
      <svg
        className="map-preview-svg"
        viewBox="0 0 320 188"
        preserveAspectRatio="xMidYMid slice"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="map-preview-bg" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#eef3fb" />
            <stop offset="100%" stopColor="#dde6f5" />
          </linearGradient>
          <linearGradient id="map-preview-water" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#bfdbfe" />
            <stop offset="100%" stopColor="#a5cdfa" />
          </linearGradient>
          <linearGradient id="map-preview-route" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#3b82f6" />
            <stop offset="100%" stopColor="#1547c4" />
          </linearGradient>
        </defs>

        <rect width="320" height="188" fill="url(#map-preview-bg)" />

        {/* Block tints — quiet "city blocks" pattern */}
        <g opacity="0.6">
          <rect x="0" y="0" width="120" height="62" fill="#e6ecf6" />
          <rect x="120" y="0" width="120" height="62" fill="#dde5f1" />
          <rect x="240" y="0" width="80" height="62" fill="#e2e8f4" />
          <rect x="0" y="122" width="200" height="66" fill="#dde5f1" />
          <rect x="200" y="122" width="120" height="66" fill="#e6ecf6" />
        </g>

        {/* Park / green area */}
        <rect x="24" y="14" width="74" height="42" rx="6" fill="#d4ebde" opacity="0.85" />
        <g fill="#a3d9b6" opacity="0.7">
          <circle cx="42" cy="32" r="4" />
          <circle cx="58" cy="38" r="3" />
          <circle cx="74" cy="28" r="4" />
          <circle cx="82" cy="44" r="3" />
        </g>

        {/* Water edge (subtle) */}
        <path
          d="M 286 0 C 296 38 280 70 308 110 C 318 144 296 170 320 188 L 320 0 Z"
          fill="url(#map-preview-water)"
          opacity="0.6"
        />

        {/* Major roads — horizontal + vertical */}
        <g stroke="white" strokeLinecap="butt">
          <line x1="0" y1="62" x2="320" y2="62" strokeWidth="9" />
          <line x1="0" y1="122" x2="320" y2="122" strokeWidth="9" />
          <line x1="120" y1="0" x2="120" y2="188" strokeWidth="9" />
          <line x1="240" y1="0" x2="240" y2="188" strokeWidth="9" />
        </g>
        <g stroke="#c8d2e2" strokeWidth="0.5" opacity="0.6" fill="none">
          <line x1="0" y1="57.5" x2="320" y2="57.5" />
          <line x1="0" y1="66.5" x2="320" y2="66.5" />
          <line x1="0" y1="117.5" x2="320" y2="117.5" />
          <line x1="0" y1="126.5" x2="320" y2="126.5" />
          <line x1="115.5" y1="0" x2="115.5" y2="188" />
          <line x1="124.5" y1="0" x2="124.5" y2="188" />
        </g>

        {/* Route — from current location (lower-left) to entrance (upper-right) */}
        <path
          d="M 46 158 L 46 122 L 120 122 L 120 62 L 232 62"
          stroke="url(#map-preview-route)"
          strokeWidth="5"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />
        <path
          d="M 46 158 L 46 122 L 120 122 L 120 62 L 232 62"
          stroke="rgba(25, 118, 255, 0.18)"
          strokeWidth="11"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />

        {/* Current location dot */}
        <g transform="translate(46, 158)">
          <circle r="11" fill="rgba(25, 118, 255, 0.20)" />
          <circle r="6" fill="#1976ff" />
          <circle r="3" fill="white" />
        </g>

        {/* Destination pin */}
        <g transform="translate(232, 62)">
          <ellipse cx="0" cy="22" rx="9" ry="3" fill="rgba(10,31,68,0.18)" />
          <path
            d="M 0 -24 C -10 -24 -16 -16 -16 -8 c 0 12 16 28 16 28 s 16 -16 16 -28 C 16 -16 10 -24 0 -24 Z"
            fill="#1547c4"
          />
          <circle cx="0" cy="-9" r="6" fill="white" />
          <text
            x="0"
            y="-6.5"
            textAnchor="middle"
            fontSize="9"
            fontWeight="700"
            fill="#1547c4"
          >
            P
          </text>
        </g>
      </svg>

      <span className="map-preview-overlay">{label}</span>
    </section>
  );
}
