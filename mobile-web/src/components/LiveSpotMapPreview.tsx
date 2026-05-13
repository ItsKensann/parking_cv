import { Link } from "react-router-dom";
import type { FacilitySlug, SectionKind } from "../lib/types";

interface LiveSpotMapPreviewProps {
  facilitySlug: FacilitySlug;
  sectionKind: SectionKind;
}

/**
 * Compact preview card pointing at /f/:slug/map. Renders a stylized
 * miniature parking lot (two rows of stalls separated by a painted
 * lane, a handful of cars in red/silver/navy, one stall lit up in
 * SwiftPark blue as the "selected" state) so the card teases the
 * spot-visualization feature without committing to GLB / live data
 * wiring. Phase 7B can swap `preview-card__visual` for the real
 * SpotMap3D without touching the surrounding card chrome.
 */
export function LiveSpotMapPreview({
  facilitySlug,
  sectionKind,
}: LiveSpotMapPreviewProps) {
  const sectionWord = sectionKind === "Zone" ? "zone" : "level";
  return (
    <Link
      to={`/f/${facilitySlug}/map`}
      className="preview-card preview-card--map"
      aria-label="Open the live spot map"
    >
      <div className="preview-card__visual preview-card__visual--map">
        <svg
          viewBox="0 0 220 110"
          className="preview-card__lot-svg"
          aria-hidden="true"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <linearGradient id="lot-selected" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" />
              <stop offset="100%" stopColor="#1547c4" />
            </linearGradient>
            <linearGradient id="lot-car-navy" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#1e3a8a" />
              <stop offset="100%" stopColor="#0a1f44" />
            </linearGradient>
            <linearGradient id="lot-car-silver" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#cbd5e1" />
              <stop offset="100%" stopColor="#94a3b8" />
            </linearGradient>
            <linearGradient id="lot-car-red" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#fb7185" />
              <stop offset="100%" stopColor="#e11d48" />
            </linearGradient>
            <linearGradient id="lot-floor" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f4f7fc" />
              <stop offset="100%" stopColor="#e6ecf6" />
            </linearGradient>
          </defs>

          <rect width="220" height="110" fill="url(#lot-floor)" />

          {/* Drive lane with painted dashes */}
          <rect x="0" y="49" width="220" height="12" fill="#d6dde9" />
          <g stroke="white" strokeWidth="1.4" strokeDasharray="8 6">
            <line x1="0" y1="55" x2="220" y2="55" />
          </g>

          {/* Painted stalls — top row */}
          <g stroke="white" strokeWidth="1.2" fill="rgba(255,255,255,0.55)">
            {[10, 38, 66, 94, 122, 150, 178].map((x) => (
              <rect key={`top-${x}`} x={x} y="8" width="22" height="34" rx="2" />
            ))}
          </g>

          {/* Cars in the top row */}
          {/* navy sedan */}
          <g transform="translate(11, 12)">
            <rect width="20" height="26" rx="4" fill="url(#lot-car-navy)" />
            <rect x="3" y="3" width="14" height="9" rx="2" fill="rgba(255,255,255,0.18)" />
          </g>
          {/* empty stall (38-60) */}
          {/* silver suv */}
          <g transform="translate(67, 11)">
            <rect width="20" height="28" rx="4" fill="url(#lot-car-silver)" />
            <rect x="3" y="3" width="14" height="9" rx="2" fill="rgba(255,255,255,0.30)" />
          </g>
          {/* red coupe */}
          <g transform="translate(95, 12)">
            <rect width="20" height="26" rx="4" fill="url(#lot-car-red)" />
            <rect x="3" y="3" width="14" height="9" rx="2" fill="rgba(255,255,255,0.18)" />
          </g>
          {/* selected stall (122-144) - blue spotlight, no car */}
          <g transform="translate(122, 8)">
            <rect width="22" height="34" rx="3" fill="url(#lot-selected)" />
            <rect x="3" y="6" width="16" height="22" rx="3" fill="rgba(255,255,255,0.18)" />
            <text
              x="11"
              y="22"
              textAnchor="middle"
              fontFamily="Inter, system-ui, sans-serif"
              fontSize="9"
              fontWeight="700"
              fill="white"
            >
              P
            </text>
          </g>
          {/* empty (150-172) */}
          {/* navy sedan */}
          <g transform="translate(179, 12)">
            <rect width="20" height="26" rx="4" fill="url(#lot-car-navy)" />
            <rect x="3" y="3" width="14" height="9" rx="2" fill="rgba(255,255,255,0.18)" />
          </g>

          {/* Painted stalls — bottom row */}
          <g stroke="white" strokeWidth="1.2" fill="rgba(255,255,255,0.55)">
            {[10, 38, 66, 94, 122, 150, 178].map((x) => (
              <rect key={`bot-${x}`} x={x} y="68" width="22" height="34" rx="2" />
            ))}
          </g>

          {/* Cars in the bottom row */}
          <g transform="translate(11, 72)">
            <rect width="20" height="26" rx="4" fill="url(#lot-car-silver)" />
            <rect x="3" y="3" width="14" height="9" rx="2" fill="rgba(255,255,255,0.30)" />
          </g>
          <g transform="translate(39, 72)">
            <rect width="20" height="26" rx="4" fill="url(#lot-car-navy)" />
            <rect x="3" y="3" width="14" height="9" rx="2" fill="rgba(255,255,255,0.18)" />
          </g>
          {/* empty 66-88 */}
          <g transform="translate(95, 72)">
            <rect width="20" height="26" rx="4" fill="url(#lot-car-red)" />
            <rect x="3" y="3" width="14" height="9" rx="2" fill="rgba(255,255,255,0.18)" />
          </g>
          {/* empty 122-144 */}
          <g transform="translate(151, 72)">
            <rect width="20" height="26" rx="4" fill="url(#lot-car-navy)" />
            <rect x="3" y="3" width="14" height="9" rx="2" fill="rgba(255,255,255,0.18)" />
          </g>
          <g transform="translate(179, 72)">
            <rect width="20" height="26" rx="4" fill="url(#lot-car-silver)" />
            <rect x="3" y="3" width="14" height="9" rx="2" fill="rgba(255,255,255,0.30)" />
          </g>
        </svg>
        <span className="preview-card__badge">Signature feature</span>
      </div>
      <div className="preview-card__body">
        <h3>Live spot map</h3>
        <p>See open {sectionWord}s and pick a specific spot before you enter.</p>
        <span className="preview-card__cta">View Spot Map →</span>
      </div>
    </Link>
  );
}
