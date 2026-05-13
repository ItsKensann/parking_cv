import { Link } from "react-router-dom";
import { StatusBadge } from "./StatusBadge";
import type { FacilitySlug, ParkingSection } from "../lib/types";

interface SectionCardProps {
  section: ParkingSection;
  facilitySlug: FacilitySlug;
  isRecommended?: boolean;
}

/**
 * Zone/Level card — also the row's tap target. Clicking the card
 * deep-links to the spot map with the section pre-selected via the
 * `?section=` query param. Visually unchanged from before plus a
 * small chevron affordance on the right of the title row.
 */
export function SectionCard({
  section,
  facilitySlug,
  isRecommended = false,
}: SectionCardProps) {
  const pct = Math.min(100, Math.max(0, section.occupancyPct));
  const totalSpaces =
    section.mappedSpaces > 0 ? section.mappedSpaces : section.capacity;
  const to = `/f/${facilitySlug}/map?section=${encodeURIComponent(section.id)}`;
  return (
    <Link
      to={to}
      className={`section-card ${isRecommended ? "section-card-recommended" : ""}`}
      aria-label={`Open ${section.label} in the live spot map`}
    >
      <div className="section-card-title-row">
        <div className="section-card-title">
          <h3>{section.label}</h3>
          {isRecommended ? <span className="mini-pill">Best now</span> : null}
        </div>
        <span className="section-card-chevron" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path
              d="m9 6 6 6-6 6"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </div>
      <div className="section-card-metrics">
        <div>
          <strong>{section.available} open</strong>
          <p>
            {section.sourceLabel} · {totalSpaces} spaces
          </p>
        </div>
        <StatusBadge status={section.status} compact />
      </div>
      <div
        className="section-progress"
        data-status={section.status}
        aria-hidden="true"
      >
        <span style={{ width: `${pct}%` }} />
      </div>
    </Link>
  );
}
