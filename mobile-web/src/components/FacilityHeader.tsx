import { Link } from "react-router-dom";
import { LiveUpdatePill } from "./LiveUpdatePill";
import { StatusBadge } from "./StatusBadge";
import { buildGoogleMapsDirectionsUrl } from "../lib/navigation";
import type {
  FacilityOccupancy,
  FacilitySlug,
  Recommendation,
  Spot,
} from "../lib/types";

interface FacilityHeaderProps {
  occupancy: FacilityOccupancy;
  facilitySlug: FacilitySlug;
  recommendation: Recommendation | null;
  selectedSpot?: Spot;
}

/**
 * Consolidated facility hero: header info + integrated recommendation
 * + primary actions in a single softly-framed card. Replaces the prior
 * facility-header → recommendation-card split that made the top of the
 * page feel like two competing blocks. Data flow is unchanged — the
 * caller still passes the resolved `recommendation` from
 * `getRecommendation(occupancy)`.
 */
export function FacilityHeader({
  occupancy,
  facilitySlug,
  recommendation,
  selectedSpot,
}: FacilityHeaderProps) {
  const directionsUrl = buildGoogleMapsDirectionsUrl(facilitySlug);
  const spotQuery = selectedSpot
    ? `?spot=${encodeURIComponent(selectedSpot.label || selectedSpot.id)}`
    : "";

  return (
    <header className="facility-hero">
      <div className="facility-hero__top">
        <LiveUpdatePill
          dataState={occupancy.dataState}
          updatedAt={occupancy.updatedAt}
        />
        <StatusBadge status={occupancy.status} compact />
      </div>

      <div className="facility-hero__title">
        <h1>{occupancy.facility.publicName}</h1>
        <p>{occupancy.facility.location}</p>
      </div>

      {occupancy.message ? (
        <p className="degraded-note">{occupancy.message}</p>
      ) : null}

      <div className="facility-hero__divider" aria-hidden="true" />

      {recommendation ? (
        <div className="facility-hero__recommendation">
          <span className="facility-hero__rec-eyebrow">
            Best option right now
          </span>
          <p className="facility-hero__rec-line">
            <strong>{recommendation.section.label}</strong>
            <span className="dot-sep">·</span>
            {recommendation.section.available} spaces open
          </p>
          <p className="facility-hero__rec-reason">{recommendation.reason}</p>
        </div>
      ) : (
        <p className="facility-hero__empty">
          Live availability will appear here as soon as the facility reports in.
        </p>
      )}

      <div className="facility-hero__actions action-row">
        <a
          className="primary-button"
          href={directionsUrl}
          target="_blank"
          rel="noreferrer"
        >
          Get Directions
        </a>
        <Link
          className="secondary-button"
          to={`/f/${facilitySlug}/map${spotQuery}`}
        >
          View Live Spot Map
        </Link>
      </div>
    </header>
  );
}
