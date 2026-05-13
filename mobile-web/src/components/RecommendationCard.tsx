import { Link } from "react-router-dom";
import { buildGoogleMapsDirectionsUrl } from "../lib/navigation";
import type {
  FacilitySlug,
  Recommendation,
  SectionKind,
  Spot,
} from "../lib/types";

interface RecommendationCardProps {
  recommendation: Recommendation | null;
  facilitySlug: FacilitySlug;
  sectionKind: SectionKind;
  selectedSpot?: Spot;
}

/**
 * Compact branded recommendation hero. Describes ONLY the recommended
 * section — whole-facility totals live in their own neutral summary
 * strip below so this card stays focused on the decision: which
 * section to drive to and the two primary actions.
 */
export function RecommendationCard({
  recommendation,
  facilitySlug,
  sectionKind,
  selectedSpot,
}: RecommendationCardProps) {
  const directionsUrl = buildGoogleMapsDirectionsUrl(facilitySlug);
  const spotQuery = selectedSpot
    ? `?spot=${encodeURIComponent(selectedSpot.label || selectedSpot.id)}`
    : "";

  if (!recommendation) {
    return (
      <section className="recommendation-banner recommendation-banner--empty">
        <div className="recommendation-banner__head">
          <span className="recommendation-banner__eyebrow">
            Live availability
          </span>
          <p className="recommendation-banner__line">
            Watching every {sectionKind.toLowerCase()} for open spaces.
          </p>
          <p className="recommendation-banner__reason">
            A recommendation appears here as soon as a section has open spaces.
          </p>
        </div>
        <div className="recommendation-banner__actions">
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
            to={`/f/${facilitySlug}/map`}
          >
            View Live Spot Map
          </Link>
        </div>
      </section>
    );
  }

  const section = recommendation.section;
  const occupiedPct = computeSectionOccupiedPct(section);

  return (
    <section className="recommendation-banner">
      <div className="recommendation-banner__head">
        <span className="recommendation-banner__eyebrow">
          Best option right now
        </span>
        <div className="recommendation-banner__title-row">
          <p className="recommendation-banner__line">
            <strong>{section.label}</strong>
            <span className="recommendation-banner__sep">·</span>
            {section.available} spaces available
          </p>
          {occupiedPct !== null ? (
            <span
              className="recommendation-banner__pct"
              aria-label={`${occupiedPct}% occupied in ${section.label}`}
            >
              {occupiedPct}% occupied
            </span>
          ) : (
            <span
              className="recommendation-banner__pct recommendation-banner__pct--live"
              aria-label={`Live data for ${section.label}`}
            >
              Live
            </span>
          )}
        </div>
        <p className="recommendation-banner__reason">{recommendation.reason}</p>
      </div>
      <div className="recommendation-banner__actions">
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
          View Live Map
        </Link>
      </div>
    </section>
  );
}

/**
 * Compute the recommended section's own occupied-percentage using the
 * section stats source of truth — never mixing in whole-facility
 * totals. Denominator preference:
 *   1. mappedSpaces (calibrated camera stalls)
 *   2. capacity (operational/declared capacity)
 *   3. available + occupied + unknown (last-resort headcount sum)
 * Returns null when no usable denominator is available so the caller
 * can render the "Live" fallback chip instead of a misleading 0%.
 */
function computeSectionOccupiedPct(section: {
  mappedSpaces?: number;
  capacity?: number;
  available: number;
  occupied: number;
  unknown: number;
}): number | null {
  const denominator =
    (section.mappedSpaces && section.mappedSpaces > 0
      ? section.mappedSpaces
      : 0) ||
    (section.capacity && section.capacity > 0 ? section.capacity : 0) ||
    section.available + section.occupied + section.unknown;
  if (!denominator || denominator <= 0) return null;
  const raw = (section.occupied / denominator) * 100;
  if (!Number.isFinite(raw)) return null;
  return Math.min(100, Math.max(0, Math.round(raw)));
}
