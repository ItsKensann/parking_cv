import { Link } from "react-router-dom";
import { buildGoogleMapsDirectionsUrl } from "../lib/navigation";
import type { FacilitySlug, Recommendation, Spot } from "../lib/types";

interface RecommendationCardProps {
  facilitySlug: FacilitySlug;
  recommendation: Recommendation | null;
  selectedSpot?: Spot;
}

export function RecommendationCard({
  facilitySlug,
  recommendation,
  selectedSpot,
}: RecommendationCardProps) {
  if (!recommendation) {
    return (
      <section className="recommendation-card">
        <p className="card-kicker">Recommended</p>
        <h2>Check live availability</h2>
        <p>No recommended section is available yet.</p>
      </section>
    );
  }

  const section = recommendation.section;
  const spotQuery = selectedSpot
    ? `?spot=${encodeURIComponent(selectedSpot.label || selectedSpot.id)}`
    : "";

  return (
    <section className="recommendation-card">
      <p className="card-kicker">Recommended</p>
      <div className="recommendation-main">
        <div>
          <h2>{section.label}</h2>
          <p>
            <strong>{section.available}</strong> spaces available
          </p>
          <span>{recommendation.reason}</span>
        </div>
        <div className="recommendation-meter">
          <span>{Math.round(section.occupancyPct)}%</span>
          <small>occupied</small>
        </div>
      </div>
      <div className="action-row">
        <a
          className="primary-button"
          href={buildGoogleMapsDirectionsUrl(facilitySlug)}
          target="_blank"
          rel="noreferrer"
        >
          Get Directions
        </a>
        <Link className="secondary-button" to={`/f/${facilitySlug}/map${spotQuery}`}>
          View Live Spot Map
        </Link>
      </div>
    </section>
  );
}

