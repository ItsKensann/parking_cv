import { Link, useParams } from "react-router-dom";
import { FacilityHeader } from "../components/FacilityHeader";
import { LiveSpotMapPreview } from "../components/LiveSpotMapPreview";
import { NavigationHandoffPreview } from "../components/NavigationHandoffPreview";
import { PoweredBySwiftPark } from "../components/PoweredBySwiftPark";
import { RecommendationCard } from "../components/RecommendationCard";
import { ZoneCard } from "../components/ZoneCard";
import { parseFacilitySlug } from "../lib/api";
import { getRecommendation } from "../lib/recommendations";
import { firstAvailableSpot } from "../lib/spotLookup";
import { useFacilityOccupancy } from "../lib/useFacilityOccupancy";
import type { FacilityOccupancy } from "../lib/types";

export function FacilityPage() {
  const { facilitySlug: facilityParam } = useParams();
  const facilitySlug = parseFacilitySlug(facilityParam);
  const { occupancy, loading } = useFacilityOccupancy(facilitySlug);

  if (!facilitySlug) return <UnknownFacility />;
  if (loading || !occupancy) return <FacilitySkeleton />;

  const recommendation = getRecommendation(occupancy);
  const recommendedSpot = firstAvailableSpot(recommendation?.section);

  return (
    <main className="page-stack">
      <FacilityHeader occupancy={occupancy} />

      <RecommendationCard
        recommendation={recommendation}
        facilitySlug={facilitySlug}
        sectionKind={occupancy.facility.sectionKind}
        selectedSpot={recommendedSpot}
      />

      <FacilitySummaryStrip occupancy={occupancy} />

      <section className="content-card">
        <div className="section-heading-row">
          <div>
            <p className="eyebrow">{occupancy.facility.sectionKind} availability</p>
            <h2>Choose where to go</h2>
          </div>
          <Link to={`/f/${facilitySlug}/map`}>Spot map</Link>
        </div>
        <div className="section-list">
          {occupancy.sections.map((section) => (
            <ZoneCard
              key={section.id}
              section={section}
              facilitySlug={facilitySlug}
              isRecommended={recommendation?.section.id === section.id}
            />
          ))}
        </div>
      </section>

      <LiveSpotMapPreview
        facilitySlug={facilitySlug}
        sectionKind={occupancy.facility.sectionKind}
      />

      <NavigationHandoffPreview facilitySlug={facilitySlug} />

      <PoweredBySwiftPark />
    </main>
  );
}

/**
 * Whole-facility totals strip. Lives below the recommendation card so
 * the recommendation stays focused on the recommended section. All
 * three numbers are derived from `occupancy.sections` — the same
 * single source of truth the per-section cards render from.
 */
function FacilitySummaryStrip({ occupancy }: { occupancy: FacilityOccupancy }) {
  const totals = occupancy.sections.reduce(
    (acc, section) => {
      acc.available += section.available;
      acc.occupied += section.occupied;
      acc.mapped += section.mappedSpaces || section.capacity;
      return acc;
    },
    { available: 0, occupied: 0, mapped: 0 },
  );
  const mappedLabel =
    occupancy.facility.type === "surface_lot" ? "Mapped" : "Capacity";

  return (
    <section className="facility-summary" aria-label="Facility totals">
      <div>
        <strong>{totals.available}</strong>
        <span>Available</span>
      </div>
      <div>
        <strong>{totals.occupied}</strong>
        <span>Occupied</span>
      </div>
      <div>
        <strong>{totals.mapped}</strong>
        <span>{mappedLabel}</span>
      </div>
    </section>
  );
}

function FacilitySkeleton() {
  return (
    <main className="page-stack facility-skeleton" aria-busy="true">
      <div className="skeleton-card skeleton-card--hero">
        <div className="skeleton-line skeleton-line--brand" />
        <div className="skeleton-line skeleton-line--title" />
        <div className="skeleton-line skeleton-line--meta" />
        <div className="skeleton-line skeleton-line--pill" />
      </div>
      <div className="skeleton-card skeleton-card--banner">
        <div className="skeleton-line skeleton-line--small" />
        <div className="skeleton-line skeleton-line--title" />
        <div className="skeleton-line skeleton-line--meta" />
        <div className="skeleton-row">
          <div className="skeleton-button" />
          <div className="skeleton-button" />
        </div>
      </div>
      <div className="skeleton-card">
        <div className="skeleton-row skeleton-row--stats">
          <div className="skeleton-stat" />
          <div className="skeleton-stat" />
          <div className="skeleton-stat" />
        </div>
      </div>
      <div className="skeleton-card">
        <div className="skeleton-line skeleton-line--small" />
        <div className="skeleton-line skeleton-line--title" />
        <div className="skeleton-row skeleton-row--rows">
          <div className="skeleton-line skeleton-line--row" />
          <div className="skeleton-line skeleton-line--row" />
          <div className="skeleton-line skeleton-line--row" />
        </div>
      </div>
    </main>
  );
}

function UnknownFacility() {
  return (
    <main className="page-stack">
      <section className="content-card center-card">
        <h1>Facility not found</h1>
        <p>This SwiftPark link is not configured yet.</p>
        <Link className="primary-button" to="/f/brighton-ski-resort">
          View Brighton Parking
        </Link>
      </section>
    </main>
  );
}
