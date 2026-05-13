import { Link, useParams } from "react-router-dom";
import { FacilityHeader } from "../components/FacilityHeader";
import { PoweredBySwiftPark } from "../components/PoweredBySwiftPark";
import { RecommendationCard } from "../components/RecommendationCard";
import { ZoneCard } from "../components/ZoneCard";
import { parseFacilitySlug } from "../lib/api";
import { getRecommendation } from "../lib/recommendations";
import { firstAvailableSpot } from "../lib/spotLookup";
import { useFacilityOccupancy } from "../lib/useFacilityOccupancy";

export function FacilityPage() {
  const { facilitySlug: facilityParam } = useParams();
  const facilitySlug = parseFacilitySlug(facilityParam);
  const { occupancy, loading } = useFacilityOccupancy(facilitySlug);

  if (!facilitySlug) return <UnknownFacility />;
  if (loading || !occupancy) return <LoadingPage label="Loading parking availability" />;

  const recommendation = getRecommendation(occupancy);
  const recommendedSpot = firstAvailableSpot(recommendation?.section);

  return (
    <main className="page-stack">
      <FacilityHeader occupancy={occupancy} />
      <RecommendationCard
        facilitySlug={facilitySlug}
        recommendation={recommendation}
        selectedSpot={recommendedSpot}
      />

      <section className="summary-card">
        <div>
          <span>Available</span>
          <strong>{occupancy.available}</strong>
        </div>
        <div>
          <span>Occupied</span>
          <strong>{occupancy.occupied}</strong>
        </div>
        <div>
          <span>{occupancy.facility.type === "surface_lot" ? "Mapped" : "Capacity"}</span>
          <strong>{occupancy.capacity}</strong>
        </div>
      </section>

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
              isRecommended={recommendation?.section.id === section.id}
            />
          ))}
        </div>
      </section>

      <PoweredBySwiftPark />
    </main>
  );
}

function LoadingPage({ label }: { label: string }) {
  return (
    <main className="page-stack loading-page">
      <div className="loading-ring" />
      <p>{label}</p>
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
