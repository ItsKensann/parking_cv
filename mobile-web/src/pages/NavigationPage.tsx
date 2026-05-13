import { Link, useParams, useSearchParams } from "react-router-dom";
import { DirectionsButton } from "../components/DirectionsButton";
import { PoweredBySwiftPark } from "../components/PoweredBySwiftPark";
import { parseFacilitySlug } from "../lib/api";
import { buildFinalGuidance, buildSpotRoute } from "../lib/navigation";
import { getRecommendation } from "../lib/recommendations";
import { findSectionForSpot, findSpot, firstAvailableSpot } from "../lib/spotLookup";
import { useFacilityOccupancy } from "../lib/useFacilityOccupancy";

export function NavigationPage() {
  const { facilitySlug: facilityParam } = useParams();
  const [searchParams] = useSearchParams();
  const facilitySlug = parseFacilitySlug(facilityParam);
  const { occupancy, loading } = useFacilityOccupancy(facilitySlug);

  if (!facilitySlug) return <NavigationMessage title="Facility not found" />;
  if (loading || !occupancy)
    return <NavigationMessage title="Preparing navigation" loading />;

  const recommendation = getRecommendation(occupancy);
  const selectedSpot =
    findSpot(occupancy, searchParams.get("spot")) ??
    firstAvailableSpot(recommendation?.section);
  const selectedSection =
    findSectionForSpot(occupancy, selectedSpot) ?? recommendation?.section;

  return (
    <main className="page-stack">
      <header className="compact-page-header">
        <Link to={`/f/${facilitySlug}`} className="back-link" aria-label="Back">
          Back
        </Link>
        <div>
          <p className="eyebrow">{occupancy.facility.publicName}</p>
          <h1>Navigate to Parking</h1>
        </div>
      </header>

      <section className="content-card route-preview-card">
        <span className="step-pill">
          <span>1</span> Drive to the facility
        </span>
        <h2>Open Maps to the entrance</h2>
        <p>
          Apple or Google Maps gets you to the entrance. SwiftPark guides the
          final parking step from there.
        </p>
        <div className="action-row" style={{ marginTop: 16 }}>
          <DirectionsButton facilitySlug={facilitySlug} label="Open in Maps" />
        </div>
      </section>

      <section className="content-card">
        <span className="step-pill">
          <span>2</span> Park on-site
        </span>
        <h2>{selectedSection ? selectedSection.label : "Recommended parking area"}</h2>
        <p>{buildFinalGuidance(selectedSection)}</p>
        {selectedSpot ? (
          <div className="spot-mini-summary">
            <span>Your spot</span>
            <strong>{selectedSpot.label}</strong>
          </div>
        ) : null}
      </section>

      {selectedSpot ? (
        <Link
          className="primary-button full-width"
          to={buildSpotRoute(facilitySlug, "parked", selectedSpot)}
        >
          I've Parked
        </Link>
      ) : (
        <Link className="primary-button full-width" to={`/f/${facilitySlug}/map`}>
          Choose a Spot
        </Link>
      )}

      <PoweredBySwiftPark />
    </main>
  );
}

function NavigationMessage({
  title,
  loading = false,
}: {
  title: string;
  loading?: boolean;
}) {
  return (
    <main className="page-stack loading-page">
      {loading ? <div className="loading-ring" /> : null}
      <p>{title}</p>
    </main>
  );
}
