import { Link, useParams } from "react-router-dom";
import { PoweredBySwiftPark } from "../components/PoweredBySwiftPark";
import { StatusBadge } from "../components/StatusBadge";
import { parseFacilitySlug } from "../lib/api";
import { buildSpotRoute } from "../lib/navigation";
import { findSectionForSpot, findSpot } from "../lib/spotLookup";
import { useFacilityOccupancy } from "../lib/useFacilityOccupancy";

export function SpotDetailsPage() {
  const { facilitySlug: facilityParam, spotId } = useParams();
  const facilitySlug = parseFacilitySlug(facilityParam);
  const { occupancy, loading } = useFacilityOccupancy(facilitySlug);

  if (!facilitySlug) return <DetailsMessage title="Facility not found" />;
  if (loading || !occupancy)
    return <DetailsMessage title="Loading spot details" loading />;

  const spot = findSpot(occupancy, spotId);
  const section = findSectionForSpot(occupancy, spot);

  if (!spot || !section) {
    return (
      <main className="page-stack">
        <section className="content-card center-card">
          <h1>Spot not found</h1>
          <p>That spot isn't in the current SwiftPark view.</p>
          <Link className="primary-button" to={`/f/${facilitySlug}/map`}>
            View Spot Map
          </Link>
        </section>
      </main>
    );
  }

  const isAvailable = spot.status === "available";

  return (
    <main className="page-stack">
      <header className="compact-page-header">
        <Link
          to={`/f/${facilitySlug}/map?spot=${encodeURIComponent(spot.label)}`}
          className="back-link"
          aria-label="Back to spot map"
        >
          Back
        </Link>
        <div>
          <p className="eyebrow">{occupancy.facility.publicName}</p>
          <h1>{spot.label}</h1>
        </div>
      </header>

      <section className="content-card spot-details-card">
        <div className="large-spot-badge">{spot.label}</div>
        <div>
          <p className="eyebrow">{section.label}</p>
          <h2>{isAvailable ? "Ready to park" : "Spot status"}</h2>
          <StatusBadge status={isAvailable ? "open" : "busy"} />
        </div>
        <p>
          SwiftPark hands off to Maps for the facility entrance, then guides the
          final parking step on site.
        </p>
      </section>

      <section className="content-card">
        <p className="eyebrow">Parking details</p>
        <div className="detail-list">
          <span>Facility</span>
          <strong>{occupancy.facility.name}</strong>
          <span>{occupancy.facility.sectionKind}</span>
          <strong>{section.label}</strong>
          <span>Status</span>
          <strong>{spot.status}</strong>
          <span>Source</span>
          <strong>{section.sourceLabel}</strong>
        </div>
      </section>

      <Link
        className="primary-button full-width"
        to={buildSpotRoute(facilitySlug, "navigate", spot)}
      >
        Navigate to Spot
      </Link>
      <Link
        className="secondary-button full-width"
        to={buildSpotRoute(facilitySlug, "parked", spot)}
      >
        Confirm Parked
      </Link>

      <PoweredBySwiftPark />
    </main>
  );
}

function DetailsMessage({
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
