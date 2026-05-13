import { Link, useParams, useSearchParams } from "react-router-dom";
import { PoweredBySwiftPark } from "../components/PoweredBySwiftPark";
import { parseFacilitySlug } from "../lib/api";
import { findSectionForSpot, findSpot } from "../lib/spotLookup";
import { useFacilityOccupancy } from "../lib/useFacilityOccupancy";

export function ParkedPage() {
  const { facilitySlug: facilityParam } = useParams();
  const [searchParams] = useSearchParams();
  const facilitySlug = parseFacilitySlug(facilityParam);
  const { occupancy, loading } = useFacilityOccupancy(facilitySlug);

  if (!facilitySlug) return <ParkedMessage title="Facility not found" />;
  if (loading || !occupancy) return <ParkedMessage title="Confirming parking" loading />;

  const spot = findSpot(occupancy, searchParams.get("spot"));
  const section = findSectionForSpot(occupancy, spot);

  return (
    <main className="page-stack parked-page">
      <section className="parked-hero">
        <div className="confetti confetti-one" />
        <div className="confetti confetti-two" />
        <div className="confetti confetti-three" />
        <div className="check-badge" aria-hidden="true" />
        <p className="eyebrow">Parking confirmed</p>
        <h1>You're Parked</h1>
        <p>Your parking location is saved for this SwiftPark session.</p>
      </section>

      <section className="content-card">
        <div className="detail-list">
          <span>Facility</span>
          <strong>{occupancy.facility.name}</strong>
          <span>Area</span>
          <strong>{section?.label ?? "Selected area"}</strong>
          <span>Spot</span>
          <strong>{spot?.label ?? searchParams.get("spot") ?? "Not selected"}</strong>
        </div>
      </section>

      <section className="content-card">
        <p className="eyebrow">Reminder</p>
        <h2>Check posted signs before leaving your car</h2>
        <p>SwiftPark helps with live availability. Facility rules and posted signs still apply.</p>
      </section>

      <Link className="secondary-button full-width" to={`/f/${facilitySlug}`}>
        Back to Facility
      </Link>
      <PoweredBySwiftPark />
    </main>
  );
}

function ParkedMessage({ title, loading = false }: { title: string; loading?: boolean }) {
  return (
    <main className="page-stack loading-page">
      {loading ? <div className="loading-ring" /> : null}
      <p>{title}</p>
    </main>
  );
}

