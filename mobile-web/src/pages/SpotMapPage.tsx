import { useEffect, useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { PoweredBySwiftPark } from "../components/PoweredBySwiftPark";
import { SpotMapPreview } from "../components/SpotMapPreview";
import { parseFacilitySlug } from "../lib/api";
import { buildSpotRoute } from "../lib/navigation";
import { getRecommendation } from "../lib/recommendations";
import { findSpot, firstAvailableSpot } from "../lib/spotLookup";
import type { Spot } from "../lib/types";
import { useFacilityOccupancy } from "../lib/useFacilityOccupancy";

export function SpotMapPage() {
  const { facilitySlug: facilityParam } = useParams();
  const [searchParams] = useSearchParams();
  const facilitySlug = parseFacilitySlug(facilityParam);
  const { occupancy, loading } = useFacilityOccupancy(facilitySlug);
  const [selectedSectionId, setSelectedSectionId] = useState<string | null>(null);
  const [selectedSpot, setSelectedSpot] = useState<Spot | null>(null);

  const recommendation = occupancy ? getRecommendation(occupancy) : null;
  const routeSpot = searchParams.get("spot");
  const routeSection = searchParams.get("section");

  useEffect(() => {
    if (!occupancy) return;
    const spotFromRoute = findSpot(occupancy, routeSpot);
    // Match the `?section=` query param against either the section's
    // canonical id (e.g. "Z1", "1") or its short label (e.g. "Z1", "L1"),
    // case-insensitively, so deep links from the facility page's zone
    // cards land on the right section.
    const sectionQuery = routeSection?.trim().toUpperCase() ?? "";
    const sectionFromQuery = sectionQuery
      ? occupancy.sections.find(
          (s) =>
            s.id.toUpperCase() === sectionQuery ||
            s.shortLabel.toUpperCase() === sectionQuery,
        )
      : undefined;
    const initialSection =
      sectionFromQuery?.id ??
      spotFromRoute?.level ??
      recommendation?.section.id ??
      occupancy.sections[0]?.id;
    setSelectedSectionId((current) => current ?? initialSection ?? null);
    setSelectedSpot((current) => current ?? spotFromRoute ?? null);
  }, [occupancy, recommendation, routeSpot, routeSection]);

  const selectedSection = useMemo(() => {
    if (!occupancy) return undefined;
    return (
      occupancy.sections.find((section) => section.id === selectedSectionId) ??
      recommendation?.section ??
      occupancy.sections[0]
    );
  }, [occupancy, recommendation, selectedSectionId]);

  if (!facilitySlug) return <MapMessage title="Facility not found" />;
  if (loading || !occupancy || !selectedSection) {
    return <MapMessage title="Loading live spot map" loading />;
  }

  const nextSpot = selectedSpot ?? firstAvailableSpot(selectedSection);

  return (
    <main className="page-stack">
      <header className="compact-page-header">
        <Link to={`/f/${facilitySlug}`} className="back-link">
          Back
        </Link>
        <div>
          <p className="eyebrow">{occupancy.facility.publicName}</p>
          <h1>Live Spot Map</h1>
        </div>
      </header>

      <section className="content-card">
        <div className="segmented-control">
          {occupancy.sections.map((section) => (
            <button
              key={section.id}
              type="button"
              className={section.id === selectedSection.id ? "active" : ""}
              onClick={() => {
                setSelectedSectionId(section.id);
                setSelectedSpot(null);
              }}
            >
              {section.shortLabel}
            </button>
          ))}
        </div>
        <div className="map-legend">
          <span className="legend-available">Available</span>
          <span className="legend-occupied">Occupied</span>
          <span className="legend-unknown">Unknown</span>
          <span className="legend-selected">Selected</span>
        </div>
        <SpotMapPreview
          section={selectedSection}
          selectedSpotId={selectedSpot?.id ?? selectedSpot?.label}
          onSelectSpot={setSelectedSpot}
        />
      </section>

      <section className="selected-spot-panel">
        <div>
          <p className="eyebrow">Selected spot</p>
          <h2>{nextSpot ? nextSpot.label : "Choose an open spot"}</h2>
          <p>
            {selectedSection.label} · {selectedSection.available} spaces open
          </p>
        </div>
        <div className="action-row vertical-actions">
          {nextSpot ? (
            <>
              <Link className="primary-button" to={buildSpotRoute(facilitySlug, "spot", nextSpot)}>
                View Spot Details
              </Link>
              <Link className="secondary-button" to={buildSpotRoute(facilitySlug, "navigate", nextSpot)}>
                Navigate
              </Link>
            </>
          ) : (
            <button className="primary-button" type="button" disabled>
              No open spot selected
            </button>
          )}
        </div>
      </section>

      <PoweredBySwiftPark />
    </main>
  );
}

function MapMessage({ title, loading = false }: { title: string; loading?: boolean }) {
  return (
    <main className="page-stack loading-page">
      {loading ? <div className="loading-ring" /> : null}
      <p>{title}</p>
    </main>
  );
}
