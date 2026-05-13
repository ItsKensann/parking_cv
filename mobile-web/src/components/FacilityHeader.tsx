import { LiveUpdatePill } from "./LiveUpdatePill";
import { StatusBadge } from "./StatusBadge";
import { SwiftParkMark } from "./SwiftParkMark";
import type { FacilityOccupancy } from "../lib/types";

interface FacilityHeaderProps {
  occupancy: FacilityOccupancy;
}

/**
 * Facility identity card: SwiftPark brand stripe + facility name +
 * location + live/status pills + (optionally) a compact degraded
 * notice. The primary CTAs and recommendation copy now live in the
 * RecommendationCard below this header, so this card stays focused on
 * "where am I" rather than "where should I go".
 */
export function FacilityHeader({ occupancy }: FacilityHeaderProps) {
  return (
    <header className="facility-hero">
      <div className="facility-hero__brand-row">
        <SwiftParkMark size="sm" />
        <StatusBadge status={occupancy.status} compact />
      </div>

      <div className="facility-hero__title">
        <h1>{occupancy.facility.publicName}</h1>
        <p>{occupancy.facility.location}</p>
      </div>

      <LiveUpdatePill
        dataState={occupancy.dataState}
        updatedAt={occupancy.updatedAt}
      />

      {occupancy.message ? (
        <p className="degraded-note">{occupancy.message}</p>
      ) : null}
    </header>
  );
}
