import { LiveUpdatePill } from "./LiveUpdatePill";
import { StatusBadge } from "./StatusBadge";
import type { FacilityOccupancy } from "../lib/types";

interface FacilityHeaderProps {
  occupancy: FacilityOccupancy;
}

export function FacilityHeader({ occupancy }: FacilityHeaderProps) {
  return (
    <header className="facility-header">
      <div>
        <p className="eyebrow">SwiftPark public parking</p>
        <h1>{occupancy.facility.publicName}</h1>
        <p className="facility-location">{occupancy.facility.location}</p>
      </div>
      <div className="header-status-row">
        <LiveUpdatePill dataState={occupancy.dataState} updatedAt={occupancy.updatedAt} />
        <StatusBadge status={occupancy.status} compact />
      </div>
      {occupancy.message ? (
        <p className="degraded-note">{occupancy.message}</p>
      ) : null}
    </header>
  );
}

