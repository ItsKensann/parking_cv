import type { ParkingSection, Spot } from "../lib/types";

interface SpotMapPreviewProps {
  section: ParkingSection;
  selectedSpotId?: string;
  onSelectSpot?: (spot: Spot) => void;
}

export function SpotMapPreview({
  section,
  selectedSpotId,
  onSelectSpot,
}: SpotMapPreviewProps) {
  return (
    <div className="spot-map-preview" aria-label={`${section.label} spot map`}>
      <div className="map-lane map-lane-top" />
      <div className="spot-grid">
        {section.spots.map((spot) => {
          const isSelected = spot.id === selectedSpotId || spot.label === selectedSpotId;
          const isAvailable = spot.status === "available";
          return (
            <button
              key={`${spot.level}-${spot.id}-${spot.label}`}
              className={`spot-cell spot-${spot.status} ${isSelected ? "spot-selected" : ""}`}
              type="button"
              disabled={!isAvailable}
              onClick={() => onSelectSpot?.(spot)}
              aria-pressed={isSelected}
              aria-label={`${spot.label}, ${spot.status}`}
            >
              <span>{spot.label.replace(`${section.shortLabel}-`, "")}</span>
            </button>
          );
        })}
      </div>
      <div className="map-lane map-lane-bottom" />
    </div>
  );
}

