import type { FacilityOccupancy, ParkingSection, Spot } from "./types";

export function findSpot(
  occupancy: FacilityOccupancy,
  spotId: string | null | undefined,
): Spot | undefined {
  if (!spotId) return undefined;
  const decoded = decodeURIComponent(spotId);
  return occupancy.spots.find(
    (spot) =>
      spot.id.toLowerCase() === decoded.toLowerCase() ||
      spot.label.toLowerCase() === decoded.toLowerCase(),
  );
}

export function findSectionForSpot(
  occupancy: FacilityOccupancy,
  spot: Spot | undefined,
): ParkingSection | undefined {
  if (!spot) return undefined;
  return occupancy.sections.find((section) => section.id === spot.level);
}

export function firstAvailableSpot(section: ParkingSection | undefined): Spot | undefined {
  return section?.spots.find((spot) => spot.status === "available");
}

