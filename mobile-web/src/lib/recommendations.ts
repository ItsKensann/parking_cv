import type { FacilityOccupancy, ParkingSection, Recommendation } from "./types";

export function getRecommendation(
  occupancy: FacilityOccupancy,
): Recommendation | null {
  if (occupancy.sections.length === 0) return null;

  const usableSections = occupancy.sections.filter(
    (section) => section.available > 0 && section.status !== "full",
  );
  const lessCrowded = usableSections.filter(
    (section) => section.status !== "nearly_full",
  );
  const candidates =
    lessCrowded.length > 0 ? lessCrowded : usableSections.length > 0 ? usableSections : occupancy.sections;

  const section = [...candidates].sort(compareSectionsForRecommendation)[0];

  return {
    section,
    reason: recommendationReason(section, occupancy.facility.type),
  };
}

function compareSectionsForRecommendation(a: ParkingSection, b: ParkingSection): number {
  if (b.available !== a.available) return b.available - a.available;
  if (a.occupancyPct !== b.occupancyPct) return a.occupancyPct - b.occupancyPct;
  return a.shortLabel.localeCompare(b.shortLabel);
}

function recommendationReason(
  section: ParkingSection,
  facilityType: "surface_lot" | "garage",
): string {
  if (section.sourceLabel === "Live camera" && facilityType === "surface_lot") {
    return "Best availability with live camera coverage";
  }

  if (section.occupancyPct <= 50) {
    return "Lowest occupancy right now";
  }

  return "Best availability right now";
}

