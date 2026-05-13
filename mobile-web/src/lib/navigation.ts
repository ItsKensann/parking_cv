import { getFacility } from "./facilities";
import type { FacilitySlug, ParkingSection, Spot } from "./types";

export function buildGoogleMapsDirectionsUrl(facilitySlug: FacilitySlug): string {
  const facility = getFacility(facilitySlug);
  const destination = `${facility.entrance.lat},${facility.entrance.lng}`;
  const params = new URLSearchParams({
    api: "1",
    destination,
    travelmode: "driving",
  });

  return `https://www.google.com/maps/dir/?${params.toString()}`;
}

export function buildSpotRoute(
  facilitySlug: FacilitySlug,
  path: "spot" | "navigate" | "parked",
  spot: Spot,
): string {
  const spotId = encodeURIComponent(spot.label || spot.id);
  if (path === "spot") return `/f/${facilitySlug}/spot/${spotId}`;
  return `/f/${facilitySlug}/${path}?spot=${spotId}`;
}

export function buildFinalGuidance(section: ParkingSection | undefined): string {
  if (!section) {
    return "Follow facility signs, then use SwiftPark to finish the last step.";
  }

  if (section.kind === "Zone") {
    return `Enter ${section.label}. Follow the main aisle and look for open spaces highlighted in blue.`;
  }

  return `Enter ${section.label}. Follow posted garage signs and look for open spaces highlighted in blue.`;
}

