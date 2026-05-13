import { buildGoogleMapsDirectionsUrl } from "../lib/navigation";
import type { FacilitySlug } from "../lib/types";

interface DirectionsButtonProps {
  facilitySlug: FacilitySlug;
  label?: string;
}

export function DirectionsButton({
  facilitySlug,
  label = "Open Google Maps",
}: DirectionsButtonProps) {
  return (
    <a
      className="primary-button full-width"
      href={buildGoogleMapsDirectionsUrl(facilitySlug)}
      target="_blank"
      rel="noreferrer"
    >
      {label}
    </a>
  );
}

