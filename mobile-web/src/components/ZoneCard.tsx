import { SectionCard } from "./SectionCard";
import type { FacilitySlug, ParkingSection } from "../lib/types";

interface ZoneCardProps {
  section: ParkingSection;
  facilitySlug: FacilitySlug;
  isRecommended?: boolean;
}

export function ZoneCard({
  section,
  facilitySlug,
  isRecommended = false,
}: ZoneCardProps) {
  return (
    <SectionCard
      section={section}
      facilitySlug={facilitySlug}
      isRecommended={isRecommended}
    />
  );
}
