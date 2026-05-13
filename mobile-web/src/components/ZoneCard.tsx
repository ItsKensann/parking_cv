import { SectionCard } from "./SectionCard";
import type { ParkingSection } from "../lib/types";

interface ZoneCardProps {
  section: ParkingSection;
  isRecommended?: boolean;
}

export function ZoneCard({ section, isRecommended = false }: ZoneCardProps) {
  return <SectionCard section={section} isRecommended={isRecommended} />;
}

