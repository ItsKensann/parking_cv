import { StatusBadge } from "./StatusBadge";
import type { ParkingSection } from "../lib/types";

interface SectionCardProps {
  section: ParkingSection;
  isRecommended?: boolean;
}

export function SectionCard({ section, isRecommended = false }: SectionCardProps) {
  return (
    <article className={`section-card ${isRecommended ? "section-card-recommended" : ""}`}>
      <div>
        <div className="section-card-title-row">
          <h3>{section.label}</h3>
          {isRecommended ? <span className="mini-pill">Best now</span> : null}
        </div>
        <p>{section.sourceLabel}</p>
      </div>
      <div className="section-card-metrics">
        <strong>{section.available} open</strong>
        <StatusBadge status={section.status} compact />
      </div>
    </article>
  );
}

