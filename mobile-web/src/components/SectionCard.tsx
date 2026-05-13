import { StatusBadge } from "./StatusBadge";
import type { ParkingSection } from "../lib/types";

interface SectionCardProps {
  section: ParkingSection;
  isRecommended?: boolean;
}

export function SectionCard({ section, isRecommended = false }: SectionCardProps) {
  const pct = Math.min(100, Math.max(0, section.occupancyPct));
  return (
    <article
      className={`section-card ${isRecommended ? "section-card-recommended" : ""}`}
    >
      <div className="section-card-title-row">
        <h3>{section.label}</h3>
        {isRecommended ? <span className="mini-pill">Best now</span> : null}
      </div>
      <div className="section-card-metrics">
        <div>
          <strong>{section.available} open</strong>
          <p>{section.sourceLabel}</p>
        </div>
        <StatusBadge status={section.status} compact />
      </div>
      <div
        className="section-progress"
        data-status={section.status}
        aria-hidden="true"
      >
        <span style={{ width: `${pct}%` }} />
      </div>
    </article>
  );
}
