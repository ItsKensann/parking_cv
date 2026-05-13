import { statusLabel } from "../lib/normalize";
import type { FacilityStatus } from "../lib/types";

interface StatusBadgeProps {
  status: FacilityStatus;
  compact?: boolean;
}

export function StatusBadge({ status, compact = false }: StatusBadgeProps) {
  return (
    <span className={`status-badge status-${status} ${compact ? "status-compact" : ""}`}>
      {statusLabel(status)}
    </span>
  );
}

