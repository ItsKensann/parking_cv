import type { DataState } from "../lib/types";

interface LiveUpdatePillProps {
  dataState: DataState;
  updatedAt: string;
}

export function LiveUpdatePill({ dataState, updatedAt }: LiveUpdatePillProps) {
  const label = dataState === "fallback" ? "Offline fallback" : "LIVE";
  return (
    <span className={`live-pill live-pill-${dataState}`}>
      <span className="live-dot" />
      {label} · Updated {formatUpdatedAt(updatedAt)}
    </span>
  );
}

function formatUpdatedAt(value: string): string {
  const updated = new Date(value).getTime();
  if (!Number.isFinite(updated)) return "just now";

  const diffSeconds = Math.max(Math.round((Date.now() - updated) / 1000), 0);
  if (diffSeconds < 5) return "just now";
  if (diffSeconds < 60) return `${diffSeconds} sec ago`;
  const diffMinutes = Math.round(diffSeconds / 60);
  if (diffMinutes < 60) return `${diffMinutes} min ago`;
  return `${Math.round(diffMinutes / 60)} hr ago`;
}

