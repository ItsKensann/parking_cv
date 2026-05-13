import { BRIGHTON_FACILITY_SLUG, getFacility, OSU_FACILITY_SLUG } from "./facilities";
import type {
  DataState,
  FacilityOccupancy,
  FacilityStatus,
  ParkingSection,
  SectionDataSource,
  SectionKind,
  Spot,
  SpotStatus,
} from "./types";

interface RawSpot {
  id?: string;
  label?: string;
  level?: string;
  status?: string;
  confidence?: number;
}

export interface RawOccupancy {
  lot_id?: string;
  lot_slug?: string;
  lot_name?: string;
  location?: string;
  facility_status?: string;
  timestamp?: string;
  capacity?: number;
  available?: number;
  occupied?: number;
  unknown?: number;
  occupancy_pct?: number;
  spots?: RawSpot[];
}

export interface BrightonMockZonesPayload {
  zones?: Array<{
    level: string;
    label_prefix?: string;
    capacity: number;
    occupied?: number;
    unknown?: number;
    confidence?: number;
  }>;
  capacity?: number;
  available?: number;
  occupied?: number;
  unknown?: number;
  spots?: RawSpot[];
  __source?: "backend_mock" | "fallback";
}

export const BRIGHTON_LOCAL_MOCK_ZONES: BrightonMockZonesPayload = {
  __source: "fallback",
  zones: [
    {
      level: "Z2",
      label_prefix: "Z2",
      capacity: 30,
      occupied: 12,
      unknown: 2,
      confidence: 0.88,
    },
    {
      level: "Z3",
      label_prefix: "Z3",
      capacity: 20,
      occupied: 7,
      unknown: 1,
      confidence: 0.85,
    },
  ],
};

export function deriveFacilityStatus(occupancyPct: number): FacilityStatus {
  if (occupancyPct >= 98) return "full";
  if (occupancyPct >= 85) return "nearly_full";
  if (occupancyPct >= 60) return "busy";
  return "open";
}

export function statusFromAvailable(
  available: number,
  capacity: number,
  occupancyPct: number,
): FacilityStatus {
  if (capacity <= 0) return "unknown";
  if (available <= 0 || occupancyPct >= 98) return "full";
  return deriveFacilityStatus(occupancyPct);
}

export function statusLabel(status: FacilityStatus): string {
  switch (status) {
    case "open":
      return "Open";
    case "busy":
      return "Busy";
    case "nearly_full":
      return "Nearly Full";
    case "full":
      return "Full";
    case "unknown":
      return "Unknown";
  }
}

export function sectionDisplayLabel(kind: SectionKind, sectionId: string): string {
  const normalized = normalizeSectionId(sectionId);
  if (kind === "Zone") {
    return `Zone ${normalized.replace(/^Z/i, "")}`;
  }
  return `Level ${normalized}`;
}

export function normalizeSectionId(value: string | undefined): string {
  const raw = (value ?? "").trim().toUpperCase();
  if (raw === "1" || raw === "ZONE 1") return "Z1";
  if (raw === "2" || raw === "ZONE 2") return "Z2";
  if (raw === "3" || raw === "ZONE 3") return "Z3";
  if (raw === "LEVEL 1") return "L1";
  if (raw === "LEVEL 2") return "L2";
  if (raw === "LEVEL 3") return "L3";
  if (/^[123]$/.test(raw)) return `L${raw}`;
  return raw || "UNKNOWN";
}

export function normalizePct(value: unknown, occupied: number, capacity: number): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    const pct = value <= 1 ? value * 100 : value;
    return roundPct(clamp(pct, 0, 100));
  }

  if (capacity <= 0) return 0;
  return roundPct(clamp((occupied / capacity) * 100, 0, 100));
}

export function normalizeSpotStatus(value: unknown): SpotStatus {
  return value === "available" || value === "occupied" || value === "unknown"
    ? value
    : "unknown";
}

export function countSpots(spots: Spot[]) {
  return spots.reduce(
    (counts, spot) => {
      counts[spot.status] += 1;
      return counts;
    },
    { available: 0, occupied: 0, unknown: 0 },
  );
}

export function normalizeOsuOccupancy(raw: RawOccupancy, dataState: DataState): FacilityOccupancy {
  const facility = getFacility(OSU_FACILITY_SLUG);
  const spots = normalizeSpots(raw.spots ?? [], "L");
  const sections = buildSectionStats({
    kind: facility.sectionKind,
    sectionIds: facility.sectionIds,
    spots,
    source: "demo",
  });
  const totals = getFacilitySummaryFromSections(sections);

  return {
    facility,
    status: statusFromAvailable(totals.available, totals.capacity, totals.occupancyPct),
    capacity: totals.capacity,
    available: totals.available,
    occupied: totals.occupied,
    unknown: totals.unknown,
    occupancyPct: totals.occupancyPct,
    updatedAt: new Date().toISOString(),
    dataState,
    sections,
    spots,
  };
}

export function normalizeBrightonOccupancy(
  yolo: RawOccupancy,
  mockPayload: BrightonMockZonesPayload,
  dataState: DataState,
  zoneOneSource: SectionDataSource = "live",
): FacilityOccupancy {
  const facility = getFacility(BRIGHTON_FACILITY_SLUG);
  const zoneOne = buildBrightonZoneOneSection(yolo, zoneOneSource);
  const mockSections = buildBrightonMockSections(mockPayload);
  const sections = [zoneOne, ...mockSections];
  const totals = getFacilitySummaryFromSections(sections);
  const spots = sections.flatMap((section) => section.spots);

  return {
    facility,
    status: statusFromAvailable(totals.available, totals.capacity, totals.occupancyPct),
    capacity: totals.capacity,
    available: totals.available,
    occupied: totals.occupied,
    unknown: totals.unknown,
    occupancyPct: totals.occupancyPct,
    updatedAt: yolo.timestamp ?? new Date().toISOString(),
    dataState,
    sections,
    spots,
  };
}

export function buildFallbackOccupancy(slug: typeof BRIGHTON_FACILITY_SLUG | typeof OSU_FACILITY_SLUG): FacilityOccupancy {
  if (slug === BRIGHTON_FACILITY_SLUG) {
    return normalizeBrightonOccupancy(
      {
        capacity: 50,
        available: 6,
        occupied: 44,
        unknown: 0,
        occupancy_pct: 0.88,
        spots: [],
      },
      BRIGHTON_LOCAL_MOCK_ZONES,
      "fallback",
      "fallback",
    );
  }

  return normalizeOsuOccupancy(
    {
      spots: buildSpotsFromCounts("L1", "L1", 12, 24, 4, 0.86)
        .concat(buildSpotsFromCounts("L2", "L2", 10, 27, 3, 0.88))
        .concat(buildSpotsFromCounts("L3", "L3", 17, 21, 2, 0.91)),
    },
    "fallback",
  );
}

function buildBrightonZoneOneSection(
  yolo: RawOccupancy,
  source: SectionDataSource,
): ParkingSection {
  const rawSpots = Array.isArray(yolo.spots) ? yolo.spots : [];
  const hasRealSpots = rawSpots.length > 0;
  const spots = hasRealSpots
    ? normalizeSpots(rawSpots, "Z", "Z1")
    : buildBrightonZoneOnePseudoSpots(
        toCount(yolo.available),
        toCount(yolo.occupied),
        toCount(yolo.unknown),
        0.9,
      );
  const spotCounts = countSpots(spots);
  const capacity = spots.length;
  const operationalCapacity = toCount(yolo.capacity);
  const occupancyPct = normalizePct(undefined, spotCounts.occupied, capacity);

  return {
    id: "Z1",
    label: "Zone 1",
    shortLabel: "Z1",
    kind: "Zone",
    capacity,
    operationalCapacity:
      operationalCapacity > capacity ? operationalCapacity : undefined,
    mappedSpaces: spots.length,
    available: spotCounts.available,
    occupied: spotCounts.occupied,
    unknown: spotCounts.unknown,
    occupancyPct,
    status: statusFromAvailable(spotCounts.available, capacity, occupancyPct),
    source,
    sourceLabel:
      source === "live" ? "Live camera data" : "Latest available data",
    spots,
  };
}

function buildBrightonMockSections(payload: BrightonMockZonesPayload): ParkingSection[] {
  const payloadSpots = Array.isArray(payload.spots) ? payload.spots : [];
  const source = payload.__source ?? "backend_mock";
  if (payloadSpots.length > 0) {
    return buildSectionStats({
      kind: "Zone",
      sectionIds: ["Z2", "Z3"],
      spots: normalizeSpots(payloadSpots, "Z"),
      source,
    });
  }

  const zones =
    Array.isArray(payload.zones) && payload.zones.length > 0
      ? payload.zones
      : BRIGHTON_LOCAL_MOCK_ZONES.zones ?? [];

  return zones.map((zone) => {
    const capacity = toCount(zone.capacity);
    const occupied = clampCount(toCount(zone.occupied), capacity);
    const unknown = clampCount(toCount(zone.unknown), Math.max(capacity - occupied, 0));
    const available = Math.max(capacity - occupied - unknown, 0);
    const level = normalizeSectionId(zone.level);
    const spots = buildSpotsFromCounts(
      level,
      zone.label_prefix ?? level,
      available,
      occupied,
      unknown,
      normalizeConfidence(zone.confidence),
    );
    const occupancyPct = normalizePct(undefined, occupied, capacity);

    return {
      id: level,
      label: sectionDisplayLabel("Zone", level),
      shortLabel: level,
      kind: "Zone" as const,
      capacity,
      mappedSpaces: spots.length,
      available,
      occupied,
      unknown,
      occupancyPct,
      status: statusFromAvailable(available, capacity, occupancyPct),
      source,
      sourceLabel: sourceLabelForSection(source, spots.length),
      spots,
    };
  });
}

export function buildSectionStats({
  kind,
  sectionIds,
  spots,
  source,
}: {
  kind: SectionKind;
  sectionIds: string[];
  spots: Spot[];
  source: SectionDataSource;
}): ParkingSection[] {
  return sectionIds.map((sectionId) => {
    const sectionSpots = spots.filter((spot) => normalizeSectionId(spot.level) === sectionId);
    const counts = countSpots(sectionSpots);
    const capacity = sectionSpots.length;
    const occupancyPct = normalizePct(undefined, counts.occupied, capacity);

    return {
      id: sectionId,
      label: sectionDisplayLabel(kind, sectionId),
      shortLabel: sectionId,
      kind,
      capacity,
      mappedSpaces: capacity,
      available: counts.available,
      occupied: counts.occupied,
      unknown: counts.unknown,
      occupancyPct,
      status: statusFromAvailable(counts.available, capacity, occupancyPct),
      source,
      sourceLabel: sourceLabelForSection(source, capacity),
      spots: sectionSpots,
    };
  });
}

function normalizeSpots(rawSpots: RawSpot[], defaultPrefix: "L" | "Z", forcedLevel?: string): Spot[] {
  return rawSpots.map((spot, index) => {
    const level = forcedLevel ?? normalizeRawLevel(spot.level, defaultPrefix);
    const fallbackLabel = `${level}-${String(index + 1).padStart(3, "0")}`;
    const label = (spot.label ?? spot.id ?? fallbackLabel).trim();

    return {
      id: (spot.id ?? label).trim(),
      label,
      level,
      status: normalizeSpotStatus(spot.status),
      confidence: normalizeConfidence(spot.confidence),
    };
  });
}

function normalizeRawLevel(value: string | undefined, defaultPrefix: "L" | "Z"): string {
  const raw = (value ?? "").trim().toUpperCase();
  if (/^[123]$/.test(raw)) return `${defaultPrefix}${raw}`;

  const normalized = normalizeSectionId(value);
  if (normalized === "UNKNOWN") return `${defaultPrefix}1`;
  return normalized;
}

function buildBrightonZoneOnePseudoSpots(
  available: number,
  occupied: number,
  unknown: number,
  confidence: number,
): Spot[] {
  const statuses: SpotStatus[] = [
    ...Array<SpotStatus>(Math.max(available, 0)).fill("available"),
    ...Array<SpotStatus>(Math.max(occupied, 0)).fill("occupied"),
    ...Array<SpotStatus>(Math.max(unknown, 0)).fill("unknown"),
  ];

  return statuses.map((status, index) => {
    const label = `S${String(index + 1).padStart(2, "0")}`;
    return {
      id: label,
      label,
      level: "Z1",
      status,
      confidence,
    };
  });
}

function buildSpotsFromCounts(
  level: string,
  labelPrefix: string,
  available: number,
  occupied: number,
  unknown: number,
  confidence: number,
): Spot[] {
  const statuses: SpotStatus[] = [
    ...Array<SpotStatus>(Math.max(available, 0)).fill("available"),
    ...Array<SpotStatus>(Math.max(occupied, 0)).fill("occupied"),
    ...Array<SpotStatus>(Math.max(unknown, 0)).fill("unknown"),
  ];

  return statuses.map((status, index) => {
    const label = `${labelPrefix}-${String(index + 1).padStart(3, "0")}`;
    return {
      id: label,
      label,
      level,
      status,
      confidence,
    };
  });
}

export function getFacilitySummaryFromSections(sections: ParkingSection[]) {
  const totals = sections.reduce(
    (acc, section) => {
      acc.capacity += section.capacity;
      acc.available += section.available;
      acc.occupied += section.occupied;
      acc.unknown += section.unknown;
      return acc;
    },
    { capacity: 0, available: 0, occupied: 0, unknown: 0 },
  );

  return {
    ...totals,
    occupancyPct: normalizePct(undefined, totals.occupied, totals.capacity),
  };
}

function sourceLabelForSection(source: SectionDataSource, _mappedSpaces: number): string {
  // Public-facing zone/level cards keep the source label short and
  // jargon-free. The numeric count is already shown beside this label
  // ("X open" / progress bar), so we don't repeat capacity counts here.
  switch (source) {
    case "live":
      return "Live camera data";
    case "backend_mock":
      return "Latest zone data";
    case "demo":
      return "Live facility data";
    case "fallback":
      return "Latest available data";
  }
}

function toCount(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(Math.round(value), 0)
    : fallback;
}

function clampCount(value: number, capacity: number): number {
  return Math.min(Math.max(Math.round(value), 0), Math.max(capacity, 0));
}

function normalizeConfidence(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value)
    ? clamp(value, 0, 1)
    : 1;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function roundPct(value: number): number {
  return Math.round(value * 10) / 10;
}
