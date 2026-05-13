export type FacilitySlug = "brighton-ski-resort" | "osu-structure-1";

export type FacilityType = "surface_lot" | "garage";

export type SectionKind = "Zone" | "Level";

export type SpotStatus = "available" | "occupied" | "unknown";

export type FacilityStatus =
  | "open"
  | "busy"
  | "nearly_full"
  | "full"
  | "unknown";

export type DataState = "live" | "demo" | "fallback";

export type SectionDataSource = "live" | "backend_mock" | "demo" | "fallback";

export interface FacilityCoordinates {
  lat: number;
  lng: number;
}

export interface FacilityConfig {
  slug: FacilitySlug;
  name: string;
  publicName: string;
  location: string;
  type: FacilityType;
  sectionKind: SectionKind;
  sectionIds: string[];
  entrance: FacilityCoordinates;
  dataSourceLabel: string;
  finalGuidance: string;
}

export interface Spot {
  id: string;
  label: string;
  level: string;
  status: SpotStatus;
  confidence: number;
}

export interface ParkingSection {
  id: string;
  label: string;
  shortLabel: string;
  kind: SectionKind;
  capacity: number;
  operationalCapacity?: number;
  mappedSpaces: number;
  available: number;
  occupied: number;
  unknown: number;
  occupancyPct: number;
  status: FacilityStatus;
  source: SectionDataSource;
  sourceLabel: string;
  spots: Spot[];
}

export interface FacilityOccupancy {
  facility: FacilityConfig;
  status: FacilityStatus;
  capacity: number;
  available: number;
  occupied: number;
  unknown: number;
  occupancyPct: number;
  updatedAt: string;
  dataState: DataState;
  message?: string;
  sections: ParkingSection[];
  spots: Spot[];
}

export interface Recommendation {
  section: ParkingSection;
  reason: string;
}
