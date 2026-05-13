import type { FacilityConfig, FacilitySlug } from "./types";

export const BRIGHTON_FACILITY_SLUG = "brighton-ski-resort";
export const OSU_FACILITY_SLUG = "osu-structure-1";

export const FACILITIES: Record<FacilitySlug, FacilityConfig> = {
  [BRIGHTON_FACILITY_SLUG]: {
    slug: BRIGHTON_FACILITY_SLUG,
    name: "Brighton Ski Resort",
    publicName: "Brighton Ski Resort Parking",
    location: "Brighton, UT",
    type: "surface_lot",
    sectionKind: "Zone",
    sectionIds: ["Z1", "Z2", "Z3"],
    entrance: {
      lat: 40.5984,
      lng: -111.5832,
    },
    dataSourceLabel: "Camera and facility data",
    finalGuidance:
      "Enter the recommended zone, follow the main aisle, and look for open spaces highlighted in blue.",
  },
  [OSU_FACILITY_SLUG]: {
    slug: OSU_FACILITY_SLUG,
    name: "OSU Parking Structure 1",
    publicName: "OSU Parking Structure 1",
    location: "Oregon State University, Corvallis, OR",
    type: "garage",
    sectionKind: "Level",
    sectionIds: ["L1", "L2", "L3"],
    entrance: {
      lat: 44.5638,
      lng: -123.2794,
    },
    dataSourceLabel: "SwiftPark demo data",
    finalGuidance:
      "Enter the structure, follow posted level signs, and use SwiftPark for the final spot choice.",
  },
};

export function isFacilitySlug(value: string | undefined): value is FacilitySlug {
  return value === BRIGHTON_FACILITY_SLUG || value === OSU_FACILITY_SLUG;
}

export function getFacility(slug: FacilitySlug): FacilityConfig {
  return FACILITIES[slug];
}

