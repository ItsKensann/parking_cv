import {
  BRIGHTON_FACILITY_SLUG,
  isFacilitySlug,
  OSU_FACILITY_SLUG,
} from "./facilities";
import {
  BRIGHTON_LOCAL_MOCK_ZONES,
  buildFallbackOccupancy,
  type BrightonMockZonesPayload,
  normalizeBrightonOccupancy,
  normalizeOsuOccupancy,
  type RawOccupancy,
} from "./normalize";
import { getRecommendation } from "./recommendations";
import type { FacilityOccupancy, FacilitySlug } from "./types";

const RAW_DEMO_BASE =
  import.meta.env.VITE_DEMO_API_BASE_URL ?? "http://127.0.0.1:8000";
const RAW_YOLO_BASE =
  import.meta.env.VITE_YOLO_API_BASE_URL ?? "http://127.0.0.1:8001";

export const DEMO_API_BASE_URL = RAW_DEMO_BASE.replace(/\/$/, "");
export const YOLO_API_BASE_URL = RAW_YOLO_BASE.replace(/\/$/, "");

export class MobileWebApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MobileWebApiError";
  }
}

export function parseFacilitySlug(value: string | undefined): FacilitySlug | null {
  return isFacilitySlug(value) ? value : null;
}

export async function fetchFacilityOccupancy(
  facilitySlug: FacilitySlug,
): Promise<FacilityOccupancy> {
  try {
    if (facilitySlug === OSU_FACILITY_SLUG) {
      const raw = await request<RawOccupancy>(DEMO_API_BASE_URL, "/demo/occupancy");
      return normalizeOsuOccupancy(raw, "demo");
    }

    if (facilitySlug === BRIGHTON_FACILITY_SLUG) {
      return fetchBrightonOccupancy();
    }

    throw new MobileWebApiError(`Unsupported facility: ${facilitySlug}`);
  } catch (error) {
    const fallback = buildFallbackOccupancy(facilitySlug);
    return {
      ...fallback,
      message:
        error instanceof Error
          ? `Live data is temporarily unavailable. Showing a safe demo fallback. ${error.message}`
          : "Live data is temporarily unavailable. Showing a safe demo fallback.",
    };
  }
}

async function fetchBrightonOccupancy(): Promise<FacilityOccupancy> {
  const mockZones = await fetchBrightonMockZones();

  try {
    const yolo = await request<RawOccupancy>(YOLO_API_BASE_URL, "/status");
    const occupancy = normalizeBrightonOccupancy(yolo, mockZones, "live", "live");
    logBrightonDebug(occupancy);
    return occupancy;
  } catch (error) {
    warnApiFallback(
      "Brighton Zone 1 YOLO /status is unavailable. Using fallback Zone 1 data and keeping Zones 2/3 mock data.",
      error,
    );

    return {
      ...normalizeBrightonOccupancy(
        {
          capacity: 50,
          available: 6,
          occupied: 44,
          unknown: 0,
          occupancy_pct: 0.88,
          spots: [],
        },
        mockZones,
        "fallback",
        "fallback",
      ),
      message:
        error instanceof Error
          ? `Brighton live camera data is temporarily unavailable. Showing fallback Zone 1 data. ${error.message}`
          : "Brighton live camera data is temporarily unavailable. Showing fallback Zone 1 data.",
    };
  }
}

async function fetchBrightonMockZones(): Promise<BrightonMockZonesPayload> {
  try {
    const payload = await request<BrightonMockZonesPayload>(
      DEMO_API_BASE_URL,
      "/demo/brighton-mock-zones",
    );
    return { ...payload, __source: "backend_mock" };
  } catch (error) {
    warnApiFallback(
      "Brighton Zones 2/3 mock endpoint is unavailable. Using bundled mock zone data.",
      error,
    );
    return { ...BRIGHTON_LOCAL_MOCK_ZONES, __source: "fallback" };
  }
}

async function request<T>(baseUrl: string, path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${baseUrl}${path}`);
  } catch {
    throw new MobileWebApiError(`Could not reach ${baseUrl}.`);
  }

  if (!response.ok) {
    throw new MobileWebApiError(`${path} returned ${response.status}.`);
  }

  return response.json() as Promise<T>;
}

function warnApiFallback(message: string, error: unknown) {
  if (typeof console === "undefined") return;
  const detail = error instanceof Error ? error.message : String(error);
  console.warn(`[SwiftPark mobile-web] ${message}`, detail);
}

function logBrightonDebug(occupancy: FacilityOccupancy) {
  if (!import.meta.env.DEV || typeof console === "undefined") return;

  const rows = occupancy.sections.map((section) => ({
    section: section.shortLabel,
    source: section.source,
    available: section.available,
    occupied: section.occupied,
    unknown: section.unknown,
    mappedSpaces: section.mappedSpaces,
    operationalCapacity: section.operationalCapacity ?? "",
  }));
  const recommendation = getRecommendation(occupancy);

  console.info("[SwiftPark mobile-web] Brighton section stats", rows);
  console.info(
    "[SwiftPark mobile-web] Brighton recommendation",
    recommendation?.section.shortLabel ?? "none",
  );
}
