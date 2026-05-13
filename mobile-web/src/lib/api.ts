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

/**
 * Per-request timeout. Brighton's YOLO worker can be slow to wake on
 * first hit — 2.5s is enough for warm responses without blocking the
 * facility page on a cold/unavailable camera. Slow paths still show
 * cached or fallback data instead of a blank wait.
 */
const FETCH_TIMEOUT_MS = 2500;

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
      message: buildDegradedMessage(error),
    };
  }
}

async function fetchBrightonOccupancy(): Promise<FacilityOccupancy> {
  // Parallel fetch: the mock zones endpoint and the YOLO /status camera
  // feed are independent backends. Running them in series (the prior
  // shape) meant a slow YOLO would also delay Zones 2/3 data. With
  // `Promise.allSettled` each backend's failure is isolated.
  const [mockZonesResult, yoloResult] = await Promise.allSettled([
    fetchBrightonMockZones(),
    request<RawOccupancy>(YOLO_API_BASE_URL, "/status"),
  ]);

  const mockZones =
    mockZonesResult.status === "fulfilled"
      ? mockZonesResult.value
      : ({ ...BRIGHTON_LOCAL_MOCK_ZONES, __source: "fallback" } as BrightonMockZonesPayload);

  if (mockZonesResult.status === "rejected") {
    warnApiFallback(
      "Brighton Zones 2/3 mock endpoint is unavailable. Using bundled mock zone data.",
      mockZonesResult.reason,
    );
  }

  if (yoloResult.status === "fulfilled") {
    const occupancy = normalizeBrightonOccupancy(
      yoloResult.value,
      mockZones,
      "live",
      "live",
    );
    logBrightonDebug(occupancy);
    return occupancy;
  }

  warnApiFallback(
    "Brighton Zone 1 YOLO /status is unavailable. Using fallback Zone 1 data and keeping Zones 2/3 mock data.",
    yoloResult.reason,
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
    message: "Live camera unavailable · Showing latest zone data",
  };
}

async function fetchBrightonMockZones(): Promise<BrightonMockZonesPayload> {
  const payload = await request<BrightonMockZonesPayload>(
    DEMO_API_BASE_URL,
    "/demo/brighton-mock-zones",
  );
  return { ...payload, __source: "backend_mock" };
}

async function request<T>(baseUrl: string, path: string): Promise<T> {
  const url = `${baseUrl}${path}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(url, { signal: controller.signal });
  } catch (err) {
    if ((err as Error | undefined)?.name === "AbortError") {
      throw new MobileWebApiError(`Timed out reaching ${url}.`);
    }
    throw new MobileWebApiError(`Could not reach ${baseUrl}.`);
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    throw new MobileWebApiError(`${path} returned ${response.status}.`);
  }

  return response.json() as Promise<T>;
}

function buildDegradedMessage(error: unknown): string {
  const detail = error instanceof Error ? error.message : "";
  if (/timed out/i.test(detail)) {
    return "Live data slow to respond · Showing latest available";
  }
  return "Live data unavailable · Showing latest available";
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
