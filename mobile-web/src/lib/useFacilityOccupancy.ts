import { useEffect, useState } from "react";
import { fetchFacilityOccupancy } from "./api";
import type { FacilityOccupancy, FacilitySlug } from "./types";

interface FacilityOccupancyState {
  occupancy: FacilityOccupancy | null;
  loading: boolean;
}

interface CacheEntry {
  data: FacilityOccupancy;
  fetchedAt: number;
}

/**
 * Per-facility, in-memory occupancy cache shared across page mounts.
 * Navigating between facility / map / spot / navigate / parked within
 * the same session reuses the most recent fetch instead of reloading
 * every page. Stale entries trigger a background refresh while the
 * stale data renders immediately, so the user never sees a blank wait
 * just because they tapped "Back".
 */
const CACHE_TTL_MS = 20_000;
const cache = new Map<FacilitySlug, CacheEntry>();
const inflight = new Map<FacilitySlug, Promise<FacilityOccupancy>>();

function isFresh(entry: CacheEntry | undefined): entry is CacheEntry {
  return entry !== undefined && Date.now() - entry.fetchedAt < CACHE_TTL_MS;
}

function getInflight(facilitySlug: FacilitySlug): Promise<FacilityOccupancy> {
  const existing = inflight.get(facilitySlug);
  if (existing) return existing;

  const promise = fetchFacilityOccupancy(facilitySlug)
    .then((data) => {
      cache.set(facilitySlug, { data, fetchedAt: Date.now() });
      return data;
    })
    .finally(() => {
      inflight.delete(facilitySlug);
    });

  inflight.set(facilitySlug, promise);
  return promise;
}

export function useFacilityOccupancy(
  facilitySlug: FacilitySlug | null,
): FacilityOccupancyState {
  const [occupancy, setOccupancy] = useState<FacilityOccupancy | null>(() => {
    if (!facilitySlug) return null;
    return cache.get(facilitySlug)?.data ?? null;
  });
  const [loading, setLoading] = useState<boolean>(() => {
    if (!facilitySlug) return false;
    const entry = cache.get(facilitySlug);
    return !entry;
  });

  useEffect(() => {
    let cancelled = false;

    if (!facilitySlug) {
      setOccupancy(null);
      setLoading(false);
      return;
    }

    const cached = cache.get(facilitySlug);

    if (cached) {
      // Always render whatever cached data we have — even stale — so the
      // page paints immediately. The skeleton state only shows on a true
      // cold start.
      setOccupancy(cached.data);
      setLoading(false);
    } else {
      setOccupancy(null);
      setLoading(true);
    }

    // Skip the background refresh if the cache entry is still fresh.
    if (isFresh(cached)) {
      return () => {
        cancelled = true;
      };
    }

    getInflight(facilitySlug)
      .then((next) => {
        if (cancelled) return;
        setOccupancy(next);
        setLoading(false);
      })
      .catch(() => {
        if (cancelled) return;
        // fetchFacilityOccupancy never throws — it always resolves to a
        // fallback occupancy. This branch only fires if the cache layer
        // itself throws (shouldn't happen). Just stop the spinner.
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [facilitySlug]);

  return { occupancy, loading };
}

/** Test/escape-hatch — clears the in-memory cache. */
export function clearFacilityOccupancyCache(): void {
  cache.clear();
  inflight.clear();
}
