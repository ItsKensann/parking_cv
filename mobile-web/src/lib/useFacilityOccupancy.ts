import { useEffect, useState } from "react";
import { fetchFacilityOccupancy } from "./api";
import type { FacilityOccupancy, FacilitySlug } from "./types";

interface FacilityOccupancyState {
  occupancy: FacilityOccupancy | null;
  loading: boolean;
}

export function useFacilityOccupancy(
  facilitySlug: FacilitySlug | null,
): FacilityOccupancyState {
  const [occupancy, setOccupancy] = useState<FacilityOccupancy | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    if (!facilitySlug) {
      setOccupancy(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    fetchFacilityOccupancy(facilitySlug)
      .then((nextOccupancy) => {
        if (!cancelled) setOccupancy(nextOccupancy);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [facilitySlug]);

  return { occupancy, loading };
}

