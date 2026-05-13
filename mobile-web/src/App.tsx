import { Navigate, Route, Routes } from "react-router-dom";
import { BRIGHTON_FACILITY_SLUG } from "./lib/facilities";
import { FacilityPage } from "./pages/FacilityPage";
import { NavigationPage } from "./pages/NavigationPage";
import { ParkedPage } from "./pages/ParkedPage";
import { SpotDetailsPage } from "./pages/SpotDetailsPage";
import { SpotMapPage } from "./pages/SpotMapPage";

export default function App() {
  return (
    <div className="mobile-shell">
      <Routes>
        <Route path="/" element={<Navigate to={`/f/${BRIGHTON_FACILITY_SLUG}`} replace />} />
        <Route path="/f/:facilitySlug" element={<FacilityPage />} />
        <Route path="/f/:facilitySlug/map" element={<SpotMapPage />} />
        <Route path="/f/:facilitySlug/spot/:spotId" element={<SpotDetailsPage />} />
        <Route path="/f/:facilitySlug/navigate" element={<NavigationPage />} />
        <Route path="/f/:facilitySlug/parked" element={<ParkedPage />} />
        <Route path="*" element={<Navigate to={`/f/${BRIGHTON_FACILITY_SLUG}`} replace />} />
      </Routes>
    </div>
  );
}

