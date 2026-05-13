import { Link, useParams, useSearchParams } from "react-router-dom";
import { PoweredBySwiftPark } from "../components/PoweredBySwiftPark";
import { SwiftParkMark } from "../components/SwiftParkMark";
import { parseFacilitySlug } from "../lib/api";
import {
  BRIGHTON_FACILITY_SLUG,
  OSU_FACILITY_SLUG,
} from "../lib/facilities";
import { findSectionForSpot, findSpot } from "../lib/spotLookup";
import { useFacilityOccupancy } from "../lib/useFacilityOccupancy";
import type { FacilitySlug } from "../lib/types";

export function ParkedPage() {
  const { facilitySlug: facilityParam } = useParams();
  const [searchParams] = useSearchParams();
  const facilitySlug = parseFacilitySlug(facilityParam);
  const { occupancy, loading } = useFacilityOccupancy(facilitySlug);

  if (!facilitySlug) return <ParkedMessage title="Facility not found" />;
  if (loading || !occupancy) return <ParkedMessage title="Confirming parking" loading />;

  const spot = findSpot(occupancy, searchParams.get("spot"));
  const section = findSectionForSpot(occupancy, spot);
  const services = getGuestServices(facilitySlug);

  return (
    <main className="page-stack parked-page">
      <section className="parked-hero">
        <div className="confetti confetti-one" />
        <div className="confetti confetti-two" />
        <div className="confetti confetti-three" />
        <div className="confetti confetti-four" />
        <SwiftParkMark size="sm" variant="on-dark" />
        <div className="check-badge" aria-hidden="true" />
        <p className="eyebrow">Parking confirmed</p>
        <h1>You're Parked</h1>
        <p>Your parking location is saved for this SwiftPark session.</p>
      </section>

      <section className="content-card">
        <p className="eyebrow">Session details</p>
        <div className="detail-list">
          <span>Facility</span>
          <strong>{occupancy.facility.name}</strong>
          <span>{occupancy.facility.sectionKind}</span>
          <strong>{section?.label ?? "Selected area"}</strong>
          <span>Spot</span>
          <strong>{spot?.label ?? searchParams.get("spot") ?? "Not selected"}</strong>
        </div>
      </section>

      <section className="guest-services">
        <div className="guest-services__head">
          <p className="eyebrow">Helpful info</p>
          <h2>Guest services</h2>
        </div>
        <div className="guest-services__list">
          {services.map((card) => (
            <article key={card.title} className="guest-card">
              <span className="guest-card__icon" aria-hidden="true">
                <GuestIcon name={card.icon} />
              </span>
              <div className="guest-card__body">
                <h3>{card.title}</h3>
                <p>{card.body}</p>
              </div>
            </article>
          ))}
        </div>
      </section>

      <Link className="secondary-button full-width" to={`/f/${facilitySlug}`}>
        Back to facility
      </Link>
      <PoweredBySwiftPark />
    </main>
  );
}

interface GuestServiceCard {
  title: string;
  body: string;
  icon: GuestIconName;
}

type GuestIconName = "clock" | "shuttle" | "snowflake" | "campus" | "compass" | "phone" | "info";

function getGuestServices(facilitySlug: FacilitySlug): GuestServiceCard[] {
  const shared: GuestServiceCard[] = [
    {
      title: "Set a parking reminder",
      body: "SwiftPark can ping you before facility time limits expire so you don't have to keep checking.",
      icon: "clock",
    },
  ];

  if (facilitySlug === BRIGHTON_FACILITY_SLUG) {
    return [
      ...shared,
      {
        title: "Resort shuttle",
        body: "Complimentary shuttle loops between the lot and all three lifts every 10 minutes during operating hours.",
        icon: "shuttle",
      },
      {
        title: "Overnight & lodging",
        body: "Overnight parking is reserved for lodge guests. Day visitors must clear the lot by 11:00 PM.",
        icon: "snowflake",
      },
      {
        title: "Need help?",
        body: "Brighton Resort guest services is available at the main lodge or by phone for parking assistance.",
        icon: "phone",
      },
    ];
  }

  if (facilitySlug === OSU_FACILITY_SLUG) {
    return [
      ...shared,
      {
        title: "Visitor parking",
        body: "Visitors should pick up a paid permit at the entry kiosk. Faculty and staff lot passes are honored 24/7.",
        icon: "campus",
      },
      {
        title: "Find your way back",
        body: "Re-scan any SwiftPark QR sign near the entrance to walk back to your saved spot.",
        icon: "compass",
      },
      {
        title: "Facility help desk",
        body: "Campus parking services can dispatch a representative if you need a jump, locked out, or have lost a ticket.",
        icon: "phone",
      },
    ];
  }

  return [
    ...shared,
    {
      title: "Facility info",
      body: "Check posted signs for time limits, permit requirements, and overnight policies.",
      icon: "info",
    },
  ];
}

function GuestIcon({ name }: { name: GuestIconName }) {
  const stroke = "currentColor";
  switch (name) {
    case "clock":
      return (
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="9" stroke={stroke} strokeWidth="1.6" />
          <path d="M12 7v5l3 2" stroke={stroke} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      );
    case "shuttle":
      return (
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <rect x="4" y="6" width="16" height="11" rx="2.5" stroke={stroke} strokeWidth="1.6" />
          <path d="M4 12h16" stroke={stroke} strokeWidth="1.6" />
          <circle cx="8" cy="19" r="1.6" fill={stroke} />
          <circle cx="16" cy="19" r="1.6" fill={stroke} />
        </svg>
      );
    case "snowflake":
      return (
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M12 3v18M3 12h18M5.5 5.5l13 13M18.5 5.5l-13 13" stroke={stroke} strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      );
    case "campus":
      return (
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M3 11l9-6 9 6" stroke={stroke} strokeWidth="1.6" strokeLinejoin="round" />
          <path d="M5 11v8h14v-8" stroke={stroke} strokeWidth="1.6" strokeLinejoin="round" />
          <path d="M10 19v-5h4v5" stroke={stroke} strokeWidth="1.6" strokeLinejoin="round" />
        </svg>
      );
    case "compass":
      return (
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="9" stroke={stroke} strokeWidth="1.6" />
          <path d="M15 9l-2 5-5 2 2-5 5-2z" stroke={stroke} strokeWidth="1.6" strokeLinejoin="round" />
        </svg>
      );
    case "phone":
      return (
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M5 4h3l2 5-2 1a11 11 0 005 5l1-2 5 2v3a2 2 0 01-2 2A16 16 0 013 6a2 2 0 012-2z"
            stroke={stroke}
            strokeWidth="1.6"
            strokeLinejoin="round"
          />
        </svg>
      );
    case "info":
    default:
      return (
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="9" stroke={stroke} strokeWidth="1.6" />
          <path d="M12 11v5M12 8v.5" stroke={stroke} strokeWidth="1.8" strokeLinecap="round" />
        </svg>
      );
  }
}

function ParkedMessage({
  title,
  loading = false,
}: {
  title: string;
  loading?: boolean;
}) {
  return (
    <main className="page-stack loading-page">
      {loading ? <div className="loading-ring" /> : null}
      <p>{title}</p>
    </main>
  );
}
