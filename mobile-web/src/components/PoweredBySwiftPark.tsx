import { SwiftParkMark } from "./SwiftParkMark";

/**
 * Shared SwiftPark footer for every mobile-web page. The brand pin
 * sits on the left next to a stacked text block:
 *   Swift Park           ← two-tone wordmark
 *   Stress less. Park better.
 *   Live parking guidance from facility cameras.
 *
 * Layout is a flex row centered inside the mobile shell so it never
 * feels cramped against the previous section. The pin is small so the
 * footer reads as a polite sign-off, not a marketing banner.
 */
export function PoweredBySwiftPark() {
  return (
    <footer className="powered-footer">
      <span className="powered-footer__pin" aria-hidden="true">
        <SwiftParkMark size="sm" withWordmark={false} />
      </span>
      <div className="powered-footer__text">
        <p className="powered-footer__brand">
          <span className="powered-footer__swift">Swift</span>
          <span className="powered-footer__park">Park</span>
        </p>
        <p className="powered-footer__tagline">Stress less. Park better.</p>
        <p className="powered-footer__helper">
          Live parking guidance from facility cameras.
        </p>
      </div>
    </footer>
  );
}
