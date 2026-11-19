# Tracking UI review

This review covers the changed tracking/provenance components, not a full WCAG
audit of every route. Reading this as a field notebook for campus coaches and
operators: ENERGY 1, RHYTHM 2, MOTION 1, following the existing application.

## Purpose of the changes

- Keep the pitch and footage as the main evidence; use one flat coverage strip
  to expose missing positions, unknown teams and unconfirmed identities.
- Keep existing typography and field colours. Tabular digits make counts easy
  to compare; darker secondary text improves readability.
- Use `T` for tracker hypotheses and `#` for jersey evidence. Never truncate an
  ID into a different-looking identity.
- Draw equal metre scales on both pitch axes. Resize from observed canvas width;
  reuse the actual draw transform for pointer hit testing.
- Provide native buttons for player selection so a canvas is usable without a
  mouse. Keep navigation names available when mobile icon labels are hidden.
- Describe recorded footage, connected transport, measured occupancy and event
  candidates separately. Insufficient team evidence withholds possession display.

## Delivery checks for the changed components

- Hard gate PASS: no invented statistics, testimonials or assets were introduced;
  the browser exercised real SNMOT footage and the actual unavailable-data state.
- Functional states PASS: unavailable data, saved/live provenance, mismatched kit
  calibration and observed/rejected pins have explicit labels. Recorded footage
  no longer acquires a generic live-data label.
- Interaction PASS: map/heatmap controls changed the view; Space selected a player
  and the observed `aria-pressed` state became true; data notes expanded. Network
  mode remains disabled when no passing network exists.
- Geometry PASS: pitch metres have equal axis scale; hit testing uses the stored
  render geometry. Invalid players are skipped rather than pinned to the edge.
- Mobile PASS: at a 390 px viewport override, the document client width and
  scroll width were both 375 px. Player buttons wrap; navigation retains accessible
  names and 44 px targets. The temporary viewport override was reset.
- Contrast PASS for changed text/field pairings: secondary text `#65645f` on white
  5.93:1; lighter text `#706f69` on `#f7f6f3` 4.66:1; white player labels on the two
  field colours 6.04:1 and 5.72:1. Existing focus outline against white is 3.62:1.
- Craftsmanship PASS: TypeScript type check and Vite production build pass. Browser
  console inspection found routing warnings from the initial incorrect `/live`
  test URL and React Router future notices, not a rendering exception on the
  observed `/recordings` workflow.

The review found rapid touchline crossings generating repeated restart candidates.
The underlying event logic was then given sustained-observation, elapsed-time,
location and cooldown checks, covered by a dedicated jitter regression test.
