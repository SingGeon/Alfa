// Shared helpers for both lightweight-charts pages (dashboard's app.js and
// the Scout AI detail.js) - loaded as a plain <script> before either, since
// this project has no bundler/module system to import between them.

// Pan + zoom interaction options, shared by every chart instance (the main
// dashboard/detail candlestick chart and the accuracy panel) so the three
// don't drift out of sync with each other.
//
// Left to lightweight-charts' own native handling rather than anything
// custom: wheel-zoom is already cursor-anchored by the library itself (it
// keeps the time under the cursor fixed while rescaling, not the chart's
// center), and the price (Y) axis already auto-scales to whatever's
// visible - there's nothing to reimplement for either.
//
// mouseWheel is split across two options that sound alike but read
// different axes of the same wheel event: handleScale.mouseWheel (vertical
// wheel = zoom) and handleScroll.mouseWheel (horizontal wheel/trackpad
// swipe = pan). Zoom stays on ordinary vertical scroll, so
// handleScroll.mouseWheel stays off - otherwise a normal mouse wheel would
// both zoom AND pan on the same gesture. vertTouchDrag stays off too: this
// chart's Y axis is not something a user ever manually pans (it auto-
// scales), so capturing vertical touch drags here would just steal the
// page's own scroll gesture on mobile for no benefit.
const CHART_PAN_ZOOM_OPTIONS = {
  handleScroll: {
    mouseWheel: false,
    pressedMouseMove: true, // click-and-drag (mouse) to pan
    horzTouchDrag: true, // swipe (touch) to pan
    vertTouchDrag: false,
  },
  handleScale: {
    mouseWheel: true, // wheel to zoom, anchored at the cursor
    pinch: true, // pinch-to-zoom (touch)
    axisPressedMouseMove: true,
  },
  // Momentum after a released drag/swipe - only for touch (a flick that
  // keeps gliding is the expected mobile feel); off for mouse, where a
  // drag that keeps sliding after release feels like losing control of a
  // precise navigation action instead.
  kineticScroll: { touch: true, mouse: false },
};

// Zooming out can still push the visible window's left edge before the very
// first real bar (index 0) into empty space, though - and once there, a
// refresh's "keep the user's exact scroll position" logic faithfully re-pins
// the view to that same empty window on every subsequent refresh, forever,
// which looks exactly like "the chart froze". This keeps that one guard -
// still needed with panning enabled (arguably more so, now that dragging
// past the start is as easy as zooming out past it always was).
function installLeftEdgeClamp(chartInstance) {
  const MIN_FROM = -10;
  chartInstance.timeScale().subscribeVisibleLogicalRangeChange((range) => {
    if (range && range.from < MIN_FROM) {
      const width = range.to - range.from;
      chartInstance.timeScale().setVisibleLogicalRange({ from: MIN_FROM, to: MIN_FROM + width });
    }
  });
}
