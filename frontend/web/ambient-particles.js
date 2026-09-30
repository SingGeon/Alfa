// Ambient page-wide particle field: a fixed, full-viewport layer of soft
// glowing orbs behind the "How it works"/"Features"/footer sections (the
// hero has its own dense WebGL coin+ember scene and paints over this one
// with its own opaque background, so there's no overlap/competition - this
// is what makes the rest of the landing page feel like part of the same
// particle field instead of the hero being an isolated box).
//
// Deliberately plain canvas 2D, not WebGL: a few dozen soft circles is
// trivial either way, and this layer has no rotation/projection/mesh math
// to justify the extra setup - keeping it separate also means a failure
// here (or prefers-reduced-motion) never touches the hero's own canvases.
(function initAmbientField() {
  const canvas = document.getElementById("ambientField");
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext("2d");

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const ACCENT_A = [143, 163, 255];
  const ACCENT_B = [185, 198, 255];
  const WHITE = [255, 255, 255];
  function lerp3(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }

  const COUNT = 70;
  let width = 0, height = 0, dpr = 1;
  let orbs = [];

  function makeOrb() {
    // Size and radiance are independent rolls - "bilute" of every
    // combination (small+dim, small+bright, big+dim, big+bright), not
    // just one size scaling brightness with it.
    const sizeRoll = Math.random();
    const radius = sizeRoll < 0.12 ? 5.5 + Math.random() * 6.5 : sizeRoll < 0.5 ? 2.2 + Math.random() * 2.2 : 1 + Math.random() * 1.4;
    const radiance = 0.25 + Math.random() * 0.85;
    const color = Math.random() < 0.18 ? lerp3(WHITE, ACCENT_B, 0.4) : lerp3(ACCENT_A, ACCENT_B, Math.random());
    return {
      x: Math.random() * width,
      y: Math.random() * height,
      radius,
      radiance,
      color,
      vx: (Math.random() - 0.5) * 6, // px/s, gentle drift
      vy: (Math.random() - 0.5) * 6,
      phase: Math.random() * Math.PI * 2,
      pulseSpeed: 0.3 + Math.random() * 0.6,
    };
  }

  function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function ensurePopulation() {
    if (orbs.length === COUNT) return;
    orbs = new Array(COUNT).fill(null).map(makeOrb);
  }

  function draw(t) {
    ctx.clearRect(0, 0, width, height);
    for (const o of orbs) {
      const pulse = 0.75 + 0.25 * Math.sin(t * o.pulseSpeed + o.phase);
      const r = o.radius * pulse;
      const a = o.radiance * pulse;
      const grad = ctx.createRadialGradient(o.x, o.y, 0, o.x, o.y, r * 3.2);
      grad.addColorStop(0, `rgba(${o.color[0]}, ${o.color[1]}, ${o.color[2]}, ${a})`);
      grad.addColorStop(1, `rgba(${o.color[0]}, ${o.color[1]}, ${o.color[2]}, 0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(o.x, o.y, r * 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function step(dt) {
    for (const o of orbs) {
      o.x += o.vx * dt;
      o.y += o.vy * dt;
      // Wrap around the viewport instead of respawning - keeps the same
      // population (and its size/radiance mix) drifting indefinitely.
      const pad = 40;
      if (o.x < -pad) o.x = width + pad; else if (o.x > width + pad) o.x = -pad;
      if (o.y < -pad) o.y = height + pad; else if (o.y > height + pad) o.y = -pad;
    }
  }

  let rafId = null;
  let lastT = null;
  function frame(now) {
    const t = now / 1000;
    const dt = lastT == null ? 0 : Math.min(0.05, t - lastT);
    lastT = t;
    step(dt);
    draw(t);
    rafId = requestAnimationFrame(frame);
  }

  function start() {
    if (rafId != null || prefersReducedMotion) return;
    lastT = null;
    rafId = requestAnimationFrame(frame);
  }
  function stop() {
    if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; }
  }

  resize();
  ensurePopulation();
  if (prefersReducedMotion) {
    draw(0);
  } else {
    start();
    document.addEventListener("visibilitychange", () => (document.hidden ? stop() : start()));
  }
  window.addEventListener("resize", () => {
    resize();
    ensurePopulation();
    if (prefersReducedMotion) draw(0);
  });
})();
