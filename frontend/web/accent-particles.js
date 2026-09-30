// The accent phrase in the hero title ("growth opportunities") emanates
// the same kind of glowing embers as the coin animation - a small, local
// particle field spawning from wherever that span's own line boxes
// actually sit. Uses getClientRects() rather than measuring the text
// itself (no offscreen glyph sampling needed): a wrapped inline span
// reports one rect per line it occupies, which is already exactly the
// area the words cover, wrap and all.
(function initAccentParticles() {
  const canvas = document.getElementById("accentEmitCanvas");
  const accent = document.querySelector(".hero-title-accent");
  const title = document.querySelector(".hero-title");
  if (!canvas || !accent || !title || !canvas.getContext) return;
  const ctx = canvas.getContext("2d");

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (prefersReducedMotion) return; // a static hero doesn't need static embers either - just skip

  const ACCENT_A = [143, 163, 255];
  const ACCENT_B = [185, 198, 255];
  const WHITE = [255, 255, 255];
  function lerp3(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }

  let width = 0, height = 0, dpr = 1;
  let lineRects = []; // relative to the canvas/title box
  const COUNT = 26;
  const particles = new Array(COUNT).fill(null).map(() => ({ x: 0, y: 0, vx: 0, vy: 0, age: 0, life: 1, active: false, size: 1, glow: 0.6 }));

  function measure() {
    const rect = title.getBoundingClientRect();
    width = rect.width;
    height = rect.height;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    lineRects = Array.from(accent.getClientRects()).map((r) => ({
      x: r.left - rect.left, y: r.top - rect.top, w: r.width, h: r.height,
    }));
  }

  function spawn(p) {
    if (lineRects.length === 0) { p.active = false; return; }
    const line = lineRects[(Math.random() * lineRects.length) | 0];
    // Bias toward the line's vertical center (roughly where the glyphs'
    // ink actually is) rather than the full line-height box.
    p.x = line.x + Math.random() * line.w;
    p.y = line.y + line.h * (0.35 + Math.random() * 0.3);
    const angle = -Math.PI / 2 + (Math.random() - 0.5) * 1.8; // mostly upward, imprecise
    const speed = 8 + Math.random() * 22;
    p.vx = Math.cos(angle) * speed;
    p.vy = Math.sin(angle) * speed;
    p.age = 0;
    p.life = 0.9 + Math.random() * 1.4;
    const sizeRoll = Math.random();
    p.size = sizeRoll < 0.15 ? 3.2 + Math.random() * 2.4 : 1.1 + Math.random() * 1.6;
    p.glow = 0.35 + Math.random() * 0.65;
    p.color = Math.random() < 0.25 ? lerp3(WHITE, ACCENT_B, 0.3) : lerp3(ACCENT_A, ACCENT_B, Math.random());
    p.active = true;
  }

  function step(dt) {
    for (const p of particles) {
      if (!p.active) {
        if (Math.random() < dt * 2.2) spawn(p); // irregular respawn timing, not synced
        continue;
      }
      p.age += dt;
      if (p.age >= p.life) { p.active = false; continue; }
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.vx *= 0.98;
      p.vy *= 0.98;
    }
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);
    for (const p of particles) {
      if (!p.active) continue;
      const t = p.age / p.life;
      const envelope = t < 0.15 ? t / 0.15 : 1 - (t - 0.15) / 0.85;
      const r = p.size * (0.8 + envelope * 0.5);
      const a = p.glow * envelope;
      const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 3);
      grad.addColorStop(0, `rgba(${p.color[0]}, ${p.color[1]}, ${p.color[2]}, ${a})`);
      grad.addColorStop(1, `rgba(${p.color[0]}, ${p.color[1]}, ${p.color[2]}, 0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r * 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  let rafId = null;
  let lastT = null;
  function frame(now) {
    const t = now / 1000;
    const dt = lastT == null ? 0 : Math.min(0.05, t - lastT);
    lastT = t;
    step(dt);
    draw();
    rafId = requestAnimationFrame(frame);
  }
  function start() { if (rafId == null) { lastT = null; rafId = requestAnimationFrame(frame); } }
  function stop() { if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; } }

  measure();
  start();
  window.addEventListener("resize", measure);
  // Fonts loading after first paint can shift line-wrap positions.
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(measure);
  document.addEventListener("visibilitychange", () => (document.hidden ? stop() : start()));
  const io = new IntersectionObserver((entries) => entries.forEach((e) => (e.isIntersecting ? start() : stop())), { threshold: 0.01 });
  io.observe(canvas);
})();
