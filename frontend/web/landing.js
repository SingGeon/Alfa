// Hero visual: two concentric, independently-spinning Ethereum "coins" -
// the same octahedron the real ETH diamond logo traces, each subdivided
// into a dense triangulated point/line mesh - filling the whole hero
// section as its background layer, with text floating above it.
//
// Outer coin: large, spins slowly around the vertical (Y) axis only,
// dimmer/more violet, wireframe-dense. Inner coin: smaller, tumbles on
// both Y and X at a different rate (an independent axis, not just a
// different speed on the same one - the "two objects rotating
// independently of each other" / pseudo-4D-projection look), brighter
// and whiter, reads as an energetic core inside the outer shell. Both
// continuously shed ember particles in irregular, asynchronous bursts
// ("waves") rather than a constant even stream.
//
// Deliberately hand-rolled WebGL (no three.js/GPU library): the whole
// scene is a few thousand points/line-vertices, trivial for any GPU, so a
// compact custom renderer stays far lighter than a 600KB+ 3D engine
// dependency for a decorative hero. A cheap two-canvas trick stands in for
// real bloom: the sharp canvas renders the scene at full resolution, a
// second canvas renders the *same* scene at a fraction of the resolution
// and sits on top with a CSS blur + "screen" blend - upscaled+blurred
// bright points read as a soft glow without a real post-process pass.
(function initEthHero() {
  const sharpCanvas = document.getElementById("ethHero");
  const glowCanvas = document.getElementById("ethHeroGlow");
  if (!sharpCanvas || !sharpCanvas.getContext) return;

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // -- palette (exact hex from the brief) ---------------------------------
  const ACCENT_A = hex("#8fa3ff");   // electric blue-violet
  const ACCENT_B = hex("#b9c6ff");   // lighter violet
  const WHITE = [1, 1, 1];

  function hex(h) {
    const n = parseInt(h.slice(1), 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  }
  function lerp3(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }

  // -- tiny column-major mat4 helpers (WebGL convention) -------------------
  const mat4 = {
    perspective(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2);
      const out = new Float32Array(16);
      out[0] = f / aspect; out[5] = f;
      out[10] = (far + near) / (near - far); out[11] = -1;
      out[14] = (2 * far * near) / (near - far);
      return out;
    },
    translate(x, y, z) {
      const out = new Float32Array(16);
      out[0] = out[5] = out[10] = out[15] = 1;
      out[12] = x; out[13] = y; out[14] = z;
      return out;
    },
    rotationY(theta) {
      const c = Math.cos(theta), s = Math.sin(theta);
      const out = new Float32Array(16);
      out[0] = c; out[2] = -s; out[5] = 1; out[8] = s; out[10] = c; out[15] = 1;
      return out;
    },
    rotationX(theta) {
      const c = Math.cos(theta), s = Math.sin(theta);
      const out = new Float32Array(16);
      out[0] = 1; out[5] = c; out[6] = s; out[9] = -s; out[10] = c; out[15] = 1;
      return out;
    },
    multiply(a, b) {
      const out = new Float32Array(16);
      for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
          let sum = 0;
          for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[col * 4 + k];
          out[col * 4 + row] = sum;
        }
      }
      return out;
    },
  };
  // Plain-JS equivalents (no matrix needed) for spawning sparks at a
  // world-space position consistent with the shader's rotation order.
  function rotY(p, c, s) { return [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c]; }
  function rotX(p, c, s) { return [p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c]; }

  // -- geometry: the octahedron ("diamond") the real ETH logo traces -------
  const BASE_V = [
    [0, 1, 0], [0, -1, 0],
    [1, 0, 0], [0, 0, 1], [-1, 0, 0], [0, 0, -1],
  ];
  const FACES = [
    [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 2],
    [1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 2],
  ];

  function randomInTriangle(a, b, c) {
    let u = Math.random(), v = Math.random();
    if (u + v > 1) { u = 1 - u; v = 1 - v; }
    return [
      a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0]),
      a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1]),
      a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2]),
    ];
  }

  // Builds one coin's static mesh: a barycentric triangular grid over each
  // of the 8 faces (subdiv = grid resolution per face) gives both the
  // point-cloud *and* the dense internal triangulation lines (not just the
  // 12 outer edges) that make it read as a constellation/neural-net, plus
  // a light dusting of extra random points per face for volumetric fill.
  function buildCoin(radius, subdiv, extraFillPerFace, tint, stretchY) {
    const positions = [], colors = [], sizes = [], alphas = [];
    const linePositions = [], lineColors = [];
    // Brighter, more saturated toward white so the mesh actually reads
    // against the point glow instead of disappearing under it.
    const lineColor = lerp3(ACCENT_A, WHITE, 0.55 + tint * 0.3);

    // Adjacent faces share an entire boundary edge of grid points (and the
    // 6 apex/equator vertices are each shared by 4 faces) - without
    // deduplication every one of those gets drawn 2-4x on top of itself,
    // which is exactly what was blowing the boundary/vertex points up into
    // oversized blobs that drowned out the interior triangulation. One
    // shared point registry (keyed by rounded position) fixes that: each
    // unique point is colored/sized once and referenced by index from
    // however many faces touch it.
    const pointIndex = new Map();
    function keyOf(p) { return Math.round(p[0] * 2000) + "," + Math.round(p[1] * 2000) + "," + Math.round(p[2] * 2000); }
    function addPoint(p, role) {
      const key = keyOf(p);
      const existing = pointIndex.get(key);
      if (existing != null) return existing;
      const idx = positions.length / 3;
      pointIndex.set(key, idx);
      positions.push(p[0], p[1], p[2]);
      if (role === 2) { // apex/equator vertex
        const c = lerp3(WHITE, ACCENT_B, 0.15);
        colors.push(c[0], c[1], c[2]);
        sizes.push(3.2 + tint * 1.0);
        alphas.push(1.0);
      } else if (role === 1) { // face-boundary edge point
        const c = lerp3(ACCENT_B, WHITE, 0.5 + tint * 0.35);
        colors.push(c[0], c[1], c[2]);
        sizes.push(1.6 + Math.random() * 0.55 + tint * 0.4);
        alphas.push((0.8 + Math.random() * 0.2) * (0.85 + tint * 0.15));
      } else { // interior point
        const c = lerp3(ACCENT_A, ACCENT_B, Math.random() * 0.7 + tint * 0.3);
        colors.push(c[0], c[1], c[2]);
        sizes.push(1.0 + Math.random() * 0.75 + tint * 0.35);
        alphas.push((0.4 + Math.random() * 0.32) * (0.8 + tint * 0.2));
      }
      return idx;
    }
    function addLine(iA, iB) {
      linePositions.push(
        positions[iA * 3], positions[iA * 3 + 1], positions[iA * 3 + 2],
        positions[iB * 3], positions[iB * 3 + 1], positions[iB * 3 + 2]
      );
      lineColors.push(lineColor[0], lineColor[1], lineColor[2], lineColor[0], lineColor[1], lineColor[2]);
    }

    // Stretching Y (not X/Z) turns the regular octahedron into a taller,
    // less "square" gem - elongated top-to-bottom rather than uniformly
    // scaled, and it fills more of a tall hero on its own.
    const scaleV = (v) => [v[0] * radius, v[1] * radius * stretchY, v[2] * radius];

    FACES.forEach(([ia, ib, ic]) => {
      const A = scaleV(BASE_V[ia]);
      const B = scaleV(BASE_V[ib]);
      const C = scaleV(BASE_V[ic]);

      const idxGrid = new Map(); // "i,j" -> point index
      for (let i = 0; i <= subdiv; i++) {
        for (let j = 0; j <= subdiv - i; j++) {
          const p = [
            A[0] + (i / subdiv) * (B[0] - A[0]) + (j / subdiv) * (C[0] - A[0]),
            A[1] + (i / subdiv) * (B[1] - A[1]) + (j / subdiv) * (C[1] - A[1]),
            A[2] + (i / subdiv) * (B[2] - A[2]) + (j / subdiv) * (C[2] - A[2]),
          ];
          const isCorner = (i === 0 && j === 0) || (i === subdiv && j === 0) || (i === 0 && j === subdiv);
          const isBoundary = i === 0 || j === 0 || i + j === subdiv;
          idxGrid.set(i + "," + j, addPoint(p, isCorner ? 2 : isBoundary ? 1 : 0));
        }
      }
      // The 3 triangular-lattice directions cover every small triangle's
      // edges - the real octahedron silhouette edges end up drawn once per
      // adjacent face (so they pop slightly brighter than interior
      // cross-hatch lines, a nice hierarchy rather than a bug).
      for (let i = 0; i <= subdiv; i++) {
        for (let j = 0; j <= subdiv - i; j++) {
          const cur = idxGrid.get(i + "," + j);
          if (i + 1 + j <= subdiv) addLine(cur, idxGrid.get((i + 1) + "," + j));
          if (i + j + 1 <= subdiv) addLine(cur, idxGrid.get(i + "," + (j + 1)));
          if (j - 1 >= 0) addLine(cur, idxGrid.get((i + 1) + "," + (j - 1)));
        }
      }

      for (let k = 0; k < extraFillPerFace; k++) {
        const p = randomInTriangle(A, B, C);
        positions.push(p[0], p[1], p[2]);
        const c = lerp3(ACCENT_A, ACCENT_B, Math.random());
        colors.push(c[0], c[1], c[2]);
        sizes.push(1.0 + Math.random() * 0.7);
        alphas.push((0.2 + Math.random() * 0.2) * (0.7 + tint * 0.3));
      }
    });

    return {
      positions: new Float32Array(positions),
      colors: new Float32Array(colors),
      sizes: new Float32Array(sizes),
      alphas: new Float32Array(alphas),
      count: positions.length / 3,
      linePositions: new Float32Array(linePositions),
      lineColors: new Float32Array(lineColors),
      lineCount: linePositions.length / 3,
    };
  }

  const STRETCH_Y = 1.6; // elongates both coins vertically - a taller gem, not a square diamond
  const OUTER_RADIUS = 1.3, OUTER_SUBDIV = 5, OUTER_FILL = 70;
  const INNER_RADIUS = 0.52, INNER_SUBDIV = 3, INNER_FILL = 34;
  const outerMesh = buildCoin(OUTER_RADIUS, OUTER_SUBDIV, OUTER_FILL, 0.1, STRETCH_Y);
  const innerMesh = buildCoin(INNER_RADIUS, INNER_SUBDIV, INNER_FILL, 0.85, STRETCH_Y);

  // -- irregular "wave" ember-particle systems (one per coin) --------------
  // Instead of a constant even drizzle, most of the pool sits dormant; a
  // randomly-timed wave activates a cluster of them at once from one
  // "epicenter" face (plus a few strays from elsewhere), each with its own
  // jittered direction/speed/life/start-delay - organic bursts, not a
  // mechanical stream, and the two coins' waves never sync with each other.
  function makeSparkSystem(count) {
    return {
      items: new Array(count).fill(null).map(() => ({
        pos: [0, 0, 0], vel: [0, 0, 0], age: 0, life: 1, delay: 0, active: false, seed: Math.random(),
        sizeMul: 1, glow: 0.7,
      })),
      waveTimer: Math.random() * 0.6,
    };
  }
  function activateSpark(s, radius, worldPoint, delay) {
    const radial = Math.hypot(worldPoint[0], worldPoint[1], worldPoint[2]) || 1;
    // Fast, mostly-straight-line motion (a real velocity integrated every
    // frame, so it reads as a streak once the trail-fade catches up to it)
    // but pointed in an imprecise direction - the jitter scales *with*
    // speed so it stays a meaningful fraction of the vector at any speed,
    // instead of vanishing into a near-perfect radial burst.
    const speed = radius * (0.75 + Math.random() * 1.15);
    const jitter = speed * 0.85;
    s.pos[0] = worldPoint[0]; s.pos[1] = worldPoint[1]; s.pos[2] = worldPoint[2];
    s.vel[0] = (worldPoint[0] / radial) * speed + (Math.random() - 0.5) * jitter;
    s.vel[1] = (worldPoint[1] / radial) * speed + (Math.random() - 0.5) * jitter;
    s.vel[2] = (worldPoint[2] / radial) * speed + (Math.random() - 0.5) * jitter;
    s.age = 0;
    s.life = 2.0 + Math.random() * 2.8;
    s.delay = delay;
    s.active = true;
    s.seed = Math.random();
    // Size/radiance vary a lot more than before - most embers are small
    // and modest, a few are noticeably bigger, glowing "orbs"; brightness
    // is its own independent roll so a small one can still glow hard and a
    // big one can be dim, instead of size and glow always moving together.
    const sizeRoll = Math.random();
    s.sizeMul = sizeRoll < 0.1 ? 2.4 + Math.random() * 1.6 : sizeRoll < 0.4 ? 1.1 + Math.random() * 0.6 : 0.45 + Math.random() * 0.5;
    s.glow = 0.5 + Math.random() * 0.9;
  }
  // Shared by the periodic "wave" bursts and the one-off intro "big bang" -
  // just factored out so the intro can fire a much bigger, near-simultanous
  // burst across (almost) the whole dormant pool instead of the usual
  // fraction, using the same face-sampling/rotation logic.
  function triggerBurst(sys, radius, stretchY, cy, sy, cx, sx, applyX, burstFrac, maxDelay) {
    const inactive = sys.items.filter((s) => !s.active);
    const burstSize = Math.max(3, Math.round(inactive.length * burstFrac));
    const epicenter = FACES[(Math.random() * FACES.length) | 0];
    for (let k = 0; k < burstSize && k < inactive.length; k++) {
      const s = inactive[k];
      const face = Math.random() < 0.5 ? epicenter : FACES[(Math.random() * FACES.length) | 0];
      const [ia, ib, ic] = face;
      const scaleV = (v) => [v[0] * radius, v[1] * radius * stretchY, v[2] * radius];
      let local = randomInTriangle(scaleV(BASE_V[ia]), scaleV(BASE_V[ib]), scaleV(BASE_V[ic]));
      local = rotY(local, cy, sy);
      if (applyX) local = rotX(local, cx, sx);
      activateSpark(s, radius, local, Math.random() * maxDelay);
    }
  }
  function updateSparkSystem(sys, dt, radius, stretchY, cy, sy, cx, sx, applyX) {
    sys.waveTimer -= dt;
    if (sys.waveTimer <= 0) {
      sys.waveTimer = 0.015 + Math.random() * 0.12; // frequent, irregular bursts - not one steady stream
      const burstFrac = 0.55 + Math.random() * 0.45;
      triggerBurst(sys, radius, stretchY, cy, sy, cx, sx, applyX, burstFrac, 0.3);
    }
    for (const s of sys.items) {
      if (!s.active) continue;
      if (s.delay > 0) { s.delay -= dt; continue; }
      s.age += dt;
      if (s.age >= s.life) { s.active = false; continue; }
      s.pos[0] += s.vel[0] * dt;
      s.pos[1] += s.vel[1] * dt;
      s.pos[2] += s.vel[2] * dt;
    }
  }

  const OUTER_SPARKS = 950, INNER_SPARKS = 560;
  const outerSparks = makeSparkSystem(OUTER_SPARKS);
  const innerSparks = makeSparkSystem(INNER_SPARKS);

  // -- rotation state -------------------------------------------------------
  let outerAngleY = 0.5;
  let innerAngleY = 2.1;
  let innerAngleX = 0.3;

  // -- shaders --------------------------------------------------------------
  const POINT_VS = `
    attribute vec3 aPosition;
    attribute vec3 aColor;
    attribute float aSize;
    attribute float aAlpha;
    uniform mat4 uProj;
    uniform mat4 uModelView;
    uniform float uSizeScale;
    varying vec3 vColor;
    varying float vAlpha;
    void main() {
      vec4 viewPos = uModelView * vec4(aPosition, 1.0);
      gl_Position = uProj * viewPos;
      float depthAtten = clamp(uSizeScale / -viewPos.z, 0.15, 3.0);
      gl_PointSize = aSize * depthAtten;
      float depthFade = clamp((-viewPos.z - 2.6) / 3.4, 0.0, 1.0);
      vAlpha = aAlpha * (1.0 - depthFade * 0.6);
      vColor = aColor;
    }
  `;
  const POINT_FS = `
    precision mediump float;
    varying vec3 vColor;
    varying float vAlpha;
    void main() {
      vec2 uv = gl_PointCoord * 2.0 - 1.0;
      float d = length(uv);
      float glow = smoothstep(1.0, 0.0, d);
      glow = pow(glow, 1.7);
      gl_FragColor = vec4(vColor * glow, glow * vAlpha);
    }
  `;
  const LINE_VS = `
    attribute vec3 aPosition;
    attribute vec3 aColor;
    uniform mat4 uProj;
    uniform mat4 uModelView;
    varying vec3 vColor;
    void main() {
      vec4 viewPos = uModelView * vec4(aPosition, 1.0);
      gl_Position = uProj * viewPos;
      vColor = aColor;
    }
  `;
  const LINE_FS = `
    precision mediump float;
    uniform float uAlpha;
    varying vec3 vColor;
    void main() { gl_FragColor = vec4(vColor, uAlpha); }
  `;
  const QUAD_VS = `
    attribute vec2 aPosition;
    void main() { gl_Position = vec4(aPosition, 0.0, 1.0); }
  `;
  const QUAD_FS = `
    precision mediump float;
    uniform float uAlpha;
    void main() { gl_FragColor = vec4(0.0, 0.0, 0.02, uAlpha); }
  `;

  function compile(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.warn("ETH hero shader error:", gl.getShaderInfoLog(sh));
      gl.deleteShader(sh);
      return null;
    }
    return sh;
  }
  function link(gl, vsSrc, fsSrc) {
    const vs = compile(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = compile(gl, gl.FRAGMENT_SHADER, fsSrc);
    if (!vs || !fs) return null;
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.warn("ETH hero program link error:", gl.getProgramInfoLog(prog));
      return null;
    }
    return prog;
  }
  function makeBuffer(gl, data, usage) {
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, usage);
    return buf;
  }
  function attrib(gl, prog, buf, name, size) {
    const loc = gl.getAttribLocation(prog, name);
    if (loc < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  }

  function createScene(canvas, resScale) {
    const gl = canvas.getContext("webgl", { alpha: false, antialias: true, premultipliedAlpha: false })
      || canvas.getContext("experimental-webgl");
    if (!gl) return null;

    const pointProg = link(gl, POINT_VS, POINT_FS);
    const lineProg = link(gl, LINE_VS, LINE_FS);
    const quadProg = link(gl, QUAD_VS, QUAD_FS);
    if (!pointProg || !lineProg || !quadProg) return null;

    function uploadMesh(mesh) {
      return {
        posBuf: makeBuffer(gl, mesh.positions, gl.STATIC_DRAW),
        colorBuf: makeBuffer(gl, mesh.colors, gl.STATIC_DRAW),
        sizeBuf: makeBuffer(gl, mesh.sizes, gl.STATIC_DRAW),
        alphaBuf: makeBuffer(gl, mesh.alphas, gl.STATIC_DRAW),
        count: mesh.count,
        linePosBuf: makeBuffer(gl, mesh.linePositions, gl.STATIC_DRAW),
        lineColorBuf: makeBuffer(gl, mesh.lineColors, gl.STATIC_DRAW),
        lineCount: mesh.lineCount,
      };
    }
    const outerGpu = uploadMesh(outerMesh);
    const innerGpu = uploadMesh(innerMesh);

    function makeSparkBuffers(count) {
      return {
        pos: new Float32Array(count * 3), posBuf: gl.createBuffer(),
        color: new Float32Array(count * 3), colorBuf: gl.createBuffer(),
        size: new Float32Array(count), sizeBuf: gl.createBuffer(),
        alpha: new Float32Array(count), alphaBuf: gl.createBuffer(),
      };
    }
    const outerSparkGpu = makeSparkBuffers(OUTER_SPARKS);
    const innerSparkGpu = makeSparkBuffers(INNER_SPARKS);

    const quadBuf = makeBuffer(gl, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);

    function resize() {
      const rect = canvas.parentElement.getBoundingClientRect();
      const w = Math.max(1, Math.round(rect.width * resScale));
      const h = Math.max(1, Math.round(rect.height * resScale));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      gl.viewport(0, 0, canvas.width, canvas.height);
    }

    function drawMesh(gpu, proj, modelView, sizeScale, lineAlpha) {
      gl.useProgram(lineProg);
      attrib(gl, lineProg, gpu.linePosBuf, "aPosition", 3);
      attrib(gl, lineProg, gpu.lineColorBuf, "aColor", 3);
      gl.uniformMatrix4fv(gl.getUniformLocation(lineProg, "uProj"), false, proj);
      gl.uniformMatrix4fv(gl.getUniformLocation(lineProg, "uModelView"), false, modelView);
      gl.uniform1f(gl.getUniformLocation(lineProg, "uAlpha"), lineAlpha);
      gl.drawArrays(gl.LINES, 0, gpu.lineCount);

      gl.useProgram(pointProg);
      attrib(gl, pointProg, gpu.posBuf, "aPosition", 3);
      attrib(gl, pointProg, gpu.colorBuf, "aColor", 3);
      attrib(gl, pointProg, gpu.sizeBuf, "aSize", 1);
      attrib(gl, pointProg, gpu.alphaBuf, "aAlpha", 1);
      gl.uniformMatrix4fv(gl.getUniformLocation(pointProg, "uProj"), false, proj);
      gl.uniformMatrix4fv(gl.getUniformLocation(pointProg, "uModelView"), false, modelView);
      gl.uniform1f(gl.getUniformLocation(pointProg, "uSizeScale"), sizeScale);
      gl.drawArrays(gl.POINTS, 0, gpu.count);
    }

    function drawSparks(gpu, sys, proj, view, sizeScale) {
      for (let i = 0; i < sys.items.length; i++) {
        const s = sys.items[i];
        const active = s.active && s.delay <= 0;
        const t = active ? s.age / s.life : 0;
        const envelope = active ? (t < 0.1 ? t / 0.1 : Math.max(0, 1 - (t - 0.1) / 0.9)) : 0;
        gpu.pos[i * 3] = s.pos[0]; gpu.pos[i * 3 + 1] = s.pos[1]; gpu.pos[i * 3 + 2] = s.pos[2];
        const c = lerp3(WHITE, ACCENT_B, 0.5);
        gpu.color[i * 3] = c[0]; gpu.color[i * 3 + 1] = c[1]; gpu.color[i * 3 + 2] = c[2];
        gpu.size[i] = (2.1 * (1 - t) + 0.6) * s.sizeMul;
        gpu.alpha[i] = envelope * s.glow;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, gpu.posBuf); gl.bufferData(gl.ARRAY_BUFFER, gpu.pos, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, gpu.colorBuf); gl.bufferData(gl.ARRAY_BUFFER, gpu.color, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, gpu.sizeBuf); gl.bufferData(gl.ARRAY_BUFFER, gpu.size, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, gpu.alphaBuf); gl.bufferData(gl.ARRAY_BUFFER, gpu.alpha, gl.DYNAMIC_DRAW);

      gl.useProgram(pointProg);
      attrib(gl, pointProg, gpu.posBuf, "aPosition", 3);
      attrib(gl, pointProg, gpu.colorBuf, "aColor", 3);
      attrib(gl, pointProg, gpu.sizeBuf, "aSize", 1);
      attrib(gl, pointProg, gpu.alphaBuf, "aAlpha", 1);
      gl.uniformMatrix4fv(gl.getUniformLocation(pointProg, "uProj"), false, proj);
      gl.uniformMatrix4fv(gl.getUniformLocation(pointProg, "uModelView"), false, view);
      gl.uniform1f(gl.getUniformLocation(pointProg, "uSizeScale"), sizeScale);
      gl.drawArrays(gl.POINTS, 0, gpu.pos.length / 3);
    }

    function draw(fadeAlpha, viewX, viewZ, tiltY, tiltX) {
      resize();
      const aspect = canvas.width / canvas.height || 1;
      const proj = mat4.perspective((42 * Math.PI) / 180, aspect, 0.1, 14);
      // Resting position is shifted right (not centered) - the structure
      // sits over the right half of the hero, with the text column free
      // on the left, rather than a centered blob the copy has to sit
      // beside. On first load it starts centered and closer (viewX/viewZ
      // animate in from frame()) and eases into that resting spot. The
      // mouse tilt (tiltY/tiltX) rotates the whole assembly around its own
      // center *before* that translation, so it leans toward the cursor
      // without disturbing where it sits on screen.
      const tilt = mat4.multiply(mat4.rotationY(tiltY || 0), mat4.rotationX(tiltX || 0));
      const view = mat4.multiply(mat4.translate(viewX, 0, viewZ), tilt);

      const outerRot = mat4.rotationY(outerAngleY);
      const outerMV = mat4.multiply(view, outerRot);

      const innerRot = mat4.multiply(mat4.rotationY(innerAngleY), mat4.rotationX(innerAngleX));
      const innerMV = mat4.multiply(view, innerRot);

      // 1) trail fade - a near-black translucent quad, normal alpha blend.
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(quadProg);
      attrib(gl, quadProg, quadBuf, "aPosition", 2);
      gl.uniform1f(gl.getUniformLocation(quadProg, "uAlpha"), fadeAlpha);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      // 2) additive glow pass.
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      const sizeScale = canvas.height * 0.115;

      drawMesh(outerGpu, proj, outerMV, sizeScale, 0.62);
      drawSparks(outerSparkGpu, outerSparks, proj, view, sizeScale);

      drawMesh(innerGpu, proj, innerMV, sizeScale, 0.8);
      drawSparks(innerSparkGpu, innerSparks, proj, view, sizeScale);
    }

    return { draw };
  }

  const sharpScene = createScene(sharpCanvas, Math.min(window.devicePixelRatio || 1, 2));
  const glowScene = glowCanvas ? createScene(glowCanvas, 0.35) : null;
  if (!sharpScene) return; // no WebGL - leave the dark hero background only

  let rafId = null;
  let lastT = null;

  // Intro "fly into place": on first load the coin sits centered and close
  // (as if the visitor opens the page right inside the animation), then
  // eases out to its normal resting spot on the right over ~1.9s. Timed
  // from the first rAF timestamp (not wall-clock `start()` calls), so
  // later stop/start cycles from tab-visibility or scroll-in/out never
  // replay it - introStart is only ever set once per page load.
  let introStart = null;
  const INTRO_DURATION = 1900;
  const INTRO_FROM = { x: 0, z: -4.1 };
  const INTRO_TO = { x: 1.55, z: -6.4 };
  function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }
  // Overshoots past 1 then settles back - the coin "lands" past its resting
  // spot and eases back into it instead of just gliding to a stop.
  function easeOutBack(t) {
    const c1 = 1.70158, c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  }

  // Rotation "spins up": both coins spin several times faster than their
  // resting rate at the very start (a blur-like tumble) and ease down to
  // normal speed over roughly the same window the position takes to land -
  // ties the spin to the arrival instead of it just running underneath at
  // a constant rate the whole time.
  const SPIN_UP_DURATION = 1500;
  const SPIN_UP_FROM = 5.5;

  // Quick opacity fade-in so the coin doesn't read as already sitting there
  // a frame before anything moves.
  const FADE_IN_DURATION = 350;

  // One synchronized, oversized particle burst - a "big bang" - the instant
  // the intro starts, on top of (not instead of) the two systems' own
  // ongoing async waves. Fires once per page load.
  let bigBangFired = false;

  // Mouse-follow tilt: the whole assembly (both coins + their sparks) leans
  // toward wherever the cursor is on the page, layered on top of each
  // coin's own independent spin rather than replacing it. Smoothed toward
  // the raw pointer position so it reads as inertial drift, not a snap.
  let mouseTargetX = 0, mouseTargetY = 0;
  let mouseSmoothX = 0, mouseSmoothY = 0;
  const MOUSE_TILT_Y = 0.34;
  const MOUSE_TILT_X = 0.2;
  if (!prefersReducedMotion) {
    window.addEventListener("mousemove", (e) => {
      mouseTargetX = (e.clientX / window.innerWidth) * 2 - 1;
      mouseTargetY = (e.clientY / window.innerHeight) * 2 - 1;
    });
  }

  function frame(now) {
    const t = now / 1000;
    const dt = lastT == null ? 0 : Math.min(0.05, t - lastT);
    lastT = t;

    if (introStart == null) introStart = now;
    const elapsed = now - introStart;

    if (!bigBangFired) {
      bigBangFired = true;
      const ocy0 = Math.cos(outerAngleY), osy0 = Math.sin(outerAngleY);
      const icy0 = Math.cos(innerAngleY), isy0 = Math.sin(innerAngleY);
      const icx0 = Math.cos(innerAngleX), isx0 = Math.sin(innerAngleX);
      triggerBurst(outerSparks, OUTER_RADIUS, STRETCH_Y, ocy0, osy0, 1, 0, false, 0.95, 0.5);
      triggerBurst(innerSparks, INNER_RADIUS, STRETCH_Y, icy0, isy0, icx0, isx0, true, 0.95, 0.5);
    }

    const positionT = easeOutBack(Math.min(1, elapsed / INTRO_DURATION));
    const viewX = INTRO_FROM.x + (INTRO_TO.x - INTRO_FROM.x) * positionT;
    const viewZ = INTRO_FROM.z + (INTRO_TO.z - INTRO_FROM.z) * positionT;

    const spinT = easeOutCubic(Math.min(1, elapsed / SPIN_UP_DURATION));
    const spinMul = SPIN_UP_FROM + (1 - SPIN_UP_FROM) * spinT;

    outerAngleY += dt * 0.2 * spinMul; // slow, single-axis spin - fast at intro, settles to base rate
    innerAngleY -= dt * 0.34 * spinMul; // opposite direction, faster - independent of the outer shell
    innerAngleX += dt * 0.13 * spinMul; // a second axis the outer coin never uses -> genuinely independent tumble

    const ocy = Math.cos(outerAngleY), osy = Math.sin(outerAngleY);
    const icy = Math.cos(innerAngleY), isy = Math.sin(innerAngleY);
    const icx = Math.cos(innerAngleX), isx = Math.sin(innerAngleX);
    updateSparkSystem(outerSparks, dt, OUTER_RADIUS, STRETCH_Y, ocy, osy, 1, 0, false);
    updateSparkSystem(innerSparks, dt, INNER_RADIUS, STRETCH_Y, icy, isy, icx, isx, true);

    // Exponential smoothing toward the raw pointer target, independent of
    // frame rate - reads as inertial "catch-up" drift rather than jitter.
    const smoothing = 1 - Math.pow(0.0005, dt);
    mouseSmoothX += (mouseTargetX - mouseSmoothX) * smoothing;
    mouseSmoothY += (mouseTargetY - mouseSmoothY) * smoothing;
    const tiltY = mouseSmoothX * MOUSE_TILT_Y;
    const tiltX = -mouseSmoothY * MOUSE_TILT_X;

    const fadeInT = Math.min(1, elapsed / FADE_IN_DURATION);
    sharpCanvas.style.opacity = fadeInT;
    if (glowCanvas) glowCanvas.style.opacity = fadeInT;

    const fade = 1 - Math.pow(0.9, dt * 60); // slower fade -> longer visible streak trails
    sharpScene.draw(fade, viewX, viewZ, tiltY, tiltX);
    if (glowScene) glowScene.draw(fade, viewX, viewZ, tiltY, tiltX);
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

  if (prefersReducedMotion) {
    // Skip the fly-in and mouse tilt for reduced motion - render straight at the resting pose.
    sharpScene.draw(0, INTRO_TO.x, INTRO_TO.z, 0, 0);
    if (glowScene) glowScene.draw(0, INTRO_TO.x, INTRO_TO.z, 0, 0);
  } else {
    document.addEventListener("visibilitychange", () => (document.hidden ? stop() : start()));
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => (e.isIntersecting ? start() : stop())),
      { threshold: 0.01 }
    );
    io.observe(sharpCanvas);
    start();
  }
})();

// Warm up the ETH dashboard while the visitor is still reading the landing
// page, so clicking through to it doesn't hit a cold start. Two separate
// things were both slow on a fresh visit:
//   1. dashboard.html loads the charting library from a CDN with a plain,
//      render-blocking <script> tag in <head> - nothing on that page paints
//      until that fetch finishes, so a slow/first-time fetch of that one
//      file made the *whole page* look stuck loading, not just the chart.
//   2. The model prediction itself is cached server-side (~5min warm-up
//      cycle - see run_api.py), but a visitor arriving inside a cold gap
//      (e.g. right after a server restart) still pays the full training
//      cost themselves.
// Prefetching the CDN script (so the browser already has it cached - the
// blocking <script> tag on the next page becomes a cache hit instead of a
// network round trip) and firing the same requests dashboard.html makes on
// load (so the server-side cache is warm too) turns both into non-issues
// by the time the click actually happens - done at low priority and
// swallowing any failure, since this is purely an optimization, never
// something the landing page itself should wait on or break over.
(function warmUpDashboard() {
  const prefetch = (href, as) => {
    const link = document.createElement("link");
    link.rel = "prefetch";
    link.href = href;
    if (as) link.as = as;
    document.head.appendChild(link);
  };
  prefetch("https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js", "script");
  prefetch("/chart-utils.js", "script");
  prefetch("/app.js", "script");
  prefetch("/dashboard.html", "document");

  // Same default (interval=1h, steps=24, sentiment on) dashboard.html's
  // own first load requests - a real hit against the exact cache key it'll
  // need, not just a resource fetch.
  fetch("/api/predict?interval=1h&steps=24&use_sentiment=true").catch(() => {});
  fetch("/api/outlook").catch(() => {});
})();
