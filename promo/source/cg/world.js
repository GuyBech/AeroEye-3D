// AeroEye-3D promo: the CG world's motion. Everything the shader and the HUD need comes from here.
const V = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  mul: (a, k) => [a[0] * k, a[1] * k, a[2] * k],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: a => Math.hypot(a[0], a[1], a[2]),
  norm: a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
  lerp: (a, b, u) => [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u],
};
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const ss = (a, b, x) => { const u = clamp((x - a) / (b - a)); return u * u * (3 - 2 * u); };
const lerp = (a, b, u) => a + (b - a) * u;
const PIV = [0, 1.62, 0];
const SB = [-1.75, .86, .15];
const DRONE_S = 2.1;

// flight path: Catmull-Rom through waypoints, one per 5 s of engagement time
const WP = [[420, 150, 700], [300, 128, 560], [190, 112, 440], [92, 100, 340], [6, 92, 262], [-72, 88, 196], [-150, 86, 138], [-230, 86, 96], [-310, 88, 70]];
function path(tau) {
  const u = clamp(tau / 5, 0, WP.length - 1.001), i = Math.floor(u), f = u - i;
  const p0 = WP[Math.max(i - 1, 0)], p1 = WP[i], p2 = WP[Math.min(i + 1, WP.length - 1)], p3 = WP[Math.min(i + 2, WP.length - 1)];
  const c = k => .5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * f + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * f * f + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * f * f * f);
  const w = [Math.sin(tau * .9) * 5, Math.sin(tau * 1.3 + 1) * 3, Math.cos(tau * .7) * 3];
  return [c(0) + w[0], c(1) + w[1], c(2) + w[2]];
}
const vel = tau => V.mul(V.sub(path(tau + .05), path(tau - .05)), 10);

function droneRot(v, roll = 0, pitch = null) {
  // world -> drone local (columns of the transpose are the body axes)
  const f0 = V.norm([v[0], 0, v[2]]);
  const sp = V.len(v);
  const p = pitch === null ? clamp(sp * .012, 0, .3) : pitch;
  const f = V.norm(V.add(f0, [0, -Math.sin(p), 0]));
  let r = V.norm(V.cross([0, 1, 0], f));
  let u = V.cross(f, r);
  const c = Math.cos(roll), s = Math.sin(roll);
  const r2 = V.add(V.mul(r, c), V.mul(u, s)), u2 = V.add(V.mul(u, c), V.mul(r, -s));
  return { r: r2, u: u2, f };
}

function aimAt(q) {
  const d = V.sub(q, PIV);
  return [Math.atan2(d[0], d[2]), Math.atan2(d[1], Math.hypot(d[0], d[2]))];
}
function barrelTip(pan, tilt) {
  const ct = Math.cos(tilt), st = Math.sin(tilt), cp = Math.cos(pan), sp = Math.sin(pan);
  const f = [sp * ct, st, cp * ct], u = [-sp * st, ct, -cp * st];
  return V.add(V.add(PIV, V.mul(f, .85)), V.mul(u, .40));
}
function lenses() {
  const a = .12, c = Math.cos(a), s = Math.sin(a);
  const w = l => [SB[0] + l[0] * c + l[2] * s, SB[1] + l[1], SB[2] - l[0] * s + l[2] * c];
  return [w([-.58, .02, .44]), w([.58, .02, .44])];
}

const HIT_TAU = 30.0;          // drone is hit at this engagement time
function fallen(tau) {
  // after the hit: momentum + gravity + tumble
  const dt = tau - HIT_TAU, p0 = path(HIT_TAU), v0 = vel(HIT_TAU);
  return [p0[0] + v0[0] * dt * .8, p0[1] + v0[1] * dt - 4.9 * .9 * dt * dt, p0[2] + v0[2] * dt * .8];
}
function dronePos(tau) { return tau <= HIT_TAU ? path(tau) : fallen(tau); }

const smooth = (a, b, u) => lerp(a, b, u * u * (3 - 2 * u));

// camera helpers
const look = (ro, ta, fov) => ({ ro, ta, fov });
const orbit = (c, r, a, h, ta, fov) => look([c[0] + Math.sin(a) * r, c[1] + h, c[2] + Math.cos(a) * r], ta, fov);

// each shot: frames, and state(u, s) with u in 0..1, s seconds into the shot
const SHOTS = {
  hook1: {
    n: 72, state: (u, s) => {
      const tau = -6 + s * 3.5;
      const d = path(0), p = V.add(V.lerp([980, 330, 1600], d, ss(0, 1, u) * .45), [0, 0, 0]);
      return { cam: look([-30 + s * 4, 3.2 + s * .3, -22], [160, 92 + s * 3, 700], 30), tau, dp: p, showT: 0, rotor: 1 };
    },
  },
  hook2: {
    n: 54, sub: 4, state: (u, s) => {
      const ro = [8, 118, 300];
      const dp = V.lerp([-62, 128, 372], [42, 113, 262], u);
      const v = V.mul(V.sub([42, 113, 262], [-62, 128, 372]), 1 / 1.8);
      return { cam: look(ro, V.lerp(dp, [8, 118, 340], .22), 46), tau: 10, dp, dvel: v, rotor: 1, roll: -.22 };
    },
  },
  scan: {
    n: 72, state: (u, s) => {
      const pan = .42 * Math.sin(s * 1.35 + .4) + .2, tilt = .22 + .05 * Math.sin(s * 2.1);
      const a = 2.15 + s * .11;
      return { cam: orbit([0, 0, 0], 7.4, a, 1.15, [-.45, 1.5, 0], 34), tau: -30, showD: 0, pan, tilt };
    },
  },
  detect: {
    n: 144, state: (u, s) => {
      const tau = 3.6 + s * 1.1;
      const d = dronePos(tau);
      const [ap, at] = aimAt(d);
      const lock = ss(1.7, 2.9, s);
      const sweepP = .42 * Math.sin((s + 2.4) * 1.35 + .4) + .2, sweepT = .22 + .05 * Math.sin((s + 2.4) * 2.1);
      return { cam: look([-2.6, 2.1, -6.6], V.lerp([26, 46, 190], [44, 52, 210], ss(0, 4.8, s)), lerp(40, 33, ss(.3, 4.8, s))), tau, pan: lerp(sweepP, ap, lock), tilt: lerp(sweepT, at, lock) };
    },
  },
  eyes: {
    n: 144, w: 960, h: 720, eyes: true, state: (u, s, eye) => {
      const tau = 9.2 + s * 1.1;
      const d = dronePos(tau);
      const base = 7.5;                  // visibly exaggerated stereo baseline
      const ro = [eye * base / 2 - 1.75, 3.5, .15];
      const [ap, at] = aimAt(d);
      const m = path(11.9), fw = V.norm(V.sub(m, [-1.75, 3.5, .15]));
      return { cam: look(ro, V.add(ro, fw), 8.2), tau, pan: ap, tilt: at, showT: 0 };
    },
  },
  depth: {
    n: 144, state: (u, s) => {
      const tau = 14.5 + s * 1.1;
      const d = dronePos(tau);
      const [ap, at] = aimAt(d);
      return { cam: look([-2.55 + s * .04, 1.78 + s * .02, -4.5], [-2.2 + d[0] * .02, 3.1, 26], 47), tau, pan: ap, tilt: at, overlayLens: 1 };
    },
  },
  track: {
    n: 288, sub: 2, state: (u, s) => {
      const tau = 19.8 + s * .8;
      const d = dronePos(tau), v = vel(tau);
      const side = V.norm(V.cross(v, [0, 1, 0]));
      const a = lerp(-.7, .9, ss(0, 1, u));
      const off = V.add(V.add(V.mul(side, Math.cos(a) * 19), V.mul(V.norm(v), Math.sin(a) * -19)), [0, 7.5 - 8 * u, 0]);
      const [ap, at] = aimAt(d);
      return { cam: look(V.add(d, off), V.add(d, V.mul(V.norm(v), 2.5)), 36), tau, pan: ap, tilt: at, trail: 1 };
    },
  },
  turret: {
    n: 72, state: (u, s) => {
      const tau = 27.6 + s * .25;
      const d = dronePos(tau);
      const [ap, at] = aimAt(d);
      const k = ss(.1, .85, s);
      const pan = lerp(ap - .75, ap, k), tilt = lerp(at - .28, at, k);
      const on = ss(.9, 1.05, s);
      const dir = V.norm(V.sub(V.add(d, [9, 7, 0]), barrelTip(pan, tilt)));
      return { cam: orbit([0, 0, 0], 3.9 + s * .12, ap + 1.72 + s * .10, .55, [.05, 1.92, 0], 40), tau, pan, tilt, beam: on, beamDir: dir, showD: 0 };
    },
  },
  chase: {
    n: 90, sub: 2, state: (u, s) => {
      const tau = lerp(28.2, HIT_TAU, u);
      const d = dronePos(tau), v = vel(tau);
      const k = 1 - ss(0, 1, u) ** 1.7;
      const miss = V.add(V.mul([9, 7.5, 2], k), [Math.sin(s * 3) * 1.2 * k, Math.cos(s * 2.3) * 1.0 * k, 0]);
      const aim = V.add(d, miss);
      const [ap, at] = aimAt(aim);
      const tip = barrelTip(ap, at);
      const side = V.norm(V.cross(v, [0, 1, 0]));
      const ro = V.add(V.add(d, V.mul(side, -lerp(13, 8.5, u))), [lerp(4, -2.5, u), lerp(-5.5, -4.2, u), 0]);
      return { cam: look(ro, V.lerp(d, aim, .1), 38), tau, pan: ap, tilt: at, beam: 1, beamEnd: V.add(tip, V.mul(V.norm(V.sub(aim, tip)), 900)), hitNow: u >= 1 };
    },
  },
  hit: {
    n: 54, state: (u, s) => {
      const tau = HIT_TAU + s * .42;
      const d = dronePos(tau);
      const p0 = dronePos(HIT_TAU);
      const v = vel(HIT_TAU);
      const side = V.norm(V.cross(v, [0, 1, 0]));
      const ro = V.add(V.add(p0, V.mul(side, -8.5)), [-2.5, -4.2, 0]);
      const [ap, at] = aimAt(p0);
      const on = 1 - ss(.75, .95, s);
      return { cam: look(ro, V.lerp(p0, d, .75), 38), tau, pan: ap, tilt: at, beam: on, beamEnd: d, hit: on, burn: 1, rotor: 1 - ss(0, .9, s), dvelOverride: 1, tumble: s };
    },
  },
  recap: {
    n: 144, state: (u, s) => {
      const tau = HIT_TAU + 1.8 * .42 + s * .75;
      const d = dronePos(tau);
      const [ap, at] = aimAt(dronePos(HIT_TAU));
      return { cam: look([2.5, 2.2, -9], [-150, lerp(34, 16, ss(0, 1, u)), 190], 40), tau, pan: lerp(ap, .1, ss(.1, .8, u)), tilt: lerp(at, .2, ss(.1, .8, u)), burn: 1, rotor: 0, tumble: 1.8 + s };
    },
  },
};

function frameState(name, f, eye = 0) {
  const sh = SHOTS[name], s = f / 30, u = sh.n > 1 ? f / (sh.n - 1) : 0;
  const st = sh.state(u, s, eye);
  const tau = st.tau;
  const dp = st.dp || dronePos(tau);
  const v = st.dvel || (tau > HIT_TAU ? V.sub(dronePos(tau + .05), dronePos(tau - .05)) : vel(tau));
  const R = droneRot(tau > HIT_TAU ? vel(HIT_TAU) : v, st.roll === undefined ? -v[0] * .002 : st.roll);
  if (st.tumble !== undefined) {
    const a = st.tumble * 1.5, c = Math.cos(a), s_ = Math.sin(a);
    const r2 = V.add(V.mul(R.r, c), V.mul(R.u, s_)), u2 = V.add(V.mul(R.u, c), V.mul(R.r, -s_));
    const c2 = Math.cos(a * .6), s2 = Math.sin(a * .6);
    R.u = V.add(V.mul(u2, c2), V.mul(R.f, s2)); R.f = V.add(V.mul(R.f, c2), V.mul(u2, -s2)); R.r = r2;
  }
  const [ap, at] = aimAt(dp);
  const pan = st.pan === undefined ? ap : st.pan, tilt = st.tilt === undefined ? at : st.tilt;
  const tip = barrelTip(pan, tilt);
  const beam = st.beam || 0;
  let end = st.beamEnd;
  if (st.beamDir) end = V.add(tip, V.mul(st.beamDir, 900));
  if (!end) end = dp;
  return {
    ...st, dp, v, R, pan, tilt, tip, beam, end,
    hit: st.hit !== undefined ? st.hit : (st.hitNow ? 1 : 0),
    burn: st.burn || 0, rotor: st.rotor === undefined ? 1 : st.rotor,
    showD: st.showD === undefined ? 1 : st.showD,
    clock: 40 + tau * 1.0,
  };
}

function camBasis(c) {
  const f = V.norm(V.sub(c.ta, c.ro));
  const r = V.norm(V.cross([0, 1, 0], f));
  const u = V.cross(f, r);
  return { r, u, f, F: .5 / Math.tan(c.fov * Math.PI / 360) };
}
function project(p, c, b, W, H) {
  const q = V.sub(p, c.ro);
  const z = V.dot(q, b.f);
  return [W / 2 + V.dot(q, b.r) / z * b.F * H, H / 2 - V.dot(q, b.u) / z * b.F * H, z];
}

if (typeof module !== 'undefined') module.exports = { SHOTS, frameState };
