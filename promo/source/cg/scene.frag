#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 outC;
uniform sampler2D uN;
uniform vec2 uRes, uJit;
uniform vec3 uRo;
uniform mat3 uCam;
uniform float uF, uT;
uniform vec3 uDP;
uniform mat3 uDR;
uniform float uDS, uRotA, uRotOn, uBurn, uShowD;
uniform float uPan, uTilt;
uniform vec3 uBA, uBB;
uniform float uBOn, uBHit;

const vec3 SUN = normalize(vec3(-.30, .034, .95));
const vec3 SUNC = vec3(3.0, 1.25, .42);
const vec3 PIV = vec3(0., 1.62, 0.);
const vec3 SB = vec3(-1.75, .86, .15);
const mat2 M2 = mat2(.8, .6, -.6, .8);

float n2(vec2 p) { return textureLod(uN, p / 64., 0.).r; }
float fbm(vec2 p, int o) { float a = .5, s = 0.; for (int i = 0; i < o; i++) { s += a * n2(p); a *= .5; p = M2 * p * 2.03; } return s; }
float ridge(vec2 p, int o) {
  float a = .5, s = 0., w = 1.;
  for (int i = 0; i < o; i++) { float n = 1. - abs(2. * n2(p) - 1.); n *= n; s += a * n * w; w = clamp(n * 1.7, 0., 1.); a *= .5; p = M2 * p * 2.07; }
  return s;
}

float H(vec2 p, int o) {
  float d = length(p);
  float r = ridge(p * .00026 + vec2(.17, .31), o);
  float big = pow(r, 1.6) * 880.;
  float mid = fbm(p * .0019 + .3, min(o, 6)) * 86. - 48.;
  float dun = ridge(p * .0058, min(o, 5)) * 16.;
  float valley = smoothstep(140., 2300., d);
  float hh = mid + dun - 26. + big * valley;
  float plat = 1. - smoothstep(15., 70., d);
  return mix(hh, -.0, plat);
}

// ---------- sky ----------
vec3 skyBase(vec3 rd) {
  float y = max(rd.y, 0.);
  vec3 h = normalize(vec3(rd.x, 0., rd.z));
  float az = max(dot(h, normalize(vec3(SUN.x, 0., SUN.z))), 0.);
  float s = max(dot(rd, SUN), 0.);
  vec3 hor = mix(vec3(.13, .15, .26), vec3(1.25, .36, .085), pow(az, 3.2));
  hor = mix(hor, vec3(.62, .20, .17), (1. - pow(az, 3.2)) * pow(az, .6) * .55);
  vec3 c = mix(hor, vec3(.055, .105, .23), pow(clamp(y * 4.2, 0., 1.), .58));
  c = mix(c, vec3(.010, .024, .072), smoothstep(.10, .78, y));
  c += vec3(1.7, .52, .11) * pow(s, 18.) * .55;
  c += vec3(3.0, 1.3, .36) * pow(s, 90.) * .9;
  return c;
}

vec3 sky(vec3 rd, vec3 ro) {
  vec3 c = skyBase(rd);
  float s = max(dot(rd, SUN), 0.);
  c += vec3(30., 16., 7.) * smoothstep(.99972, .99990, s);
  if (rd.y > .012) {
    float t = (1500. - ro.y) / rd.y;
    vec2 q = (ro + rd * t).xz;
    float w = uT * .0009;
    float d = fbm(q * .00042 + vec2(w, w * .4), 7);
    float d2 = fbm(q * .00042 + vec2(w, w * .4) + SUN.xz * .09, 5);
    float cov = smoothstep(.40, .70, d);
    float lit = clamp((d - d2) * 7. + .55, 0., 1.);
    vec3 cc = mix(vec3(.018, .022, .05), vec3(1.5, .44, .15), lit * lit * (.05 + .95 * pow(s, 2.4)));
    cc += vec3(.12, .05, .07) * lit * (1. - s);
    float fade = smoothstep(.012, .10, rd.y) * (1. - smoothstep(.55, .95, rd.y) * .55);
    c = mix(c, cc, cov * fade * .92);
    // high cirrus
    float t2 = (4200. - ro.y) / rd.y;
    vec2 q2 = (ro + rd * t2).xz * vec2(.00011, .0006);
    float ci = smoothstep(.46, .78, fbm(q2 + w * .3, 6));
    c += ci * fade * vec3(.8, .28, .16) * (.03 + .55 * pow(s, 5.));
    // stars
    vec2 sp = floor(rd.xz / (rd.y + .35) * 620.);
    vec3 s3 = fract(vec3(sp.xyx) * .1031);
    s3 += dot(s3, s3.yzx + 33.33);
    float hs = fract((s3.x + s3.y) * s3.z);
    c += step(.9975, hs) * (.25 + 2.2 * (hs - .9975) * 400.) * .32 * smoothstep(.16, .6, rd.y) * (1. - cov * .9) * vec3(.8, .9, 1.);
  }
  return c;
}

// ---------- SDF kit ----------
float sdBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q, 0.)) + min(max(q.x, max(q.y, q.z)), 0.); }
float sdRB(vec3 p, vec3 b, float r) { return sdBox(p, b - r) - r; }
float sdCY(vec3 p, float r, float h) { vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h); return min(max(d.x, d.y), 0.) + length(max(d, 0.)); }
float sdCZ(vec3 p, float r, float h) { return sdCY(p.xzy, r, h); }
float sdCp(vec3 p, vec3 a, vec3 b, float r) { vec3 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0., 1.); return length(pa - ba * h) - r; }
vec2 U(vec2 a, vec2 b) { return a.x < b.x ? a : b; }

// materials: 1 ground 2 concrete 3 black polymer 4 purple 5 copper 6 glass 7 carbon 8 metal 9 pcb 10 burn
vec2 mapTurret(vec3 p) {
  vec2 r = vec2(1e5, 0.);
  r = U(r, vec2(sdRB(p - vec3(0., .09, 0.), vec3(.74, .07, .70), .03), 3.));
  r = U(r, vec2(sdCY(p - vec3(0., .28, 0.), .46, .12), 3.));
  r = U(r, vec2(sdCY(p - vec3(0., .54, 0.), .17, .18), 8.));
  // pan stage
  vec3 q = p - PIV;
  float cp = cos(uPan), sp = sin(uPan), ct = cos(uTilt), st = sin(uTilt);
  vec3 a = vec3(q.x * cp - q.z * sp, q.y, q.x * sp + q.z * cp);
  vec3 s = vec3(abs(a.x), a.y, a.z);
  r = U(r, vec2(sdRB(a - vec3(0., -.82, 0.), vec3(.60, .09, .40), .04), 3.));
  vec3 ps = s - vec3(.52, -.37, .0);
  float plate = sdRB(ps, vec3(.05, .52, .40), .025);
  plate = max(plate, dot(ps - vec3(0., .34, .0), normalize(vec3(0., .72, .69))));
  plate = max(plate, dot(ps - vec3(0., .34, .0), normalize(vec3(0., .72, -.69))));
  vec3 cut = ps - vec3(0., -.2, 0.);
  float tri = max(abs(cut.x) - 1., max(-cut.y - .17, max(dot(cut.yz - vec2(.2, 0.), normalize(vec2(.72, .69))), dot(cut.yz - vec2(.2, 0.), normalize(vec2(.72, -.69))))));
  plate = max(plate, -tri);
  r = U(r, vec2(plate, 3.));
  r = U(r, vec2(sdCZ((s - vec3(.53, -.78, .30)).zyx, .022, .08), 8.));
  r = U(r, vec2(sdCZ((s - vec3(.53, -.78, -.30)).zyx, .022, .08), 8.));
  r = U(r, vec2(sdRB(a - vec3(0., -.56, -.05), vec3(.25, .18, .28), .03), 7.));
  r = U(r, vec2(sdCZ((s - vec3(.60, .0, .0)).zyx, .10, .05), 8.));
  // tilt head
  vec3 b = vec3(a.x, a.y * ct - a.z * st, a.y * st + a.z * ct);
  r = U(r, vec2(sdRB(b - vec3(0., 0., .02), vec3(.34, .22, .40), .05), 3.));
  r = U(r, vec2(sdRB(b - vec3(0., .25, .05), vec3(.27, .045, .35), .02), 4.));
  r = U(r, vec2(sdRB(b - vec3(0., .30, .08), vec3(.20, .018, .25), .01), 9.));
  r = U(r, vec2(sdCZ(b - vec3(0., .40, .25), .115, .52), 5.));
  r = U(r, vec2(sdCZ(b - vec3(0., .40, -.02), .135, .08), 8.));
  r = U(r, vec2(sdCZ(b - vec3(0., .40, .80), .128, .03), 8.));
  r = U(r, vec2(sdCZ(b - vec3(0., .40, .835), .07, .012), 6.));
  // stereo bar, fixed
  vec3 c = p - SB;
  float sy = .12, c0 = cos(sy), s0 = sin(sy);
  c = vec3(c.x * c0 - c.z * s0, c.y, c.x * s0 + c.z * c0);
  vec3 w = c - vec3(0., -.40, -.08);
  float wedge = max(sdRB(w, vec3(.84, .46, .22), .02), dot(w - vec3(0., .46, .12), normalize(vec3(0., .42, .91))));
  r = U(r, vec2(wedge, 4.));
  r = U(r, vec2(sdRB(c - vec3(0., .02, .13), vec3(.80, .18, .022), .015), 9.));
  vec3 l = vec3(abs(c.x), c.yz) - vec3(.58, .02, .28);
  r = U(r, vec2(sdCZ(l, .135, .14), 3.));
  r = U(r, vec2(sdCZ(l - vec3(0., 0., .15), .095, .012), 6.));
  // cable from the bar, and the servo leads
  r = U(r, vec2(sdCp(p, vec3(-1.3, .1, -.1), vec3(-.7, .06, -.5), .025), 7.));
  return r;
}

vec2 mapDrone(vec3 pw) {
  vec3 p = uDR * (pw - uDP) / uDS;
  vec3 s = vec3(abs(p.x), p.y, abs(p.z));
  vec2 r = vec2(sdRB(p - vec3(0., .0, .02), vec3(.20, .085, .36), .07), 7.);
  r = U(r, vec2(sdRB(p - vec3(0., .12, -.05), vec3(.13, .06, .22), .04), 3.));
  r = U(r, vec2(length((p - vec3(0., -.13, .30)) * vec3(1., 1., 1.)) - .095, 3.));
  r = U(r, vec2(length(p - vec3(0., -.14, .38)) - .045, 6.));
  r = U(r, vec2(sdCp(s, vec3(.10, .0, .10), vec3(.62, .03, .50), .028), 7.));
  r = U(r, vec2(sdCY(s - vec3(.62, .06, .50), .068, .055), 8.));
  r = U(r, vec2(sdCY(s - vec3(.62, .125, .50), .018, .03), 8.));
  // skids
  vec3 q = vec3(abs(p.x), p.yz);
  r = U(r, vec2(sdCp(q, vec3(.14, -.06, .15), vec3(.26, -.32, .20), .012), 7.));
  r = U(r, vec2(sdCp(q, vec3(.14, -.06, -.15), vec3(.26, -.32, -.20), .012), 7.));
  r = U(r, vec2(sdCp(q, vec3(.26, -.32, -.34), vec3(.26, -.32, .34), .014), 7.));
  if (uBurn > 0.) {
    float b = length(p - vec3(.25, .05, .3)) - .22;
    if (r.x < .02 && b < 0.) r.y = 10.;
  }
  r.x *= uDS;
  return r;
}

vec2 mapAll(vec3 p) {
  vec2 r = mapTurret(p);
  if (uShowD > .5) r = U(r, mapDrone(p));
  return r;
}

float sphIn(vec3 ro, vec3 rd, vec3 c, float R, out float tf) {
  vec3 o = ro - c;
  float b = dot(o, rd), cc = dot(o, o) - R * R, h = b * b - cc;
  tf = 0.;
  if (h < 0.) return -1.;
  h = sqrt(h);
  tf = -b + h;
  return max(-b - h, 0.);
}

vec2 traceObj(vec3 ro, vec3 rd, float tmax, vec3 c, float R, int which) {
  float tf;
  float t = sphIn(ro, rd, c, R, tf);
  if (t < 0. || t > tmax) return vec2(-1.);
  tf = min(tf, tmax);
  for (int i = 0; i < 96; i++) {
    vec3 p = ro + rd * t;
    vec2 h = which == 0 ? mapTurret(p) : mapDrone(p);
    if (h.x < .0006 * t + .0008) return vec2(t, h.y);
    t += h.x;
    if (t > tf) break;
  }
  return vec2(-1.);
}

vec3 nrm(vec3 p, int which, float e) {
  vec2 k = vec2(1., -1.);
  if (which == 0) return normalize(k.xyy * mapTurret(p + k.xyy * e).x + k.yyx * mapTurret(p + k.yyx * e).x + k.yxy * mapTurret(p + k.yxy * e).x + k.xxx * mapTurret(p + k.xxx * e).x);
  return normalize(k.xyy * mapDrone(p + k.xyy * e).x + k.yyx * mapDrone(p + k.yyx * e).x + k.yxy * mapDrone(p + k.yxy * e).x + k.xxx * mapDrone(p + k.xxx * e).x);
}

float softT(vec3 ro, vec3 rd) {
  float tf;
  float t = sphIn(ro, rd, vec3(-.4, 1.2, 0.), 3.6, tf);
  if (t < 0.) return 1.;
  float res = 1.;
  t = max(t, .02);
  for (int i = 0; i < 40; i++) {
    float h = mapTurret(ro + rd * t).x;
    res = min(res, 9. * h / t);
    t += clamp(h, .02, .4);
    if (res < .004 || t > tf) break;
  }
  return clamp(res, 0., 1.);
}

float traceTerrain(vec3 ro, vec3 rd, float tmax) {
  float t = .4;
  if (ro.y > 640. && rd.y >= 0.) return -1.;
  for (int i = 0; i < 170; i++) {
    vec3 p = ro + rd * t;
    if (p.y > 640. && rd.y >= 0.) return -1.;
    float h = p.y - H(p.xz, t < 500. ? 7 : 5);
    if (h < .0008 * t) return t;
    t += max(h * .42, .012 * t * .1 + .03);
    if (t > tmax) return -1.;
  }
  return -1.;
}

float terrShadow(vec3 p) {
  float res = 1., t = 2.;
  for (int i = 0; i < 26; i++) {
    vec3 q = p + SUN * t;
    if (q.y > 640.) break;
    float h = q.y - H(q.xz, 5);
    res = min(res, 5. * h / t);
    t += max(h * .6, 3. + t * .05);
    if (res < .01 || t > 3500.) break;
  }
  return clamp(res, 0., 1.);
}

vec3 envR(vec3 r) {
  vec3 c = skyBase(r);
  if (r.y < 0.) c = mix(vec3(.03, .025, .025), vec3(.3, .14, .08), exp(r.y * 7.));
  return c;
}

vec3 shadeObj(vec3 p, vec3 rd, vec3 n, float m) {
  vec3 alb = vec3(.1);
  float rough = .5, metal = 0.;
  if (m == 3.) { alb = vec3(.022, .023, .026); rough = .42; }
  else if (m == 4.) { alb = vec3(.20, .085, .62); rough = .34; }
  else if (m == 5.) { alb = vec3(.86, .52, .20); rough = .22; metal = 1.; }
  else if (m == 6.) { alb = vec3(.01, .012, .02); rough = .04; metal = .4; }
  else if (m == 7.) { alb = vec3(.042, .045, .05); rough = .38; }
  else if (m == 8.) { alb = vec3(.42, .43, .46); rough = .3; metal = 1.; }
  else if (m == 9.) { alb = vec3(.012, .05, .036); rough = .35; }
  else if (m == 10.) { alb = vec3(.02); rough = .9; }
  else if (m == 11.) { alb = vec3(.75, .30, .05); rough = .4; }
  float ndl = max(dot(n, SUN), 0.);
  float sh = softT(p + n * .01, SUN);
  vec3 h = normalize(SUN - rd);
  float ndh = max(dot(n, h), 0.);
  float sp = pow(ndh, mix(420., 24., rough)) * mix(.4, 3., 1. - rough);
  vec3 r = reflect(rd, n);
  float fr = pow(1. - max(dot(n, -rd), 0.), 5.);
  vec3 F0 = mix(vec3(.04), alb, metal);
  vec3 F = F0 + (1. - F0) * fr * (1. - rough);
  vec3 col = alb * (1. - metal) * (SUNC * ndl * sh + vec3(.085, .13, .25) * (.55 + .45 * n.y) + vec3(.13, .07, .04) * (.5 - .5 * n.y));
  col += envR(r) * F * (1. - rough * .55);
  col += SUNC * sp * sh * F0 * 2.2;
  return col;
}

// translucent rotor blur over everything behind it
vec4 rotors(vec3 ro, vec3 rd) {
  vec3 o = uDR * (ro - uDP) / uDS, d = uDR * rd;
  vec4 acc = vec4(0.);
  for (int i = 0; i < 4; i++) {
    float sx = (i & 1) == 0 ? 1. : -1., sz = (i & 2) == 0 ? 1. : -1.;
    vec3 c = vec3(.62 * sx, .155, .50 * sz);
    float t = (c.y - o.y) / d.y;
    if (t <= 0.) continue;
    vec2 q = (o + d * t).xz - c.xz;
    float rr = length(q);
    if (rr > .40) continue;
    float a = atan(q.y, q.x) + uRotA * sx * sz + float(i) * 1.3;
    float blur = (.10 + .06 * sin(rr * 90.)) * smoothstep(.40, .37, rr) * smoothstep(.02, .05, rr);
    float blade = pow(max(cos(a * 1.), 0.), 26.) + pow(max(cos(a + 3.1416), 0.), 26.);
    float alpha = mix(blade * .9 * smoothstep(.40, .38, rr), blur, uRotOn);
    alpha = clamp(alpha, 0., 1.);
    vec3 c2 = vec3(.09, .08, .08) + SUNC * .05;
    acc.rgb += (1. - acc.a) * c2 * alpha;
    acc.a += (1. - acc.a) * alpha;
  }
  return acc;
}

float glowPt(vec3 ro, vec3 rd, vec3 c, float w) {
  float t = max(dot(c - ro, rd), 0.);
  vec3 q = ro + rd * t - c;
  return w * w / (dot(q, q) + w * w * .02);
}

vec3 render(vec3 ro, vec3 rd) {
  float tmax = 9000.;
  float tt = traceTerrain(ro, rd, tmax);
  vec2 hT = traceObj(ro, rd, tt > 0. ? tt : tmax, vec3(-.4, 1.2, 0.), 3.6, 0);
  vec2 hD = vec2(-1.);
  if (uShowD > .5) hD = traceObj(ro, rd, hT.x > 0. ? hT.x : (tt > 0. ? tt : tmax), uDP, uDS * 1.25, 1);
  vec3 col;
  float t = tmax;
  if (hD.x > 0.) {
    t = hD.x;
    vec3 p = ro + rd * t;
    vec3 n = nrm(p, 1, .0012 * uDS + .00025 * t);
    col = shadeObj(p, rd, n, hD.y);
    if (uBHit > 0.) {
      vec3 l = uBB - p;
      float dd = dot(l, l);
      col += vec3(9., .9, .35) * uBHit * max(dot(n, normalize(l)), .08) / (dd * 18. + .05);
    }
    if (hD.y == 10.) {
      float e = fbm(p.xz * 4. + uT * .4, 3);
      col += vec3(6., 1.5, .25) * pow(e, 3.) * 2.2 * uBurn;
    }
  } else if (hT.x > 0.) {
    t = hT.x;
    vec3 p = ro + rd * t;
    vec3 n = nrm(p, 0, .0015);
    col = shadeObj(p, rd, n, hT.y);
    if (uBOn > 0.) {
      float dl = length(p - uBA);
      col += vec3(4., .22, .1) * uBOn * exp(-dl * dl * 9.) * .5;
    }
  } else if (tt > 0.) {
    t = tt;
    vec3 p = ro + rd * t;
    float e = max(.0007 * t, .012);
    int o = t < 400. ? 10 : (t < 1500. ? 8 : 6);
    float h0 = H(p.xz, o);
    vec3 n = normalize(vec3(h0 - H(p.xz + vec2(e, 0.), o), e, h0 - H(p.xz + vec2(0., e), o)));
    float d = length(p.xz);
    float slope = 1. - n.y;
    float st = n2(vec2(p.y * .0035, p.x * .00012)) * .7 + .3 * n2(p.xz * .0021);
    vec3 rock = mix(vec3(.17, .105, .075), vec3(.34, .205, .125), st);
    vec3 sand = mix(vec3(.40, .265, .17), vec3(.50, .335, .205), fbm(p.xz * .013, 4));
    vec3 alb = mix(sand, rock, smoothstep(.04, .22, slope));
    float gr = fbm(p.xz * 1.9, 4);
    alb *= .78 + .44 * gr;
    float rough = 1.;
    if (d < 13.5) {
      vec2 g = abs(fract(p.xz / 3.) - .5);
      float seam = smoothstep(.0, .012, min(g.x, g.y));
      alb = mix(vec3(.17), vec3(.33, .33, .32) * (.82 + .3 * fbm(p.xz * .7, 4)), seam);
      float ring = smoothstep(.09, .06, abs(d - 12.6));
      float dash = step(.5, fract(atan(p.z, p.x) * 12.));
      alb = mix(alb, vec3(.82, .58, .08) * dash + vec3(.03) * (1. - dash), ring);
      n = vec3(0., 1., 0.);
    }
    float sh = terrShadow(p + n * .2);
    if (d < 22.) sh *= softT(p + n * .02, SUN);
    float ndl = max(dot(n, SUN), 0.);
    vec3 am = vec3(.07, .105, .20) * (.45 + .55 * n.y) + vec3(.09, .05, .03) * (1. - n.y);
    col = alb * (SUNC * ndl * sh * .62 + am);
    col *= .82 + .18 * smoothstep(0., 30., t * .0 + p.y + 70.);
    if (uBOn > 0.) {
      float dl = length(p - uBA);
      col += vec3(.5, .02, .01) * uBOn * 1.2 / (dl * dl + 3.);
    }
  } else {
    col = sky(rd, ro);
  }
  // aerial perspective
  if (t < tmax) {
    float f = 1. - exp(-pow(t * .00030, 1.25));
    vec3 fc = skyBase(normalize(vec3(rd.x, max(rd.y, .01) * .3 + .008, rd.z))) * .86;
    col = mix(col, fc, f);
    float low = exp(-max(ro.y + rd.y * t + 50., 0.) * .012);
    col = mix(col, fc, clamp(low * (1. - exp(-t * .0012)), 0., .6));
  }
  if (uShowD > .5) {
    vec4 ro_ = rotors(ro, rd);
    float tr = dot(uDP - ro, rd);
    if (tr > 0. && tr < t + uDS * 2.) col = mix(col, ro_.rgb / max(ro_.a, .001), ro_.a);
    float blink = step(.72, fract(uT * 1.3));
    vec3 c0 = uDP + transpose(uDR) * (vec3(-.62, .02, .50) * uDS);
    vec3 c1 = uDP + transpose(uDR) * (vec3(.62, .02, .50) * uDS);
    vec3 c2 = uDP + transpose(uDR) * (vec3(0., -.05, -.42) * uDS);
    float w = uDS * .018 + .0006 * length(uDP - ro);
    col += vec3(9., .5, .25) * glowPt(ro, rd, c0, w) * .05 * uRotOn;
    col += vec3(.3, 8., 1.6) * glowPt(ro, rd, c1, w) * .05 * uRotOn;
    col += vec3(9., 8., 7.) * glowPt(ro, rd, c2, w) * .18 * blink * uRotOn;
  }
  if (uBOn > 0.) {
    vec3 ab = uBB - uBA, ao = ro - uBA;
    float L = length(ab);
    vec3 u = ab / L;
    float rdu = dot(rd, u), rda = dot(rd, ao), uao = dot(u, ao);
    float den = 1. - rdu * rdu + 1e-6;
    float s = (uao - rdu * rda) / den, tb = (rdu * uao - rda) / den;
    s = clamp(s, 0., L);
    tb = max(dot(uBA + u * s - ro, rd), .02);
    if (tb < t + 1.) {
      vec3 cl = uBA + u * s - (ro + rd * tb);
      float d = length(cl);
      float w = .012 + tb * .00055;
      float dust = .72 + .46 * n2(vec2(s * .018 - uT * .22, tb * .003));
      float core = exp(-d * d / (w * w));
      float halo = w * w * 2.6 / (d * d + w * w * 1.4);
      col += vec3(11., .7, .32) * uBOn * core * 3.0;
      col += vec3(14., 12., 12.) * uBOn * pow(core, 6.) * .9;
      col += vec3(4.8, .16, .10) * uBOn * halo * dust * .34;
    }
    // aperture and impact glare
    col += vec3(14., 2., 1.2) * uBOn * glowPt(ro, rd, uBA, .05 + length(uBA - ro) * .0035) * .03;
    if (uBHit > 0.) {
      float dh = length(uBB - ro);
      if (dh < t + 6.) {
        float g = glowPt(ro, rd, uBB, .05 + dh * .0034);
        col += vec3(16., 5., 3.) * uBHit * g * .12;
        col += vec3(18., 3., 1.2) * uBHit * pow(g, .34) * .036;
      }
    }
  }
  return col;
}

void main() {
  vec2 uv = (gl_FragCoord.xy + uJit - .5 * uRes) / uRes.y;
  vec3 rd = normalize(uCam * vec3(uv, uF));
  outC = vec4(max(render(uRo, rd), vec3(0.)), 1.);
}
