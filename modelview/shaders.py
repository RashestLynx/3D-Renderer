"""GLSL for the real-time viewer: physically based shading (GGX + Smith + Schlick, metallic-roughness, clearcoat),
a procedural studio environment for reflections, a shadow-mapped key light, ACES tone mapping."""

COMMON = """
const float PI = 3.14159265;
uniform float env_mode;        // 0 studio, 1 overcast, 2 dark, 3 white
vec3 env(vec3 d) {
    // car-photo studio: dark floor and walls, a big overhead softbox, two long horizontal light strips that run
    // along a car's flanks in its reflections, and three tall vertical strips around the set
    float y = d.y;
    float az = atan(d.z, d.x);
    vec3 c = mix(vec3(0.015, 0.015, 0.018), vec3(0.07, 0.072, 0.08), smoothstep(-0.2, 0.05, y));
    c = mix(c, vec3(0.13, 0.135, 0.15), smoothstep(0.05, 0.9, y));
    float ceiling = smoothstep(0.84, 0.88, y) * smoothstep(0.62, 0.55, abs(d.x)) * smoothstep(0.42, 0.35, abs(d.z));
    float strip1 = smoothstep(0.035, 0.0, abs(y - 0.30)) * (0.75 + 0.25 * cos(az * 2.0));
    float strip2 = smoothstep(0.02, 0.0, abs(y - 0.12)) * 0.6;
    float v1 = smoothstep(0.07, 0.04, abs(az - 0.9)) * smoothstep(-0.05, 0.05, y) * smoothstep(0.75, 0.6, y);
    float v2 = smoothstep(0.07, 0.04, abs(az + 2.2)) * smoothstep(-0.05, 0.05, y) * smoothstep(0.75, 0.6, y);
    float v3 = smoothstep(0.05, 0.03, abs(az - 2.6)) * smoothstep(0.0, 0.1, y) * smoothstep(0.6, 0.5, y);
    c += vec3(5.0) * ceiling + vec3(3.4, 3.35, 3.3) * strip1 + vec3(2.2) * strip2
       + vec3(3.0, 2.95, 2.85) * v1 + vec3(2.4, 2.5, 2.7) * v2 + vec3(1.8) * v3;
    if (env_mode > 0.5 && env_mode < 1.5) c = mix(vec3(0.20), vec3(1.1), smoothstep(-0.3, 1.0, y));
    if (env_mode > 1.5 && env_mode < 2.5) c *= 0.35;
    if (env_mode > 2.5) c = mix(vec3(0.5), vec3(1.6), smoothstep(-0.5, 1.0, y));
    return c;
}
vec3 env_irr(vec3 n) {             // diffuse light from the environment (cosine-blurred, approximated)
    vec3 c = mix(vec3(0.04, 0.04, 0.045), vec3(0.30, 0.31, 0.34), n.y * 0.5 + 0.5);
    c += vec3(0.15) * max(dot(n, normalize(vec3(-0.9, 0.35, 0.3))), 0.0);
    if (env_mode > 0.5 && env_mode < 1.5) c = mix(vec3(0.25), vec3(0.8), n.y * 0.5 + 0.5);
    if (env_mode > 1.5 && env_mode < 2.5) c *= 0.35;
    if (env_mode > 2.5) c = mix(vec3(0.6), vec3(1.2), n.y * 0.5 + 0.5);
    return c;
}
vec3 env_spec(vec3 r, float rough) {  // rough reflections fade toward the blurred environment
    float k = clamp(rough * 1.4, 0.0, 1.0);
    return mix(env(r), env_irr(r) * 1.05, k * k * (3.0 - 2.0 * k));
}
vec2 env_brdf(float nv, float rough) {   // Karis' analytic fit of the split-sum BRDF
    vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}
float D_ggx(float nh, float a) { float a2 = a * a; float d = nh * nh * (a2 - 1.0) + 1.0; return a2 / (PI * d * d); }
float V_smith(float nv, float nl, float a) {
    float k = a * 0.5;
    return 0.25 / max((nv * (1.0 - k) + k) * (nl * (1.0 - k) + k), 1e-4);
}
vec3 F_schlick(vec3 f0, float vh) { return f0 + (1.0 - f0) * pow(1.0 - vh, 5.0); }
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
"""

VS = """
#version 330
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in vec2 in_uv2; in vec4 in_tan;
uniform mat4 mvp; uniform mat4 light_mvp; uniform bool mirror; uniform float floor_y;
out vec3 v_pos; out vec3 v_n; out vec4 v_t; out vec2 v_uv; out vec2 v_uv2; out vec4 v_ls;
void main() {
    vec3 p = in_pos; vec3 n = in_nrm; vec4 t = in_tan;
    if (mirror) {                     // the floor reflection: the model flipped about the floor plane
        p.y = 2.0 * floor_y - p.y; n.y = -n.y; t.y = -t.y; t.w = -t.w;
    }
    v_pos = p; v_n = n; v_t = t; v_uv = in_uv; v_uv2 = in_uv2;
    v_ls = light_mvp * vec4(p, 1.0);
    gl_Position = mvp * vec4(p, 1.0);
}
"""

FS = """
#version 330
""" + COMMON + """
in vec3 v_pos; in vec3 v_n; in vec4 v_t; in vec2 v_uv; in vec2 v_uv2; in vec4 v_ls;
uniform sampler2D tex_c; uniform sampler2D tex_n; uniform sampler2D tex_mr; uniform sampler2D tex_e;
uniform sampler2D tex_o; uniform sampler2D shadow_map;
uniform bool has_c; uniform bool has_n; uniform bool has_mr; uniform bool has_e; uniform bool has_o; uniform bool o_uv2;
uniform bool use_tex; uniform bool use_nmap; uniform bool lit; uniform bool flip_g; uniform bool shadows;
uniform vec4 base; uniform float metallic; uniform float roughness; uniform vec3 emissive;
uniform float clearcoat; uniform float cc_rough; uniform float transmission; uniform float cutoff; uniform bool blend;
uniform vec3 cam; uniform vec3 key_dir; uniform float exposure; uniform vec2 shadow_texel;
uniform mat3 tf_c; uniform mat3 tf_n; uniform mat3 tf_mr; uniform mat3 tf_e; uniform mat3 tf_o;   // KHR_texture_transform
uniform bool uv2_c; uniform bool uv2_n; uniform bool uv2_mr; uniform bool uv2_e;
out vec4 f_color;
vec2 tuv(mat3 tf, bool second) { return (tf * vec3(second ? v_uv2 : v_uv, 1.0)).xy; }

float shadow_vis() {
    vec3 p = v_ls.xyz / v_ls.w * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0) return 1.0;
    float vis = 0.0;
    for (int x = -2; x <= 2; x++) for (int y = -2; y <= 2; y++) {
        float d = texture(shadow_map, p.xy + vec2(x, y) * shadow_texel).r;
        vis += (p.z - 0.0015 > d) ? 0.0 : 1.0;
    }
    return vis / 25.0;
}

void main() {
    vec4 c = (has_c && use_tex) ? texture(tex_c, tuv(tf_c, uv2_c)) * base : (use_tex ? base : vec4(0.75, 0.75, 0.77, base.a));
    if (c.a < cutoff) discard;
    vec3 n = normalize(v_n);
    vec3 v = normalize(cam - v_pos);
    if (!gl_FrontFacing) n = -n;
    vec3 ng = n;                          // the smooth surface: the clearcoat layer lies on this, not on the bumps
    float nlen = 1.0;
    if (has_n && use_nmap) {
        vec3 t = normalize(v_t.xyz - n * dot(n, v_t.xyz));
        vec3 b = cross(n, t) * (v_t.w < 0.0 ? -1.0 : 1.0);
        // sampled a little blurrier than the colour: bump detail finer than a pixel sparkles as noise
        vec3 tn = texture(tex_n, tuv(tf_n, uv2_n), 1.5).xyz * 2.0 - 1.0;
        if (flip_g) tn.y = -tn.y;
        if (clearcoat > 0.0) tn.xy *= 0.3;   // paint flakes sit under the lacquer: a sheen, not grit
        nlen = length(tn);                // < 1 where the mip-filtered bumps disagree (fine flakes far away)
        n = normalize(mat3(t, b, n) * tn);
    }
    vec3 albedo = pow(c.rgb, vec3(2.2));
    if (!lit) { f_color = vec4(c.rgb, blend ? c.a : 1.0); return; }
    float metal = metallic, rough = roughness;
    if (has_mr && use_tex) { vec4 mr = texture(tex_mr, tuv(tf_mr, uv2_mr)); rough *= mr.g; metal *= mr.b; }
    // Toksvig: bump detail too fine for the pixel widens the highlight instead of sparkling
    rough = clamp(sqrt(rough * rough + clamp((1.0 - nlen) / max(nlen, 1e-3), 0.0, 1.0) * 10.0), 0.04, 1.0);
    float ao = (has_o && use_tex) ? texture(tex_o, tuv(tf_o, o_uv2)).r : 1.0;
    vec3 f0 = mix(vec3(0.04), albedo, metal);
    vec3 diff = albedo * (1.0 - metal);
    float nv = clamp(dot(n, v), 1e-3, 1.0);
    float a = rough * rough;
    vec3 col = vec3(0.0);
    // three lights: shadowed key, cool fill, rim
    vec3 L[3] = vec3[](key_dir, normalize(vec3(-0.8, 0.3, -0.35)), normalize(vec3(-0.25, 0.45, -1.0)));
    vec3 C[3] = vec3[](vec3(2.6, 2.5, 2.35), vec3(0.45, 0.5, 0.6), vec3(1.0, 1.0, 1.05));
    float vis = shadows ? shadow_vis() : 1.0;
    for (int i = 0; i < 3; i++) {
        vec3 l = L[i];
        float nl = max(dot(n, l), 0.0);
        if (nl <= 0.0) continue;
        vec3 h = normalize(l + v);
        float nh = max(dot(n, h), 0.0), vh = max(dot(v, h), 0.0);
        vec3 F = F_schlick(f0, vh);
        // GGX's long tail on fully rough surfaces turns dark matte things (black hair, cloth) milky grey under a
        // bright key: damp it the way the environment term is damped
        vec3 spec = D_ggx(nh, a) * V_smith(nv, nl, a) * F * mix(1.0, 0.15, rough * rough);
        vec3 kd = (1.0 - F);
        float s = (i == 0) ? vis : 1.0;
        col += (kd * diff / PI + spec) * C[i] * nl * s * PI * 0.55;
        if (clearcoat > 0.0) {
            float ca = cc_rough * cc_rough + 0.002;
            float gl = max(dot(ng, l), 0.0), gh = max(dot(ng, h), 0.0), gv = clamp(dot(ng, v), 1e-3, 1.0);
            col += clearcoat * D_ggx(gh, ca) * V_smith(gv, gl, ca) * F_schlick(vec3(0.04), vh) * C[i] * gl * s * PI * 0.55;
        }
    }
    vec2 ab = env_brdf(nv, rough);
    vec3 r = reflect(-v, n);
    float horizon = clamp(1.0 + dot(r, normalize(v_n)), 0.0, 1.0);    // no reflections from below the surface
    // rough surfaces barely mirror the room: fading the split-sum term keeps matte cloth/hair/skin from going milky
    vec3 spec_env = env_spec(r, rough) * (f0 * ab.x + ab.y) * horizon * horizon * mix(1.0, 0.25, rough);
    col += (diff * env_irr(n) * 0.75 + spec_env) * ao * mix(0.45, 1.0, vis);
    if (clearcoat > 0.0) {
        float gv = clamp(dot(ng, v), 1e-3, 1.0);
        vec3 rg = reflect(-v, ng);
        float fc = 0.04 + 0.96 * pow(1.0 - gv, 5.0);
        col = col * (1.0 - clearcoat * fc) + clearcoat * fc * env_spec(rg, cc_rough) * clamp(1.0 + dot(rg, ng), 0.0, 1.0);
    }
    vec3 e = emissive * ((has_e && use_tex) ? pow(texture(tex_e, tuv(tf_e, uv2_e)).rgb, vec3(2.2)) : vec3(1.0));
    col += e * 2.0;
    float alpha = blend ? c.a : 1.0;
    if (transmission > 0.0) {          // glass: mostly reflection, a tinted see-through body
        float fr = 0.04 + 0.96 * pow(1.0 - nv, 5.0);
        alpha = clamp(mix(0.18, 1.0, fr) + (1.0 - transmission) * 0.6, 0.0, 1.0) * c.a;
        col = col * 0.35 + env_spec(r, rough) * fr;
    }
    f_color = vec4(pow(aces(col * exposure), vec3(1.0 / 2.2)), alpha);
}
"""

DEPTH_VS = """
#version 330
in vec3 in_pos; in vec2 in_uv; uniform mat4 light_mvp; out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = light_mvp * vec4(in_pos, 1.0); }
"""
DEPTH_FS = """
#version 330
in vec2 v_uv; uniform sampler2D tex_c; uniform bool has_c; uniform float cutoff; out vec4 f;
void main() { if (has_c && cutoff > 0.0 && texture(tex_c, v_uv).a < cutoff) discard; f = vec4(1.0); }
"""

BG_VS = """
#version 330
out vec2 ndc;
void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); ndc = p * 2.0 - 1.0; gl_Position = vec4(ndc, 0.9999, 1.0); }
"""
BG_FS = """
#version 330
in vec2 ndc; uniform mat4 inv_vp; uniform vec3 top; uniform vec3 bottom; uniform bool show_env; uniform float exposure;
""" + COMMON + """
out vec4 f_color;
void main() {
    vec2 uv = ndc * 0.5 + 0.5;
    vec2 d = uv - vec2(0.5, 0.55);
    vec3 c = mix(bottom, top, uv.y) * (1.0 - dot(d, d) * 0.6);
    if (show_env) {
        vec4 p = inv_vp * vec4(ndc, 1.0, 1.0);
        vec4 o = inv_vp * vec4(ndc, -1.0, 1.0);
        vec3 dir = normalize(p.xyz / p.w - o.xyz / o.w);
        c = pow(aces(env(dir) * exposure * 0.6), vec3(1.0 / 2.2));
    }
    f_color = vec4(c, 1.0);
}
"""

GROUND_VS = """
#version 330
in vec2 in_xz; uniform mat4 mvp; uniform mat4 light_mvp; uniform mat4 contact_mvp; uniform float floor_y;
out vec2 v_xz; out vec4 v_ls; out vec4 v_cs;
void main() {
    vec4 p = vec4(in_xz.x, floor_y, in_xz.y, 1.0);
    v_xz = in_xz; v_ls = light_mvp * p; v_cs = contact_mvp * p; gl_Position = mvp * p;
}
"""
GROUND_FS = """
#version 330
// The floor: drawn over the mirrored model (a glossy showroom floor whose reflection fades with distance), darkened
// by the key light's shadow and by a soft contact shadow (anything above the floor, seen from straight overhead).
in vec2 v_xz; in vec4 v_ls; in vec4 v_cs;
uniform sampler2D shadow_map; uniform sampler2D contact_map; uniform vec2 shadow_texel; uniform vec2 contact_texel;
uniform vec2 centre; uniform float radius; uniform vec3 tint; uniform float reflect_amount; uniform bool shadows;
uniform float exposure; out vec4 f_color;
void main() {
    float vis = 1.0, occ = 0.0;
    if (shadows) {
        vec3 p = v_ls.xyz / v_ls.w * 0.5 + 0.5;
        if (p.x > 0.0 && p.x < 1.0 && p.y > 0.0 && p.y < 1.0) {
            vis = 0.0;
            for (int x = -3; x <= 3; x++) for (int y = -3; y <= 3; y++)
                vis += (p.z - 0.002 > texture(shadow_map, p.xy + vec2(x, y) * shadow_texel * 1.6).r) ? 0.0 : 1.0;
            vis /= 49.0;
        }
        vec3 q = v_cs.xyz / v_cs.w * 0.5 + 0.5;
        for (int x = -4; x <= 4; x++) for (int y = -4; y <= 4; y++) {
            vec2 o = vec2(x, y) * contact_texel * 5.0;
            float w = exp(-float(x * x + y * y) / 12.0);
            occ += (texture(contact_map, q.xy + o).r < 0.9995 ? 1.0 : 0.0) * w;
        }
        occ /= 23.3;
    }
    float r = length(v_xz - centre) / radius;
    vec3 col = tint * (0.45 + 0.55 * vis) * (1.0 - 0.8 * clamp(occ, 0.0, 1.0));
    float shine = reflect_amount * (1.0 - smoothstep(0.02, 0.55, r));
    float alpha = (1.0 - shine) * (1.0 - smoothstep(0.55, 1.0, r));
    // where the floor fades out, keep the shadows a little longer so the model never floats
    float dark = (1.0 - vis) * 0.5 + occ * 0.7;
    alpha = max(alpha, clamp(dark, 0.0, 0.85) * (1.0 - smoothstep(0.4, 1.0, r)));
    f_color = vec4(col, alpha);
}
"""

LINE_VS = """
#version 330
in vec3 in_pos; uniform mat4 mvp; out float fade;
void main() { fade = in_pos.y; gl_Position = mvp * vec4(in_pos.x, 0.0, in_pos.z, 1.0); }
"""
LINE_FS = """
#version 330
in float fade; uniform vec4 color; out vec4 f_color;
void main() { f_color = vec4(color.rgb, color.a * fade); }
"""

UI_VS = """
#version 330
in vec2 in_pos; in vec2 in_uv; uniform vec2 screen; out vec2 uv;
void main() { uv = in_uv; gl_Position = vec4(in_pos / screen * 2.0 - 1.0, 0.0, 1.0); }
"""
UI_FS = """
#version 330
in vec2 uv; uniform sampler2D tex; out vec4 f_color;
void main() { f_color = texture(tex, uv); }
"""
