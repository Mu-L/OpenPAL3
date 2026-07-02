#version 450
#extension GL_ARB_separate_shader_objects : enable

// PAL3-specific actor (skin) shader — vertex stage. Faithfully reproduces the
// original PAL3 `gfxscript/skin_1L.gbf` / `skin_2L.gbf` (`vs_1_1`) lighting
// model, which is computed **per vertex** (Gouraud) and modulated by the
// texture in the fragment stage:
//
//   perLight_i = atten_i * max(N·L_i, 0) * diffuse_i.rgb + ambient_i.rgb
//   color.rgb  = saturate( SUM_{i in 1..2} perLight_i )   // vs_1_1 clamps COLOR
//   color.a    = diffuse[0].w
//   finalPixel = texture * color                          // ColorOp = Modulate
//
// The original evaluates at most the two nearest omni point lights with the
// Direct3D attenuation `atten = 1/(a0 + a1·d + a2·d²)`. PAL3 ships un-attenuated
// omni lights (FLT_MAX range ⇒ `a1=a2=0` ⇒ atten≈1). Doing this per-vertex —
// rather than per-pixel — is what gives the original its characteristically
// flat, evenly-lit characters (a per-pixel N·L looks "too realistic").
//
// The `ambient` term stands in for the character's material ambient
// (`MtlAmbient`), which is high for PAL3 roles (skin is fully readable on
// ambient alone); it is distinct from the dim ~0.10 scene ambient used for
// scenery, hence the 0.55 floor. World/normal transform follows radiance's
// row-vector convention (`v * M`).
layout(set = 0, binding = 0) uniform PerFrameUbo {
    mat4 view;
    mat4 proj;
    vec4 ambient;            // rgb = ambient color, w = light count
    vec4 lightPos[16];       // xyz = world position, w = outer range
    vec4 lightColor[16];     // rgb = color, w = inner range
} perFrameUbo;

layout(set = 1, binding = 0) uniform PerInstanceUbo {
    mat4 model;
} perInstanceUbo;

layout(set = 3, binding = 0) uniform MaterialParams {
    vec4 tint;
    vec4 misc;
    vec4 uv_xform;
} mat;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 fragWorldPos;
layout(location = 2) out vec3 fragColor;

mat4 clip = mat4(vec4(1.0, 0.0, 0.0, 0.0),
                 vec4(0.0, -1.0, 0.0, 0.0),
                 vec4(0.0, 0.0, 0.5, 0.5),
                 vec4(0.0, 0.0, 0, 1.0));

void main() {
    vec4 world = vec4(position, 1.0) * perInstanceUbo.model;
    gl_Position = world * perFrameUbo.view * perFrameUbo.proj * clip;

    vec3 worldPos = world.xyz;
    vec3 N = normalize(normal * mat3(perInstanceUbo.model));

    // Character material ambient: high (skin reads fully on ambient), and
    // separate from the dim scene ambient used for scenery.
    vec3 ambient = max(perFrameUbo.ambient.rgb, vec3(0.55));

    // Pick the two nearest enabled lights, matching the original's 1-2 light
    // skin shaders. The full table is at most 16 entries.
    int count = int(perFrameUbo.ambient.w);
    int best0 = -1, best1 = -1;
    float d0 = 1e30, d1 = 1e30;
    for (int i = 0; i < count; i++) {
        float dist = distance(perFrameUbo.lightPos[i].xyz, worldPos);
        if (dist < d0) {
            d1 = d0; best1 = best0;
            d0 = dist; best0 = i;
        } else if (dist < d1) {
            d1 = dist; best1 = i;
        }
    }

    // Per-vertex Gouraud accumulation, exactly as skin_1L/skin_2L.gbf.
    vec3 lit = ambient;
    int idx[2] = int[2](best0, best1);
    for (int k = 0; k < 2; k++) {
        int i = idx[k];
        if (i < 0) continue;
        vec3 d = perFrameUbo.lightPos[i].xyz - worldPos;
        float dist = length(d);
        vec3 L = dist > 0.0 ? d / dist : vec3(0.0, 1.0, 0.0);

        // PAL3 omni lights ship FLT_MAX range (no attenuation); treat any very
        // large outer radius as infinite, else use the D3D point-light cutoff.
        float outer = perFrameUbo.lightPos[i].w;
        float inner = perFrameUbo.lightColor[i].w;
        float atten = 1.0;
        if (outer < 1.0e18) {
            float edge0 = max(inner, outer * 0.85);
            atten = 1.0 - smoothstep(edge0, outer, dist);
        }

        lit += perFrameUbo.lightColor[i].rgb * max(dot(N, L), 0.0) * atten;
    }

    // vs_1_1 clamps the interpolated COLOR output to [0,1] before it reaches the
    // texture stage; replicate that here so bright keys don't blow out the
    // texture modulation.
    fragColor = clamp(lit, 0.0, 1.0);

    fragWorldPos = worldPos;
    fragTexCoord = inTexCoord * mat.uv_xform.xy + mat.uv_xform.zw;
}
