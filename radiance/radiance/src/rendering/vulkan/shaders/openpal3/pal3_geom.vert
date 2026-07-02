#version 450
#extension GL_ARB_separate_shader_objects : enable

// PAL3 static geometry shader (POL/CVD) — vertex stage. Faithfully reproduces
// the original PAL3 `gfxscript/geom_1L.gbf` / `geom_2L.gbf` (`vs_1_1`) model:
// lighting is computed **per vertex** (Gouraud) and modulated by the texture in
// the fragment stage, exactly like the skin shaders but without bone skinning:
//
//   perLight_i = atten_i * max(N·L_i, 0) * diffuse_i.rgb + ambient.rgb
//   color.rgb  = saturate( SUM_{i in 1..2} perLight_i )   // vs_1_1 clamps COLOR
//   finalPixel = texture * color                          // ColorOp = Modulate
//
// World/normal transform follows radiance's row-vector convention (`v * M`).
// Lightmapped POL surfaces use `LightMapMaterialDef` instead of this path.
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

    // Scenery uses the dim scene ambient directly (unlike actors, which carry a
    // high material ambient).
    vec3 ambient = perFrameUbo.ambient.rgb;

    // Pick the two nearest enabled lights (geom_1L/geom_2L).
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

    // Per-vertex Gouraud accumulation, exactly as geom_1L/geom_2L.gbf.
    vec3 lit = ambient;
    int idx[2] = int[2](best0, best1);
    for (int k = 0; k < 2; k++) {
        int i = idx[k];
        if (i < 0) continue;
        vec3 d = perFrameUbo.lightPos[i].xyz - worldPos;
        float dist = length(d);
        vec3 L = dist > 0.0 ? d / dist : vec3(0.0, 1.0, 0.0);

        float outer = perFrameUbo.lightPos[i].w;
        float inner = perFrameUbo.lightColor[i].w;
        float atten = 1.0;
        if (outer < 1.0e18) {
            float edge0 = max(inner, outer * 0.85);
            atten = 1.0 - smoothstep(edge0, outer, dist);
        }

        lit += perFrameUbo.lightColor[i].rgb * max(dot(N, L), 0.0) * atten;
    }

    // vs_1_1 clamps the interpolated COLOR output to [0,1].
    fragColor = clamp(lit, 0.0, 1.0);

    fragWorldPos = worldPos;
    fragTexCoord = inTexCoord * mat.uv_xform.xy + mat.uv_xform.zw;
}
