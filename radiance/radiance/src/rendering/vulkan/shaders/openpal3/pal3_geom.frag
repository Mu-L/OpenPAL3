#version 450
#extension GL_ARB_separate_shader_objects : enable

// PAL3 static geometry shader (POL/CVD) — fragment stage. Lighting is computed
// per-vertex in `pal3_geom.vert` (faithful to the original `vs_1_1`
// geom_1L/geom_2L.gbf Gouraud model); this stage only performs the texture
// combine the effect script declares: finalPixel = texture * vertexColor
// (ColorOp[0] = Modulate). Per-vertex (not per-pixel) shading is what keeps the
// original's flat look.

layout(constant_id = 0) const bool ALPHA_TEST = true;

layout(set = 0, binding = 0) uniform PerFrameUbo {
    mat4 view;
    mat4 proj;
    vec4 ambient;            // rgb = ambient color, w = light count
    vec4 lightPos[16];       // xyz = world position, w = outer range
    vec4 lightColor[16];     // rgb = color, w = inner range
    vec4 sunDir;
    vec4 sunColor;
    mat4 lightViewProj[3];
    vec4 cascadeSplits;
    vec4 shadowParams;
    vec4 fogColor;           // rgb = linear fog color
    vec4 fogParams;          // x = enabled, y = start depth, z = end depth
} perFrameUbo;

layout(set = 2, binding = 0) uniform sampler2D texSampler;
layout(set = 3, binding = 0) uniform MaterialParams {
    vec4 tint;
    vec4 misc;               // x = alpha_ref (when ALPHA_TEST), w = fog_exempt
    vec4 uv_xform;
} mat;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec3 fragWorldPos;
layout(location = 2) in vec3 fragColor;   // per-vertex Gouraud color (clamped)

layout(location = 0) out vec4 outColor;

void main() {
    vec4 sampled = texture(texSampler, fragTexCoord);
    if (ALPHA_TEST && sampled.a < mat.misc.x) {
        discard;
    }

    // finalPixel = texture * interpolated vertex color (ColorOp = Modulate).
    vec3 rgb = sampled.rgb * fragColor * mat.tint.rgb * mat.tint.a;
    outColor = vec4(rgb, sampled.a * mat.tint.a);

    // Linear distance fog, matching the shared shaders' eye-space fog.
    if (perFrameUbo.fogParams.x > 0.5 && mat.misc.w < 0.5) {
        float dist = -(vec4(fragWorldPos, 1.0) * perFrameUbo.view).z;
        float fStart = perFrameUbo.fogParams.y;
        float fEnd = perFrameUbo.fogParams.z;
        float vis = clamp((fEnd - dist) / max(fEnd - fStart, 1e-4), 0.0, 1.0);
        outColor.rgb = mix(perFrameUbo.fogColor.rgb * outColor.a, outColor.rgb, vis);
    }
}
