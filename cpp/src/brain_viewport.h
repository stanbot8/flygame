#pragma once
// Brain viewport: renders a spiking brain point cloud to an FBO texture.
//
// Used by the flygame viewer to show brain activity in an ImGui panel,
// matching the fwmc viewer's brain visualization. When NMFLY_BRAIN is
// enabled, neuron positions come from the ParametricGenerator; otherwise
// the BrainSDF is used to generate geometry.
//
// Requires an OpenGL 3.3+ context (separate from the MuJoCo compat context).

#ifdef NMFLY_VIEWER

#include <glad/gl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#ifdef NMFLY_BRAIN
#include "tissue/brain_sdf.h"
#endif

namespace nmfly {

// Point vertex: position + color + size
struct PointVertex {
    float x, y, z;
    float r, g, b;
    float size;
    int region;
};

// Simple 4x4 matrix math (avoids GLM dependency)
struct Mat4 {
    float m[16] = {};
    static Mat4 Identity() {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }
    static Mat4 Perspective(float fov_rad, float aspect, float z_near, float z_far) {
        Mat4 r;
        float f = 1.0f / std::tan(fov_rad * 0.5f);
        r.m[0] = f / aspect;
        r.m[5] = f;
        r.m[10] = (z_far + z_near) / (z_near - z_far);
        r.m[11] = -1.0f;
        r.m[14] = (2.0f * z_far * z_near) / (z_near - z_far);
        return r;
    }
    static Mat4 LookAt(float ex, float ey, float ez,
                       float cx, float cy, float cz,
                       float ux, float uy, float uz) {
        float fx = cx - ex, fy = cy - ey, fz = cz - ez;
        float fl = std::sqrt(fx*fx + fy*fy + fz*fz);
        fx /= fl; fy /= fl; fz /= fl;
        float rx = fy*uz - fz*uy, ry = fz*ux - fx*uz, rz = fx*uy - fy*ux;
        float rl = std::sqrt(rx*rx + ry*ry + rz*rz);
        rx /= rl; ry /= rl; rz /= rl;
        ux = ry*fz - rz*fy; uy = rz*fx - rx*fz; uz = rx*fy - ry*fx;
        Mat4 r;
        r.m[0]=rx; r.m[4]=ry; r.m[8]=rz;
        r.m[1]=ux; r.m[5]=uy; r.m[9]=uz;
        r.m[2]=-fx; r.m[6]=-fy; r.m[10]=-fz;
        r.m[12]=-(rx*ex+ry*ey+rz*ez);
        r.m[13]=-(ux*ex+uy*ey+uz*ez);
        r.m[14]=(fx*ex+fy*ey+fz*ez);
        r.m[15]=1.0f;
        return r;
    }
};

// Orbit camera
struct OrbitCamera {
    float azimuth = 0.3f;
    float elevation = 0.4f;
    float radius = 500.0f;
    float target_x = 250.0f;
    float target_y = 150.0f;
    float target_z = 100.0f;

    void Eye(float& ex, float& ey, float& ez) const {
        ex = target_x + radius * std::cos(elevation) * std::sin(azimuth);
        ey = target_y + radius * std::sin(elevation);
        ez = target_z + radius * std::cos(elevation) * std::cos(azimuth);
    }

    Mat4 ViewMatrix() const {
        float ex, ey, ez;
        Eye(ex, ey, ez);
        return Mat4::LookAt(ex, ey, ez, target_x, target_y, target_z, 0, 1, 0);
    }

    void Reset() {
        azimuth = 0.3f; elevation = 0.4f; radius = 500.0f;
        target_x = 250.0f; target_y = 150.0f; target_z = 100.0f;
    }
};

// Region colors (distinct hues for each brain region)
inline void RegionColor(int region, float& r, float& g, float& b) {
    static const float colors[][3] = {
        {0.9f, 0.7f, 0.5f},  // 0 central_brain: warm beige
        {0.2f, 0.6f, 0.9f},  // 1 optic_lobe_L: blue
        {0.2f, 0.6f, 0.9f},  // 2 optic_lobe_R: blue
        {0.9f, 0.3f, 0.5f},  // 3 mb_calyx_L: pink
        {0.9f, 0.3f, 0.5f},  // 4 mb_calyx_R: pink
        {0.8f, 0.2f, 0.4f},  // 5 mb_lobe_L: darker pink
        {0.8f, 0.2f, 0.4f},  // 6 mb_lobe_R: darker pink
        {0.3f, 0.9f, 0.4f},  // 7 antennal_lobe_L: green
        {0.3f, 0.9f, 0.4f},  // 8 antennal_lobe_R: green
        {1.0f, 0.8f, 0.2f},  // 9 central_complex: gold
        {0.6f, 0.3f, 0.9f},  // 10 lateral_horn_L: purple
        {0.6f, 0.3f, 0.9f},  // 11 lateral_horn_R: purple
        {0.4f, 0.8f, 0.8f},  // 12 sez: teal
    };
    int idx = (region >= 0 && region < 13) ? region : 0;
    r = colors[idx][0];
    g = colors[idx][1];
    b = colors[idx][2];
}

// Shader sources
static const char* kBrainVertSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in float aSize;
uniform mat4 uView;
uniform mat4 uProj;
out vec3 vColor;
void main() {
  gl_Position = uProj * uView * vec4(aPos, 1.0);
  gl_PointSize = aSize;
  vColor = aColor;
}
)";

static const char* kBrainFragSrc = R"(
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() {
  vec2 coord = gl_PointCoord * 2.0 - 1.0;
  if (dot(coord, coord) > 1.0) discard;
  FragColor = vec4(vColor, 1.0);
}
)";

// Shader compilation
inline GLuint CompileBrainShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        fprintf(stderr, "Shader error: %s\n", log);
    }
    return s;
}

inline GLuint LinkBrainProgram(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glBindAttribLocation(p, 0, "aPos");
    glBindAttribLocation(p, 1, "aColor");
    glBindAttribLocation(p, 2, "aSize");
    glLinkProgram(p);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(p, 512, nullptr, log);
        fprintf(stderr, "Link error: %s\n", log);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

// Brain viewport: manages FBO, shaders, point cloud, and rendering.
class BrainViewport {
public:
    OrbitCamera camera;
    std::vector<PointVertex> points;

    struct BaseColor { float r, g, b; };
    std::vector<BaseColor> base_colors;

    bool initialized = false;

    // Initialize GL resources. Must be called from a GL 3.3+ context.
    bool Init() {
        program_ = LinkBrainProgram(
            CompileBrainShader(GL_VERTEX_SHADER, kBrainVertSrc),
            CompileBrainShader(GL_FRAGMENT_SHADER, kBrainFragSrc));
        loc_view_ = glGetUniformLocation(program_, "uView");
        loc_proj_ = glGetUniformLocation(program_, "uProj");

        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        // Position
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PointVertex),
                              (void*)offsetof(PointVertex, x));
        // Color
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(PointVertex),
                              (void*)offsetof(PointVertex, r));
        // Size
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(PointVertex),
                              (void*)offsetof(PointVertex, size));
        glBindVertexArray(0);

        CreateFBO(512, 384);
        initialized = true;
        return true;
    }

    // Build point cloud from BrainSDF (proper brain shape).
    // Uses the same SDF sampling as the fwmc viewer.
    void BuildFromSDF(float spacing = 3.0f) {
#ifdef NMFLY_BRAIN
        mechabrain::BrainSDF sdf;
        sdf.InitDrosophila();

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> jitter(
            -spacing * 0.4f, spacing * 0.4f);

        points.clear();
        for (float z = 0; z < 200; z += spacing) {
            for (float y = 0; y < 300; y += spacing) {
                for (float x = 0; x < 500; x += spacing) {
                    float jx = x + jitter(rng);
                    float jy = y + jitter(rng);
                    float jz = z + jitter(rng);
                    float d = sdf.Evaluate(jx, jy, jz);
                    if (d < -spacing * 0.3f) {
                        int region = sdf.NearestRegion(jx, jy, jz);
                        PointVertex pv;
                        pv.x = jx; pv.y = jy; pv.z = jz;
                        pv.region = region;
                        pv.size = 4.0f;
                        RegionColor(region, pv.r, pv.g, pv.b);
                        points.push_back(pv);
                    }
                }
            }
        }

        base_colors.resize(points.size());
        for (size_t i = 0; i < points.size(); ++i)
            base_colors[i] = {points[i].r, points[i].g, points[i].b};

        // Count points per region for activity mapping.
        for (int r = 0; r < 13; ++r) {
            points_per_region_[r].clear();
        }
        for (size_t i = 0; i < points.size(); ++i) {
            int r = points[i].region;
            if (r >= 0 && r < 13)
                points_per_region_[r].push_back(i);
        }

        spike_vis_.assign(points.size(), 0.0f);
        UploadPoints();
        printf("[brain_vp] %zu visual points from SDF\n", points.size());
#endif
    }

    // Update point colors based on per-region spike rates.
    // region_rates: firing rate (Hz) per SDF region [0..12].
    void UpdateActivityByRegion(const float* region_rates) {
        // Sparse flash: each frame, randomly pick a fraction of points in
        // each region to flash white, matching fwmc's per-neuron sparse look.
        // fraction = rate / 1000 (at 30 Hz, ~3% of points flash per frame).
        for (int r = 0; r < 13; ++r) {
            float rate = region_rates[r];
            float frac = std::clamp(rate / 1000.0f, 0.0f, 0.15f);

            for (size_t idx : points_per_region_[r]) {
                // Chance to flash this point.
                if (frac > 0.0f) {
                    float roll = static_cast<float>(activity_rng_() & 0xFFFF) / 65535.0f;
                    if (roll < frac)
                        spike_vis_[idx] = 1.0f;  // full flash
                }

                // Decay (same 0.85 as fwmc).
                float v = spike_vis_[idx];
                points[idx].r = base_colors[idx].r * (1.0f - v) + v;
                points[idx].g = base_colors[idx].g * (1.0f - v) + v;
                points[idx].b = base_colors[idx].b * (1.0f - v) + v;
                float base_sz = region_base_size_[points[idx].region];
                points[idx].size = base_sz + v * 3.0f;
                spike_vis_[idx] *= 0.85f;
            }
        }
        UploadPoints();
    }

    // Render brain to FBO. Call from the brain GL context.
    void Render(int want_w, int want_h) {
        if (!initialized || points.empty()) return;

        if (want_w != fbo_w_ || want_h != fbo_h_)
            CreateFBO(want_w, want_h);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glViewport(0, 0, fbo_w_, fbo_h_);
        glClearColor(0.06f, 0.06f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        float aspect = fbo_w_ > 0 ? static_cast<float>(fbo_w_) / fbo_h_ : 1.0f;
        Mat4 proj = Mat4::Perspective(0.785f, aspect, 1.0f, 5000.0f);
        Mat4 view = camera.ViewMatrix();

        glUseProgram(program_);
        glUniformMatrix4fv(loc_proj_, 1, GL_FALSE, proj.m);
        glUniformMatrix4fv(loc_view_, 1, GL_FALSE, view.m);

        glBindVertexArray(vao_);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(points.size()));
        glBindVertexArray(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glFlush();
    }

    GLuint Texture() const { return tex_; }
    size_t PointCount() const { return points.size(); }

    // Update base point sizes per region (from Regions overlay sliders).
    void SetRegionSizes(const float* region_size) {
        for (int i = 0; i < 14; ++i)
            region_base_size_[i] = region_size[i];
        for (auto& pv : points)
            pv.size = region_size[pv.region];
        UploadPoints();
    }

    void Shutdown() {
        if (fbo_) { glDeleteFramebuffers(1, &fbo_); fbo_ = 0; }
        if (tex_) { glDeleteTextures(1, &tex_); tex_ = 0; }
        if (depth_) { glDeleteRenderbuffers(1, &depth_); depth_ = 0; }
        if (vao_) { glDeleteVertexArrays(1, &vao_); vao_ = 0; }
        if (vbo_) { glDeleteBuffers(1, &vbo_); vbo_ = 0; }
        if (program_) { glDeleteProgram(program_); program_ = 0; }
        initialized = false;
    }

private:
    void CreateFBO(int w, int h) {
        if (fbo_) { glDeleteFramebuffers(1, &fbo_); glDeleteTextures(1, &tex_); glDeleteRenderbuffers(1, &depth_); }
        fbo_w_ = w; fbo_h_ = h;
        glGenFramebuffers(1, &fbo_);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glGenTextures(1, &tex_);
        glBindTexture(GL_TEXTURE_2D, tex_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_, 0);
        glGenRenderbuffers(1, &depth_);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void UploadPoints() {
        if (!vbo_ || points.empty()) return;
        spike_vis_.resize(points.size(), 0.0f);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(points.size() * sizeof(PointVertex)),
                     points.data(), GL_DYNAMIC_DRAW);
    }

    GLuint fbo_ = 0, tex_ = 0, depth_ = 0;
    int fbo_w_ = 0, fbo_h_ = 0;
    GLuint vao_ = 0, vbo_ = 0;
    GLuint program_ = 0;
    GLint loc_view_ = -1, loc_proj_ = -1;
    std::vector<float> spike_vis_;
    std::vector<size_t> points_per_region_[13];
    float region_base_size_[14] = {1,2,2,4,4,4,4,4,4,4,4,4,4,4};
    std::minstd_rand activity_rng_{12345};
};

}  // namespace nmfly

#endif  // NMFLY_VIEWER
