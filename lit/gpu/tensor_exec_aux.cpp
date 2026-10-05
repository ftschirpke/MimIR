// Driver for tensor_exec.mim: checks the GPU-lowered externs against scalar references.
#include <cmath>
#include <cstdio>
#include <vector>

extern "C" {
float* mm(float*, float*);
float* mm_odd(float*, float*);
float* fc(float*, float*);
float* fc_br(float*, float*, float*);
float* bmm(float*, float*);
float* chain(float*, float*, float*);
float* cv(float*, float*);
float* cv_ps(float*, float*);
float* cv_big(float*, float*);
float* ew(float*);
}

static std::vector<float> data(size_t n, unsigned seed) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = float((i * 2654435761u + seed * 97u) % 1009u) / 1009.0f - 0.5f;
    return v;
}

static int compare(const char* name, const float* got, const std::vector<float>& ref) {
    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i)
        if (!(std::fabs(ref[i] - got[i]) <= 1e-4f + 1e-4f * std::fabs(ref[i])))
            if (++bad <= 5) std::printf("%s[%zu]: got %f want %f\n", name, i, got[i], ref[i]);
    std::printf("%-7s %s (%zu values)\n", name, bad ? "MISMATCH" : "ok", ref.size());
    return bad;
}

// out[b][m][n] = Σ_k A(b, m, k) · B(b, k, n), with A/B given as accessors.
template<class FA, class FB>
static std::vector<float> matmul(size_t Bn, size_t M, size_t K, size_t N, FA A, FB B) {
    std::vector<float> out(Bn * M * N);
    for (size_t b = 0; b < Bn; ++b)
        for (size_t m = 0; m < M; ++m)
            for (size_t n = 0; n < N; ++n) {
                double s = 0;
                for (size_t k = 0; k < K; ++k) s += double(A(b, m, k)) * double(B(b, k, n));
                out[(b * M + m) * N + n] = float(s);
            }
    return out;
}

static std::vector<float> conv(const std::vector<float>& x, const std::vector<float>& w, size_t n, size_t ci,
                               size_t h, size_t wd, size_t co, size_t k, size_t st, size_t pad) {
    size_t oh = (h + 2 * pad - k) / st + 1, ow = (wd + 2 * pad - k) / st + 1;
    std::vector<float> out(n * co * oh * ow);
    for (size_t b = 0; b < n; ++b)
        for (size_t o = 0; o < co; ++o)
            for (size_t y = 0; y < oh; ++y)
                for (size_t z = 0; z < ow; ++z) {
                    double s = 0;
                    for (size_t c = 0; c < ci; ++c)
                        for (size_t i = 0; i < k; ++i)
                            for (size_t j = 0; j < k; ++j) {
                                long iy = long(y * st + i) - long(pad), iz = long(z * st + j) - long(pad);
                                if (iy < 0 || iz < 0 || iy >= long(h) || iz >= long(wd)) continue;
                                s += double(x[((b * ci + c) * h + iy) * wd + iz]) * double(w[((o * ci + c) * k + i) * k + j]);
                            }
                    out[((b * co + o) * oh + y) * ow + z] = float(s);
                }
    return out;
}

int main() {
    int bad = 0;
    {
        auto a = data(32 * 48, 1), b = data(48 * 64, 2);
        bad += compare("mm", mm(a.data(), b.data()),
                       matmul(1, 32, 48, 64, [&](size_t, size_t m, size_t k) { return a[m * 48 + k]; },
                              [&](size_t, size_t k, size_t n) { return b[k * 64 + n]; }));
    }
    {
        auto a = data(20 * 33, 3), b = data(33 * 17, 4);
        bad += compare("mm_odd", mm_odd(a.data(), b.data()),
                       matmul(1, 20, 33, 17, [&](size_t, size_t m, size_t k) { return a[m * 33 + k]; },
                              [&](size_t, size_t k, size_t n) { return b[k * 17 + n]; }));
    }
    {
        auto x = data(32 * 64, 5), w = data(48 * 64, 6);
        auto ref = matmul(1, 32, 64, 48, [&](size_t, size_t m, size_t k) { return x[m * 64 + k]; },
                          [&](size_t, size_t k, size_t n) { return w[n * 64 + k]; });
        bad += compare("fc", fc(x.data(), w.data()), ref);
        auto bias = data(48, 7);
        for (size_t i = 0; i < ref.size(); ++i) ref[i] = std::fmax(0.0f, ref[i] + bias[i % 48]);
        bad += compare("fc_br", fc_br(x.data(), w.data(), bias.data()), ref);
    }
    {
        auto a = data(4 * 32 * 48, 8), b = data(4 * 48 * 64, 9);
        bad += compare("bmm", bmm(a.data(), b.data()),
                       matmul(4, 32, 48, 64, [&](size_t bb, size_t m, size_t k) { return a[(bb * 32 + m) * 48 + k]; },
                              [&](size_t bb, size_t k, size_t n) { return b[(bb * 48 + k) * 64 + n]; }));
    }
    {
        auto a = data(32 * 48, 10), b = data(48 * 64, 11), c = data(64 * 32, 12);
        auto ab = matmul(1, 32, 48, 64, [&](size_t, size_t m, size_t k) { return a[m * 48 + k]; },
                         [&](size_t, size_t k, size_t n) { return b[k * 64 + n]; });
        for (auto& v : ab) v = std::fmax(0.0f, v);
        bad += compare("chain", chain(a.data(), b.data(), c.data()),
                       matmul(1, 32, 64, 32, [&](size_t, size_t m, size_t k) { return ab[m * 64 + k]; },
                              [&](size_t, size_t k, size_t n) { return c[k * 32 + n]; }));
    }
    {
        auto x = data(2 * 8 * 18 * 18, 13), w = data(16 * 8 * 9, 14);
        bad += compare("cv", cv(x.data(), w.data()), conv(x, w, 2, 8, 18, 18, 16, 3, 1, 0));
    }
    {
        auto x = data(2 * 3 * 17 * 17, 15), w = data(8 * 3 * 9, 16);
        bad += compare("cv_ps", cv_ps(x.data(), w.data()), conv(x, w, 2, 3, 17, 17, 8, 3, 2, 1));
    }
    {
        auto x = data(2 * 64 * 100, 17), w = data(4 * 64 * 9, 18);
        bad += compare("cv_big", cv_big(x.data(), w.data()), conv(x, w, 2, 64, 10, 10, 4, 3, 1, 0));
    }
    {
        auto x = data(32 * 64, 19);
        auto ref = x;
        for (auto& v : ref) v = std::fmax(0.0f, v);
        bad += compare("ew", ew(x.data()), ref);
    }
    return bad ? 1 : 0;
}
