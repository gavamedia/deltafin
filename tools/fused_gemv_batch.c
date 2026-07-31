// Batched MXFP4 dequant+GEMV entry point for Kimi-K3 MoE layers.
//
// Problem this solves: tools/fast_moe.py calls libmxfp4gemv.so once per GEMV, i.e.
// 48 ctypes calls per MoE layer (16 experts x {w1,w3,w2}). Each of those calls does
// pthread_create x N + pthread_join x N -> ~192 thread create/destroy per layer, per
// token. This file keeps ONE persistent worker pool alive for the process lifetime and
// exposes one dispatch per *phase* (2 per layer) instead of 48.
//
// Bit-exactness: this TU #includes fused_gemv.c verbatim (with NO_MAIN). ARM calls the
// established mxfp4_gemv_rows2() loop. AVX2 builds call its exact 256-bit equivalent
// after preparing each distinct activation once per whole dispatch. Work is partitioned
// on 32-row boundaries, so a row's value does not depend on its worker or chunk.
//
// Build:
//   clang -O3 -mcpu=native -shared -DNO_MAIN -o libmxfp4batch.dylib fused_gemv_batch.c -lpthread
//   cc -O3 -march=native -fPIC -shared -DNO_MAIN -o libmxfp4batch.so fused_gemv_batch.c -lpthread -lm
//
// Exports:
//   int  mxfp4_pool_init(int nthreads)      // idempotent; rebuilds if size changed
//   void mxfp4_pool_shutdown(void)
//   int  mxfp4_pool_threads(void)
//   void mxfp4_gemv_batch(...)              // N independent GEMVs, one dispatch
//   void mxfp4_moe_layer(...)               // N GEMVs sharing one x, concatenated out
//   void mxfp4_situ_batch(...)              // parallel SiTU (opt-in, not bit-exact vs numpy)
//   void mxfp4_moe_expert_set(...)          // whole expert set in ONE call (opt-in)

#ifndef NO_MAIN
#define NO_MAIN 1
#endif
#include "fused_gemv.c"

#include <errno.h>

#define K3_MAX_THREADS 32
#define K3_MAX_MATS    256
#define K3_CHUNK       32     // rows per work unit; even => stays on the rows2 fast path
#define K3_SPIN_WORKER 40000  // worker spin before blocking on the condvar
#define K3_SPIN_MASTER 60000  // caller spin before blocking on the done condvar

// ---------------------------------------------------------------- job description
typedef enum { K3_JOB_GEMV = 0, K3_JOB_SITU = 1 } k3_kind_t;

typedef struct {
    k3_kind_t kind;
    // --- K3_JOB_GEMV ---
    const uint8_t *const *P;
    const uint8_t *const *S;
    const float   *const *X;
    const float *XP[K3_MAX_MATS]; // AVX2-prepared activations, owned by batch scratch
    float        *const *Y;
    const int *rows, *cols;
    int n_mats;
    int use_avx2;
    int cum[K3_MAX_MATS + 1];   // cum[m] = first unit index of matrix m
    // --- K3_JOB_SITU ---
    const float *gu;            // [n_ex][2][d_ff] interleaved gate,up
    float *h;                   // [n_ex][d_ff]
    int n_ex, d_ff;
    // --- both ---
    int n_units;
} k3_job_t;

static k3_job_t g_job;

// These fields are touched by different actors at different frequencies:
// every worker contends on g_cursor, the last workers update g_finished while
// the caller polls it, and idle workers poll g_gen.  Keeping adjacent atomics
// on one cache line makes those independent protocols invalidate each other.
// This is layout-only: it changes neither the synchronization order nor the
// kernel arithmetic, and remains useful across Apple CPU generations.
//
// The attribute leads the declaration because that is the one position both
// GCC's aligned attribute and MSVC's __declspec(align) accept.
#ifndef K3_ATOMIC_ALIGN
#define K3_ATOMIC_ALIGN K3_ALIGN_PREFIX(128)
#endif
K3_ATOMIC_ALIGN static _Atomic long g_gen      = 0;   // dispatch generation
K3_ATOMIC_ALIGN static _Atomic int  g_cursor   = 0;   // next work unit
K3_ATOMIC_ALIGN static _Atomic int  g_finished = 0;   // completed workers
K3_ATOMIC_ALIGN static _Atomic int  g_stop     = 0;
K3_ATOMIC_ALIGN static _Atomic int  g_ready    = 0;   // startup latch

static pthread_t       g_th[K3_MAX_THREADS];
static int             g_nworkers = 0;
static pthread_mutex_t g_mu      = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cv_work = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  g_cv_done = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t g_api     = PTHREAD_MUTEX_INITIALIZER;  // serializes dispatches

#if MXFP4_CAN_BUILD_AVX2
static float  *g_xscratch = NULL;
static size_t  g_xscratch_n = 0;
#endif
static int g_last_x_permutations = 0;

static inline void k3_pause(void) { mxfp4_cpu_relax(); }

#if MXFP4_CAN_BUILD_AVX2
// Prepare every distinct (pointer, length) only once.  Phase A normally maps 32
// matrices to one x; phase B maps 16 matrices to 16 h vectors.  offsets are recorded
// before a possible arena growth so no pointer into a replaced allocation can escape.
static int k3_prepare_batch_x_avx2(
        const float *const *xs, const int *cols, int n_mats) {
    size_t offsets[K3_MAX_MATS];
    size_t need = 0;
    int unique = 0;
    for (int m = 0; m < n_mats; m++) {
        if (!xs[m] || cols[m] <= 0 || cols[m] % 32 != 0) return 0;
        int prior = -1;
        for (int k = 0; k < m; k++) {
            if (xs[k] == xs[m] && cols[k] == cols[m]) {
                prior = k;
                break;
            }
        }
        if (prior >= 0) {
            offsets[m] = offsets[prior];
            continue;
        }
        size_t width = (size_t)cols[m];
        if (need > SIZE_MAX - width) return 0;
        offsets[m] = need;
        need += width;
        unique++;
    }
    if (need > SIZE_MAX / sizeof(float)) return 0;
    if (g_xscratch_n < need) {
        size_t bytes = need * sizeof(float);
        if (bytes > SIZE_MAX - 63) return 0;
        size_t aligned_bytes = (bytes + 63) & ~(size_t)63;
        float *grown = (float *)k3_aligned_alloc(64, aligned_bytes);
        if (!grown) return 0;
        k3_aligned_free(g_xscratch);
        g_xscratch = grown;
        g_xscratch_n = aligned_bytes / sizeof(float);
    }
    for (int m = 0; m < n_mats; m++) {
        g_job.XP[m] = g_xscratch + offsets[m];
        int prior = -1;
        for (int k = 0; k < m; k++) {
            if (offsets[k] == offsets[m]) {
                prior = k;
                break;
            }
        }
        if (prior < 0
                && !mxfp4_permute_x_avx2(xs[m], g_xscratch + offsets[m], cols[m]))
            return 0;
    }
    g_last_x_permutations = unique;
    return 1;
}
#endif

K3_EXPORT int mxfp4_batch_last_x_permutations(void) {
    return g_last_x_permutations;
}

// unit index -> (matrix, row range). n_mats is tiny (<=48 in practice) so a binary
// search costs nothing next to 32 rows x 3584 cols of streaming work.
static inline void k3_unit_gemv(const k3_job_t *j, int u, int *m, int *r0, int *r1) {
    int lo = 0, hi = j->n_mats - 1;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (j->cum[mid + 1] <= u) lo = mid + 1; else hi = mid; }
    int a = (u - j->cum[lo]) * K3_CHUNK;
    int b = a + K3_CHUNK;
    if (b > j->rows[lo]) b = j->rows[lo];
    *m = lo; *r0 = a; *r1 = b;
}

// SiTU, matching tools/fast_moe.py::_situ in fp32:
//   a = 4*tanh(g/4) / (1 + exp(-g));  h = a * (25*tanh(u/25))
// NOTE: libm tanhf/expf are not bit-identical to numpy's vectorized transcendentals.
// This path is opt-in; the bit-exact Python path keeps SiTU in numpy.
#define K3_SITU_BETA        4.0f
#define K3_SITU_LINEAR_BETA 25.0f
static void k3_situ_range(const float *restrict g, const float *restrict u,
                          float *restrict h, int n) {
    for (int i = 0; i < n; i++) {
        float a = K3_SITU_BETA * tanhf(g[i] / K3_SITU_BETA) / (1.0f + expf(-g[i]));
        h[i] = a * (K3_SITU_LINEAR_BETA * tanhf(u[i] / K3_SITU_LINEAR_BETA));
    }
}

static void k3_run_units(void) {
    const k3_job_t *j = &g_job;
    const int n = j->n_units;
    int u;
    if (j->kind == K3_JOB_GEMV) {
        while ((u = atomic_fetch_add_explicit(&g_cursor, 1, memory_order_relaxed)) < n) {
            int m, r0, r1;
            k3_unit_gemv(j, u, &m, &r0, &r1);
#if MXFP4_CAN_BUILD_AVX2
            if (j->use_avx2) {
                mxfp4_gemv_rows2_avx2_prepared(
                    j->P[m], j->S[m], j->XP[m], j->Y[m],
                    r0, r1, j->cols[m]);
                continue;
            }
#endif
            mxfp4_gemv_rows2(j->P[m], j->S[m], j->X[m], j->Y[m], r0, r1, j->cols[m]);
        }
    } else {
        while ((u = atomic_fetch_add_explicit(&g_cursor, 1, memory_order_relaxed)) < n) {
            const float *g = j->gu + (size_t)(2 * u) * j->d_ff;
            k3_situ_range(g, g + j->d_ff, j->h + (size_t)u * j->d_ff, j->d_ff);
        }
    }
}

// ---------------------------------------------------------------- worker
static void *k3_worker(void *arg) {
    (void)arg;
    mxfp4_qos_self();
    // Latch the start generation and announce readiness. mxfp4_pool_init() blocks until
    // every worker has done this, so no dispatch can be missed by a slow-starting thread
    // (which would hang the caller waiting for a completion that never arrives).
    long mygen = atomic_load_explicit(&g_gen, memory_order_acquire);
    atomic_fetch_add_explicit(&g_ready, 1, memory_order_release);
    for (;;) {
        // spin briefly (dispatches are back-to-back inside a layer), then block so we
        // do not burn P-cores between tokens.
        int spins = 0;
        while (atomic_load_explicit(&g_gen, memory_order_acquire) == mygen
               && !atomic_load_explicit(&g_stop, memory_order_acquire)) {
            if (++spins > K3_SPIN_WORKER) {
                pthread_mutex_lock(&g_mu);
                while (atomic_load_explicit(&g_gen, memory_order_acquire) == mygen
                       && !atomic_load_explicit(&g_stop, memory_order_acquire))
                    pthread_cond_wait(&g_cv_work, &g_mu);
                pthread_mutex_unlock(&g_mu);
                break;
            }
            k3_pause();
        }
        if (atomic_load_explicit(&g_stop, memory_order_acquire)) break;
        mygen = atomic_load_explicit(&g_gen, memory_order_acquire);

        k3_run_units();

        if (atomic_fetch_add_explicit(&g_finished, 1, memory_order_acq_rel) + 1 == g_nworkers) {
            pthread_mutex_lock(&g_mu);
            pthread_cond_broadcast(&g_cv_done);
            pthread_mutex_unlock(&g_mu);
        }
    }
    return NULL;
}

// ---------------------------------------------------------------- pool lifecycle
static void k3_pool_stop_locked(void) {
    if (!g_nworkers) return;
    pthread_mutex_lock(&g_mu);
    atomic_store_explicit(&g_stop, 1, memory_order_release);
    pthread_cond_broadcast(&g_cv_work);
    pthread_mutex_unlock(&g_mu);
    for (int i = 0; i < g_nworkers; i++) pthread_join(g_th[i], NULL);
    g_nworkers = 0;
    atomic_store_explicit(&g_stop, 0, memory_order_release);
}

K3_EXPORT int mxfp4_pool_init(int nthreads) {
    if (nthreads < 1) nthreads = 1;
    if (nthreads > K3_MAX_THREADS) nthreads = K3_MAX_THREADS;
    pthread_mutex_lock(&g_api);
    if (g_nworkers != nthreads) {
        k3_pool_stop_locked();
        // start with an empty job so a worker that wakes early finds nothing to do
        g_job.kind = K3_JOB_GEMV;
        g_job.n_units = 0;
        atomic_store_explicit(&g_gen, 0, memory_order_relaxed);
        atomic_store_explicit(&g_cursor, 0, memory_order_relaxed);
        atomic_store_explicit(&g_finished, 0, memory_order_relaxed);
        atomic_store_explicit(&g_ready, 0, memory_order_relaxed);
        pthread_attr_t at;
        pthread_attr_init(&at);
        mxfp4_qos_attr(&at);
        int made = 0;
        for (int i = 0; i < nthreads; i++) {
            if (mxfp4_pthread_create(
                    &g_th[i], &at, k3_worker, NULL) != 0) break;
            made++;
        }
        pthread_attr_destroy(&at);
        // Every worker must be parked on generation 0 before the first dispatch.
        while (atomic_load_explicit(&g_ready, memory_order_acquire) < made) sched_yield();
        g_nworkers = made;
    }
    int r = g_nworkers;
    pthread_mutex_unlock(&g_api);
    return r;
}

K3_EXPORT void mxfp4_pool_shutdown(void) {
    pthread_mutex_lock(&g_api);
    k3_pool_stop_locked();
#if MXFP4_CAN_BUILD_AVX2
    k3_aligned_free(g_xscratch);
    g_xscratch = NULL;
    g_xscratch_n = 0;
#endif
    g_last_x_permutations = 0;
    pthread_mutex_unlock(&g_api);
}

K3_EXPORT int mxfp4_pool_threads(void) { return g_nworkers; }

// ---------------------------------------------------------------- dispatch
// Caller must hold g_api. g_job must be fully populated (incl. n_units).
static void k3_dispatch(void) {
    if (g_job.n_units <= 0) return;
    if (g_nworkers <= 0) {                            // degenerate: run inline
        atomic_store_explicit(&g_cursor, 0, memory_order_relaxed);
        k3_run_units();
        return;
    }

    atomic_store_explicit(&g_cursor, 0, memory_order_relaxed);
    atomic_store_explicit(&g_finished, 0, memory_order_relaxed);

    pthread_mutex_lock(&g_mu);
    atomic_fetch_add_explicit(&g_gen, 1, memory_order_release);
    pthread_cond_broadcast(&g_cv_work);
    pthread_mutex_unlock(&g_mu);

    int spins = 0;
    for (;;) {
        if (atomic_load_explicit(&g_finished, memory_order_acquire) >= g_nworkers) return;
        if (++spins > K3_SPIN_MASTER) break;
        k3_pause();
    }
    // Bounded slices so a future protocol bug shows up as a loud stall rather than a
    // silent hang. The completion count is exact (mxfp4_pool_init parks every worker on
    // the start generation, and dispatches are serialized by g_api), so this only loops.
    pthread_mutex_lock(&g_mu);
    int stalls = 0;
    while (atomic_load_explicit(&g_finished, memory_order_acquire) < g_nworkers) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 5;
        if (pthread_cond_timedwait(&g_cv_done, &g_mu, &ts) == ETIMEDOUT && ++stalls == 1)
            fprintf(stderr, "[mxfp4batch] dispatch stalled: %d/%d workers done, "
                            "%d/%d units claimed\n",
                    atomic_load(&g_finished), g_nworkers,
                    atomic_load(&g_cursor), g_job.n_units);
    }
    pthread_mutex_unlock(&g_mu);
}

// ---------------------------------------------------------------- public: batched GEMV
// Matrix m computes ys[m][r] = dot(W_m[r,:], xs[m]) for r in [0, rows[m]).
// One dispatch for the whole batch; rows are work-stolen across all matrices, so a
// short matrix cannot leave a thread idle at the tail.
K3_EXPORT void mxfp4_gemv_batch(
        const uint8_t *const *packed, const uint8_t *const *scales,
        const float *const *xs, float *const *ys,
        const int *rows, const int *cols, int n_mats, int nthreads) {
    if (n_mats <= 0) return;
    if (n_mats > K3_MAX_MATS) n_mats = K3_MAX_MATS;
    mxfp4_pool_init(nthreads);
    pthread_mutex_lock(&g_api);
    g_job.kind = K3_JOB_GEMV;
    g_job.P = packed; g_job.S = scales; g_job.X = xs; g_job.Y = ys;
    g_job.rows = rows; g_job.cols = cols; g_job.n_mats = n_mats;
    g_job.use_avx2 = 0;
    g_last_x_permutations = 0;
#if MXFP4_CAN_BUILD_AVX2
    if (mxfp4_have_avx2())
        g_job.use_avx2 = k3_prepare_batch_x_avx2(xs, cols, n_mats);
#endif
    int c = 0;
    for (int m = 0; m < n_mats; m++) {
        g_job.cum[m] = c;
        c += (rows[m] + K3_CHUNK - 1) / K3_CHUNK;
    }
    g_job.cum[n_mats] = c;
    g_job.n_units = c;
    k3_dispatch();
    pthread_mutex_unlock(&g_api);
}

// Requested shape: n_mats matrices that all consume the SAME x (this is exactly the
// w1/w3 phase of a MoE layer: 16 experts x 2 matrices, one shared activation), with
// outputs concatenated into `out` at offset sum(rows[0..m)).
K3_EXPORT void mxfp4_moe_layer(
        const uint8_t *const *packed, const uint8_t *const *scales,
        const int *rows, const int *cols, int n_mats,
        const float *x, float *out, int nthreads) {
    if (n_mats <= 0) return;
    if (n_mats > K3_MAX_MATS) n_mats = K3_MAX_MATS;
    const float *xs[K3_MAX_MATS];
    float *ys[K3_MAX_MATS];
    size_t off = 0;
    for (int m = 0; m < n_mats; m++) {
        xs[m] = x;
        ys[m] = out + off;
        off += (size_t)rows[m];
    }
    mxfp4_gemv_batch(packed, scales, xs, ys, rows, cols, n_mats, nthreads);
}

// Parallel SiTU over n_ex experts. gu is [n_ex][2][d_ff] (gate,up interleaved per
// expert, i.e. the natural mxfp4_moe_layer output when mats are ordered w1,w3,w1,w3...).
K3_EXPORT void mxfp4_situ_batch(
        const float *gu, float *h, int n_ex, int d_ff, int nthreads) {
    if (n_ex <= 0 || d_ff <= 0) return;
    mxfp4_pool_init(nthreads);
    pthread_mutex_lock(&g_api);
    g_job.kind = K3_JOB_SITU;
    g_job.gu = gu; g_job.h = h; g_job.n_ex = n_ex; g_job.d_ff = d_ff;
    g_job.n_units = n_ex;
    k3_dispatch();
    pthread_mutex_unlock(&g_api);
}

// ---------------------------------------------------------------- public: whole layer
// ONE call for a token's entire routed expert set:
//   phase A  : 2*n_ex GEMVs (w1,w3) sharing x            -> gu   [n_ex][2][d_ff]
//   activation: SiTU                                      -> h    [n_ex][d_ff]
//   phase B  : n_ex GEMVs (w2), each with its own h       -> yb   [n_ex][d_model]
//   combine  : out = sum_e wts[e] * yb[e]                (sequential, ids order)
// p13/s13 hold 2*n_ex pointers ordered e0.w1, e0.w3, e1.w1, e1.w3, ...
// scratch must be >= n_ex*(3*d_ff + d_model) floats, or NULL to use an internal buffer.
// NOT bit-exact vs the numpy SiTU path (libm vs numpy transcendentals); the GEMVs are.
static float *g_scr = NULL;
static size_t g_scr_n = 0;
static pthread_mutex_t g_scr_mu = PTHREAD_MUTEX_INITIALIZER;

K3_EXPORT void mxfp4_moe_expert_set(
        const uint8_t *const *p13, const uint8_t *const *s13,
        const uint8_t *const *p2,  const uint8_t *const *s2,
        const float *x, const float *wts,
        int n_ex, int d_ff, int d_model,
        float *scratch, float *out, int nthreads) {
    if (n_ex <= 0 || n_ex > K3_MAX_MATS / 2
            || d_ff <= 0 || d_model <= 0) return;
    size_t dff = (size_t)d_ff, dmodel = (size_t)d_model;
    if (dff > (SIZE_MAX - dmodel) / 3) return;
    size_t per_expert = 3 * dff + dmodel;
    if ((size_t)n_ex > SIZE_MAX / per_expert / sizeof(float)) return;
    size_t need = (size_t)n_ex * per_expert;
    pthread_mutex_lock(&g_scr_mu);
    float *scr = scratch;
    if (!scr) {
        if (g_scr_n < need) {
            free(g_scr);
            g_scr = (float *)malloc(need * sizeof(float));
            g_scr_n = g_scr ? need : 0;
        }
        scr = g_scr;
        if (!scr) {
            pthread_mutex_unlock(&g_scr_mu);
            return;
        }
    }
    float *gu = scr;                                   // [n_ex][2][d_ff]
    float *h  = gu + (size_t)n_ex * 2 * d_ff;          // [n_ex][d_ff]
    float *yb = h  + (size_t)n_ex * d_ff;              // [n_ex][d_model]

    int rows13[K3_MAX_MATS], cols13[K3_MAX_MATS];
    int rows2[K3_MAX_MATS],  cols2[K3_MAX_MATS];
    const float *xs[K3_MAX_MATS]; float *ys[K3_MAX_MATS];

    int n13 = 2 * n_ex;
    for (int m = 0; m < n13; m++) { rows13[m] = d_ff; cols13[m] = d_model; xs[m] = x; ys[m] = gu + (size_t)m * d_ff; }
    mxfp4_gemv_batch(p13, s13, xs, ys, rows13, cols13, n13, nthreads);

    mxfp4_situ_batch(gu, h, n_ex, d_ff, nthreads);

    for (int e = 0; e < n_ex; e++) {
        rows2[e] = d_model; cols2[e] = d_ff;
        xs[e] = h + (size_t)e * d_ff;
        ys[e] = yb + (size_t)e * d_model;
    }
    mxfp4_gemv_batch(p2, s2, xs, ys, rows2, cols2, n_ex, nthreads);

    for (int i = 0; i < d_model; i++) out[i] = 0.0f;
    for (int e = 0; e < n_ex; e++) {
        const float w = wts[e], *ye = yb + (size_t)e * d_model;
        for (int i = 0; i < d_model; i++) out[i] += w * ye[i];
    }
    pthread_mutex_unlock(&g_scr_mu);
}
