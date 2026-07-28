// Batched MXFP4 dequant+GEMV entry point for Kimi-K3 MoE layers (Apple Silicon).
//
// Problem this solves: tools/fast_moe.py calls libmxfp4gemv.so once per GEMV, i.e.
// 48 ctypes calls per MoE layer (16 experts x {w1,w3,w2}). Each of those calls does
// pthread_create x N + pthread_join x N -> ~192 thread create/destroy per layer, per
// token. This file keeps ONE persistent worker pool alive for the process lifetime and
// exposes one dispatch per *phase* (2 per layer) instead of 48.
//
// Bit-exactness: this TU #includes fused_gemv.c verbatim (with NO_MAIN) and calls the
// very same mxfp4_gemv_rows2() row loop. The TBL nibble expansion, the e8m0
// exponent-add trick and the per-row fp32 accumulation order are therefore identical
// by construction, not by reimplementation. Work is partitioned on 32-row boundaries,
// so every row still takes the 2-row path exactly as run_gemv_mt() did; a row's value
// does not depend on which thread or which chunk computed it.
//
// Build (macOS):
//   clang -O3 -mcpu=native -shared -DNO_MAIN -o libmxfp4batch.dylib fused_gemv_batch.c -lpthread
// Build (Linux x86-64):
//   gcc -O3 -march=x86-64-v3 -shared -fPIC -DNO_MAIN -o libmxfp4batch.so fused_gemv_batch.c -lpthread
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
    float        *const *Y;
    const int *rows, *cols;
    int n_mats;
    int cum[K3_MAX_MATS + 1];   // cum[m] = first unit index of matrix m
    // --- K3_JOB_SITU ---
    const float *gu;            // [n_ex][2][d_ff] interleaved gate,up
    float *h;                   // [n_ex][d_ff]
    int n_ex, d_ff;
    // --- both ---
    int n_units;
} k3_job_t;

static k3_job_t g_job;

static _Atomic long g_gen      = 0;   // dispatch generation counter
static _Atomic int  g_cursor   = 0;   // work-stealing cursor into [0, n_units)
static _Atomic int  g_finished = 0;   // workers done with the current dispatch
static _Atomic int  g_stop     = 0;
static _Atomic int  g_ready    = 0;   // workers that have latched the start generation

static pthread_t       g_th[K3_MAX_THREADS];
static int             g_nworkers = 0;
static pthread_mutex_t g_mu      = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cv_work = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  g_cv_done = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t g_api     = PTHREAD_MUTEX_INITIALIZER;  // serializes dispatches

static inline void k3_pause(void) { k3_cpu_relax(); }   // yield / pause (fused_gemv.c)

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
    k3_qos_self();
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

int mxfp4_pool_init(int nthreads) {
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
#ifdef __APPLE__
        pthread_attr_set_qos_class_np(&at, QOS_CLASS_USER_INTERACTIVE, 0);
#endif
        int made = 0;
        for (int i = 0; i < nthreads; i++) {
            if (pthread_create(&g_th[i], &at, k3_worker, NULL) != 0) break;
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

void mxfp4_pool_shutdown(void) {
    pthread_mutex_lock(&g_api);
    k3_pool_stop_locked();
    pthread_mutex_unlock(&g_api);
}

int mxfp4_pool_threads(void) { return g_nworkers; }

// ---------------------------------------------------------------- dispatch
// Caller must hold g_api. g_job must be fully populated (incl. n_units).
static void k3_dispatch(void) {
    if (g_job.n_units <= 0) return;
    if (g_nworkers <= 0) { k3_run_units(); return; }   // degenerate: run inline

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
void mxfp4_gemv_batch(const uint8_t *const *packed, const uint8_t *const *scales,
                      const float *const *xs, float *const *ys,
                      const int *rows, const int *cols, int n_mats, int nthreads) {
    if (n_mats <= 0) return;
    if (n_mats > K3_MAX_MATS) n_mats = K3_MAX_MATS;
    mxfp4_pool_init(nthreads);
    pthread_mutex_lock(&g_api);
    g_job.kind = K3_JOB_GEMV;
    g_job.P = packed; g_job.S = scales; g_job.X = xs; g_job.Y = ys;
    g_job.rows = rows; g_job.cols = cols; g_job.n_mats = n_mats;
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
void mxfp4_moe_layer(const uint8_t *const *packed, const uint8_t *const *scales,
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
void mxfp4_situ_batch(const float *gu, float *h, int n_ex, int d_ff, int nthreads) {
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

void mxfp4_moe_expert_set(const uint8_t *const *p13, const uint8_t *const *s13,
                          const uint8_t *const *p2,  const uint8_t *const *s2,
                          const float *x, const float *wts,
                          int n_ex, int d_ff, int d_model,
                          float *scratch, float *out, int nthreads) {
    if (n_ex <= 0) return;
    size_t need = (size_t)n_ex * (3 * (size_t)d_ff + (size_t)d_model);
    float *scr = scratch;
    if (!scr) {
        if (g_scr_n < need) {
            free(g_scr);
            g_scr = (float *)malloc(need * sizeof(float));
            g_scr_n = g_scr ? need : 0;
        }
        scr = g_scr;
        if (!scr) return;
    }
    float *gu = scr;                                   // [n_ex][2][d_ff]
    float *h  = gu + (size_t)n_ex * 2 * d_ff;          // [n_ex][d_ff]
    float *yb = h  + (size_t)n_ex * d_ff;              // [n_ex][d_model]

    int rows13[K3_MAX_MATS], cols13[K3_MAX_MATS];
    int rows2[K3_MAX_MATS],  cols2[K3_MAX_MATS];
    const float *xs[K3_MAX_MATS]; float *ys[K3_MAX_MATS];

    int n13 = 2 * n_ex;
    if (n13 > K3_MAX_MATS || n_ex > K3_MAX_MATS) return;
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
}
