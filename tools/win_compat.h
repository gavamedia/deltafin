// Minimal Win32 implementations of the POSIX subset used by the MXFP4 kernels.
//
// This is intentionally narrow, exactly like neon_compat_x86.h: it implements
// only the threading, timing and aligned-allocation calls that fused_gemv.c and
// fused_gemv_batch.c actually make, so one kernel source builds on macOS, Linux
// and Windows instead of maintaining a second, easily-divergent implementation.
//
// The mutex is an SRWLOCK, which is not recursive.  The kernels never re-acquire
// a held lock, and making that assumption explicit here is safer than silently
// substituting a recursive CRITICAL_SECTION that would hide a future mistake.

#ifndef K3_WIN_COMPAT_H
#define K3_WIN_COMPAT_H

#if !defined(_WIN32)
#error "win_compat.h is only for Windows builds"
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>

#include <errno.h>
#include <intrin.h>
#include <malloc.h>
#include <process.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

// ---------------- exported symbols ----------------
// A Windows DLL exports nothing by default, so the public kernel entry points
// carry an explicit attribute.  build_native.py validates the resulting export
// table against the same symbol manifest it uses on macOS and Linux.
#define K3_EXPORT __declspec(dllexport)

// ---------------- alignment ----------------
// __declspec(align) occupies the same declaration slot as GCC's aligned
// attribute after `struct`, but must precede a variable's storage class.
#define K3_ALIGNAS(n) __declspec(align(n))
#define K3_ALIGN_PREFIX(n) __declspec(align(n))

// ---------------- aligned allocation ----------------
// The UCRT deliberately does not implement C11 aligned_alloc, because its free()
// cannot release such a block.  Blocks from this allocator must therefore be
// released with k3_aligned_free, never with free().
static inline void *k3_aligned_alloc(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}

static inline void k3_aligned_free(void *ptr) {
    _aligned_free(ptr);
}

// ---------------- timing ----------------
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 1

// 100 ns ticks between 1601-01-01 (FILETIME) and 1970-01-01 (Unix epoch).
#define K3_FILETIME_UNIX_EPOCH_DELTA 116444736000000000ULL

static inline int clock_gettime(int clock_id, struct timespec *ts) {
    if (!ts) {
        errno = EINVAL;
        return -1;
    }
    if (clock_id == CLOCK_REALTIME) {
        FILETIME ft;
        ULARGE_INTEGER ticks;
        GetSystemTimePreciseAsFileTime(&ft);
        ticks.LowPart = ft.dwLowDateTime;
        ticks.HighPart = ft.dwHighDateTime;
        ticks.QuadPart -= K3_FILETIME_UNIX_EPOCH_DELTA;
        ts->tv_sec = (time_t)(ticks.QuadPart / 10000000ULL);
        ts->tv_nsec = (long)((ticks.QuadPart % 10000000ULL) * 100ULL);
        return 0;
    }
    if (clock_id == CLOCK_MONOTONIC) {
        LARGE_INTEGER frequency, counter;
        if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0) {
            errno = EINVAL;
            return -1;
        }
        QueryPerformanceCounter(&counter);
        // Split before scaling so a long uptime cannot overflow the product.
        ts->tv_sec = (time_t)(counter.QuadPart / frequency.QuadPart);
        ts->tv_nsec = (long)(
            ((counter.QuadPart % frequency.QuadPart) * 1000000000LL)
            / frequency.QuadPart);
        return 0;
    }
    errno = EINVAL;
    return -1;
}

// ---------------- scheduling ----------------
static inline int sched_yield(void) {
    // SwitchToThread yields only to this processor's ready threads and reports
    // whether one ran.  Either outcome is a successful yield for the callers.
    SwitchToThread();
    return 0;
}

// ---------------- threads ----------------
typedef HANDLE pthread_t;
typedef int pthread_attr_t;

static inline int pthread_attr_init(pthread_attr_t *attr) {
    if (attr) *attr = 0;
    return 0;
}

static inline int pthread_attr_destroy(pthread_attr_t *attr) {
    (void)attr;
    return 0;
}

typedef struct k3_thread_start_s {
    void *(*routine)(void *);
    void *arg;
} k3_thread_start_t;

static unsigned __stdcall k3_thread_trampoline(void *raw) {
    k3_thread_start_t start = *(k3_thread_start_t *)raw;
    free(raw);
    (void)start.routine(start.arg);
    return 0u;
}

static inline int pthread_create(
        pthread_t *thread, const pthread_attr_t *attr,
        void *(*routine)(void *), void *arg) {
    (void)attr;
    if (!thread || !routine) return EINVAL;
    k3_thread_start_t *start =
        (k3_thread_start_t *)malloc(sizeof(k3_thread_start_t));
    if (!start) return EAGAIN;
    start->routine = routine;
    start->arg = arg;
    // _beginthreadex, not CreateThread: these workers call into the CRT, which
    // needs its per-thread state created and torn down by the CRT itself.
    uintptr_t handle = _beginthreadex(NULL, 0, k3_thread_trampoline, start, 0, NULL);
    if (handle == 0) {
        free(start);
        return EAGAIN;
    }
    *thread = (HANDLE)handle;
    return 0;
}

static inline int pthread_join(pthread_t thread, void **retval) {
    if (retval) *retval = NULL;
    if (!thread) return EINVAL;
    if (WaitForSingleObject(thread, INFINITE) != WAIT_OBJECT_0) return EINVAL;
    CloseHandle(thread);
    return 0;
}

// ---------------- mutexes ----------------
typedef SRWLOCK pthread_mutex_t;
#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT

static inline int pthread_mutex_lock(pthread_mutex_t *mutex) {
    AcquireSRWLockExclusive(mutex);
    return 0;
}

static inline int pthread_mutex_unlock(pthread_mutex_t *mutex) {
    ReleaseSRWLockExclusive(mutex);
    return 0;
}

// ---------------- condition variables ----------------
typedef CONDITION_VARIABLE pthread_cond_t;
#define PTHREAD_COND_INITIALIZER CONDITION_VARIABLE_INIT

static inline int pthread_cond_wait(
        pthread_cond_t *cond, pthread_mutex_t *mutex) {
    if (!SleepConditionVariableSRW(cond, mutex, INFINITE, 0)) return EINVAL;
    return 0;
}

static inline int pthread_cond_timedwait(
        pthread_cond_t *cond, pthread_mutex_t *mutex,
        const struct timespec *abstime) {
    struct timespec now;
    int64_t milliseconds;
    if (!abstime) return EINVAL;
    if (clock_gettime(CLOCK_REALTIME, &now) != 0) return EINVAL;
    milliseconds =
        (int64_t)(abstime->tv_sec - now.tv_sec) * 1000
        + ((int64_t)abstime->tv_nsec - (int64_t)now.tv_nsec) / 1000000;
    if (milliseconds < 0) milliseconds = 0;
    // INFINITE must stay reachable only through pthread_cond_wait.
    if (milliseconds > (int64_t)(INFINITE - 1)) milliseconds = (int64_t)(INFINITE - 1);
    if (!SleepConditionVariableSRW(cond, mutex, (DWORD)milliseconds, 0)) {
        return GetLastError() == ERROR_TIMEOUT ? ETIMEDOUT : EINVAL;
    }
    return 0;
}

static inline int pthread_cond_broadcast(pthread_cond_t *cond) {
    WakeAllConditionVariable(cond);
    return 0;
}

// ---------------- one-time initialization ----------------
typedef INIT_ONCE pthread_once_t;
#define PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT

static BOOL CALLBACK k3_once_trampoline(
        PINIT_ONCE once, PVOID parameter, PVOID *context) {
    (void)once;
    (void)context;
    ((void (*)(void))parameter)();
    return TRUE;
}

static inline int pthread_once(pthread_once_t *once, void (*routine)(void)) {
    if (!InitOnceExecuteOnce(once, k3_once_trampoline, (PVOID)routine, NULL)) {
        return EINVAL;
    }
    return 0;
}

// ---------------- CPU feature detection ----------------
// Mirrors what __builtin_cpu_supports("avx2") checks on GCC and Clang: the
// instruction bit alone is not enough, the OS must also preserve YMM state.
static inline int k3_cpu_supports_avx2(void) {
    int regs[4];
    __cpuid(regs, 0);
    if (regs[0] < 7) return 0;
    __cpuid(regs, 1);
    if (!(regs[2] & (1 << 27))) return 0;        // OSXSAVE
    if ((_xgetbv(0) & 0x6u) != 0x6u) return 0;   // XMM and YMM state saved
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;            // AVX2
}

#endif
