/*
 * ssa.c -- exact Gillespie simulation of the two-pole plasmid model.
 *
 * State c[p*(N+1) + j], p = 0,1, j = 1..N (index 0 unused), c[p,1] = f_p.
 *
 * Channels per pole (w_j = j^gamma), Parts IV and V of the notes:
 *   nucleation   X1+X1 -> X2        (1/2) beta f (f-1)
 *   growth       X1+Xj -> X(j+1)    beta w_j f c_j                 j >= 2
 *   shedding     Xj -> X(j-1)+X1    kappa j c_j                    j >= 3
 *   dissolution  X2 -> X1+X1        kappa c_2
 *   transfer     X1(p) -> X1(q)     rho f_p
 * and, if smol != 0 (Smoluchowski form, Part V):
 *   merging      Xi+Xj -> X(i+j)    betam w_i w_j c_i c_j  (i<j),
 *                                   (1/2) betam w_i^2 c_i (c_i-1)  (i=j)
 *   breaking     Xm -> Xi+X(m-i)    kappa m c_m (i != m-i), (1/2) kappa m c_m
 *                                   (m = 2i);   2 <= i <= m-2
 * Total merging propensity (1/2) betam (A^2 - A2), A = sum w_j c_j,
 * A2 = sum w_j^2 c_j; total breaking (1/2) kappa sum_{m>=4} m (m-3) c_m.
 *
 * Built as a shared library and called from plasmid.py through ctypes.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* ---- xoshiro256** (Blackman & Vigna), seeded by splitmix64 ---- */
typedef struct { uint64_t s[4]; } rng_t;

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t rng_next(rng_t *r) {
    uint64_t *s = r->s;
    const uint64_t res = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return res;
}

static void rng_seed(rng_t *r, uint64_t x) {
    for (int i = 0; i < 4; i++) {
        uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        r->s[i] = z ^ (z >> 31);
    }
}

/* uniform on (0, 1) */
static inline double unif(rng_t *r) { return ((rng_next(r) >> 11) + 0.5) * 0x1.0p-53; }

/* index k in [lo, hi] with probability weight[k] / total */
static int pick(const double *weight, int lo, int hi, double total, rng_t *r) {
    double u = unif(r) * total, acc = 0.0;
    int last = lo;
    for (int k = lo; k <= hi; k++) {
        if (weight[k] <= 0.0) continue;
        acc += weight[k];
        last = k;
        if (u < acc) return k;
    }
    return last;                                 /* round-off guard */
}

/*
 * Run one trajectory from state c (modified in place to the state at T).
 * The state just before time trec[i] is copied to rec[i*2*(N+1) ...].
 * Returns the number of reaction events.
 */
long ssa_run(int N, double beta, double betam, double kappa, double rho,
             double gamma, int smol, double T, int nrec, const double *trec,
             int64_t *c, int16_t *rec, uint64_t seed)
{
    const int L = N + 1;
    rng_t rng;
    rng_seed(&rng, seed);

    double *w = calloc(L, sizeof(double));
    double *wt = calloc(L, sizeof(double));      /* scratch weights */
    for (int j = 2; j <= N; j++) w[j] = pow((double)j, gamma);

    double t = 0.0;
    int ri = 0;
    long nev = 0;

    for (;;) {
        double a[2][7];      /* nuc, gro, shd, dim, trn, mer, brk */
        double A[2], tot = 0.0;
        for (int p = 0; p < 2; p++) {
            const int64_t *cp = c + p * L;
            double f = (double)cp[1], Ap = 0.0, A2 = 0.0, shd = 0.0, brk = 0.0;
            long ncl = 0;
            for (int j = 2; j <= N; j++) {
                if (!cp[j]) continue;
                double cj = (double)cp[j];
                Ap += w[j] * cj;
                A2 += w[j] * w[j] * cj;
                ncl += cp[j];
                if (j >= 3) shd += j * cj;
                if (j >= 4) brk += 0.5 * j * (j - 3.0) * cj;
            }
            A[p] = Ap;
            a[p][0] = 0.5 * beta * f * (f - 1.0);
            a[p][1] = beta * f * Ap;
            a[p][2] = kappa * shd;
            a[p][3] = kappa * (double)cp[2];
            a[p][4] = rho * f;
            a[p][5] = (smol && ncl >= 2) ? 0.5 * betam * (Ap * Ap - A2) : 0.0;
            a[p][6] = smol ? kappa * brk : 0.0;
            for (int k = 0; k < 7; k++) tot += a[p][k];
        }

        double tn = (tot > 0.0) ? t - log(unif(&rng)) / tot : INFINITY;
        while (ri < nrec && trec[ri] <= tn) {
            for (int k = 0; k < 2 * L; k++) rec[(long)ri * 2 * L + k] = (int16_t)c[k];
            ri++;
        }
        if (tn >= T) break;
        t = tn;
        nev++;

        /* choose pole and channel */
        double u = unif(&rng) * tot, acc = 0.0;
        int p = 1, ch = 6;
        for (int pp = 0; pp < 2; pp++)
            for (int k = 0; k < 7; k++) {
                acc += a[pp][k];
                if (u < acc && a[pp][k] > 0.0) { p = pp; ch = k; goto chosen; }
            }
        /* round-off: last channel with positive propensity */
        for (int pp = 1; pp >= 0; pp--)
            for (int k = 6; k >= 0; k--)
                if (a[pp][k] > 0.0) { p = pp; ch = k; goto chosen; }
chosen:;
        int64_t *cp = c + p * L;
        int j, i, m;
        switch (ch) {
        case 0:                                  /* nucleation */
            cp[1] -= 2; cp[2] += 1; break;
        case 1:                                  /* growth */
            for (j = 2; j <= N; j++) wt[j] = w[j] * (double)cp[j];
            j = pick(wt, 2, N, A[p], &rng);
            cp[1] -= 1; cp[j] -= 1; cp[j + 1] += 1; break;
        case 2: {                                /* shedding, j >= 3 */
            double s = 0.0;
            for (j = 3; j <= N; j++) { wt[j] = j * (double)cp[j]; s += wt[j]; }
            j = pick(wt, 3, N, s, &rng);
            cp[j] -= 1; cp[j - 1] += 1; cp[1] += 1; break; }
        case 3:                                  /* dimer dissolution */
            cp[2] -= 1; cp[1] += 2; break;
        case 4:                                  /* transfer */
            cp[1] -= 1; c[(1 - p) * L + 1] += 1; break;
        case 5: {                                /* merging */
            /* first cluster a with P ~ w_a (A - w_a), partner b ~ w_b among
               the remaining clusters: ordered pair ~ w_a w_b, as required */
            double s = 0.0;
            for (j = 2; j <= N; j++) { wt[j] = w[j] * (double)cp[j] * (A[p] - w[j]); s += wt[j]; }
            i = pick(wt, 2, N, s, &rng);
            cp[i] -= 1;
            s = 0.0;
            for (j = 2; j <= N; j++) { wt[j] = w[j] * (double)cp[j]; s += wt[j]; }
            j = pick(wt, 2, N, s, &rng);
            cp[j] -= 1; cp[i + j] += 1; break; }
        case 6: {                                /* breaking */
            /* m ~ m (m-3) c_m; then one of the m-3 labels i = 2..m-2, each
               at rate kappa m / 2, gives the pieces {i, m-i} */
            double s = 0.0;
            for (m = 2; m <= N; m++) { wt[m] = (m >= 4) ? m * (m - 3.0) * (double)cp[m] : 0.0; s += wt[m]; }
            m = pick(wt, 4, N, s, &rng);
            i = 2 + (int)(unif(&rng) * (m - 3));
            if (i > m - 2) i = m - 2;
            cp[m] -= 1; cp[i] += 1; cp[m - i] += 1; break; }
        }
    }
    while (ri < nrec) {
        for (int k = 0; k < 2 * L; k++) rec[(long)ri * 2 * L + k] = (int16_t)c[k];
        ri++;
    }
    free(w); free(wt);
    return nev;
}
