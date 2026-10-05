//! JavaScript number semantics the reference relies on.
//!
//! `Math.min`/`Math.max` propagate NaN and order signed zeros; `Math.round`
//! rounds halves toward +Infinity. `Math.sin` and `Math.cos` are not the
//! platform libm: Node builds V8 without `V8_USE_LIBM_TRIG_FUNCTIONS`, so they
//! are V8's port of fdlibm (`src/base/ieee754.cc`, V8 14.6), ported here
//! statement by statement so automation values match the reference bit for bit.
//!
//! The arithmetic also follows the compiler: clang compiles C/C++ with
//! `-ffp-contract=on`, fusing a multiply and an add of one expression into an
//! FMA wherever the target has one (AArch64; x86-64 builds target a baseline
//! without FMA). [`fmuladd`] marks exactly the sites clang fuses (the
//! `llvm.fmuladd` formation of `CGExprScalar.cpp`: the left multiply is
//! preferred, compound `+=`/`-=` included), so the port matches Node on both
//! architectures (verified against Node 26 on macOS arm64 by
//! parity/check_automation.mjs).
//!
//! The fdlibm code carries this notice:
//!
//! ```text
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunSoft, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ```

// fdlibm is kept as written in V8's ieee754.cc: its constants (each parses to
// the documented bit pattern), `x - x` for NaN results, and its loops.
#![allow(
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::eq_op,
    clippy::explicit_counter_loop
)]

/// `Math.min(a, b)`: NaN if either is NaN; -0 is below +0.
pub fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() { a } else { b }
    } else if a < b {
        a
    } else {
        b
    }
}

/// `Math.max(a, b)`: NaN if either is NaN; +0 is above -0.
pub fn js_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_positive() { a } else { b }
    } else if a > b {
        a
    } else {
        b
    }
}

/// `Math.max(lo, Math.min(hi, x))`.
pub fn js_clamp(x: f64, lo: f64, hi: f64) -> f64 {
    js_max(lo, js_min(hi, x))
}

/// `Math.round(x)`: the nearest integer, halves toward +Infinity, keeping
/// -0 for values in [-0.5, 0).
pub fn js_round(x: f64) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let ceil = x.ceil();
    if ceil - 0.5 > x { ceil - 1.0 } else { ceil }
}

/// `Number.isInteger(x)` for a number.
pub fn is_integer(x: f64) -> bool {
    x.is_finite() && x.trunc() == x
}

// ----------------------------------------------------------------------------
// fdlibm sin/cos as V8 builds them.

/// `a * b + c` as clang emits `llvm.fmuladd`: one fused operation on AArch64,
/// a rounded multiply and add elsewhere.
#[inline(always)]
fn fmuladd(a: f64, b: f64, c: f64) -> f64 {
    if cfg!(target_arch = "aarch64") {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

fn high_word(x: f64) -> i32 {
    (x.to_bits() >> 32) as u32 as i32
}

fn low_word(x: f64) -> u32 {
    (x.to_bits() & 0xFFFF_FFFF) as u32
}

fn insert_words(high: i32, low: u32) -> f64 {
    f64::from_bits((u64::from(high as u32) << 32) | u64::from(low))
}

fn set_high_word(x: f64, high: i32) -> f64 {
    insert_words(high, low_word(x))
}

fn set_low_word(x: f64, low: u32) -> f64 {
    insert_words(high_word(x), low)
}

/// C `scalbn(x, n)` (musl's exact implementation).
fn scalbn(x: f64, mut n: i32) -> f64 {
    let x1p1023 = f64::from_bits(0x7fe0_0000_0000_0000); // 0x1p1023
    let x1p53 = f64::from_bits(0x4340_0000_0000_0000); // 0x1p53
    let x1p_1022 = f64::from_bits(0x0010_0000_0000_0000); // 0x1p-1022
    let mut y = x;
    if n > 1023 {
        y *= x1p1023;
        n -= 1023;
        if n > 1023 {
            y *= x1p1023;
            n -= 1023;
            if n > 1023 {
                n = 1023;
            }
        }
    } else if n < -1022 {
        // Make sure the final n < -53 to avoid double rounding in the
        // subnormal range.
        y *= x1p_1022 * x1p53;
        n += 1022 - 53;
        if n < -1022 {
            y *= x1p_1022 * x1p53;
            n += 1022 - 53;
            if n < -1022 {
                n = -1022;
            }
        }
    }
    y * f64::from_bits(((0x3ff + n) as u64) << 52)
}

const TWO_OVER_PI: [i32; 66] = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163,
    0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
    0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C,
    0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
    0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7, 0x8A5292,
    0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
    0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA,
    0x73A8C9, 0x60E27B, 0xC08C6B,
];

const NPIO2_HW: [i32; 32] = [
    0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C, 0x4025FDBB, 0x402921FB,
    0x402C463A, 0x402F6A7A, 0x4031475C, 0x4032D97C, 0x40346B9C, 0x4035FDBB, 0x40378FDB, 0x403921FB,
    0x403AB41B, 0x403C463A, 0x403DD85A, 0x403F6A7A, 0x40407E4C, 0x4041475C, 0x4042106C, 0x4042D97C,
    0x4043A28C, 0x40446B9C, 0x404534AC, 0x4045FDBB, 0x4046C6CB, 0x40478FDB, 0x404858EB, 0x404921FB,
];

/// `__ieee754_rem_pio2(x, y)`: x rem pi/2 in y[0] + y[1]; returns n.
fn rem_pio2(x: f64, y: &mut [f64; 2]) -> i32 {
    const ZERO: f64 = 0.0;
    const HALF: f64 = 0.5;
    const TWO24: f64 = 1.67772160000000000000e+07;
    const INVPIO2: f64 = 6.36619772367581382433e-01;
    const PIO2_1: f64 = 1.57079632673412561417e+00;
    const PIO2_1T: f64 = 6.07710050650619224932e-11;
    const PIO2_2: f64 = 6.07710050630396597660e-11;
    const PIO2_2T: f64 = 2.02226624879595063154e-21;
    const PIO2_3: f64 = 2.02226624871116645580e-21;
    const PIO2_3T: f64 = 8.47842766036889956997e-32;

    let mut z: f64 = 0.0;
    let hx = high_word(x);
    let ix = hx & 0x7FFFFFFF;
    if ix <= 0x3FE921FB {
        // |x| ~<= pi/4, no need for reduction
        y[0] = x;
        y[1] = 0.0;
        return 0;
    }
    if ix < 0x4002D97C {
        // |x| < 3pi/4, special case with n=+-1
        if hx > 0 {
            z = x - PIO2_1;
            if ix != 0x3FF921FB {
                y[0] = z - PIO2_1T;
                y[1] = (z - y[0]) - PIO2_1T;
            } else {
                z -= PIO2_2;
                y[0] = z - PIO2_2T;
                y[1] = (z - y[0]) - PIO2_2T;
            }
            return 1;
        } else {
            z = x + PIO2_1;
            if ix != 0x3FF921FB {
                y[0] = z + PIO2_1T;
                y[1] = (z - y[0]) + PIO2_1T;
            } else {
                z += PIO2_2;
                y[0] = z + PIO2_2T;
                y[1] = (z - y[0]) + PIO2_2T;
            }
            return -1;
        }
    }
    if ix <= 0x413921FB {
        // |x| ~<= 2^19*(pi/2), medium size
        let t = x.abs();
        let n = fmuladd(t, INVPIO2, HALF) as i32;
        let fn_ = f64::from(n);
        let mut r = fmuladd(-fn_, PIO2_1, t);
        let mut w = fn_ * PIO2_1T; // 1st round good to 85 bit
        if n < 32 && ix != NPIO2_HW[(n - 1) as usize] {
            y[0] = r - w; // quick check no cancellation
        } else {
            let j = ix >> 20;
            y[0] = r - w;
            let high = high_word(y[0]) as u32;
            let mut i = j - ((high >> 20) & 0x7FF) as i32;
            if i > 16 {
                // 2nd iteration needed, good to 118
                let t = r;
                w = fn_ * PIO2_2;
                r = t - w;
                w = fmuladd(fn_, PIO2_2T, -((t - r) - w));
                y[0] = r - w;
                let high = high_word(y[0]) as u32;
                i = j - ((high >> 20) & 0x7FF) as i32;
                if i > 49 {
                    // 3rd iteration need, 151 bits acc
                    let t = r;
                    w = fn_ * PIO2_3;
                    r = t - w;
                    w = fmuladd(fn_, PIO2_3T, -((t - r) - w));
                    y[0] = r - w;
                }
            }
        }
        y[1] = (r - y[0]) - w;
        if hx < 0 {
            y[0] = -y[0];
            y[1] = -y[1];
            return -n;
        }
        return n;
    }
    // all other (large) arguments
    if ix >= 0x7FF00000 {
        // x is inf or NaN
        y[0] = x - x;
        y[1] = y[0];
        return 0;
    }
    // set z = scalbn(|x|,ilogb(x)-23)
    let low = low_word(x);
    z = set_low_word(z, low);
    let e0 = (ix >> 20) - 1046; // e0 = ilogb(z)-23;
    z = set_high_word(z, ix - ((e0 as u32) << 20) as i32);
    let mut tx = [0.0f64; 3];
    for item in tx.iter_mut().take(2) {
        *item = f64::from(z as i32);
        z = (z - *item) * TWO24;
    }
    tx[2] = z;
    let mut nx = 3;
    while tx[nx - 1] == ZERO {
        nx -= 1; // skip zero term
    }
    let n = kernel_rem_pio2(&tx[..nx], y, e0, 2);
    if hx < 0 {
        y[0] = -y[0];
        y[1] = -y[1];
        return -n;
    }
    n
}

/// `__kernel_rem_pio2(x, y, e0, nx, prec, two_over_pi)` for prec <= 2.
fn kernel_rem_pio2(x: &[f64], y: &mut [f64; 2], e0: i32, prec: usize) -> i32 {
    const INIT_JK: [i32; 4] = [2, 3, 4, 6];
    const PIO2: [f64; 8] = [
        1.57079625129699707031e+00,
        7.54978941586159635335e-08,
        5.39030252995776476554e-15,
        3.28200341580791294123e-22,
        1.27065575308067607349e-29,
        1.22933308981111328932e-36,
        2.73370053816464559624e-44,
        2.16741683877804819444e-51,
    ];
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const TWO24: f64 = 1.67772160000000000000e+07;
    const TWON24: f64 = 5.96046447753906250000e-08;
    let ipio2 = &TWO_OVER_PI;

    let nx = x.len() as i32;
    let mut iq = [0i32; 20];
    let mut f = [0.0f64; 20];
    let mut fq = [0.0f64; 20];
    let mut q = [0.0f64; 20];

    // initialize jk
    let jk = INIT_JK[prec];
    let jp = jk;

    // determine jx,jv,q0, note that 3>q0
    let jx = nx - 1;
    let mut jv = (e0 - 3) / 24;
    if jv < 0 {
        jv = 0;
    }
    let mut q0 = e0 - 24 * (jv + 1);

    // set up f[0] to f[jx+jk] where f[jx+jk] = ipio2[jv+jk]
    let mut j = jv - jx;
    let m = jx + jk;
    for i in 0..=m {
        f[i as usize] = if j < 0 {
            ZERO
        } else {
            f64::from(ipio2[j as usize])
        };
        j += 1;
    }

    // compute q[0],q[1],...q[jk]
    for i in 0..=jk {
        let mut fw = 0.0;
        for j in 0..=jx {
            fw = fmuladd(x[j as usize], f[(jx + i - j) as usize], fw);
        }
        q[i as usize] = fw;
    }

    let mut jz = jk;
    let mut z;
    let mut n;
    let mut ih;
    loop {
        // distill q[] into iq[] reversingly
        let mut i = 0usize;
        let mut j = jz;
        z = q[jz as usize];
        while j > 0 {
            let fw = f64::from((TWON24 * z) as i32);
            iq[i] = fmuladd(-TWO24, fw, z) as i32;
            z = q[(j - 1) as usize] + fw;
            i += 1;
            j -= 1;
        }

        // compute n
        z = scalbn(z, q0); // actual value of z
        z = fmuladd(-8.0, (z * 0.125).floor(), z); // trim off integer >= 8
        n = z as i32;
        z -= f64::from(n);
        ih = 0;
        if q0 > 0 {
            // need iq[jz-1] to determine n
            let i = iq[(jz - 1) as usize] >> (24 - q0);
            n += i;
            iq[(jz - 1) as usize] -= i << (24 - q0);
            ih = iq[(jz - 1) as usize] >> (23 - q0);
        } else if q0 == 0 {
            ih = iq[(jz - 1) as usize] >> 23;
        } else if z >= 0.5 {
            ih = 2;
        }

        if ih > 0 {
            // q > 0.5
            n += 1;
            let mut carry = 0;
            for item in iq.iter_mut().take(jz as usize) {
                // compute 1-q
                let j = *item;
                if carry == 0 {
                    if j != 0 {
                        carry = 1;
                        *item = 0x1000000 - j;
                    }
                } else {
                    *item = 0xFFFFFF - j;
                }
            }
            if q0 > 0 {
                // rare case: chance is 1 in 12
                match q0 {
                    1 => iq[(jz - 1) as usize] &= 0x7FFFFF,
                    2 => iq[(jz - 1) as usize] &= 0x3FFFFF,
                    _ => {}
                }
            }
            if ih == 2 {
                z = ONE - z;
                if carry != 0 {
                    z -= scalbn(ONE, q0);
                }
            }
        }

        // check if recomputation is needed
        if z == ZERO {
            let mut j = 0;
            let mut i = jz - 1;
            while i >= jk {
                j |= iq[i as usize];
                i -= 1;
            }
            if j == 0 {
                // need recomputation
                let mut k = 1;
                while jk >= k && iq[(jk - k) as usize] == 0 {
                    k += 1; // k = no. of terms needed
                }
                for i in (jz + 1)..=(jz + k) {
                    // add q[jz+1] to q[jz+k]
                    f[(jx + i) as usize] = f64::from(ipio2[(jv + i) as usize]);
                    let mut fw = 0.0;
                    for j in 0..=jx {
                        fw = fmuladd(x[j as usize], f[(jx + i - j) as usize], fw);
                    }
                    q[i as usize] = fw;
                }
                jz += k;
                continue;
            }
        }
        break;
    }

    // chop off zero terms
    if z == 0.0 {
        jz -= 1;
        q0 -= 24;
        while iq[jz as usize] == 0 {
            jz -= 1;
            q0 -= 24;
        }
    } else {
        // break z into 24-bit if necessary
        z = scalbn(z, -q0);
        if z >= TWO24 {
            let fw = f64::from((TWON24 * z) as i32);
            iq[jz as usize] = fmuladd(-TWO24, fw, z) as i32;
            jz += 1;
            q0 += 24;
            iq[jz as usize] = fw as i32;
        } else {
            iq[jz as usize] = z as i32;
        }
    }

    // convert integer "bit" chunk to floating-point value
    let mut fw = scalbn(ONE, q0);
    let mut i = jz;
    while i >= 0 {
        q[i as usize] = fw * f64::from(iq[i as usize]);
        fw *= TWON24;
        i -= 1;
    }

    // compute PIo2[0,...,jp]*q[jz,...,0]
    let mut i = jz;
    while i >= 0 {
        let mut fw = 0.0;
        let mut k = 0;
        while k <= jp && k <= jz - i {
            fw = fmuladd(PIO2[k as usize], q[(i + k) as usize], fw);
            k += 1;
        }
        fq[(jz - i) as usize] = fw;
        i -= 1;
    }

    // compress fq[] into y[] (prec 1 and 2)
    let mut fw = 0.0;
    let mut i = jz;
    while i >= 0 {
        fw += fq[i as usize];
        i -= 1;
    }
    y[0] = if ih == 0 { fw } else { -fw };
    fw = fq[0] - fw;
    for i in 1..=jz {
        fw += fq[i as usize];
    }
    y[1] = if ih == 0 { fw } else { -fw };
    n & 7
}

/// `__kernel_cos(x, y)` on [-pi/4, pi/4].
fn kernel_cos(x: f64, y: f64) -> f64 {
    const ONE: f64 = 1.00000000000000000000e+00;
    const C1: f64 = 4.16666666666666019037e-02;
    const C2: f64 = -1.38888888888741095749e-03;
    const C3: f64 = 2.48015872894767294178e-05;
    const C4: f64 = -2.75573143513906633035e-07;
    const C5: f64 = 2.08757232129817482790e-09;
    const C6: f64 = -1.13596475577881948265e-11;

    let ix = high_word(x) & 0x7FFFFFFF; // ix = |x|'s high word
    if ix < 0x3E400000 && (x as i32) == 0 {
        // if x < 2**27, generate inexact
        return ONE;
    }
    let z = x * x;
    // r = z*(C1+z*(C2+z*(C3+z*(C4+z*(C5+z*C6)))))
    let r = z * fmuladd(
        z,
        fmuladd(z, fmuladd(z, fmuladd(z, fmuladd(z, C6, C5), C4), C3), C2),
        C1,
    );
    // z*r - x*y
    let zr_xy = fmuladd(z, r, -(x * y));
    if ix < 0x3FD33333 {
        // if |x| < 0.3: one - (0.5*z - (z*r - x*y))
        ONE - fmuladd(0.5, z, -zr_xy)
    } else {
        let qx = if ix > 0x3FE90000 {
            // x > 0.78125
            0.28125
        } else {
            insert_words(ix - 0x00200000, 0) // x/4
        };
        let iz = fmuladd(0.5, z, -qx);
        let a = ONE - qx;
        a - (iz - zr_xy)
    }
}

/// `__kernel_sin(x, y, iy)` on [-pi/4, pi/4].
fn kernel_sin(x: f64, y: f64, iy: i32) -> f64 {
    const HALF: f64 = 5.00000000000000000000e-01;
    const S1: f64 = -1.66666666666666324348e-01;
    const S2: f64 = 8.33333333332248946124e-03;
    const S3: f64 = -1.98412698298579493134e-04;
    const S4: f64 = 2.75573137070700676789e-06;
    const S5: f64 = -2.50507602534068634195e-08;
    const S6: f64 = 1.58969099521155010221e-10;

    let ix = high_word(x) & 0x7FFFFFFF; // high word of x
    if ix < 0x3E400000 && (x as i32) == 0 {
        // |x| < 2**-27, generate inexact
        return x;
    }
    let z = x * x;
    let v = z * x;
    // r = S2+z*(S3+z*(S4+z*(S5+z*S6)))
    let r = fmuladd(z, fmuladd(z, fmuladd(z, fmuladd(z, S6, S5), S4), S3), S2);
    if iy == 0 {
        // x + v*(S1 + z*r)
        fmuladd(v, fmuladd(z, r, S1), x)
    } else {
        // x - ((z*(half*y - v*r) - y) - v*S1)
        x - fmuladd(-v, S1, fmuladd(z, fmuladd(HALF, y, -(v * r)), -y))
    }
}

/// `Math.cos(x)` as V8 computes it (fdlibm `cos`).
pub fn js_cos(x: f64) -> f64 {
    let ix = high_word(x) & 0x7FFFFFFF;
    if ix <= 0x3FE921FB {
        return kernel_cos(x, 0.0);
    }
    if ix >= 0x7FF00000 {
        // cos(Inf or NaN) is NaN
        return x - x;
    }
    let mut y = [0.0f64; 2];
    let n = rem_pio2(x, &mut y);
    match n & 3 {
        0 => kernel_cos(y[0], y[1]),
        1 => -kernel_sin(y[0], y[1], 1),
        2 => -kernel_cos(y[0], y[1]),
        _ => kernel_sin(y[0], y[1], 1),
    }
}

/// `Math.sin(x)` as V8 computes it (fdlibm `sin`).
pub fn js_sin(x: f64) -> f64 {
    let ix = high_word(x) & 0x7FFFFFFF;
    if ix <= 0x3FE921FB {
        return kernel_sin(x, 0.0, 0);
    }
    if ix >= 0x7FF00000 {
        // sin(Inf or NaN) is NaN
        return x - x;
    }
    let mut y = [0.0f64; 2];
    let n = rem_pio2(x, &mut y);
    match n & 3 {
        0 => kernel_sin(y[0], y[1], 1),
        1 => kernel_cos(y[0], y[1]),
        2 => -kernel_sin(y[0], y[1], 1),
        _ => -kernel_cos(y[0], y[1]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_max_follow_javascript() {
        assert!(js_min(1.0, f64::NAN).is_nan());
        assert!(js_max(f64::NAN, 0.0).is_nan());
        assert!(js_min(0.0, -0.0).is_sign_negative());
        assert!(js_max(-0.0, 0.0).is_sign_positive());
        assert_eq!(js_clamp(1.5, 0.0, 1.0), 1.0);
        assert!(js_clamp(-0.0, 0.0, 1.0).is_sign_positive());
    }

    #[test]
    fn round_halves_toward_positive_infinity() {
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0);
        assert_eq!(js_round(0.49999999999999994), 0.0);
        assert!(js_round(-0.4).is_sign_negative());
        assert_eq!(js_round(-0.6), -1.0);
    }

    // Values printed by Node 26 (V8 14.6): `Math.cos(1e10)` etc.
    #[test]
    fn trig_matches_v8() {
        assert_eq!(js_cos(1e10), 0.873119622676856);
        assert_eq!(js_sin(0.5), 0.479425538604203);
        assert_eq!(js_cos(2.4), -0.7373937155412454);
        assert_eq!(js_sin(1e300), -0.8178819121159085);
        assert!(js_sin(f64::INFINITY).is_nan());
        assert_eq!(js_cos(0.0), 1.0);
        assert!(js_sin(-0.0).is_sign_negative());
    }
}
