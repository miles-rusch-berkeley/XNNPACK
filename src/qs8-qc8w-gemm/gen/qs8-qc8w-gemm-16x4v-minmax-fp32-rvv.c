// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/opu-rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 SiFive, Inc.
// Copyright 2024 Microchip
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/bme.h"
#include "xnnpack/gemm.h"
#include "xnnpack/math.h"

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(nc != 0);
  assert(kc != 0);

  int8_t* a0 = a;
  int8_t* c0 = c;
  
  size_t nr;
  __asm__ volatile("vsetvli %0, zero, e32, m4, ta, ma" : "=r"(nr));
  size_t vl = nr;

  const int32_t output_min_less_zero_point = (int32_t) params->fp32_scalar.output_min - (int32_t) params->fp32_scalar.output_zero_point;
  const int32_t output_max_less_zero_point = (int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point;
  const int32_t output_zero_point = params->fp32_scalar.output_zero_point;
  do {
    if XNN_UNLIKELY(nc < nr) {
      __asm__ volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl) : "r"(nc));
    }
    nc = nc - vl;

    __asm__ volatile("vle32.v v0, (%0)" : : "r"((const int32_t*)w));
    for (size_t r = 0; r < mr; r++) {
      VMV_RV(m1, r, v0); // move v0 into row r of m1
    }
    w = (const int32_t*) w + nr;

    size_t k = 0;
    do {
      // opacc_e8m1w4(a0, w, a_stride, mr, vl);
      __asm__ volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(mr));
      __asm__ volatile("vlse8.v v8, (%0), %1" : : "r"(a0), "r"(a_stride));
      __asm__ volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(vl));
      __asm__ volatile("vle8.v v9, (%0)" : : "r"(w));
      VOPACC(m1, v4, v0);
      a0++;
      w = (const int8_t*) w + nr;
      k += sizeof(int8_t);
    } while (k != kc);
 
    //v4 <- vscale
    __asm__ volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl));
    __asm__ volatile("vle32.v v4, (%0)" : : "r"((const float*) w));
    w = (const float*) w + nr;

    for (size_t r=0; r<mr; r++) {
      //v0 <- vopacc
      VMV_VR(v0, r, m1); // move row r of m1 into v0
      __asm__ volatile("vfcvt.f.x.v	v0,v0");
      //v0 <- vopacc * vscale
      __asm__ volatile("vfmul.vv	v0,v0,v4");
      //v0 <- minmax
      __asm__ volatile("vfmax.vf	v0,v0,%0" : : "f"((float) output_min_less_zero_point));
      __asm__ volatile("vfmin.vf	v0,v0,%0" : : "f"((float) output_max_less_zero_point));
      
      __asm__ volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl));
      __asm__ volatile("vfncvt.x.f.w	v0,v0");
      __asm__ volatile("vadd.vx	v0,v0,%0" : : "r"((int16_t) output_zero_point));
      
      __asm__ volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(vl));
      __asm__ volatile("vncvt.x.x.w	v0,v0");
      __asm__ volatile("vse8.v	v0, (%0)" : : "r"(c0));
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
    }
  } while (nc != 0);
}
