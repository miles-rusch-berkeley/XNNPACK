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
#include <stdio.h>
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
  assert(mr <= 16);
  assert(nc != 0);
  assert(kc != 0);
  // printf("xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x4v__rvv: mr=%zu, nc=%zu, kc=%zu\n", mr, nc, kc);
  int8_t* a0 = a;
  int8_t* c0 = c;
  
  size_t nr;
  __asm__ volatile("vsetvli %0, zero, e32, m4, ta, ma" : "=r"(nr));
  
  const int32_t output_min_less_zero_point = (int32_t) params->fp32_scalar.output_min - (int32_t) params->fp32_scalar.output_zero_point;
  const int32_t output_max_less_zero_point = (int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point;
  const int32_t output_zero_point = params->fp32_scalar.output_zero_point;
  while (nc >= 2*nr) {
    nc = nc - 2*nr;
    
    __asm__ volatile("vsetvli zero, zero, e32, m4, ta, ma" : : "r"(nr));
    __asm__ volatile("vle32.v v0, (%0)" : : "r"((const int32_t*)w));
    OPMVINBCAST(m0, v0); // broadcast channel-wise bias
    w = (const int32_t*) w + nr;
    void* w2 = (const int8_t*) w + nr*kc;
    w2 = (const float*) w + nr;
    __asm__ volatile("vle32.v v4, (%0)" : : "r"((const int32_t*)w2));
    OPMVINBCAST(m1, v4); // broadcast channel-wise bias

    __asm__ volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(nr));
    for (int k = 0; k+2 <= kc; k+=2) {
      __asm__ volatile("vle8.v v4, (%0)" : : "r"(a0));
      __asm__ volatile("vle8.v v5, (%0)" : : "r"(w));
      w = (const int8_t*) w + nr;
      VOPACC(m0, v4, v5);
      __asm__ volatile("vle8.v v6, (%0)" : : "r"(w2));
      VOPACC(m1, v4, v6);
      w2 = (const int8_t*) w2 + nr;
      a0 = (const int8_t*) a0 + a_stride;

      __asm__ volatile("vle8.v v4, (%0)" : : "r"(a0));
      __asm__ volatile("vle8.v v5, (%0)" : : "r"(w));
      w = (const int8_t*) w + nr;
      VOPACC(m0, v4, v5);
      __asm__ volatile("vle8.v v6, (%0)" : : "r"(w2));
      VOPACC(m1, v4, v6);
      w2 = (const int8_t*) w2 + nr;
      a0 = (const int8_t*) a0 + a_stride;
    }
    if XNN_UNLIKELY((kc%2) != 0) {
      __asm__ volatile("vle8.v v4, (%0)" : : "r"(a0));
      __asm__ volatile("vle8.v v5, (%0)" : : "r"(w));
      w = (const int8_t*) w + nr;
      VOPACC(m0, v4, v5);
      __asm__ volatile("vle8.v v6, (%0)" : : "r"(w2));
      VOPACC(m1, v4, v6);
      w2 = (const int8_t*) w2 + nr;
      a0 = (const int8_t*) a0 + a_stride;

    }
    
    //v4 <- vscale
    __asm__ volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(nr));
    __asm__ volatile("vle32.v v8, (%0)" : : "r"((const float*) w));
    __asm__ volatile("vle32.v v12, (%0)" : : "r"((const float*) w2));
    w = (const float*) w2 + nr;

    int8_t* cm = c0;
    for (size_t r=0; r<mr; r++) {
      // printf("      r=%zu\n", r);
      //v12 <- vopacc
      __asm__ volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(nr));
      VMV_VR(v16, r, m0); // move row r of m1 into v12
      VMV_VR(v20, r, m0); // move row r of m1 into v12
      __asm__ volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(2*nr));
      __asm__ volatile("vfcvt.f.x.v	v16,v16");
      //v16 <- vopacc * vscale
      __asm__ volatile("vfmul.vv	v16,v16,v8");
      //v16 <- minmax
      __asm__ volatile("vfmax.vf	v16,v16,%0" : : "f"((float) output_min_less_zero_point));
      __asm__ volatile("vfmin.vf	v16,v16,%0" : : "f"((float) output_max_less_zero_point));
      __asm__ volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(2*nr));
      __asm__ volatile("vfncvt.x.f.w	v16,v16");
      __asm__ volatile("vadd.vx	v16,v16,%0" : : "r"((int16_t) output_zero_point));
      __asm__ volatile("vsetvli zero, %0, e8, m2, ta, ma" : : "r"(2*nr));
      __asm__ volatile("vncvt.x.x.w	v16,v16"); 

      __asm__ volatile("vse8.v	v0, (%0)" : : "r"(cm));
      cm = (int8_t*) ((uintptr_t) cm + cm_stride);
    }
    c0 = (int8_t*) ((uintptr_t) c0 + 2*cn_stride);
    a0 = (const int8_t*)  ((uintptr_t) a0 - kc*a_stride);
  }

  __asm__ volatile("vsetvli %0, zero, e32, m4, ta, ma" : "=r"(nr));
  size_t vl = nr;
  do {
    // printf("  nc=%zu, vl=%zu\n", nc, vl);
    if XNN_UNLIKELY(nc < nr) {
      __asm__ volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl) : "r"(nc));

    }
    nc = nc - vl;

    __asm__ volatile("vle32.v v0, (%0)" : : "r"((const int32_t*)w));
    OPMVINBCAST(m0, v0); // broadcast channel-wise bias
    w = (const int32_t*) w + nr;

    __asm__ volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(vl));
    for (int k = 0; k+2 <= kc; k+=2) {
      __asm__ volatile("vle8.v v4, (%0)" : : "r"(a0));
      a0 = (const int8_t*) a0 + a_stride;
      __asm__ volatile("vle8.v v5, (%0)" : : "r"(w));
      w = (const int8_t*) w + nr;
      VOPACC(m0, v4, v5);

      __asm__ volatile("vle8.v v6, (%0)" : : "r"(a0));
      a0 = (const int8_t*) a0 + a_stride;
      __asm__ volatile("vle8.v v7, (%0)" : : "r"(w));
      w = (const int8_t*) w + nr;
      VOPACC(m0, v6, v7);
    }
    if XNN_UNLIKELY((kc%2) != 0) {
      __asm__ volatile("vle8.v v4, (%0)" : : "r"(a0));
      a0 = (const int8_t*) a0 + a_stride;
      __asm__ volatile("vle8.v v5, (%0)" : : "r"(w));
      w = (const int8_t*) w + nr;
      VOPACC(m0, v4, v5);
    }
    
    //v4 <- vscale
    __asm__ volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl));
    __asm__ volatile("vle32.v v8, (%0)" : : "r"((const float*) w));
    w = (const float*) w + nr;

    int8_t* cm = c0;
    for (size_t r=0; r<mr; r++) {
      // printf("      r=%zu\n", r);
      //v12 <- vopacc
      __asm__ volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl));
      VMV_VR(v12, r, m0); // move row r of m1 into v12
      __asm__ volatile("vfcvt.f.x.v	v12,v12");
      //v12 <- vopacc * vscale
      __asm__ volatile("vfmul.vv	v12,v12,v8");
      
      //v12 <- minmax
      __asm__ volatile("vfmax.vf	v12,v12,%0" : : "f"((float) output_min_less_zero_point));
      __asm__ volatile("vfmin.vf	v12,v12,%0" : : "f"((float) output_max_less_zero_point));
      
      __asm__ volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl));
      __asm__ volatile("vfncvt.x.f.w	v12,v12");
      __asm__ volatile("vadd.vx	v12,v12,%0" : : "r"((int16_t) output_zero_point));
      
      __asm__ volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(vl));
      __asm__ volatile("vncvt.x.x.w	v12,v12");
    
      __asm__ volatile("vse8.v	v0, (%0)" : : "r"(cm));
      cm = (int8_t*) ((uintptr_t) cm + cm_stride);
    }
    c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
    a0 = (const int8_t*)  ((uintptr_t) a0 - kc*a_stride);
  } while (nc != 0);
}
