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
  assert(nc != 0);
  assert(kc != 0);
  printf("xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x4v__rvv: mr=%zu, nc=%zu, kc=%zu\n", mr, nc, kc);
  printf("a_stride=%zu, cm_stride=%zu, cn_stride=%zu\n", a_stride, cm_stride, cn_stride);
  int8_t* c0 = c;
  size_t nr;
  __asm__ volatile("vsetvli %0, zero, e32, m4, ta, ma" : "=r"(nr));
  const int32_t output_min_less_zero_point = (int32_t) params->fp32_scalar.output_min - (int32_t) params->fp32_scalar.output_zero_point;
  const int32_t output_max_less_zero_point = (int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point;
  const int32_t output_zero_point = params->fp32_scalar.output_zero_point;
  while (nc >= 2*nr) {
    printf("nc=%zu  w=%p, \n", nc, w);
    nc = nc - 2*nr;
    __asm__ volatile("vsetvli zero, zero, e32, m8, ta, ma" : : "r"(2*nr));
    __asm__ volatile("vle32.v v0, (%0)" : : "r"((const int32_t*)w));
    __asm__ volatile("vsetvli zero, zero, e32, m4, ta, ma" : : "r"(nr));
    OPMVINBCAST(m0, v0); // broadcast channel-wise bias
    OPMVINBCAST(m1, v4); // broadcast channel-wise bias
    w = (const int32_t*) w + nr;
    
    int8_t* am = a;
    __asm__ volatile("vsetvli zero, %0, e8, m2, ta, ma" : : "r"(nr));
    for (int k = 0; k+2 <= kc; k+=2) {
      __asm__ volatile("vle8.v v8, (%0)" : : "r"(am)); // TODO: add mask for vl/2
      __asm__ volatile("vle8.v v10, (%0)" : : "r"(w));
      VOPACC(m0, v8, v9);
      __asm__ volatile("vle8.v v10, (%0)" : : "r"(w2));
      VOPACC(m1, v8, v10);
      am = (const int8_t*) am + a_stride;
      w = (const int8_t*) w + nr;
      w2 = (const int8_t*) w2 + nr;
      // k+1
      __asm__ volatile("vle8.v v11, (%0)" : : "r"(am));
      __asm__ volatile("vle8.v v12, (%0)" : : "r"(w));
      VOPACC(m0, v11, v12);
      __asm__ volatile("vle8.v v13, (%0)" : : "r"(w2));
      VOPACC(m1, v11, v13);
      am = (const int8_t*) am + a_stride;
      w = (const int8_t*) w + nr;
      w2 = (const int8_t*) w2 + nr;
    }
    if XNN_UNLIKELY((kc%2) != 0) {
      __asm__ volatile("vle8.v v8, (%0)" : : "r"(am));
      __asm__ volatile("vle8.v v9, (%0)" : : "r"(w));
      VOPACC(m0, v8, v9);
      __asm__ volatile("vle8.v v10, (%0)" : : "r"(w2));
      VOPACC(m1, v8, v10);
      w = (const int8_t*) w + nr;
      w2 = (const int8_t*) w2 + nr;
    }
  
    //v16 <- vscale
    __asm__ volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(nr));
    __asm__ volatile("vle32.v v16, (%0)" : : "r"((const float*) w));
    __asm__ volatile("vle32.v v20, (%0)" : : "r"((const float*) w2));
    w = (const float*) w + nr;
    w2 = (const float*) w2 + nr;

    int8_t* cm = c0;
    for (size_t r=0; r<mr; r++) {
      // printf("      r=%zu\n", r);
      //v12 <- vopacc
      __asm__ volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(nr));
      VMV_VR(v24, r, m0); // move row r of m1 into v12
      VMV_VR(v28, r, m1); // move row r of m1 into v12
      __asm__ volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(2*nr));
      __asm__ volatile("vfcvt.f.x.v	v24,v24");
      //v24 <- vopacc * vscale
      __asm__ volatile("vfmul.vv	v24,v24,v16");
      //v24 <- minmax
      __asm__ volatile("vfmax.vf	v24,v24,%0" : : "f"((float) output_min_less_zero_point));
      __asm__ volatile("vfmin.vf	v24,v24,%0" : : "f"((float) output_max_less_zero_point));
      __asm__ volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(2*nr));
      __asm__ volatile("vfncvt.x.f.w	v24,v24");
      __asm__ volatile("vadd.vx	v24,v24,%0" : : "r"((int16_t) output_zero_point));
      __asm__ volatile("vsetvli zero, %0, e8, m2, ta, ma" : : "r"(2*nr));
      __asm__ volatile("vncvt.x.x.w	v24,v24"); 

      __asm__ volatile("vse8.v	v24, (%0)" : : "r"(cm));
      cm = (int8_t*) ((uintptr_t) cm + cm_stride);
    }
    c0 = (int8_t*) ((uintptr_t) c0 + 2*cn_stride);
    printf("  w=%p, w2=%p\n", w, w2);
    printf("  c0=%p, cm_stride=%zu, cn_stride=%zu\n", c0, cm_stride, cn_stride);
  }
  printf("nc=%zu  w=%p, \n", nc, w);
}
