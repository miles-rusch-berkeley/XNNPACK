// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "xnnpack/common.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/math.h"
#include "xnnpack/pack-lh.h"
#include "xnnpack/transpose.h"

// Static helper function for transpose
static void xnn_x8_transpose_ukernel__16x8_rvv(
  const uint8_t* input,
  uint8_t* output,
  size_t input_stride,
  size_t output_stride,
  size_t block_width,
  size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint8_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint8_t));

  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  assert(hardware_config != NULL);
  const size_t mr = 4 * hardware_config->vlenb / sizeof(uint8_t);
  const size_t kc = 8 * hardware_config->vlenb / sizeof(uint8_t);
  const size_t tile_height = mr;
  // Process kc in chunks of 8 (vlsseg8e8 limitation)
  const size_t tile_width = 8;
  const size_t tile_hbytes = tile_height * sizeof(uint8_t);
  const size_t tile_wbytes = tile_width * sizeof(uint8_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint8_t);

  const uint8_t* i0 = input;

  uint8_t* o0 = (uint8_t*) output;
  uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + output_stride);
  uint8_t* o2 = (uint8_t*) ((uintptr_t) o1 + output_stride);
  uint8_t* o3 = (uint8_t*) ((uintptr_t) o2 + output_stride);
  uint8_t* o4 = (uint8_t*) ((uintptr_t) o3 + output_stride);
  uint8_t* o5 = (uint8_t*) ((uintptr_t) o4 + output_stride);
  uint8_t* o6 = (uint8_t*) ((uintptr_t) o5 + output_stride);
  uint8_t* o7 = (uint8_t*) ((uintptr_t) o6 + output_stride);

  do {
    size_t bh = block_height;
    size_t vl = __riscv_vsetvl_e8m1(tile_height);
    for (; bh >= tile_height; bh -= tile_height) {
      if (block_width >= tile_width) {
        vuint8m1x8_t tuple = __riscv_vlsseg8e8_v_u8m1x8(i0, input_stride, vl);

        vuint8m1_t v_d0 = __riscv_vget_v_u8m1x8_u8m1(tuple, 0);
        __riscv_vse8_v_u8m1(o0, v_d0, vl);
        vuint8m1_t v_d1 = __riscv_vget_v_u8m1x8_u8m1(tuple, 1);
        __riscv_vse8_v_u8m1(o1, v_d1, vl);
        vuint8m1_t v_d2 = __riscv_vget_v_u8m1x8_u8m1(tuple, 2);
        __riscv_vse8_v_u8m1(o2, v_d2, vl);
        vuint8m1_t v_d3 = __riscv_vget_v_u8m1x8_u8m1(tuple, 3);
        __riscv_vse8_v_u8m1(o3, v_d3, vl);
        vuint8m1_t v_d4 = __riscv_vget_v_u8m1x8_u8m1(tuple, 4);
        __riscv_vse8_v_u8m1(o4, v_d4, vl);
        vuint8m1_t v_d5 = __riscv_vget_v_u8m1x8_u8m1(tuple, 5);
        __riscv_vse8_v_u8m1(o5, v_d5, vl);
        vuint8m1_t v_d6 = __riscv_vget_v_u8m1x8_u8m1(tuple, 6);
        __riscv_vse8_v_u8m1(o6, v_d6, vl);
        vuint8m1_t v_d7 = __riscv_vget_v_u8m1x8_u8m1(tuple, 7);
        __riscv_vse8_v_u8m1(o7, v_d7, vl);

      } else {
        switch (block_width) {
          case 7: {
            vuint8m1x7_t tuple = __riscv_vlsseg7e8_v_u8m1x7(i0, input_stride, vl);

            vuint8m1_t v_d0 = __riscv_vget_v_u8m1x7_u8m1(tuple, 0);
            __riscv_vse8_v_u8m1(o0, v_d0, vl);
            vuint8m1_t v_d1 = __riscv_vget_v_u8m1x7_u8m1(tuple, 1);
            __riscv_vse8_v_u8m1(o1, v_d1, vl);
            vuint8m1_t v_d2 = __riscv_vget_v_u8m1x7_u8m1(tuple, 2);
            __riscv_vse8_v_u8m1(o2, v_d2, vl);
            vuint8m1_t v_d3 = __riscv_vget_v_u8m1x7_u8m1(tuple, 3);
            __riscv_vse8_v_u8m1(o3, v_d3, vl);
            vuint8m1_t v_d4 = __riscv_vget_v_u8m1x7_u8m1(tuple, 4);
            __riscv_vse8_v_u8m1(o4, v_d4, vl);
            vuint8m1_t v_d5 = __riscv_vget_v_u8m1x7_u8m1(tuple, 5);
            __riscv_vse8_v_u8m1(o5, v_d5, vl);
            vuint8m1_t v_d6 = __riscv_vget_v_u8m1x7_u8m1(tuple, 6);
            __riscv_vse8_v_u8m1(o6, v_d6, vl);
            break;
          }

          case 6: {
            vuint8m1x6_t tuple = __riscv_vlsseg6e8_v_u8m1x6(i0, input_stride, vl);

            vuint8m1_t v_d0 = __riscv_vget_v_u8m1x6_u8m1(tuple, 0);
            __riscv_vse8_v_u8m1(o0, v_d0, vl);
            vuint8m1_t v_d1 = __riscv_vget_v_u8m1x6_u8m1(tuple, 1);
            __riscv_vse8_v_u8m1(o1, v_d1, vl);
            vuint8m1_t v_d2 = __riscv_vget_v_u8m1x6_u8m1(tuple, 2);
            __riscv_vse8_v_u8m1(o2, v_d2, vl);
            vuint8m1_t v_d3 = __riscv_vget_v_u8m1x6_u8m1(tuple, 3);
            __riscv_vse8_v_u8m1(o3, v_d3, vl);
            vuint8m1_t v_d4 = __riscv_vget_v_u8m1x6_u8m1(tuple, 4);
            __riscv_vse8_v_u8m1(o4, v_d4, vl);
            vuint8m1_t v_d5 = __riscv_vget_v_u8m1x6_u8m1(tuple, 5);
            __riscv_vse8_v_u8m1(o5, v_d5, vl);
            break;
          }

          case 5: {
            vuint8m1x5_t tuple = __riscv_vlsseg5e8_v_u8m1x5(i0, input_stride, vl);

            vuint8m1_t v_d0 = __riscv_vget_v_u8m1x5_u8m1(tuple, 0);
            __riscv_vse8_v_u8m1(o0, v_d0, vl);
            vuint8m1_t v_d1 = __riscv_vget_v_u8m1x5_u8m1(tuple, 1);
            __riscv_vse8_v_u8m1(o1, v_d1, vl);
            vuint8m1_t v_d2 = __riscv_vget_v_u8m1x5_u8m1(tuple, 2);
            __riscv_vse8_v_u8m1(o2, v_d2, vl);
            vuint8m1_t v_d3 = __riscv_vget_v_u8m1x5_u8m1(tuple, 3);
            __riscv_vse8_v_u8m1(o3, v_d3, vl);
            vuint8m1_t v_d4 = __riscv_vget_v_u8m1x5_u8m1(tuple, 4);
            __riscv_vse8_v_u8m1(o4, v_d4, vl);
            break;
          }

          case 4: {
            vuint8m1x4_t tuple = __riscv_vlsseg4e8_v_u8m1x4(i0, input_stride, vl);

            vuint8m1_t v_d0 = __riscv_vget_v_u8m1x4_u8m1(tuple, 0);
            __riscv_vse8_v_u8m1(o0, v_d0, vl);
            vuint8m1_t v_d1 = __riscv_vget_v_u8m1x4_u8m1(tuple, 1);
            __riscv_vse8_v_u8m1(o1, v_d1, vl);
            vuint8m1_t v_d2 = __riscv_vget_v_u8m1x4_u8m1(tuple, 2);
            __riscv_vse8_v_u8m1(o2, v_d2, vl);
            vuint8m1_t v_d3 = __riscv_vget_v_u8m1x4_u8m1(tuple, 3);
            __riscv_vse8_v_u8m1(o3, v_d3, vl);
            break;
          }

          case 3: {
            vuint8m1x3_t tuple = __riscv_vlsseg3e8_v_u8m1x3(i0, input_stride, vl);

            vuint8m1_t v_d0 = __riscv_vget_v_u8m1x3_u8m1(tuple, 0);
            __riscv_vse8_v_u8m1(o0, v_d0, vl);
            vuint8m1_t v_d1 = __riscv_vget_v_u8m1x3_u8m1(tuple, 1);
            __riscv_vse8_v_u8m1(o1, v_d1, vl);
            vuint8m1_t v_d2 = __riscv_vget_v_u8m1x3_u8m1(tuple, 2);
            __riscv_vse8_v_u8m1(o2, v_d2, vl);
            break;
          }

          case 2: {
            vuint8m1x2_t tuple = __riscv_vlsseg2e8_v_u8m1x2(i0, input_stride, vl);

            vuint8m1_t v_d0 = __riscv_vget_v_u8m1x2_u8m1(tuple, 0);
            __riscv_vse8_v_u8m1(o0, v_d0, vl);
            vuint8m1_t v_d1 = __riscv_vget_v_u8m1x2_u8m1(tuple, 1);
            __riscv_vse8_v_u8m1(o1, v_d1, vl);
            break;
          }

          case 1: {
            vuint8m1_t v_d0 = __riscv_vlse8_v_u8m1(i0, input_stride, vl);
            __riscv_vse8_v_u8m1(o0, v_d0, vl);
            break;
          }

          default:
            XNN_UNREACHABLE;
        }
      }

      i0 = (uint8_t*) ((uintptr_t) i0 + input_offset);
      o7 = (uint8_t*) ((uintptr_t) o7 + tile_hbytes);
      o6 = (uint8_t*) ((uintptr_t) o6 + tile_hbytes);
      o5 = (uint8_t*) ((uintptr_t) o5 + tile_hbytes);
      o4 = (uint8_t*) ((uintptr_t) o4 + tile_hbytes);
      o3 = (uint8_t*) ((uintptr_t) o3 + tile_hbytes);
      o2 = (uint8_t*) ((uintptr_t) o2 + tile_hbytes);
      o1 = (uint8_t*) ((uintptr_t) o1 + tile_hbytes);
      o0 = (uint8_t*) ((uintptr_t) o0 + tile_hbytes);
    }

    if (bh != 0) {
      const uint8_t* i = i0;
      vl = __riscv_vsetvl_e8m1(bh);
      if (block_width >= tile_width) {
        vuint8m1x8_t tuple = __riscv_vlsseg8e8_v_u8m1x8(i, input_stride, vl);

        vuint8m1_t v_d0 = __riscv_vget_v_u8m1x8_u8m1(tuple, 0);
        __riscv_vse8_v_u8m1(o0, v_d0, vl);
        vuint8m1_t v_d1 = __riscv_vget_v_u8m1x8_u8m1(tuple, 1);
        __riscv_vse8_v_u8m1(o1, v_d1, vl);
        vuint8m1_t v_d2 = __riscv_vget_v_u8m1x8_u8m1(tuple, 2);
        __riscv_vse8_v_u8m1(o2, v_d2, vl);
        vuint8m1_t v_d3 = __riscv_vget_v_u8m1x8_u8m1(tuple, 3);
        __riscv_vse8_v_u8m1(o3, v_d3, vl);
        vuint8m1_t v_d4 = __riscv_vget_v_u8m1x8_u8m1(tuple, 4);
        __riscv_vse8_v_u8m1(o4, v_d4, vl);
        vuint8m1_t v_d5 = __riscv_vget_v_u8m1x8_u8m1(tuple, 5);
        __riscv_vse8_v_u8m1(o5, v_d5, vl);
        vuint8m1_t v_d6 = __riscv_vget_v_u8m1x8_u8m1(tuple, 6);
        __riscv_vse8_v_u8m1(o6, v_d6, vl);
        vuint8m1_t v_d7 = __riscv_vget_v_u8m1x8_u8m1(tuple, 7);
        __riscv_vse8_v_u8m1(o7, v_d7, vl);
      } else {
        switch(block_width) {
          case 7: {
            vuint8m1x7_t tuple = __riscv_vlsseg7e8_v_u8m1x7(i, input_stride, vl);

            vuint8m1_t v_d0 = __riscv_vget_v_u8m1x7_u8m1(tuple, 0);
            __riscv_vse8_v_u8m1(o0, v_d0, vl);
            vuint8m1_t v_d1 = __riscv_vget_v_u8m1x7_u8m1(tuple, 1);
            __riscv_vse8_v_u8m1(o1, v_d1, vl);
            vuint8m1_t v_d2 = __riscv_vget_v_u8m1x7_u8m1(tuple, 2);
            __riscv_vse8_v_u8m1(o2, v_d2, vl);
            vuint8m1_t v_d3 = __riscv_vget_v_u8m1x7_u8m1(tuple, 3);
            __riscv_vse8_v_u8m1(o3, v_d3, vl);
            vuint8m1_t v_d4 = __riscv_vget_v_u8m1x7_u8m1(tuple, 4);
            __riscv_vse8_v_u8m1(o4, v_d4, vl);
            vuint8m1_t v_d5 = __riscv_vget_v_u8m1x7_u8m1(tuple, 5);
            __riscv_vse8_v_u8m1(o5, v_d5, vl);
            vuint8m1_t v_d6 = __riscv_vget_v_u8m1x7_u8m1(tuple, 6);
            __riscv_vse8_v_u8m1(o6, v_d6, vl);
            break;
          }
          case 6: {
            vuint8m1x6_t tuple = __riscv_vlsseg6e8_v_u8m1x6(i, input_stride, vl);

            vuint8m1_t v_d0 = __riscv_vget_v_u8m1x6_u8m1(tuple, 0);
            __riscv_vse8_v_u8m1(o0, v_d0, vl);
            vuint8m1_t v_d1 = __riscv_vget_v_u8m1x6_u8m1(tuple, 1);
            __riscv_vse8_v_u8m1(o1, v_d1, vl);
            vuint8m1_t v_d2 = __riscv_vget_v_u8m1x6_u8m1(tuple, 2);
            __riscv_vse8_v_u8m1(o2, v_d2, vl);
            vuint8m1_t v_d3 = __riscv_vget_v_u8m1x6_u8m1(tuple, 3);
            __riscv_vse8_v_u8m1(o3, v_d3, vl);
            vuint8m1_t v_d4 = __riscv_vget_v_u8m1x6_u8m1(tuple, 4);
            __riscv_vse8_v_u8m1(o4, v_d4, vl);
            vuint8m1_t v_d5 = __riscv_vget_v_u8m1x6_u8m1(tuple, 5);
            __riscv_vse8_v_u8m1(o5, v_d5, vl);
            break;
          }
          case 5: {
            vuint8m1x5_t tuple = __riscv_vlsseg5e8_v_u8m1x5(i, input_stride, vl);

            vuint8m1_t v_d0 = __riscv_vget_v_u8m1x5_u8m1(tuple, 0);
            __riscv_vse8_v_u8m1(o0, v_d0, vl);
            vuint8m1_t v_d1 = __riscv_vget_v_u8m1x5_u8m1(tuple, 1);
            __riscv_vse8_v_u8m1(o1, v_d1, vl);
            vuint8m1_t v_d2 = __riscv_vget_v_u8m1x5_u8m1(tuple, 2);
            __riscv_vse8_v_u8m1(o2, v_d2, vl);
            vuint8m1_t v_d3 = __riscv_vget_v_u8m1x5_u8m1(tuple, 3);
            __riscv_vse8_v_u8m1(o3, v_d3, vl);
            vuint8m1_t v_d4 = __riscv_vget_v_u8m1x5_u8m1(tuple, 4);
            __riscv_vse8_v_u8m1(o4, v_d4, vl);
            break;
          }
          case 4: {
            vuint8m1x4_t tuple = __riscv_vlsseg4e8_v_u8m1x4(i, input_stride, vl);

            vuint8m1_t v_d0 = __riscv_vget_v_u8m1x4_u8m1(tuple, 0);
            __riscv_vse8_v_u8m1(o0, v_d0, vl);
            vuint8m1_t v_d1 = __riscv_vget_v_u8m1x4_u8m1(tuple, 1);
            __riscv_vse8_v_u8m1(o1, v_d1, vl);
            vuint8m1_t v_d2 = __riscv_vget_v_u8m1x4_u8m1(tuple, 2);
            __riscv_vse8_v_u8m1(o2, v_d2, vl);
            vuint8m1_t v_d3 = __riscv_vget_v_u8m1x4_u8m1(tuple, 3);
            __riscv_vse8_v_u8m1(o3, v_d3, vl);
            break;
          }
          case 3: {
            vuint8m1x3_t tuple = __riscv_vlsseg3e8_v_u8m1x3(i, input_stride, vl);

            vuint8m1_t v_d0 = __riscv_vget_v_u8m1x3_u8m1(tuple, 0);
            __riscv_vse8_v_u8m1(o0, v_d0, vl);
            vuint8m1_t v_d1 = __riscv_vget_v_u8m1x3_u8m1(tuple, 1);
            __riscv_vse8_v_u8m1(o1, v_d1, vl);
            vuint8m1_t v_d2 = __riscv_vget_v_u8m1x3_u8m1(tuple, 2);
            __riscv_vse8_v_u8m1(o2, v_d2, vl);
            break;
          }
          case 2: {
            vuint8m1x2_t tuple = __riscv_vlsseg2e8_v_u8m1x2(i, input_stride, vl);

            vuint8m1_t v_d0 = __riscv_vget_v_u8m1x2_u8m1(tuple, 0);
            __riscv_vse8_v_u8m1(o0, v_d0, vl);
            vuint8m1_t v_d1 = __riscv_vget_v_u8m1x2_u8m1(tuple, 1);
            __riscv_vse8_v_u8m1(o1, v_d1, vl);
            break;
          }

          case 1: {
            vuint8m1_t v_d0 = __riscv_vlse8_v_u8m1(i, input_stride, vl);
            __riscv_vse8_v_u8m1(o0, v_d0, vl);
            break;
          }

          default:
            XNN_UNREACHABLE;
        }
      }

      if (bh & 8) {
        o7 += 8;
        o6 += 8;
        o5 += 8;
        o4 += 8;
        o3 += 8;
        o2 += 8;
        o1 += 8;
        o0 += 8;
        i = (uint8_t*) ((uintptr_t) i + input_stride * 8);
      }
      if (bh & 4) {
        o7 += 4;
        o6 += 4;
        o5 += 4;
        o4 += 4;
        o3 += 4;
        o2 += 4;
        o1 += 4;
        o0 += 4;
        i = (uint8_t*) ((uintptr_t) i + input_stride * 4);
      }
      if (bh & 2) {
        o7 += 2;
        o6 += 2;
        o5 += 2;
        o4 += 2;
        o3 += 2;
        o2 += 2;
        o1 += 2;
        o0 += 2;
        i = (uint8_t*) ((uintptr_t) i + input_stride * 2);
      }
    }

    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);

    o0 = (uint8_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint8_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint8_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint8_t*) ((uintptr_t) o3 + output_reset);
    o4 = (uint8_t*) ((uintptr_t) o4 + output_reset);
    o5 = (uint8_t*) ((uintptr_t) o5 + output_reset);
    o6 = (uint8_t*) ((uintptr_t) o6 + output_reset);
    o7 = (uint8_t*) ((uintptr_t) o7 + output_reset);

    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}

void xnn_x8_pack_lh_ukernel__16x8_rvv(size_t m, size_t k, size_t mr_packed,
                                  size_t kr, size_t sr, size_t m_idx_start,
                                  const int8_t* XNN_RESTRICT lhs,
                                  size_t lhs_stride,
                                  void* XNN_RESTRICT lhs_packed) {
  if (mr_packed == 1) {
    memcpy(lhs_packed, lhs, sizeof(int8_t) * k);
  } else {
    // For RISC-V, use transpose kernel for packing
    const size_t m_end = m_idx_start + mr_packed;
    const size_t m_actual = (m_end > m) ? (m - m_idx_start) : mr_packed;
    
    const int8_t* input = lhs + m_idx_start * lhs_stride;
    int8_t* output = (int8_t*) lhs_packed;
    
    // Call transpose kernel: transpose k columns x m_actual rows
    xnn_x8_transpose_ukernel__16x8_rvv(
      (const uint8_t*) input,  // input
      (uint8_t*) output,        // output
      lhs_stride,               // input_stride
      k,                        // output_stride (packed data is contiguous)
      k,                        // block_width
      m_actual);                // block_height
  }
}

size_t xnn_x8_pack_lh_size__16x8_rvv(size_t m, size_t k, size_t mr_packed,
                                size_t kr, size_t sr) {
  if (mr_packed == 1) {
    return m * sizeof(int8_t) * k;
  } else {
    const size_t num_tiles = round_up(m, mr_packed) / mr_packed;
    return num_tiles * mr_packed * sizeof(int8_t) * k;
  }
}

size_t xnn_x8_pack_lh_offset__16x8_rvv(size_t m, size_t k, size_t mr_packed,
                                  size_t kr, size_t sr) {
  if (mr_packed == 1) {
    return m * sizeof(int8_t) * k;
  } else {
    const size_t tile_idx = m / mr_packed;
    return tile_idx * mr_packed * sizeof(int8_t) * k;
  }
}

size_t xnn_x8_pack_lh_size__rvv(size_t m, size_t k, size_t mr_packed,
                               size_t kr, size_t sr) {
  return xnn_x8_pack_lh_size__16x8_rvv(m, k, mr_packed, kr, sr);
}

size_t xnn_x8_pack_lh_offset__rvv(size_t m, size_t k, size_t mr_packed,
                                 size_t kr, size_t sr) {
  return xnn_x8_pack_lh_offset__16x8_rvv(m, k, mr_packed, kr, sr);
}

