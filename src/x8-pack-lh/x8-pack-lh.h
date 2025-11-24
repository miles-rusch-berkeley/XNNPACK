// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, size_fn, packed_offset_fn, params_type, \
                                init_params)                              \
  XNN_UKERNEL(arch_flags, ukernel, size_fn, packed_offset_fn)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, size_fn, packed_offset_fn) \
  XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, size_fn, packed_offset_fn)
#define XNN_DEFINED_UKERNEL
#endif

// arch_flags, ukernel, size_fn, packed_offset_fn

#if XNN_ENABLE_KLEIDIAI
XNN_UKERNEL(xnn_arch_arm_sme, xnn_x8_pack_lh_ukernel__neonsme2,
            xnn_x8_pack_lh_size__neonsme2, xnn_x8_pack_lh_offset__neonsme2)
#endif  // XNN_ENABLE_KLEIDIAI

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_x8_pack_lh_ukernel__16x8_rvv,
            xnn_x8_pack_lh_size__16x8_rvv, xnn_x8_pack_lh_offset__16x8_rvv)
#endif  // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
