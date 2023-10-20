// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023, Microchip Technology Inc
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_RVV_H
#define EIGEN_PACKET_MATH_RVV_H

#include "../../InternalHeaderCheck.h"
#include <float.h>

namespace Eigen
{
namespace internal
{
#ifndef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 8
#endif

#ifndef EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#define EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#endif

#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS 32

#ifndef MAX_INT32
#define MAX_INT32   (INT_MAX)
#endif

#ifndef MIN_INT32
#define MIN_INT32   (INT_MIN)
#endif

#ifndef MAX_FLOAT32
#define MAX_FLOAT32  (FLT_MAX)
#endif

#ifndef MIN_FLOAT32
#define MIN_FLOAT32  (FLT_MIN)
#endif

template <typename Scalar, int VectorLength>
struct rvv_packet_size_selector {
  enum { size = VectorLength / (sizeof(Scalar) * CHAR_BIT) };
};

/********************************* int32 **************************************/
typedef vint32m1_t PacketXi __attribute__((riscv_rvv_vector_bits(__RISCV_V_VECTOR_BITS_MIN)));



template <>
struct packet_traits<numext::int32_t> : default_packet_traits {
  typedef PacketXi type;
  typedef PacketXi half;  // Half not implemented yet
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<numext::int32_t, EIGEN_RISCV_RVV_VL>::size,
    HasHalfPacket = 0,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0  // Not implemented
  };
};

template <>
struct unpacket_traits<PacketXi> {
  typedef numext::int32_t type;
  typedef PacketXi half;  // Half not yet implemented
  enum {
    size = rvv_packet_size_selector<numext::int32_t, EIGEN_RISCV_RVV_VL>::size,
    alignment = Aligned64,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
EIGEN_STRONG_INLINE void prefetch<numext::int32_t>(const numext::int32_t* addr)
{
    (void)addr;
}

template <>
EIGEN_STRONG_INLINE PacketXi pset1<PacketXi>(const numext::int32_t& from)
{
    return __riscv_vmv_v_x_i32m1(from, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi plset<PacketXi>(const numext::int32_t& a)
{
    vint32m1_t index  = __riscv_vid_v_i32m1(packet_traits<numext::int32_t>::size);
    return __riscv_vadd_vx_i32m1(index, a, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi padd<PacketXi>(const PacketXi& a, const PacketXi& b)
{
  return __riscv_vadd_vv_i32m1(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi psub<PacketXi>(const PacketXi& a, const PacketXi& b)
{
  return __riscv_vsub_vv_i32m1(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pnegate(const PacketXi& a)
{
  return __riscv_vneg_v_i32m1(a, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pconj(const PacketXi& a)
{

  return a;
}

template <>
EIGEN_STRONG_INLINE PacketXi pmul<PacketXi>(const PacketXi& a, const PacketXi& b)
{
  return __riscv_vmul_vv_i32m1(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pdiv<PacketXi>(const PacketXi& a, const PacketXi& b)
{
  return __riscv_vdiv_vv_i32m1(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pmadd(const PacketXi& a, const PacketXi& b, const PacketXi& c)
{
  return __riscv_vmacc_vv_i32m1(c, a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pmin<PacketXi>(const PacketXi& a, const PacketXi& b)
{
  return __riscv_vmin_vv_i32m1(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pmax<PacketXi>(const PacketXi& a, const PacketXi& b)
{
  return __riscv_vmax_vv_i32m1(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pcmp_le<PacketXi>(const PacketXi& a, const PacketXi& b)
{
    vbool32_t cmp = __riscv_vmsle_vv_i32m1_b32(a, b, packet_traits<numext::int32_t>::size);
    vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, packet_traits<numext::int32_t>::size);
    return __riscv_vmerge_vxm_i32m1(vzero, 0xffffffff, cmp, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pcmp_lt<PacketXi>(const PacketXi& a, const PacketXi& b)
{
    vbool32_t cmp = __riscv_vmslt_vv_i32m1_b32(a, b, packet_traits<numext::int32_t>::size);
    vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, packet_traits<numext::int32_t>::size);
    return __riscv_vmerge_vxm_i32m1(vzero, 0xffffffff, cmp, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pcmp_eq<PacketXi>(const PacketXi& a, const PacketXi& b)
{
    vbool32_t cmp = __riscv_vmseq_vv_i32m1_b32(a, b, packet_traits<numext::int32_t>::size);
    vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, packet_traits<numext::int32_t>::size);
    return __riscv_vmerge_vxm_i32m1(vzero, 0xffffffff, cmp, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi ptrue<PacketXi>(const PacketXi& /*a*/)
{
    // according test(packetmath_7) result
    // each bit of element should be set to 1
    return __riscv_vmv_v_x_i32m1(0xffffffff, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pzero<PacketXi>(const PacketXi& /*a*/)
{
    return __riscv_vmv_v_x_i32m1(0, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pand<PacketXi>(const PacketXi& a, const PacketXi& b)
{
    return __riscv_vand_vv_i32m1(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi por<PacketXi>(const PacketXi& a, const PacketXi& b)
{
    return __riscv_vor_vv_i32m1(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pxor<PacketXi>(const PacketXi& a, const PacketXi& b)
{
    return __riscv_vxor_vv_i32m1(a, b, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pandnot<PacketXi>(const PacketXi& a, const PacketXi& b)
{
    vint32m1_t bnot = __riscv_vnot_v_i32m1(b, packet_traits<numext::int32_t>::size);
    return __riscv_vand_vv_i32m1(a, bnot, packet_traits<numext::int32_t>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketXi parithmetic_shift_right(PacketXi a)
{
    return __riscv_vsra_vx_i32m1(a, N, packet_traits<numext::int32_t>::size);
}

template <int N>
EIGEN_STRONG_INLINE PacketXi plogical_shift_right(PacketXi a)
{
    vuint32m1_t b = __riscv_vreinterpret_v_i32m1_u32m1(a);
    b = __riscv_vsrl_vx_u32m1(b, N, packet_traits<numext::int32_t>::size);
    return __riscv_vreinterpret_v_u32m1_i32m1(b);
}

template <int N>
EIGEN_STRONG_INLINE PacketXi plogical_shift_left(PacketXi a)
{
    return __riscv_vsll_vx_i32m1(a, N, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pload<PacketXi>(const numext::int32_t* from)
{
    EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle32_v_i32m1(from, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi ploadu<PacketXi>(const numext::int32_t* from)
{
    EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle32_v_i32m1(from, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi ploaddup<PacketXi>(const numext::int32_t* from)
{

    vuint32m1_t indexes = __riscv_vid_v_u32m1(packet_traits<numext::int32_t>::size);
    indexes = __riscv_vdivu_vx_u32m1(indexes, 2, packet_traits<numext::int32_t>::size);
    indexes = __riscv_vmul_vx_u32m1(indexes, sizeof(int32_t), packet_traits<numext::int32_t>::size);

    return __riscv_vloxei32_v_i32m1(from, indexes, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi ploadquad<PacketXi>(const numext::int32_t* from)
{

    vuint32m1_t indexes = __riscv_vid_v_u32m1(packet_traits<numext::int32_t>::size);
    indexes = __riscv_vdivu_vx_u32m1(indexes, 4, packet_traits<numext::int32_t>::size);
    indexes = __riscv_vmul_vx_u32m1(indexes, sizeof(int32_t), packet_traits<numext::int32_t>::size);

    return __riscv_vloxei32_v_i32m1(from, indexes, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<numext::int32_t>(numext::int32_t* to, const PacketXi& from)
{
    EIGEN_DEBUG_ALIGNED_STORE __riscv_vse32_v_i32m1(to, from, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<numext::int32_t>(numext::int32_t* to, const PacketXi& from)
{
    EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse32_v_i32m1(to, from, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketXi pgather<numext::int32_t, PacketXi>(const numext::int32_t* from, Index stride)
{
    return __riscv_vlse32_v_i32m1(from, stride*sizeof(int32_t), packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<numext::int32_t, PacketXi>(numext::int32_t* to, const PacketXi& from, Index stride)
{
    return __riscv_vsse32_v_i32m1(to, stride*sizeof(int32_t), from, packet_traits<numext::int32_t>::size);
}

// TODO
template <>
EIGEN_STRONG_INLINE numext::int32_t pfirst<PacketXi>(const PacketXi& a)
{
    numext::int32_t temp[packet_traits<numext::int32_t>::size];
    __riscv_vse32_v_i32m1(temp, a, packet_traits<numext::int32_t>::size);
    return temp[0];
}

template <>
EIGEN_STRONG_INLINE PacketXi preverse(const PacketXi& a)
{
    numext::int32_t temp[packet_traits<numext::int32_t>::size];
    __riscv_vse32_v_i32m1(temp, a, packet_traits<numext::int32_t>::size);
    return __riscv_vlse32_v_i32m1(temp + packet_traits<numext::int32_t>::size - 1, sizeof(int32_t) * (-1), packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pabs(const PacketXi& a)
{
    PacketXi b = pnegate(a);
    vbool32_t cmp = __riscv_vmsle_vx_i32m1_b32(a, 0, packet_traits<numext::int32_t>::size);
    return __riscv_vmerge_vvm_i32m1(a, b, cmp, packet_traits<numext::int32_t>::size);
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux<PacketXi>(const PacketXi& a)
{
    vint32m1_t sum = __riscv_vmv_v_x_i32m1(0, packet_traits<numext::int32_t>::size);
    sum = __riscv_vredsum_vs_i32m1_i32m1(a, sum, packet_traits<numext::int32_t>::size);
    return static_cast<numext::int32_t>(__riscv_vmv_x_s_i32m1_i32(sum));
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_mul<PacketXi>(const PacketXi& a)
{
    int32_t re = 1;
    int32_t tempa[packet_traits<numext::int32_t>::size];
    __riscv_vse32_v_i32m1(tempa, a, packet_traits<numext::int32_t>::size);

    for (int i = 0; i < packet_traits<numext::int32_t>::size; i++)
    {
        re = re * tempa[i];
    }
    return re;
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_min<PacketXi>(const PacketXi& a)
{
    vint32m1_t vmax = __riscv_vmv_v_x_i32m1(MAX_INT32, packet_traits<numext::int32_t>::size);
    vmax = __riscv_vredmin_vs_i32m1_i32m1(a, vmax, packet_traits<numext::int32_t>::size);
    return static_cast<numext::int32_t>(__riscv_vmv_x_s_i32m1_i32(vmax));
}

template <>
EIGEN_STRONG_INLINE numext::int32_t predux_max<PacketXi>(const PacketXi& a)
{
    vint32m1_t vmin = __riscv_vmv_v_x_i32m1(MIN_INT32, packet_traits<numext::int32_t>::size);

    vmin = __riscv_vredmax_vs_i32m1_i32m1(a, vmin, packet_traits<numext::int32_t>::size);
    return static_cast<numext::int32_t>(__riscv_vmv_x_s_i32m1_i32(vmin));
}

template <int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketXi, N>& kernel) {
    int buffer[packet_traits<int32_t>::size * N] = {0};
    int i = 0;
    int32_t stride = sizeof(int32_t) * N;
    for (i = 0; i < N; i++) 
    {
        __riscv_vsse32_v_i32m1(buffer + i, stride, kernel.packet[i], packet_traits<int32_t>::size);
    }
    
    for (i = 0; i < N; i++) 
    {
        kernel.packet[i] = __riscv_vle32_v_i32m1(buffer + i * packet_traits<int32_t>::size, packet_traits<int32_t>::size);
    }

}

/********************************* float32 ************************************/
typedef vfloat32m1_t PacketXf __attribute__((riscv_rvv_vector_bits(__RISCV_V_VECTOR_BITS_MIN)));

template <>
struct packet_traits<float> : default_packet_traits {
  typedef PacketXf type;
  typedef PacketXf half;

  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = rvv_packet_size_selector<float, EIGEN_RISCV_RVV_VL>::size,
    HasHalfPacket = 0,

    HasAdd = 1,
    HasSub = 1,
    HasShift = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 1,
    HasMin = 1,
    HasMax = 1,
    HasConj = 1,
    HasSetLinear = 0,
    HasBlend = 0,
    HasReduxp = 0,  // Not implemented in RVV

    HasDiv = 1,
    HasFloor = 1,

    HasSin = EIGEN_FAST_MATH,
    HasCos = EIGEN_FAST_MATH,
    HasLog = 1,
    HasExp = 1,
    HasSqrt = 1,
    HasReciprocal = 0, //1,Not so accurate to pass test.
    HasTanh = EIGEN_FAST_MATH,
    HasErf = EIGEN_FAST_MATH
  };
};

template <>
struct unpacket_traits<PacketXf> {
  typedef float type;
  typedef PacketXf half;  // Half not yet implemented
  typedef PacketXi integer_packet;

  enum {
    size = rvv_packet_size_selector<float, EIGEN_RISCV_RVV_VL>::size,
    alignment = Aligned64,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
EIGEN_STRONG_INLINE PacketXf pset1<PacketXf>(const float& from)
{
    return __riscv_vfmv_v_f_f32m1(from, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pset1frombits<PacketXf>(numext::uint32_t from)
{
    vuint32m1_t a;
    a = __riscv_vmv_v_x_u32m1(from, packet_traits<float>::size);
    return __riscv_vreinterpret_v_u32m1_f32m1(a);
}

template <>
EIGEN_STRONG_INLINE PacketXf plset<PacketXf>(const float& a)
{
    vuint32m1_t index  = __riscv_vid_v_u32m1(packet_traits<float>::size);
    vfloat32m1_t findex = __riscv_vfcvt_f_xu_v_f32m1(index, packet_traits<float>::size);

    return __riscv_vfadd_vf_f32m1(findex, a, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf padd<PacketXf>(const PacketXf& a, const PacketXf& b)
{
    return __riscv_vfadd_vv_f32m1(a, b, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf psub<PacketXf>(const PacketXf& a, const PacketXf& b)
{
    return __riscv_vfsub_vv_f32m1(a, b, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pnegate(const PacketXf& a)
{
    return __riscv_vfneg_v_f32m1(a, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pconj(const PacketXf& a)
{
  return a;
}

template <>
EIGEN_STRONG_INLINE PacketXf pmul<PacketXf>(const PacketXf& a, const PacketXf& b)
{
    return __riscv_vfmul_vv_f32m1(a, b, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pdiv<PacketXf>(const PacketXf& a, const PacketXf& b)
{
    return __riscv_vfdiv_vv_f32m1(a, b, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmadd(const PacketXf& a, const PacketXf& b, const PacketXf& c)
{
    return __riscv_vfmacc_vv_f32m1(c, a, b, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmin<PacketXf>(const PacketXf& a, const PacketXf& b)
{
    vbool32_t cmp = __riscv_vmflt_vv_f32m1_b32(b, a, packet_traits<float>::size);
    return __riscv_vmerge_vvm_f32m1(a, b, cmp, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmin<PropagateNaN, PacketXf>(const PacketXf& a, const PacketXf& b)
{

    vbool32_t a_NaN  = __riscv_vmfne_vv_f32m1_b32(a, a, packet_traits<float>::size);
    vbool32_t b_NaN  = __riscv_vmfne_vv_f32m1_b32(b, b, packet_traits<float>::size);

    vbool32_t ab_or_NaN = __riscv_vmor_mm_b32(a_NaN, b_NaN, packet_traits<float>::size);
    vbool32_t  ab_not_NaN = __riscv_vmnot_m_b32(ab_or_NaN, packet_traits<float>::size);

    // if a[i] is NaN, return a
    vfloat32m1_t re = __riscv_vmerge_vvm_f32m1(b, a, a_NaN, packet_traits<float>::size);
    // if b[i] is NaN, return b
    re = __riscv_vmerge_vvm_f32m1(re, b, b_NaN, packet_traits<float>::size);
    // if a & b are not NaN, get min
    vbool32_t cmp = __riscv_vmflt_vv_f32m1_b32_m(ab_not_NaN, b, a, packet_traits<float>::size);
    vfloat32m1_t numbers = __riscv_vmerge_vvm_f32m1(a, b, cmp, packet_traits<float>::size);

    // save not NaN to result.
    return __riscv_vmerge_vvm_f32m1(re, numbers, ab_not_NaN, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmin<PropagateNumbers, PacketXf>(const PacketXf& a, const PacketXf& b)
{

    vbool32_t a_NaN  = __riscv_vmfne_vv_f32m1_b32(a, a, packet_traits<float>::size);

    vbool32_t b_NaN  = __riscv_vmfne_vv_f32m1_b32(b, b, packet_traits<float>::size);

    vbool32_t ab_or_NaN = __riscv_vmor_mm_b32(a_NaN, b_NaN, packet_traits<float>::size);
    vbool32_t  ab_not_NaN = __riscv_vmnot_m_b32(ab_or_NaN, packet_traits<float>::size);

    // if a[i] is NaN, return b
    vfloat32m1_t re = __riscv_vmerge_vvm_f32m1(a, b, a_NaN, packet_traits<float>::size);
    // if b[i] is NaN, return a
    re = __riscv_vmerge_vvm_f32m1(b, re, b_NaN, packet_traits<float>::size);
    // if a & b are not NaN, get min
    vbool32_t cmp = __riscv_vmflt_vv_f32m1_b32_m(ab_not_NaN, b, a, packet_traits<float>::size);
    vfloat32m1_t numbers = __riscv_vmerge_vvm_f32m1(a, b, cmp, packet_traits<float>::size);

    // save not NaN to result.
    return __riscv_vmerge_vvm_f32m1(re, numbers, ab_not_NaN, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmax<PacketXf>(const PacketXf& a, const PacketXf& b)
{
    vbool32_t cmp = __riscv_vmflt_vv_f32m1_b32(a, b, packet_traits<float>::size);
    return __riscv_vmerge_vvm_f32m1(a, b, cmp, packet_traits<float>::size);
}


template <>
EIGEN_STRONG_INLINE PacketXf pmax<PropagateNaN, PacketXf>(const PacketXf& a, const PacketXf& b)
{

    vbool32_t a_NaN  = __riscv_vmfne_vv_f32m1_b32(a, a, packet_traits<float>::size);
    vbool32_t b_NaN  = __riscv_vmfne_vv_f32m1_b32(b, b, packet_traits<float>::size);

    vbool32_t ab_or_NaN = __riscv_vmor_mm_b32(a_NaN, b_NaN, packet_traits<float>::size);
    vbool32_t  ab_not_NaN = __riscv_vmnot_m_b32(ab_or_NaN, packet_traits<float>::size);

    // if a[i] is NaN, return a
    vfloat32m1_t re = __riscv_vmerge_vvm_f32m1(b, a, a_NaN, packet_traits<float>::size);
    // if b[i] is NaN, return b
    re = __riscv_vmerge_vvm_f32m1(re, b, b_NaN, packet_traits<float>::size);
    // if a & b are not NaN, get min
    vbool32_t cmp = __riscv_vmflt_vv_f32m1_b32_m(ab_not_NaN, a, b, packet_traits<float>::size);
    vfloat32m1_t numbers = __riscv_vmerge_vvm_f32m1(a, b, cmp, packet_traits<float>::size);

    // save not NaN to result.
    return __riscv_vmerge_vvm_f32m1(re, numbers, ab_not_NaN, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pmax<PropagateNumbers, PacketXf>(const PacketXf& a, const PacketXf& b)
{
    vbool32_t a_NaN  = __riscv_vmfne_vv_f32m1_b32(a, a, packet_traits<float>::size);

    vbool32_t b_NaN  = __riscv_vmfne_vv_f32m1_b32(b, b, packet_traits<float>::size);

    vbool32_t ab_or_NaN = __riscv_vmor_mm_b32(a_NaN, b_NaN, packet_traits<float>::size);
    vbool32_t  ab_not_NaN = __riscv_vmnot_m_b32(ab_or_NaN, packet_traits<float>::size);

    // if a[i] is NaN, return b
    vfloat32m1_t re = __riscv_vmerge_vvm_f32m1(a, b, a_NaN, packet_traits<float>::size);
    // if b[i] is NaN, return a
    re = __riscv_vmerge_vvm_f32m1(b, re, b_NaN, packet_traits<float>::size);
    // if a & b are not NaN, get min
    vbool32_t cmp = __riscv_vmfgt_vv_f32m1_b32_m(ab_not_NaN, b, a, packet_traits<float>::size);
    vfloat32m1_t numbers = __riscv_vmerge_vvm_f32m1(a, b, cmp, packet_traits<float>::size);

    // save not NaN to result.
    return __riscv_vmerge_vvm_f32m1(re, numbers, ab_not_NaN, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pcmp_le<PacketXf>(const PacketXf& a, const PacketXf& b)
{
    float all_one;
    memset(&all_one, 0xff, sizeof(all_one));
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0, packet_traits<float>::size);
    vbool32_t cmp = __riscv_vmfle_vv_f32m1_b32(a, b, packet_traits<float>::size);
    return __riscv_vfmerge_vfm_f32m1(vzero, all_one, cmp, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pcmp_lt<PacketXf>(const PacketXf& a, const PacketXf& b)
{
    float all_one;
    memset(&all_one, 0xff, sizeof(all_one));
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0, packet_traits<float>::size);
    vbool32_t cmp = __riscv_vmflt_vv_f32m1_b32(a, b, packet_traits<float>::size);
    return __riscv_vfmerge_vfm_f32m1(vzero, all_one, cmp, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pcmp_eq<PacketXf>(const PacketXf& a, const PacketXf& b)
{

    float all_one;
    memset(&all_one, 0xff, sizeof(all_one));

    vbool32_t cmp = __riscv_vmfeq_vv_f32m1_b32(a, b, packet_traits<float>::size);
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0, packet_traits<float>::size);

    return __riscv_vfmerge_vfm_f32m1(vzero, all_one, cmp, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pcmp_lt_or_nan<PacketXf>(const PacketXf& a, const PacketXf& b)
{
    float all_one;
    memset(&all_one, 0xff, sizeof(all_one));
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0, packet_traits<float>::size);

    vbool32_t va_NaN        = __riscv_vmfne_vv_f32m1_b32(a, a, packet_traits<float>::size);
    vbool32_t vb_NaN        = __riscv_vmfne_vv_f32m1_b32(b, b, packet_traits<float>::size);
    vbool32_t ab_or_NaN     = __riscv_vmor_mm_b32(va_NaN, vb_NaN, packet_traits<float>::size);
    vbool32_t ab_not_NaN    = __riscv_vmnot_m_b32(ab_or_NaN, packet_traits<float>::size);

    vbool32_t cmp = __riscv_vmflt_vv_f32m1_b32_m(ab_not_NaN, a, b, packet_traits<float>::size);

    cmp = __riscv_vmor_mm_b32(ab_or_NaN, cmp, packet_traits<float>::size);

    return __riscv_vfmerge_vfm_f32m1(vzero, all_one, cmp, packet_traits<float>::size);
}


template <>
EIGEN_STRONG_INLINE PacketXf pfloor<PacketXf>(const PacketXf& a)
{
    float tempa[packet_traits<float>::size];

    __riscv_vse32_v_f32m1(tempa, a, packet_traits<float>::size);

    for (int i = 0; i < packet_traits<float>::size; i++)
    {
        tempa[i] = floorf(tempa[i]);
    }

    return __riscv_vle32_v_f32m1(tempa, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf ptrue<PacketXf>(const PacketXf& /*a*/)
{
    float all_one;
    memset(&all_one, 0xff, sizeof(all_one));

    return __riscv_vfmv_v_f_f32m1(all_one, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf pzero<PacketXf>(const PacketXf& /*a*/)
{
    return __riscv_vfmv_v_f_f32m1(0, packet_traits<float>::size);
}

// Logical Operations are not supported for float, so reinterpret casts
template <>
EIGEN_STRONG_INLINE PacketXf pand<PacketXf>(const PacketXf& a, const PacketXf& b)
{

    vuint32m1_t temp = __riscv_vand_vv_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(a), __riscv_vreinterpret_v_f32m1_u32m1(b), packet_traits<float>::size);

    return __riscv_vreinterpret_v_u32m1_f32m1(temp);
}

template <>
EIGEN_STRONG_INLINE PacketXf por<PacketXf>(const PacketXf& a, const PacketXf& b)
{
    vuint32m1_t temp = __riscv_vor_vv_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(a), __riscv_vreinterpret_v_f32m1_u32m1(b), packet_traits<float>::size);
    return __riscv_vreinterpret_v_u32m1_f32m1(temp);
}

template <>
EIGEN_STRONG_INLINE PacketXf pxor<PacketXf>(const PacketXf& a, const PacketXf& b)
{
    vuint32m1_t temp = __riscv_vxor_vv_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(a), __riscv_vreinterpret_v_f32m1_u32m1(b), packet_traits<float>::size);
    return __riscv_vreinterpret_v_u32m1_f32m1(temp);
}

template <>
EIGEN_STRONG_INLINE PacketXf pandnot<PacketXf>(const PacketXf& a, const PacketXf& b)
{
    vuint32m1_t bnot = __riscv_vnot_v_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(b), packet_traits<float>::size);
    
    vuint32m1_t tempvua = __riscv_vreinterpret_v_f32m1_u32m1(a);

    vuint32m1_t temp = __riscv_vand_vv_u32m1(tempvua, bnot, packet_traits<float>::size);

    return __riscv_vreinterpret_v_u32m1_f32m1(temp);
}

template <>
EIGEN_STRONG_INLINE PacketXf pload<PacketXf>(const float* from)
{

    EIGEN_DEBUG_ALIGNED_LOAD return __riscv_vle32_v_f32m1(from, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf ploadu<PacketXf>(const float* from)
{

    EIGEN_DEBUG_UNALIGNED_LOAD return __riscv_vle32_v_f32m1(from, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf ploaddup<PacketXf>(const float* from)
{
    vuint32m1_t indexes = __riscv_vid_v_u32m1(packet_traits<float>::size);
    indexes = __riscv_vdivu_vx_u32m1(indexes, 2, packet_traits<float>::size);
    indexes = __riscv_vmul_vx_u32m1(indexes, sizeof(float), packet_traits<float>::size);

    return __riscv_vloxei32_v_f32m1(from, indexes, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf ploadquad<PacketXf>(const float* from)
{
    vuint32m1_t indexes = __riscv_vid_v_u32m1(packet_traits<float>::size);
    indexes = __riscv_vdivu_vx_u32m1(indexes, 4, packet_traits<float>::size);
    indexes = __riscv_vmul_vx_u32m1(indexes, sizeof(float), packet_traits<float>::size);

    return __riscv_vloxei32_v_f32m1(from, indexes, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE void pstore<float>(float* to, const PacketXf& from)
{
    EIGEN_DEBUG_ALIGNED_STORE __riscv_vse32_v_f32m1(to, from, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const PacketXf& from)
{
    EIGEN_DEBUG_UNALIGNED_STORE __riscv_vse32_v_f32m1(to, from, packet_traits<float>::size);
}

template <>
EIGEN_DEVICE_FUNC inline PacketXf pgather<float, PacketXf>(const float* from, Index stride)
{
    return __riscv_vlse32_v_f32m1(from, stride*sizeof(float), packet_traits<float>::size);
}

template <>
EIGEN_DEVICE_FUNC inline void pscatter<float, PacketXf>(float* to, const PacketXf& from, Index stride)
{
    return __riscv_vsse32_v_f32m1(to, stride*sizeof(float), from, packet_traits<float>::size);
}

template <>
EIGEN_STRONG_INLINE float pfirst<PacketXf>(const PacketXf& a)
{

    float temp[packet_traits<float>::size];
    __riscv_vse32_v_f32m1(temp, a, packet_traits<float>::size);

    return temp[0];
}

template <>
EIGEN_STRONG_INLINE PacketXf preverse(const PacketXf& a)
{
    float temp[packet_traits<numext::int32_t>::size];
    __riscv_vse32_v_f32m1(temp, a, packet_traits<float>::size);
    return __riscv_vlse32_v_f32m1(temp + packet_traits<float>::size - 1, sizeof(float) * (-1), packet_traits<float>::size);
}
template <>
EIGEN_STRONG_INLINE PacketXf pabs(const PacketXf& a)
{
    return __riscv_vfabs_v_f32m1(a, packet_traits<float>::size);
}

// TODO(tellenbach): Should this go into MathFunctions.h? If so, change for 
// all vector extensions and the generic version.
template <>
EIGEN_STRONG_INLINE PacketXf pfrexp<PacketXf>(const PacketXf& a, PacketXf& exponent)
{
    return pfrexp_generic(a, exponent);
}

template <>
EIGEN_STRONG_INLINE float predux<PacketXf>(const PacketXf& a)
{
    vfloat32m1_t scalar = __riscv_vfmv_v_f_f32m1(0, packet_traits<float>::size);
    scalar = __riscv_vfredusum_vs_f32m1_f32m1(a, scalar, packet_traits<float>::size);
    return static_cast<float>(__riscv_vfmv_f_s_f32m1_f32(scalar));
}

// Other reduction functions:
// mul
template <>
EIGEN_STRONG_INLINE float predux_mul<PacketXf>(const PacketXf& a)
{
    float re = 1;

    float tempa[packet_traits<float>::size];
    
    __riscv_vse32_v_f32m1(tempa, a, packet_traits<float>::size);
    
    for (int i = 0; i < packet_traits<float>::size; i++)
    {
        re = re * tempa[i];
    }
    
    return re;
}

template <>
EIGEN_STRONG_INLINE float predux_min<PacketXf>(const PacketXf& a)
{

    vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(MAX_FLOAT32, packet_traits<float>::size);

    vmax = __riscv_vfredmin_vs_f32m1_f32m1(a, vmax, packet_traits<float>::size);
    return static_cast<float>(__riscv_vfmv_f_s_f32m1_f32(vmax));
}

template <>
EIGEN_STRONG_INLINE float predux_max<PacketXf>(const PacketXf& a)
{

    vfloat32m1_t vmin = __riscv_vfmv_v_f_f32m1(-MAX_FLOAT32, packet_traits<float>::size);

    vmin = __riscv_vfredmax_vs_f32m1_f32m1(a, vmin, packet_traits<float>::size);
    return static_cast<float>(__riscv_vfmv_f_s_f32m1_f32(vmin));
}

template<int N>
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<PacketXf, N>& kernel)
{
    float buffer[packet_traits<float>::size * N] = {0};
    int i = 0;
    int32_t stride = sizeof(float) * N;
    for (i = 0; i < N; i++) 
    {
        __riscv_vsse32_v_f32m1(buffer + i, stride, kernel.packet[i], packet_traits<float>::size);
    }
    
    for (i = 0; i < N; i++) 
    {
        kernel.packet[i] = __riscv_vle32_v_f32m1(buffer + i * packet_traits<float>::size, packet_traits<float>::size);
    }
}

template<>
EIGEN_STRONG_INLINE PacketXf pldexp<PacketXf>(const PacketXf& a, const PacketXf& exponent)
{
    return pldexp_generic(a, exponent);
}

template<>
EIGEN_STRONG_INLINE PacketXf psqrt<PacketXf>(const PacketXf& a)
{
    return __riscv_vfsqrt_v_f32m1(a, packet_traits<float>::size);
}

template<> 
EIGEN_STRONG_INLINE PacketXf pselect( const PacketXf& mask, const PacketXf& a, const PacketXf& b)
{
    //vbool32_t b_mask = __riscv_vreinterpret_v_u32m1_b32(u_mask); don't have this one
    vbool32_t b_mask = __riscv_vmfne_vf_f32m1_b32 (mask, 0, packet_traits<float>::size);

    return __riscv_vmerge_vvm_f32m1(b, a, b_mask, packet_traits<float>::size);
}

/*
Not so accurate to pass test.
template<> 
EIGEN_STRONG_INLINE PacketXf preciprocal(const PacketXf& a)
{
    std::cout << "function: " << __FUNCTION__ << std::endl;
    float tempa[packet_traits<float>::size];
    
    __riscv_vse32_v_f32m1(tempa, a, packet_traits<float>::size);
    std::cout << "a: [";
    
    for (int i = 0; i < packet_traits<float>::size; i++)
    {
        std::cout << tempa[i] << ", ";
    }
    
    std::cout << "]\n";

    vfloat32m1_t re = __riscv_vfrec7_v_f32m1(a, packet_traits<float>::size);
    float tempr[packet_traits<float>::size];
    std::cout << "result: [";
    
    __riscv_vse32_v_f32m1(tempr, re, packet_traits<float>::size);
    for (int i = 0; i < packet_traits<float>::size; i++)
    {
        std::cout << tempr[i] << ", ";
    }
    
    std::cout << "]\n";

    return re;

}
*/

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_PACKET_MATH_RVV_H
