/**
 *  @file python/numkong_bundle.cpp
 *  @brief A single-file C++ wrapper for NumKong C files to ensure static linkage on Android.
 */

extern "C" {
#include "../numkong/c/numkong.c"
#include "../numkong/c/dispatch_bf16.c"
#include "../numkong/c/dispatch_bf16c.c"
#include "../numkong/c/dispatch_e2m3.c"
#include "../numkong/c/dispatch_e3m2.c"
#include "../numkong/c/dispatch_e4m3.c"
#include "../numkong/c/dispatch_e5m2.c"
#include "../numkong/c/dispatch_f16.c"
#include "../numkong/c/dispatch_f16c.c"
#include "../numkong/c/dispatch_f32.c"
#include "../numkong/c/dispatch_f32c.c"
#include "../numkong/c/dispatch_f64.c"
#include "../numkong/c/dispatch_f64c.c"
#include "../numkong/c/dispatch_i16.c"
#include "../numkong/c/dispatch_i32.c"
#include "../numkong/c/dispatch_i4.c"
#include "../numkong/c/dispatch_i64.c"
#include "../numkong/c/dispatch_i8.c"
#include "../numkong/c/dispatch_other.c"
#include "../numkong/c/dispatch_u1.c"
#include "../numkong/c/dispatch_u16.c"
#include "../numkong/c/dispatch_u32.c"
#include "../numkong/c/dispatch_u4.c"
#include "../numkong/c/dispatch_u64.c"
#include "../numkong/c/dispatch_u8.c"
}
