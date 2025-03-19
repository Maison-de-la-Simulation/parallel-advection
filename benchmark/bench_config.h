#pragma once

#include <benchmark/benchmark.h>
#include <cstdint>
#include <vector>

enum AdvImpl : int {
    BR3D,           // 0
    HIER,           // 1
    NDRA,           // 2
    ADAPTWG,        // 3
    HYBRID          // 4
};

using bm_vec_t = std::vector<int64_t>;

static bm_vec_t SEQ_SIZE0 = {1};
static bm_vec_t SEQ_SIZE2 = {1};

static int64_t WG_SIZE_NVI = 128;
static int64_t WG_SIZE_AMD = 256;
static int64_t WG_SIZE_PVC = 1024;

static bm_vec_t WG_SIZES_RANGE = {1, 4, 8, 64, 128, 256, 512, 1024};

static bm_vec_t IMPL_RANGE = {AdvImpl::NDRA, AdvImpl::ADAPTWG};


struct Dim3Exp {int64_t n0_, n1_, n2_;};
using expe = Dim3Exp;
static expe e0{1<<17, 1<<14, 1};       //n1 trop grand pour local mem, acces coal
static expe e1{1<<10, 1<<11, 1<<10};   //n1 trop grand (WI per WG) mais rentre en local mem
static expe e2{1<<10, 1<<14, 1<<7};    //n1 trop grand pour fit en local +acces non coal.
static expe e3{1<<27, 1<<4,  1};       //n1 trop petit, acces coalescent en dim 1
static expe e4{1<<20, 1<<4,  1<<7};    //n1 trop petit et acces non coalescent
static expe e5{1<<21, 1<<10, 1};       //cas parfait: elements contigus
static expe e6{1<<14, 1<<10, 1<<7};    //elements espacés en memoire de plus de SIMD_Size
static expe e7{0,     1<<10, (1<<6)+1};//non aligné et pas power of two
static expe e8{1,     1<<10, 1<<21};   //un seul batch en d0

//WORKS:
static expe e9{1<<11, 1<<10, 1<<10};   //profil equilibre

static std::vector EXP_SIZES{e0, e1, e2, e3, e4, e5, e6, e7, e8, e9};
