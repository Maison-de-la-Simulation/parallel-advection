#include "AdvectionParams.hpp"
#include <iostream>

ADVParams::ADVParams(ADVParamsNonCopyable &other) {
    n1 = other.n1;
    n0 = other.n0;
    n2 = other.n2;

    maxIter = other.maxIter;
    gpu = other.gpu;
    outputSolution = other.outputSolution;

    percent_loc = other.percent_loc;
    seq_size0 = other.seq_size0;
    seq_size2 = other.seq_size2;

    pref_wg_size = other.pref_wg_size;

    dt = other.dt;

    minRealX = other.minRealX;
    maxRealX = other.maxRealX;
    minRealVx = other.minRealVx;
    maxRealVx = other.maxRealVx;

    realWidthX = other.realWidthX;
    dx = other.dx;
    dvx = other.dvx;

    inv_dx = other.inv_dx;

    nSubgroups_Local = other.nSubgroups_Local;
    nSubgroups_Global = other.nSubgroups_Global;
    seqSize_Global = other.seqSize_Global;
    seqSize_Local = other.seqSize_Local;
};

ADVParamsNonCopyable::ADVParamsNonCopyable(ADVParams &other) {
    n1 = other.n1;
    n0 = other.n0;
    n2 = other.n2;

    maxIter = other.maxIter;
    gpu = other.gpu;
    outputSolution = other.outputSolution;

    percent_loc = other.percent_loc;
    seq_size0 = other.seq_size0;
    seq_size2 = other.seq_size2;

    pref_wg_size = other.pref_wg_size;

    dt = other.dt;

    minRealX = other.minRealX;
    maxRealX = other.maxRealX;
    minRealVx = other.minRealVx;
    maxRealVx = other.maxRealVx;

    realWidthX = other.realWidthX;
    dx = other.dx;
    dvx = other.dvx;

    inv_dx = other.inv_dx;

    nSubgroups_Local = other.nSubgroups_Local;
    nSubgroups_Global = other.nSubgroups_Global;
    seqSize_Global = other.seqSize_Global;
    seqSize_Local = other.seqSize_Local;
};

// ======================================================
// ======================================================
void
ADVParamsNonCopyable::setup(const ConfigMap &configMap) {
    // problem
    n1 = configMap.getInteger("problem", "n1", 1024);
    n0 = configMap.getInteger("problem", "n0", 1024);
    n2 = configMap.getInteger("problem", "n2", 1024);
    maxIter = configMap.getInteger("problem", "maxIter", 50);

    dt = configMap.getFloat("problem", "dt", 0.0001);
    minRealX = configMap.getFloat("problem", "minRealX", 0.0);
    maxRealX = configMap.getFloat("problem", "maxRealX", 1.0);
    minRealVx = configMap.getFloat("problem", "minRealVx", -1.0);
    maxRealVx = configMap.getFloat("problem", "maxRealVx", 1.0);

    // impl
    kernelImpl = configMap.getString("impl", "kernelImpl", "AdaptiveWg");
    inplace = configMap.getBool("impl", "inplace", true);

    // optimization
    gpu = configMap.getBool("optimization", "gpu", true);
    pref_wg_size = configMap.getInteger("optimization", "pref_wg_size", 512);
    seq_size0 = configMap.getInteger("optimization", "seq_size0", 1);
    seq_size2 = configMap.getInteger("optimization", "seq_size2", 1);
    nSubgroups_Local =
        configMap.getInteger("optimization", "nSubgroups_Local", 1);
    nSubgroups_Global =
        configMap.getInteger("optimization", "nSubgroups_Global", 1);
    seqSize_Global = configMap.getInteger("optimization", "seqSize_Global", 1);
    seqSize_Local = configMap.getInteger("optimization", "seqSize_Local", 3);

    // io
    outputSolution = configMap.getBool("io", "outputSolution", false);

    update_deltas();
}   // ADVParams::setup

// ======================================================
// ======================================================
void
ADVParams::update_deltas() {
    realWidthX = maxRealX - minRealX;
    dx = realWidthX / n1;
    dvx = (maxRealVx - minRealVx) / n0;

    inv_dx = 1 / dx;
}   // ADVParams::setup

// ======================================================
// ======================================================
void
ADVParamsNonCopyable::print() {
    std::cout << "##########################" << std::endl;
    std::cout << "Runtime parameters:" << std::endl;
    std::cout << "##########################" << std::endl;
    std::cout << "kernelImpl  : " << kernelImpl << std::endl;
    std::cout << "inplace     : " << inplace << std::endl;
    std::cout << "gpu         : " << gpu << std::endl;
    std::cout << "maxIter     : " << maxIter << std::endl;
    std::cout << "n0 (nvx)    : " << n0 << std::endl;
    std::cout << "n1 (nx)     : " << n1 << std::endl;
    std::cout << "n2          : " << n2 << std::endl;
    std::cout << "local_sg    : " << nSubgroups_Local << std::endl;
    std::cout << "global_sg   : " << nSubgroups_Global << std::endl;
    std::cout << "seq_size_L  : " << seqSize_Local << std::endl;
    std::cout << "seq_size_G  : " << seqSize_Global << std::endl;
    std::cout << "pref_wg_size: " << pref_wg_size << std::endl;
    std::cout << "seq_size0   : " << seq_size0 << std::endl;
    std::cout << "seq_size2   : " << seq_size2 << std::endl;
    std::cout << "dt          : " << dt << std::endl;
    std::cout << "dx          : " << dx << std::endl;
    std::cout << "dvx         : " << dvx << std::endl;
    std::cout << "minRealX    : " << minRealX << std::endl;
    std::cout << "maxRealX    : " << maxRealX << std::endl;
    std::cout << "minRealVx   : " << minRealVx << std::endl;
    std::cout << "maxRealVx   : " << maxRealVx << std::endl;
    std::cout << std::endl;

}   // ADVParams::print
