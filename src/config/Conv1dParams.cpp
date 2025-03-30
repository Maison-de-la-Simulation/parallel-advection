#include "Conv1dParams.hpp"
#include <iostream>

Conv1dParams::Conv1dParams(Conv1dParamsNonCopyable &other) {
    length = other.length;
    channel_in = other.channel_in;
    channel_out = other.channel_out;
    k = other.k;
    total_batch_size = other.total_batch_size;
    batch_size_n2 = other.batch_size_n2;
    n0 = other.n0;
    n1 = other.n1;
    n2 = other.n2;
    n_write = other.n_write;
    gpu = other.gpu;

    pref_wg_size = other.pref_wg_size;
    seq_size0 = other.seq_size0;
    seq_size2 = other.seq_size2;
    inplace = other.inplace;
};

Conv1dParamsNonCopyable::Conv1dParamsNonCopyable(Conv1dParams &other) {
    length = other.length;
    channel_in = other.channel_in;
    channel_out = other.channel_out;
    k = other.k;
    total_batch_size = other.total_batch_size;
    batch_size_n2 = other.batch_size_n2;
    n0 = other.n0;
    n1 = other.n1;
    n2 = other.n2;
    n_write = other.n_write;
    gpu = other.gpu;

    pref_wg_size = other.pref_wg_size;
    seq_size0 = other.seq_size0;
    seq_size2 = other.seq_size2;
    inplace = other.inplace;
};

// ======================================================
// ======================================================
void Conv1dParamsNonCopyable::setup(const ConfigMap& configMap)
{   // problem
    length = configMap.getInteger("problem", "length",  1024);
    channel_in = configMap.getInteger("problem", "channel_in",  3);
    channel_out = channel_in;
    k = configMap.getInteger("problem", "k",  3);
    total_batch_size =
        configMap.getInteger("problem", "total_batch_size", 262144);
    batch_size_n2 = configMap.getInteger("problem", "batch_size_n2", 512);
    n0 = total_batch_size/batch_size_n2;
    n1 = length*channel_out;
    n2 = batch_size_n2;
    n_write = compute_output_size(length, k);

    // impl
    kernelImpl = configMap.getString("impl", "kernelImpl", "AdaptiveWg");
    inplace = configMap.getBool("impl", "inplace", true);

    // optimization
    gpu = configMap.getBool("optimization", "gpu", true);
    pref_wg_size = configMap.getInteger("optimization", "pref_wg_size", 512);
    seq_size0 = configMap.getInteger("optimization", "seq_size0", 1);
    seq_size2 = configMap.getInteger("optimization", "seq_size2", 1);

} // Conv1dParams::setup

// ======================================================
// ======================================================
size_t
Conv1dParams::compute_output_size(size_t Lin,
                                           short unsigned kernel_size) {
    return Lin - (kernel_size - 1);
}   // Conv1dParams::compute_output_size

// ======================================================
// ======================================================
void
Conv1dParamsNonCopyable::print() {
    std::cout << "##########################" << std::endl;
    std::cout << "Runtime parameters:" << std::endl;
    std::cout << "##########################" << std::endl;
    std::cout << "kernelImpl   : " << kernelImpl << std::endl;
    std::cout << "inplace      : " << inplace << std::endl;
    std::cout << "gpu          : " << gpu << std::endl;
    std::cout << "n0           : " << n0 << std::endl;
    std::cout << "n1           : " << n1 << std::endl;
    std::cout << "n2           : " << n2 << std::endl;
    std::cout << "pref_wg_size : " << pref_wg_size << std::endl;
    std::cout << "seq_size0    : " << seq_size0 << std::endl;
    std::cout << "seq_size2    : " << seq_size2 << std::endl;
    std::cout << "batch_size   : " << total_batch_size << std::endl;
    std::cout << "length       : " << length << std::endl;
    std::cout << "channels(i/o): " << channel_in << std::endl;
    std::cout << "k            : " << k << std::endl;
    std::cout << "batch_n2     : " << batch_size_n2 << std::endl;
    std::cout << "n_write      : " << n_write << std::endl;
    std::cout << std::endl;
}   // Conv1dParams::print
