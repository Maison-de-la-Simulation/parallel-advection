#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

int main(int, char**) {


    constexpr int N = 8;
    constexpr int nb_wg = 2;
    constexpr int size_wg = 4;

    int a[N];
    int sum[nb_wg];

    /* Filling global array*/
    for(int i = 0; i < N; ++i)
        a[i] = 2;
        // a[i] = i%2 == 0 ? 1 : -1;

    queue Q;

    /* Print array */
    for(int i = 0; i < N; ++i){
        std::cout << a[i];

        if(i < N-1)
            std::cout << "  ";
    }
    std::cout << std::endl;

    {
        buffer<int, 1> bufA(a, range<1>(N));

        const range<1> num_work_groups{nb_wg};
        const range<1> work_group_size{size_wg};
        
        buffer<int, 1> bufSum(sum, num_work_groups);
        
        /***********************************************************************
         *                             Hierarchical
         * ********************************************************************/
        Q.submit([&](handler& cgh){
            
            auto A      = bufA.get_access<access::mode::read>(cgh);
            auto accSum = bufSum.get_access<access::mode::read_write>(cgh);
            
            // local_accessor<int> locSum{2, cgh};
            
            cgh.parallel_for_work_group(num_work_groups, work_group_size, //here we have to specify the work group size or it's invalid code
            [=](group<1> g){

                int s = 0;
                atomic_ref<int, memory_order::seq_cst, memory_scope::work_group, access::address_space::local_space> locSum(s);

                g.parallel_for_work_item(work_group_size, [&](h_item<1> it)
                { //private variable

                    locSum += A[it.get_global_id(0)];

                    // locSum[it.get_physical_id()] += A[it.get_global_id(0)];
                }) ; // Implicit barrier

                // if (g.leader()) {
                accSum[g.get_group_id(0)] = locSum;
                // }
            });
        });


        /***********************************************************************
         *                              nd_range
         * ********************************************************************/
        // Q.submit([&](handler& cgh){
        //     auto A      = bufA.get_access<access::mode::read_write>(cgh);
        //     auto accSum = bufSum.get_access<access::mode::read_write>(cgh);

        //     /* Local accessor of size 1 to store tmpSum */
        //     local_accessor<int> locSum{1, cgh};

        //     cgh.parallel_for(sycl::nd_range<1>(range<1>(N), range<1>(size_wg)),
        //     [=](sycl::nd_item<1> it){

        //         // locSum[0] += A[it.get_local_id(0)*it.get_group().get_group_id(0)];
        //         locSum[0] += A[it.get_global_id(0)];

        //         //We have to explicitely put a barrier
        //         it.barrier(sycl::access::fence_space::local_space);
        //         // group_barrier(it.get_group());


        //         if (it.get_group().leader()) {
        //             accSum[it.get_group().get_group_id(0)] = locSum[0];
        //         }
        //     });
        // });


        /***********************************************************************
         *                               Scoped
         * ********************************************************************/
        // Q.submit([&](sycl::handler &cgh){

        //     auto A      = bufA.get_access<access::mode::read_write>(cgh);
        //     auto accSum = bufSum.get_access<access::mode::read_write>(cgh);

        //     cgh.parallel(num_work_groups, work_group_size, [&](auto group){
        //         // sycl::memory_environment(group, )

        //         local_accessor<int> locSum{1, cgh};
        //         // local_memory<int[1], decltype(group)> locSum;

        //         // sycl::distribute_groups(group, [&](auto subgroup){

        //             sycl::distribute_items_and_wait(group,
        //             [=](sycl::s_item<1> l_id){
        //             // std::cout << group.get_group_id(0) << std::endl;
        //                 locSum[0] += A[l_id.get_global_id(0)];
        //                 // std::cout << l_id.get_local_id(group) << std::endl;
        //                 // std::cout << locSum[0] << std::endl;
        //                 // std::cout << A[l_id.get_local_id(group) ] << std::endl;
        //             });

        //         // });
        //         sycl::single_item(group, [&](){
        //             // std::cout << group.get_group_id(0) << std::endl;
        //             // accSum[group.get_group_id(0)] = locSum[0];
        //         });
        //     });
        // });

        // Q.submit([&](sycl::handler& cgh){
        //     auto A      = bufA.get_access<access::mode::read_write>(cgh);
        //     auto accSum = bufSum.get_access<access::mode::read_write>(cgh);

        //     cgh.parallel<class Kernel>(num_work_groups, work_group_size, 
        //     [=](auto grp){
        //     // Outside of distribute_items(), the degree of parallelism is implementation-defined.
        //     // the implementation can use whatever is most efficient for hardware/backend.
        //     // In Open SYCL CPU, this would be executed by a single thread on CPU
        //     // and Group_size threads on Open SYCL GPU
        //     // Information about the position in the physical iteration space can be obtained
        //     // using grp.get_physical_local_id() and grp.get_physical_local_range().

        //     // sycl::memory_environment() can be used to allocate local memory 
        //     // (of compile-time size) as well as private memory that is persistent across
        //     // multiple distribute_items() calls.
        //     // Of course, local accessors can also be used.
        //     sycl::memory_environment(grp, 
        //         sycl::require_local_mem<atomic_ref<int>>(),
        //         // the requested private memory is not used in this example,
        //         // and only here to showcase how to request private memory.
        //         // sycl::require_private_mem<int>(),
        //         [&](auto& locSum/*, auto& private_mem*/){

        //         // Variables not explicitly requested as local or private memory 
        //         // will be allocated in private memory of the _physical_ work item
        //         // (see the for loop below)

        //         sycl::distribute_items_and_wait(grp, [&](sycl::s_item<1> idx){
        //             locSum += A[idx.get_global_id(0)];
        //         });
        //         // Instead of an explicit group_barrier, we could also use the
        //         // blocking distribute_items_and_wait()
        //         // sycl::group_barrier(grp);

        //         // // Can execute code e.g. for a single item of a subgroup:
        //         // sycl::distribute_groups(grp, [&](auto subgroup){
        //         //     sycl::single_item(subgroup, [&](){
        //         //         // ...
        //         //     });
        //         // });

        //         // Variables inside the parallel scope that are not explicitly local or private memory
        //         // are allowed, if they are not modified from inside `distribute_items()` scope.
        //         // The SYCL implementation will allocate those in private memory of the physical item,
        //         // so they will always be efficient. This implies that the user should not attempt to assign values
        //         // per logical work item, since they are allocated per physical item.
        //         // for(int i = Group_size / 2; i > 0; i /= 2){
        //         //     // The *_and_wait variants of distribute_groups and distribute_items
        //         //     // invoke a group_barrier at the end.
        //         //     sycl::distribute_items_and_wait(grp, 
        //         //         [&](sycl::s_item<1> idx){
        //         //         size_t lid = idx.get_innermost_local_id(0);
        //         //         if(lid < i)
        //         //         scratch[lid] += scratch[lid+i];
        //         //     });
        //         // }
                
        //         sycl::single_item_and_wait(grp, [&](){
        //             std::cout << "loc:" << locSum << "\ngroup_id:" << grp.get_group_id(0) << std::endl;
        //             // data_accessor[grp.get_group_id(0)*Group_size] = scratch[0];
        //             accSum[grp.get_group_id(0)] = locSum;
        //         });
        //     });
        //     });
        // });

    } //end of scope to sync buffers


    for(int i = 0; i < N; ++i){
        std::cout << a[i];

        if(i < N-1)
            std::cout << "  ";
    }
    std::cout << std::endl;
    
    int s=0;
    for(int i=0; i < nb_wg; ++i)
        s += sum[i];

    std::cout << "sum: " << s << std::endl;

    return EXIT_SUCCESS;
}

// #include <sycl/sycl.hpp>

// using namespace sycl;

// class SumReductionKernel;

// void sum_reduction_ndrange(sycl::queue& q, const int* input, int* output) {
//   constexpr int N = 8;

//   // Create device buffers for input and output
//   sycl::buffer<int, 1> input_buf(const_cast<int*>(input), sycl::range<1>(N));
//   sycl::buffer<int, 1> output_buf(output, sycl::range<1>(1));

//   // Submit kernel to the queue
//   q.submit([&](sycl::handler& h) {
//     // Access input and output buffers
//     auto in = input_buf.get_access<sycl::access::mode::read>(h);
//     auto out = output_buf.get_access<sycl::access::mode::write>(h);

//     // Define nd-range and work-group size
//     const sycl::range<1> ndrange(N);
//     const sycl::range<1> local_size(4);

//     // Define a kernel function to perform sum reduction
//     h.parallel_for<SumReductionKernel>(
//       ndrange, [=](sycl::nd_item<1> item) {
//         // Get the global and local IDs
//         const int gid = item.get_global_id(0);
//         const int lid = item.get_local_id(0);

//         // Load data from global memory to local memory
//         sycl::local_memory<int[4], > local;
//         local[lid] = in[gid];

//         // Synchronize to ensure all work-items have loaded data
//         item.barrier(sycl::access::fence_space::local_space);

//         // Perform parallel reduction within each work-group
//         for (int stride = 2; stride <= local_size[0]; stride *= 2) {
//           const int index = 2 * lid * stride;
//           if (index < local_size[0]) {
//             local[index] += local[index + stride];
//           }
//           // Synchronize to ensure all work-items have updated their data
//           item.barrier(sycl::access::fence_space::local_space);
//         }

//         // Write the result back to global memory
//         if (lid == 0) {
//           out[0] = local[0];
//         }
//       });
//   });

//   // Wait for the queue to finish
//   q.wait();
// }