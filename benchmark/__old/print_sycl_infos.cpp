#include <sycl/sycl.hpp>

int
main(int argc, char const *argv[]) {
    bool gpu = argc > 1 ? true : false;

    sycl::device d;

    if (gpu)
        try {
            d = sycl::device{sycl::gpu_selector_v};
        } catch (const std::runtime_error e) {
            std::cout
                << "GPU was requested but none is available. Will use CPU."
                << std::endl;

            d = sycl::device{sycl::cpu_selector_v};
            // strParams.gpu = false;
        }
    else
        d = sycl::device{sycl::cpu_selector_v};

    sycl::queue Q{d};

    /* Display infos on current device */
    std::cout << "Device: "
              << Q.get_device().get_info<sycl::info::device::name>() << "\n";

    std::cout
        << "\t max_work_group_size: "
        << Q.get_device().get_info<sycl::info::device::max_work_group_size>()
        << "\n";

    auto r1d =
        Q.get_device().get_info<sycl::info::device::max_work_item_sizes<1>>();
    auto r2d =
        Q.get_device().get_info<sycl::info::device::max_work_item_sizes<2>>();
    auto r3d =
        Q.get_device().get_info<sycl::info::device::max_work_item_sizes<3>>();

    std::cout << "\t max_work_item_sizes<1>: " << r1d[0]
              << " (total =" << r1d.size() << ")\n";
    std::cout << "\t max_work_item_sizes<3>: " << r2d[0] << ", " << r2d[1]
              << " (total =" << r2d.size() << ")\n";
    std::cout << "\t max_work_item_sizes<3>: " << r3d[0] << ", " << r3d[1]
              << ", " << r3d[2] << " (total =" << r3d.size() << ")\n";


    Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(1024), cgh);

        cgh.single_task([=]() {
            auto max_size = slice_ftmp.max_size();
            std::cout << "Local accessor max_size is: " << max_size << std::endl;
        });
    });


    return 0;
}
