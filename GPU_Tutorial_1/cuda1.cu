#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Return CUDA cores per SM based on compute capability
int getCoresPerSM(int major, int minor) {

    switch (major) {

        case 2: // Fermi
            return (minor == 1) ? 48 : 32;

        case 3: // Kepler
            return 192;

        case 5: // Maxwell
            return 128;

        case 6: // Pascal
            if (minor == 1 || minor == 2)
                return 128;
            if (minor == 0)
                return 64;
            return 128;

        case 7: // Volta / Turing
            return 64;

        case 8: // Ampere / Ada
            if (minor == 0)
                return 64;
            if (minor == 6 || minor == 9)
                return 128;
            return 64;

        case 9: // Hopper / Blackwell
            return 128;

        default:
            return 128;
    }
}

int main() {

    int deviceCount = 0;

    // Get number of CUDA devices
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        cerr << "CUDA Error: "
             << cudaGetErrorString(error)
             << endl;
        return 1;
    }

    cout << "Found " << deviceCount
         << " CUDA device(s).\n" << endl;


    // Loop through all CUDA devices
    for (int i = 0; i < deviceCount; ++i) {

        cudaDeviceProp prop;

        error = cudaGetDeviceProperties(&prop, i);

        if (error != cudaSuccess) {
            cerr << "Error getting device properties: "
                 << cudaGetErrorString(error)
                 << endl;
            continue;
        }

        int coresPerSM =
            getCoresPerSM(prop.major, prop.minor);

        int totalCores =
            prop.multiProcessorCount * coresPerSM;


        cout << "============================================"
             << endl;

        cout << "        CUDA DEVICE " << i << endl;

        cout << "============================================"
             << endl;


        // Basic information
        cout << "Device Name:                  "
             << prop.name << endl;

        cout << "Compute Capability:           "
             << prop.major << "."
             << prop.minor << endl;


        // Memory information
        cout << "Total Global Memory:          "
             << prop.totalGlobalMem / (1024 * 1024)
             << " MB" << endl;

        cout << "Shared Memory Per Block:      "
             << prop.sharedMemPerBlock / 1024
             << " KB" << endl;

        cout << "L2 Cache Size:                "
             << prop.l2CacheSize / 1024
             << " KB" << endl;


        // Processor information
        cout << "Streaming Multiprocessors:    "
             << prop.multiProcessorCount << endl;

        cout << "Cores Per SM:                 "
             << coresPerSM << endl;

        cout << "Total CUDA Cores:              "
             << totalCores << endl;


        // Thread information
        cout << "Warp Size:                    "
             << prop.warpSize << endl;

        cout << "Max Threads Per Block:        "
             << prop.maxThreadsPerBlock << endl;

        cout << "Max Threads Per SM:           "
             << prop.maxThreadsPerMultiProcessor
             << endl;


        // Block dimensions
        cout << "Max Block Dimensions:         "
             << prop.maxThreadsDim[0] << " x "
             << prop.maxThreadsDim[1] << " x "
             << prop.maxThreadsDim[2]
             << endl;


        // Grid dimensions
        cout << "Max Grid Dimensions:          "
             << prop.maxGridSize[0] << " x "
             << prop.maxGridSize[1] << " x "
             << prop.maxGridSize[2]
             << endl;


        // Register information
        cout << "Registers Per Block:          "
             << prop.regsPerBlock << endl;

        cout << "Registers Per SM:             "
             << prop.regsPerMultiprocessor
             << endl;


        // Clock information
        cout << "GPU Clock Rate:               "
             << prop.clockRate / 1000
             << " MHz" << endl;

        cout << "Memory Clock Rate:            "
             << prop.memoryClockRate / 1000
             << " MHz" << endl;


        // Memory bus
        cout << "Memory Bus Width:             "
             << prop.memoryBusWidth
             << " bits" << endl;


        // Feature support
        cout << "Concurrent Kernels:           "
             << (prop.concurrentKernels ? "Yes" : "No")
             << endl;

        cout << "ECC Enabled:                  "
             << (prop.ECCEnabled ? "Yes" : "No")
             << endl;

        cout << "Unified Addressing:           "
             << (prop.unifiedAddressing ? "Yes" : "No")
             << endl;

        cout << "Managed Memory:               "
             << (prop.managedMemory ? "Yes" : "No")
             << endl;

        cout << "Concurrent Managed Access:    "
             << (prop.concurrentManagedAccess ? "Yes" : "No")
             << endl;


        // Texture / surface limits
        cout << "Max Texture 1D:               "
             << prop.maxTexture1D
             << endl;

        cout << "Max Texture 2D:               "
             << prop.maxTexture2D[0]
             << " x "
             << prop.maxTexture2D[1]
             << endl;

        cout << "Max Texture 3D:               "
             << prop.maxTexture3D[0]
             << " x "
             << prop.maxTexture3D[1]
             << " x "
             << prop.maxTexture3D[2]
             << endl;


        cout << "============================================"
             << endl;
    }

    return 0;
}
