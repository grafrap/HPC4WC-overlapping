#include <iostream>

#include "arccos_cuda.cuh"

int main(int argc, char** argv) {

    // Check command line arguments
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << "<num_arccos_calls> <size> <num_streams> <num_repetitions>" << std::endl;
        return 1;
    }

    long num_arccos_calls = std::atoi(argv[1]);
    long size = std::atoi(argv[2]);
    long num_streams = std::atoi(argv[3]);
    long num_repetitions = std::atoi(argv[4]);

    if (size <= 0 || num_streams <= 0 || num_repetitions <= 0) {
        std::cerr << "Size, number of streams and number of repetitions must be positive integers." << std::endl;
        return 1;
    }

    if (size % num_streams != 0) {
        std::cerr << "Size must be divisible by number of streams." << std::endl;
        return 1;
    }

    // Initialize data and result arrays and streams
    int size_per_stream = size / num_streams;
    
    size_t bytes = size_per_stream * sizeof(fType);

    fType* h_data[num_streams], *h_result[num_streams], *h_reference[num_streams];
    fType* d_data[num_streams];
    cudaStream_t streams[num_streams];

    // Initialize all data and streams
    if (init_data(h_data, h_result, h_reference, d_data, bytes, streams, num_arccos_calls, num_streams, size_per_stream)) {
        std::cerr << "Error initializing data." << std::endl;
        return 1;
    }

    std::chrono::duration<double> avg_duration(0.0);
    int success = 0;
    // Run the arccos computation
    for (int i = 0; i < num_repetitions; ++i) {
        std::cerr << "Repetition " << (i + 1) << " of " << num_repetitions << std::endl;
        std::chrono::duration<double> duration;
        std::cerr << "Running arccos computation with number of arccos calls: " << num_arccos_calls << ", size: " << size << " and number of streams: " << num_streams << std::endl;
        success = run_arccos(num_arccos_calls, size, num_streams, duration, h_data, h_result, h_reference, d_data, streams);
        // Check the result
        if (success == 0) {
            // std::cerr << "All results are correct." << std::endl;
            avg_duration += duration;
        } else {
            std::cerr << "There were errors in the results or there was a runtime error." << std::endl;
            break;
        }
    }

    // Cleanup all allocated memory and destroy all streams
    cleanup(h_data, h_result, h_reference, d_data, streams, num_streams);

    // Print the duration of the computation
    avg_duration /= num_repetitions;
    std::cerr << "Average duration: " << avg_duration.count() << " seconds." << std::endl;

    // Clean output in out stream
    if (success == 0) {
        std::cout << "### " << num_arccos_calls << " " << size << " " << num_streams << " " << avg_duration.count() << std::endl;
    }

    return success;
}