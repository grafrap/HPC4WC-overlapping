#include <iostream>

#include "arccos_cuda.cuh"

int main(int argc, char** argv) {

    // Check command line arguments
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <size> <num_streams> <num_repetitions>" << std::endl;
        return 1;
    }

    long size = std::atoi(argv[1]);
    long num_streams = std::atoi(argv[2]);
    long num_repetitions = std::atoi(argv[3]);

    if (size <= 0 || num_streams <= 0 || num_repetitions <= 0) {
        std::cerr << "Size, number of streams and number of repetitions must be positive integers." << std::endl;
        return 1;
    }

    std::chrono::duration<double> avg_duration(0.0);
    int success = 0;
    // Run the arccos computation
    for (int i = 0; i < num_repetitions; ++i) {
        std::cerr << "Repetition " << (i + 1) << " of " << num_repetitions << std::endl;
        std::chrono::duration<double> duration;
        std::cerr << "Running arccos computation with size: " << size << " and number of streams: " << num_streams << std::endl;
        success = run_arccos(size, num_streams, duration);
        // Check the result
        if (success == 0) {
            // std::cerr << "All results are correct." << std::endl;
            avg_duration += duration;
        } else {
            std::cerr << "There were errors in the results or there was a runtime error." << std::endl;
            break;
        }
    }

    // Print the duration of the computation
    avg_duration /= num_repetitions;
    std::cerr << "Average duration: " << avg_duration.count() << " seconds." << std::endl;

    // Clean output in out stream
    if (success == 0) {
        std::cout << "### " << size << " " << num_streams << " " << avg_duration.count() << std::endl;
    }

    return success;
}