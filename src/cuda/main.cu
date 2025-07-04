#include <iostream>

#include "arccos_cuda.cuh"

int main(int argc, char** argv) {

    // Check command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <size> <num_streams>" << std::endl;
        return 1;
    }

    int size = std::atoi(argv[1]);
    int num_streams = std::atoi(argv[2]);

    if (size <= 0 || num_streams <= 0) {
        std::cerr << "Size and number of streams must be positive integers." << std::endl;
        return 1;
    }

    // Run the arccos computation
    std::chrono::duration<double> duration;
    std::cerr << "Running arccos computation with size: " << size << " and number of streams: " << num_streams << std::endl;
    int succes = run_arccos(size, num_streams, duration);

    // Check the result
    if (succes == 0) {
        std::cerr << "All results are correct." << std::endl;
    } else {
        std::cerr << "There were errors in the results or there was a runtime error." << std::endl;
    }

    // Print the duration of the computation
    std::cerr << "Duration: " << duration.count() << " seconds." << std::endl;

    // Clean output in out stream
    std::cout << "### " << size << " " << num_streams << " " << duration.count() << std::endl;

    return succes;
}