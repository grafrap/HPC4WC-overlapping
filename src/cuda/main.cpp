#include"cnpy.h"
#include<cstdlib>
#include<iostream>
#include<map>
#include<string>

// temp:
#include <sys/stat.h>


int main(){

    struct stat st;
    if (stat("data/ref_data.npz", &st) != 0) {
        perror("stat failed");
        return 1;
    }
    std::cout << "File size: " << st.st_size << " bytes" << std::endl;

    FILE* fp = fopen("data/ref_data.npz", "rb");
    if (!fp) {
        perror("fopen failed");
        return 1;
    }
    char buf[30];
    size_t res = fread(buf, sizeof(char), 30, fp);
    std::cout << "fread returned " << res << " bytes" << std::endl;
    fclose(fp);

    
    std::cout << "BEGIN MAIN" << std::endl;
    cnpy::NpyArray x_arr = cnpy::npz_load("data/ref_data.npz","x");
    std::cout << "BEGIN MAIN" << std::endl;
    cnpy::NpyArray ref_arr = cnpy::npz_load("data/ref_data.npz","ref");
    // cnpy::NpyArray ref_arr = cnpy::npz_load("data/ref_data.npz","ref_single");
    std::cout << "BEGIN MAIN" << std::endl;
    double* x = x_arr.data<double>();
    double* ref = ref_arr.data<double>();
    int x_dims = x_arr.shape.size();
    std::cout << "Shape of array: (";
    for (size_t i = 0; i < x_dims; ++i) {
        std::cout << x_arr.shape[i] << ", ";
    }
    std::cout << ")" << std::endl;

    std::cout << "some of its elements:" << std::endl;
    for (int i = 0; i < 18; ++i) {
        std::cout << "x[" << i << "] = " << x[i] << ", ref[" << i << "] = " << ref[i] << std::endl;
    }

    return 0;
}