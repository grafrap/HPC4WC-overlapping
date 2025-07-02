#include"cnpy.h"
#include<cstdlib>
#include<iostream>
#include<map>
#include<string>


int main(){
    cnpy::NpyArray x_arr = cnpy::npz_load("data/ref_data.npz","x");
    cnpy::NpyArray ref_arr = cnpy::npz_load("data/ref_data.npz","ref_single");
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