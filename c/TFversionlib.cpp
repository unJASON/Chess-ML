#include <iostream>
#include <tensorflow/c/c_api.h>
extern "C"{
    void version() {
        std:: cout << "Hello from TensorFlow C library version" << TF_Version();
    }
}
