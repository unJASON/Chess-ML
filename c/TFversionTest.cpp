#include <iostream>
#include <tensorflow/c/c_api.h>

int main() {
    std:: cout << "Hello from TensorFlow C library version" << TF_Version();
    return 0;
}
