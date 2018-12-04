#include <iostream>
#include <ctime>
#include <time.h>
#include<random>
#include<algorithm>
#include<cstring>
extern "C"{
    int foo(int a, int b){
        printf("you input %d and %d\n", a, b);
        return a+b;
    }
}