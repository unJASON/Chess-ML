#include <Python.h>

extern "C"{
    void callMd(){
        // 初始化Python
        //在使用Python系统前，必须使用Py_Initialize对其
        //进行初始化。它会载入Python的内建模块并添加系统路
        //径到模块搜索路径中。这个函数没有返回值，检查系统
        //是否初始化成功需要使用Py_IsInitialized。
        Py_Initialize();
        Py_Finalize();
    }
}