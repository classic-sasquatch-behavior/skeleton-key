#pragma once

//cuda
#include<cuda.h>
#include<cuda_runtime.h>

//std
#include<iostream>
#include<string>
#include<vector>

//check if arrayfire is being used
#ifdef AF_API_VERSION
#include"arrayfire.h"
#endif

//check if OpenCV is being used
#ifdef CV_VERSION
#include"opencv2/core.hpp"
#endif