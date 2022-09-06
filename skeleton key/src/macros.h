#pragma once

#pragma region dims

	//establish the coordinates of the current thread. this one is for 1d kernels								
	#define DIMS_1D(_first_dim_) \
		int _first_dim_ = (blockIdx.x * blockDim.x) + threadIdx.x; \
		int & _FIRST_ = _first_dim_;

	//establish the coordinates of the current thread. this one is for 2d kernels		
	#define DIMS_2D(_first_dim_, _second_dim_) \
		int _first_dim_ = (blockIdx.x * blockDim.x) + threadIdx.x; \
		int _second_dim_ = (blockIdx.y * blockDim.y) + threadIdx.y; \
		int & _FIRST_ = _first_dim_;\
		int & _SECOND_ = _second_dim_;

	//establish the coordinates of the current thread. this one is for 3d kernels		
	#define DIMS_3D(_first_dim_, _second_dim_, _third_dim_)\
		int _first_dim_ = (blockIdx.x * blockDim.x) + threadIdx.x; \
		int _second_dim_ = (blockIdx.y * blockDim.y) + threadIdx.y; \
		int _third_dim_ = (blockIdx.z * blockDim.z) + threadIdx.z; \
		int & _FIRST_ = _first_dim_;\
		int & _SECOND_ = _second_dim_;\
		int & _THIRD_ = _third_dim_;

#pragma endregion

#pragma region bounds

	//checks that the thread falls within the bounds of the problem. this one is for 1d kernels
	#define BOUNDS_1D(_first_span_)\
		if((_FIRST_ < 0)||(_FIRST_ >= _first_span_)){return;} 

	//checks that the thread falls within the bounds of the problem. this one is for 2d kernels
	#define BOUNDS_2D(_first_span_, _second_span_)\
		if((_FIRST_ < 0)||(_SECOND_ < 0)||(_FIRST_ >= _first_span_)||(_SECOND_ >= _second_span_)){return;} 

	//checks that the thread falls within the bounds of the problem. this one is for 3d kernels
	#define BOUNDS_3D(_first_span_, _second_span_, _third_span_)\
		if((_FIRST_ < 0)||(_SECOND_ < 0)||(_THIRD_ < 0)||(_FIRST_ >= _first_span_)||(_SECOND_ >= _second_span_)||(_THIRD_ >= _third_span_)){return;} 

#pragma endregion

#pragma region cast operations

	//casts a larger dimension to a smaller one
	#define CAST_DOWN(_old_coord_, _new_max_) \
		((_old_coord_ - (_old_coord_ % _new_max_ ))/ _new_max_)

	//casts a smaller dimension to a larger one. tries to place the new coordinate in the middle of each 'segment'
	#define CAST_UP(_old_coord_, _old_max_, _new_max_) \
		((_old_coord_*(_new_max_/_old_max_))+(((_new_max_/_old_max_)-((_new_max_/_old_max_)%2))/2))

	//virtually transform a 2d tensor into a 1d tensor, and return the resulting linear id of the element pointed to by the given coordinates
	#define LINEAR_CAST(_maj_dim_, _min_dim_, _min_max_) \
		((_maj_dim_ * _min_max_) + _min_dim_)

	#define CARTESIAN_CAST()

#pragma endregion

#pragma region spatial operations

	#define FOR_MXN_INCLUSIVE(_new_maj_, _new_min_, _M_, _N_, _maj_max_, _min_max_, _origin_maj_, _origin_min_, ...)				 \
	int _maj_limit_ = (_M_ - (_M_ % 2)) / 2;																			   			 \
	int _min_limit_ = (_N_ - (_N_ % 2)) / 2;																						 \
		__pragma(unroll) for (int _neighbor_maj_ = -_maj_limit_; _neighbor_maj_ < (_maj_limit_ + (_M_ % 2)); _neighbor_maj_++) {	 \
			__pragma(unroll) for (int _neighbor_min_ = -_min_limit_; _neighbor_min_ < (_min_limit_ + (_N_ % 2)); _neighbor_min_++) { \
				int _new_maj_ = _origin_maj_ + _neighbor_maj_;																		 \
				int _new_min_ = _origin_min_ + _neighbor_min_;																	     \
				if((_new_maj_ < 0)||(_new_min_ < 0)||(_new_maj_ >= _maj_max_)||(_new_min_ >= _min_max_ )){continue;}				 \
				__VA_ARGS__;																										 \
			}																														 \
		}

	#define FOR_MXN_EXCLUSIVE(_new_maj_, _new_min_, _M_, _N_, _maj_max_, _min_max_, _origin_maj_, _origin_min_, ...)				 \
	int _maj_limit_ = (_M_ - (_M_ % 2)) / 2;																						 \
	int _min_limit_ = (_N_ - (_N_ % 2)) / 2;																						 \
		__pragma(unroll) for (int _neighbor_maj_ = -_maj_limit_; _neighbor_maj_ < (_maj_limit_ + (_M_ % 2)); _neighbor_maj_++) {	 \
			__pragma(unroll) for (int _neighbor_min_ = -_min_limit_; _neighbor_min_ < (_min_limit_ + (_N_ % 2)); _neighbor_min_++) { \
				int _new_maj_ = _origin_maj_ + _neighbor_maj_;																		 \
				int _new_min_ = _origin_min_ + _neighbor_min_;																		 \
				if((_new_maj_ < 0)||(_new_min_ < 0)||(_new_maj_ >= _maj_max_)||(_new_min_ >= _min_max_ )							 \
								  ||((_new_maj_ == _origin_maj_)&&(_new_min_ == _origin_min_))) {continue;}							 \
				__VA_ARGS__;																										 \
			}																														 \
		}

	#define FOR_3X3_INCLUSIVE(_new_maj_, _new_min_, _maj_max_, _min_max_, _origin_maj_, _origin_min_, ...) \
		FOR_MXN_INCLUSIVE(_new_maj_, _new_min_, 3, 3, _maj_max_, _min_max_, _origin_maj_, _origin_min_, __VA_ARGS__)

	#define FOR_NEIGHBOR(_new_maj_, _new_min_, _maj_max_, _min_max_, _origin_maj_, _origin_min_, ...) \
		FOR_MXN_EXCLUSIVE(_new_maj_, _new_min_, 3, 3, _maj_max_, _min_max_, _origin_maj_, _origin_min_, __VA_ARGS__)

#pragma endregion

#pragma region error checking

	#define SYNC_KERNEL(_kernel_)																 \
	{																							 \
		cudaDeviceSynchronize();																 \
		sk::Error = cudaGetLastError();															 \
		if(sk::Error != cudaSuccess) {															 \
			std::cout << "CUDA error at " << #_kernel_ << " - "									 \
			<< cudaGetErrorName(sk::Error) << ":" << cudaGetErrorString(sk::Error) << std::endl; \
			abort();													  						 \
			}																					 \
	}		

#pragma endregion

#pragma region miscellaneous operations

	#define sk_Copy(_to_, _from_, _length_)\
		__pragma(unroll) for (int i = 0; i < _length_; i++){\
			_to_[i] = _from_[i];\
		}

 #pragma endregion






