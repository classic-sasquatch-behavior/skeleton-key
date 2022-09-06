#pragma once
#include<cuda.h>
#include<cuda_runtime.h>
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




namespace sk {

	enum host_or_device { 
		host = 0,
		device = 1,
	};


	template <typename Type>
	struct Device_Ptr {

		Type* device_data = nullptr;

		uint num_dims = 0;

		uint spans[4] = { 1, 1, 1, 1 };

		Device_Ptr(uint spans_in[4], Type* device_data_in) {
			for (int i = 0; i < 4; i++) {
				spans[i] = spans_in[i];
				if (spans[i] > 1) num_dims++;
			}

			device_data = device_data_in;

		}

		__device__ uint maj() const { return spans[0]; }
		__device__ uint min() const { return spans[1]; }
		__device__ uint cub() const { return spans[2]; }
		__device__ uint hyp() const { return spans[3]; }

		__device__ Type& operator ()(int maj_pos, int min_pos = 0, int cub_pos = 0, int hyp_pos = 0) { return device_data[(((((maj_pos * min()) + min_pos) * cub()) + cub_pos) * hyp()) + hyp_pos]; }

	};


	template<typename Type>
	struct Tensor {


		/************************ data ************************/

		Type* device_data = nullptr;
		Type* host_data = nullptr;

		//for debug purposes
		std::string name = "uninitialized";

		uint num_dims = 0;

		uint spans[4] = { 1, 1, 1, 1 };

		__host__ uint maj() const { return spans[0]; }
		__host__ uint min() const { return spans[1]; }
		__host__ uint cub() const { return spans[2]; }
		__host__ uint hyp() const { return spans[3]; }

		bool synced = false;
		sk::host_or_device up_to_date = sk::host;

		//get data from host. checks if we've been messing around with the device data, and if so synchronises
		__host__ Type& operator ()(int maj, int min = 0, int cub = 0, int hyp = 0) {
			switch (synced) {
				case true: desync(host); break;
				case false: switch (up_to_date) {
					case device: sync(); desync(host); break;
					default: break;
				} break;
			}

			return host_data[(((((maj * min) + min) * cub) + cub) * hyp) + hyp];
		}

		int num_elements() const { return maj() * min() * cub() * hyp(); }
		int bytesize() const { return (num_elements() * sizeof(Type)); }


		/************************ constructors, etc. ************************/

		//default constructor
		Tensor(Type constant = 0) {
			initialize_memory();
			fill_memory(constant);
			ready();
		}
		
		//copy constructor
		Tensor(const Tensor& input) {
			name = input.name;
			sk_Copy(spans, input.spans, 4);
			num_dims = input.num_dims;
			initialize_memory();

			//this part is a bit nonsensical, I need to rethink it. it's trying to make sure it copies the up-to-date version of the input Tensor
			if (!input.synced && (input.up_to_date == device)) {
				desync(device);
				cudaMemcpy(device_data, input.device_data, bytesize(), cudaMemcpyDeviceToDevice);
				sync();
			}

			else {
				desync(host);
				for (int i = 0; i < input.num_elements(); i++) {
					host_data[i] = input.host_data[i];
				}
				sync();
			}
		}

		//copy assignment operator
		void operator=(const Tensor& input) {
			name = input.name;
			sk_Copy(spans, input.spans, 4);
			num_dims = input.num_dims;

			//this part is a bit nonsensical, I need to rethink it. it's trying to make sure it copies the up-to-date version of the input Tensor
			if (!input.synced && (input.up_to_date == device)) {
				desync(device);
				cudaMemcpy(device_data, input.device_data, bytesize(), cudaMemcpyDeviceToDevice);
				sync();
			}

			else {
				desync(host);
				for (int i = 0; i < input.num_elements(); i++) {
					host_data[i] = input.host_data[i];
				}
				sync();
			}
		}

		//initialization with list of spans
		Tensor(std::vector<uint> in_spans, Type constant = 0, std::string in_name = "default") {
			name = in_name;
			int num_dims_in = in_spans.size();
			num_dims = 0;
			for (int i = 0; i < num_dims_in; i++) {
				int current_span = in_spans[i];
				if (current_span > 1) {
					spans[i] = in_spans[i];
					num_dims++;
				}
			}
			initialize_memory();
			fill_memory(constant);
			ready();
		}


		//destructor
		~Tensor() {
			cudaFree(device_data);
			delete[] host_data;
		}


		/************************ functions ***********************/

		//checks if host and device are the same, and if not brings them up to date with each other
		void sync() {
			if (!synced) {
				switch (up_to_date) {
				case sk::host: upload(); break;
				case sk::device: download(); break;
				}
				ready();
			}
		}

		//signals that the host and device data may be different from one another. called automatically when switching from one domain to the other
		void desync(sk::host_or_device changing) {
			up_to_date = changing;
			synced = false;
		}

		//signals that the host and device portions of the tensor are synchronized without actually doing anything, used during initialization
		void ready() {
			synced = true;
		}

		//allocates memory for host and device
		void initialize_memory() {
			initialize_host_memory();
			initialize_device_memory();
		}

		//allocates memory for host
		void initialize_host_memory() {
			host_data = new Type[num_elements()];
		}

		//allocates memory for device
		void initialize_device_memory() {
			cudaMalloc((void**)&device_data, bytesize());
		}

		//fills memory on host and device with a constant
		void fill_memory(Type input) {
			fill_host_memory(input);
			fill_device_memory(input);
		}

		//fills memory on host with a constant
		void fill_host_memory(Type input) {
			for (int i = 0; i < num_elements(); i++) {
				host_data[i] = input;
			}
		}

		//fills memory on device with a constant
		void fill_device_memory(Type input) {
			cudaMemset(device_data, input, bytesize());
		}

		//copies data from host to device
		void upload() {
			cudaMemcpy(device_data, host_data, bytesize(), cudaMemcpyHostToDevice);
		}

		//copies data from device to host
		void download() {
			cudaMemcpy(host_data, device_data, bytesize(), cudaMemcpyDeviceToHost);
		}

		//converts Tensor to Device_Ptr, called implicitly when Tensor is passed to a kernel. checks if we've been messing with the host data, and if so synchronises
		operator Device_Ptr<Type>() {
			switch (synced) {
				case true: desync(device); break;
				case false: if (up_to_date == host) {
					sync();
					desync(device);
				} break;
			}

			return(Device_Ptr<Type>(spans, device_data));
		}


		/************************ interoperability ************************/

		//to pointer - returns a pointer to the first element. uses the parenthetical operator to ensure synchronization.
		__host__ operator Type*() {
			return &((*this)(0));
		}

		//to numerical type - returns the first element. uses the parenthetical operator to ensure synchronization.
		__host__ operator Type() {
			return (*this)(0);
		}

		//comparison to numerical type
		bool operator ==(Type to_compare) {
			Type self = (*this)(0);
			return to_compare == self;
		}

		//from vector
		void operator=(std::vector<Type> input) {
			desync(host);
			num_dims = 1;
			spans[0] = input.size();

			std::copy(input.begin(), input.end(), host_data);

			sync();
		}

		//to vector
		operator std::vector<Type>() { return std::vector<Type>(host_data, host_data + num_elements()); }

		#ifdef AF_API_VERSION
		//from array
		void operator=(af::array& input) {

			num_dims = input.numdims();
			for (int i = 0; i < num_dims; i++) {
				spans[i] = input.dims(i);
			}

			desync(device);
			initialize_device_memory();
			cudaMemcpy((void*)device_data, (void*)input.device<Type>(), bytesize(), cudaMemcpyDeviceToDevice);
			input.unlock(); //probably a sloppy way to do this, but oh well
			sync();
		}

		//to array
		operator af::array() { return af::array((dim_t)maj(), (dim_t)min(), (dim_t)cub(), (dim_t)hyp(), host_data); }
		#endif

		#ifdef CV_VERSION

		//from Mat to Tensor
		void operator=(cv::Mat input) {
			num_dims = input.dims;
			bool has_channels = (input.channels() > 1);
			num_dims += has_channels;

			spans[0] = input.rows;
			spans[1] = input.cols;
			spans[2] = input.channels();

			desync(host);
			host_data = (Type*)input.data;
			sync();
		}

		//from Tensor to Mat
		operator cv::Mat() {
			return cv::Mat(spans[0], spans[1], cv::DataType<Type>::type, host_data);
		}

		#ifdef DOESNT_WORK
			//from GpuMat to Tensor
			void operator=(cv::cuda::GpuMat input) {
				cv::Mat temp;
				input.download(temp);

				num_dims = temp.dims;
				spans[0] = temp.rows;
				spans[1] = temp.cols;

				desync(host);
				std::copy((Type*)temp.data, (Type*)temp.data[temp.rows * temp.cols], host_data);
				sync();
			}

			cv::cuda::GpuMat make_gpumat() {
				cv::Mat temp;
				temp = *this;
				cv::cuda::GpuMat result = *new cv::cuda::GpuMat();
				result.upload(temp);
				return result;
			}
		

			//from Tensor to GpuMat
			operator cv::cuda::GpuMat() {
				return make_gpumat();
			}
		#endif

		#endif

	};
}