#pragma once

#include"external_libs.h"



//this is the new features branch

namespace sk {

	enum host_or_device { 
		host = 0,
		device = 1,
	};


	template <typename Type>
	struct Device_Ptr {

		Type* device_data = nullptr;

		uint num_dims = 0;

		uint dims[4] = { 1, 1, 1, 1 };

		Device_Ptr(uint dims_in[4], Type* device_data_in) {
			for (int i = 0; i < 4; i++) {
				dims[i] = dims_in[i];
				if (dims[i] > 1) num_dims++;
			}

			device_data = device_data_in;

		}

		__device__ uint first_dim() const { return dims[0]; }
		__device__ uint second_dim() const { return dims[1]; }
		__device__ uint third_dim() const { return dims[2]; }
		__device__ uint fourth_dim() const { return dims[3]; }

		__device__ Type& operator ()(int first_pos, int second_pos = 0, int third_pos = 0, int fourth_pos = 0) { return device_data[(((((first_pos * second_dim()) + second_pos) * third_dim()) + third_pos) * fourth_dim()) + fourth_pos]; }

	};


	template<typename Type>
	struct Tensor {


		/************************ data ************************/

		Type* device_data = nullptr;
		Type* host_data = nullptr;

		//for debug purposes
		std::string name = "uninitialized";

		uint num_dims = 0;

		uint dims[4] = { 1, 1, 1, 1 };

		__host__ uint first_dim() const { return dims[0]; }
		__host__ uint second_dim() const { return dims[1]; }
		__host__ uint third_dim() const { return dims[2]; }
		__host__ uint fourth_dim() const { return dims[3]; }

		bool synced = false;
		sk::host_or_device up_to_date = sk::host;

		//get data from host. checks if we've been messing around with the device data, and if so synchronises
		__host__ Type& operator ()(int first_coord, int second_coord = 0, int third_coord = 0, int fourth_coord = 0) {
			switch (synced) {
				case true: desync(host); break;
				case false: switch (up_to_date) {
					case device: sync(); desync(host); break;
					default: break;
				} break;
			}

			return host_data[(((((first_coord * second_dim()) + second_coord) * third_dim()) + third_coord) * fourth_dim()) + fourth_coord];
		}

		int num_elements() const { return first_dim() * second_dim() * third_dim() * fourth_dim(); }
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
			sk_Copy(dims, input.dims, 4);
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
			sk_Copy(dims, input.dims, 4);
			num_dims = input.num_dims;

			//this part is a bit nonsensical, I need to rethink it. it's trying to make sure it copies the up-to-date version of the input Tensor
			if ((!input.synced) && (input.up_to_date == device)) {
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

		//initialization with list of dims
		Tensor(std::vector<uint> in_dims, Type constant = 0, std::string in_name = "default") {
			name = in_name;
			int num_dims_in = in_dims.size();
			num_dims = 0;
			for (int i = 0; i < num_dims_in; i++) {
				int current_dim = in_dims[i];
				if (current_dim > 1) {
					dims[i] = in_dims[i];
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

			return(Device_Ptr<Type>(dims, device_data));
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
			dims[0] = input.size();

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
				dims[i] = input.dims(i);
			}

			desync(device);
			initialize_device_memory();
			cudaMemcpy((void*)device_data, (void*)input.device<Type>(), bytesize(), cudaMemcpyDeviceToDevice);
			input.unlock(); //probably a sloppy way to do this, but oh well
			sync();
		}

		//to array
		operator af::array() { return af::array((dim_t)first_dim(), (dim_t)second_dim(), (dim_t)third_dim(), (dim_t)fourth_dim(), host_data); }
		#endif

		#ifdef CV_VERSION

		//from Mat to Tensor
		void operator=(cv::Mat input) {
			num_dims = input.dims;
			bool has_channels = (input.channels() > 1);
			num_dims += has_channels;

			dims[0] = input.rows;
			dims[1] = input.cols;
			dims[2] = input.channels();

			desync(host);
			host_data = (Type*)input.data;
			sync();
		}

		//from Tensor to Mat
		operator cv::Mat() {
			return cv::Mat(dims[0], dims[1], cv::DataType<Type>::type, host_data);
		}

		#ifdef DOESNT_WORK
			//from GpuMat to Tensor
			void operator=(cv::cuda::GpuMat input) {
				cv::Mat temp;
				input.download(temp);

				num_dims = temp.dims;
				dims[0] = temp.rows;
				dims[1] = temp.cols;

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