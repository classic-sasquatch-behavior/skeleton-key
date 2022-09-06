#pragma once
#include<cuda.h>
#include<cuda_runtime.h>



namespace sk {
	namespace configure {
		
		inline uint block_dim_x = 0;
		inline uint block_dim_y = 0;
		inline uint block_dim_z = 0;

		inline uint grid_dim_x = 0;
		inline uint grid_dim_y = 0;
		inline uint grid_dim_z = 0;

		inline dim3 num_blocks(0,0,0);
		inline dim3 threads_per_block(0,0,0);
		
		static void kernel_1d(int x_length) {
			block_dim_x = 1024;
			block_dim_y = 1;
			block_dim_z = 1;

			grid_dim_x = ((x_length - (x_length % block_dim_x)) / block_dim_x) + 1;
			grid_dim_y = 1;
			grid_dim_z = 1;

			num_blocks = { grid_dim_x, grid_dim_y, grid_dim_z};
			threads_per_block = { block_dim_x, block_dim_y, block_dim_z };
		}

		static void kernel_2d(int x_length, int y_length) {
			block_dim_x = 32;
			block_dim_y = 32;
			block_dim_z = 1;

			grid_dim_x = ((x_length - (x_length % block_dim_x)) / block_dim_x) + 1;
			grid_dim_y = ((y_length - (y_length % block_dim_y)) / block_dim_y) + 1;
			grid_dim_z = 1;

			num_blocks = { grid_dim_x, grid_dim_y, grid_dim_z };
			threads_per_block = { block_dim_x, block_dim_y, block_dim_z };
		}

		static void kernel_3d(int x_length, int y_length, int z_length) {
			block_dim_x = 16;
			block_dim_y = 16;
			block_dim_z = 4;

			grid_dim_x = ((x_length - (x_length % block_dim_x)) / block_dim_x) + 1;
			grid_dim_y = ((y_length - (y_length % block_dim_y)) / block_dim_y) + 1;
			grid_dim_z = ((z_length - (z_length % block_dim_z)) / block_dim_z) + 1;

			num_blocks = { grid_dim_x, grid_dim_y, grid_dim_z };
			threads_per_block = { block_dim_x, block_dim_y, block_dim_z };
		}
	}

	#define LAUNCH sk::configure::num_blocks, sk::configure::threads_per_block
}
