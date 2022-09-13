

# SKELETON KEY

this library is a collection of code I've been using to reduce much of the (often relentless) boilerplate needed to write with CUDA, while still keeping 
things conceptually similar to raw CUDA. I find it particularly useful for computer vision and cellular automata, tasks which operate upon a 2d matrix. 
Specifically, the library is designed to bring into focus the abstract model of CUDA as a tool which works spatially on matrices.

This library has three parts:

The launch manager

The Tensor struct

A series of macros for use within kernels


## code example:
```
__global__ void add_by_element (sk::Device_Ptr<int> A, sk::Device_Ptr<int> B) {
	DIMS_2D(col, row);
	BOUNDS_2D(A.spans[0], A.spans[1]);

	A(col, row) += B(col, row);
}


void launch_add_by_element(sk::Tensor<int>& A){

	sk::Tensor<int> B({512, 512}, 1);

	sk::configure::kernel_2d(A.spans[0], A.spans[1]);
	add_by_element<<<LAUNCH>>>(A, B);
	SYNC_KERNEL(add_by_element);

}
```

## using the launch manager:

the launch manager is a static struct which configures and stores your launch parameters for you (i.e. <<<num_blocks, threads_per_block>>>). 
since it is a static struct, there is (in theory) only one launch manager object which is generated at compile time, and hovers around holding
your launch parameters. 

it has three functions, which can be invoked anywhere at any time: kernel_1d(span), kernel_2d(span, span), and kernel_3d(span, span, span). 
calling one of these functions will configure the launch parameters stored in the manager object to create launch parameters describing 
a kernel of the given dimensionality and the given size. when you want to actually launch the kernel, you pass in the num_blocks and the
threads_per_block stored by the launch manager. I usually do this with the LAUNCH macro (\<\<\<LAUNCH>>>), which just evaluates to
sk::configure::num_blocks, sk::configure::threads_per_block 


## using the macros:

Skeleton Key provides a series of macros to be used within kernels. The two main ones which are usually called at the start of every kernel are
the DIMS macros (DIMS_1D(dim), DIMS_2D(dim, dim), DIMS_3D(dim, dim, dim)), and the BOUNDS macros (BOUNDS_1D(), BOUNDS_2D(), BOUNDS_3D()). 
The DIMS macros accept an argument for each dimension, and aquire the coordinates of a thread for you so you dont have to derive them with
"blockIdx.x * blockDim.x + threadDim.x". simply enter the names of your dimensions as arguments (e.g. row, col), and the DIMS macro will 
define coordinate variables with those names using the traditional/verbose method mentioned previously, which you can use in the rest of the kernel.

the BOUNDS macros (BOUNDS_1D(span), BOUNDS_2D(span, span), BOUNDS_3D(span, span, span)) check to make sure a thread is within the prescribed spatial bounds 
of the problem. the BOUNDS macros are equivalent to manually writing something like: 
```if((row < 0)||(col < 0)||(row >= rows)||(col >= cols)){return;}```
Since kernels are often launched with excess threads which will cause memory errors if they fire, including a check like this ensures that these threads 
will shut themselves down before wreaking any havoc. The BOUNDS macros expect as many arguments as they have dimensions, each one an integer which 
describes the size of that dimension in elements.

there is one more series of macros included, which is more experimental. these are the FOR_MXN macros (FOR_MXN_EXCLUSIVE(), FOR_MXN_INCLUSIVE()). These create a set of two for loops which iterate over a square in space. they essentially represent a loop which plays out in 2d space as a M x N square centered on the starting thread. They are useful for 2D spatial convolution (as found in computer vision tasks). EXCLUSIVE versus INCLUSIVE changes whether the "home" coordinate is skipped or included. to use it, the first two arguments will be the names of your indices that you will be using later in the loop. These correspond to coordinates in the kernel context (i.e. your first and second dims plus some small offset). the third and fourth arguments denote the size of the loop: the values M and N as mentioned in the name of the macro. after these four arguments, enter the code to be executed at each step of the loop. example usage:

```
FOR_MXN_EXCLUSIVE(n_first, n_second, 3, 3,
	//your code goes here. don't worry about commas, the macro is magic.
);
```

there are also a couple of derived versions: FOR_3X3_INCLUSIVE and FOR_NEIGHBOR. these are simply shorthand which allows the programmer to bypass specifying the size of the loop (since conceptually the sizes are implicit). they call FOR_MXN_INCLUSIVE(3,3) and FOR_MXN_EXCLUSIVE(3,3) respectively.


## using the Tensor struct:

the Tensor struct is a straightforward container for 1d, 2d, 3d, and 4d matrices (a.k.a. tensors). It automatically manages transfer/synchronization 
of data between host and device for the user. It tries not to transfer data between the host and device until it is necessary to do so, for performance 
reasons. The memory management is automatic, and the user should never have to manually call the sync() or desync() functions in the course of normal use.

both the device and host data are accessable through the parenthetical operator, (coord, coord, ...). to access the device data from within the 
kernel, an additional step must be taken. In the kernel definition, the user must specify sk::Device_Ptr rather than sk::Tensor. this will make 
the compiler invoke the typecast between them. accessing the data in the Device_Ptr is done in the exact same way as on the host with Tensor, by
using the parenthetical operator. note that Tensor and Device_Ptr are both templates, therefore the template argument must be specified when using 
them (as for instance one does when creating a std::vector).

The long term goal for this struct is to make it interoperable with equivalent matrix containers from many different libraries through the use of
copy constructors and copy assignment operators written for those specific libraries. This will make it possible to get interoperability up and 
running quickly between any two libraries which sk::Tensor knows about. So far I've got it set up to work with the ArrayFire library and OpenCV.
Let me know your preferred library and I can try to add it for you! Or if you'd like to try yourself, you only have to write two functions. I will
write some documentation on how to do this, pending interest.


## planned features:
	
interoperability between sk::Tensor and more external libraries
	
allow user configuration for names/cardinality of dimensions within Tensor






