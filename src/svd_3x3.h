#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#define USE_SCALAR_IMPLEMENTATION
#define USE_ACCURATE_RSQRT_IN_JACOBI_CONJUGATION
#define COMPUTE_V_AS_MATRIX
#define COMPUTE_U_AS_MATRIX
#include "Singular_Value_Decomposition_Preamble.hpp"

// Require column-major layout (AoS)
using np_array_t = pybind11::array_t< float, pybind11::array::f_style | pybind11::array::forcecast >;

// Fast 3x3 SVD  
std::list< np_array_t > svd_3x3(np_array_t& x) {
	if ( x.ndim()     != 2){ throw std::runtime_error("Input should be 2-D NumPy array"); }
  if ( x.shape()[1] != 3){ throw std::runtime_error("Input should have size [3,3]"); }
	np_array_t Um({ 3, 3 });
	np_array_t Sm({ 3 });
	np_array_t Vm({ 3, 3 });
	auto A = x.mutable_unchecked< 2 >();
	auto U = Um.mutable_unchecked< 2 >();
	auto S = Sm.mutable_unchecked< 1 >();
	auto V = Vm.mutable_unchecked< 2 >();
	#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"
	ENABLE_SCALAR_IMPLEMENTATION(Sa11.f = A(0,0);) ENABLE_SCALAR_IMPLEMENTATION(Sa21.f = A(1,0);) ENABLE_SCALAR_IMPLEMENTATION(Sa31.f = A(2,0);)
	ENABLE_SCALAR_IMPLEMENTATION(Sa12.f = A(0,1);) ENABLE_SCALAR_IMPLEMENTATION(Sa22.f = A(1,1);) ENABLE_SCALAR_IMPLEMENTATION(Sa32.f = A(2,1);)
	ENABLE_SCALAR_IMPLEMENTATION(Sa13.f = A(0,2);) ENABLE_SCALAR_IMPLEMENTATION(Sa23.f = A(1,2);) ENABLE_SCALAR_IMPLEMENTATION(Sa33.f = A(2,2);)
	#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp" 
	U(0,0) = Su11.f; U(0,1) = Su12.f; U(0,2) = Su13.f;
	U(1,0) = Su21.f; U(1,1) = Su22.f; U(1,2) = Su23.f;
	U(2,0) = Su31.f; U(2,1) = Su32.f; U(2,2) = Su33.f;
	S(0) = Sa11.f; S(1) = Sa22.f; S(2) = Sa33.f; 
	// note the transpose 
	V(0,0) = Sv11.f; V(0,1) = Sv21.f; V(0,2) = Sv31.f;
	V(1,0) = Sv12.f; V(1,1) = Sv22.f; V(1,2) = Sv32.f;
	V(2,0) = Sv13.f; V(2,1) = Sv23.f; V(2,2) = Sv33.f;
	auto out = std::list< np_array_t > { Um, Sm, Vm };
	return(out);
}

// Given a (3 x m) array x, computes (m/3) 3x3 SVDs 
template<  bool compute_uv, typename Lambda >
void svd_3x3_stream(np_array_t& x, Lambda f) {
	if (x.ndim() != 2){ throw std::runtime_error("Input should be 2-D NumPy array"); }
  if (x.shape()[0] != 3){ throw std::runtime_error("Input should have size [3,m]"); }
	if (x.shape()[1] % 3 != 0){ throw std::runtime_error("Input column size should divide by 3."); }
	
	// Input matrix
	np_array_t Um({ 3, 3 });
	auto A = x.mutable_unchecked< 2 >();
	
	// Singular values to report
	np_array_t Sm(3);
	auto S = Sm.mutable_unchecked< 1 >();
	
	size_t c = 0; // current column 
	const size_t n_svds = x.shape()[1] / 3;
	if constexpr (compute_uv){
		np_array_t Um({ 3, 3 });
		np_array_t Vm({ 3, 3 });
		auto V = Vm.mutable_unchecked< 2 >();
		auto U = Um.mutable_unchecked< 2 >();
		for (size_t i = 0; i < n_svds; ++i, c += 3){
			#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"
			ENABLE_SCALAR_IMPLEMENTATION(Sa11.f = A(0,c+0);) ENABLE_SCALAR_IMPLEMENTATION(Sa21.f = A(1,c+0);) ENABLE_SCALAR_IMPLEMENTATION(Sa31.f = A(2,c+0);)
			ENABLE_SCALAR_IMPLEMENTATION(Sa12.f = A(0,c+1);) ENABLE_SCALAR_IMPLEMENTATION(Sa22.f = A(1,c+1);) ENABLE_SCALAR_IMPLEMENTATION(Sa32.f = A(2,c+1);)
			ENABLE_SCALAR_IMPLEMENTATION(Sa13.f = A(0,c+2);) ENABLE_SCALAR_IMPLEMENTATION(Sa23.f = A(1,c+2);) ENABLE_SCALAR_IMPLEMENTATION(Sa33.f = A(2,c+2);)
			#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp" 
			U(0,0) = Su11.f; U(0,1) = Su12.f; U(0,2) = Su13.f;
			U(1,0) = Su21.f; U(1,1) = Su22.f; U(1,2) = Su23.f;
			U(2,0) = Su31.f; U(2,1) = Su32.f; U(2,2) = Su33.f;
			S(0) = Sa11.f; S(1) = Sa22.f; S(2) = Sa33.f;  
			V(0,0) = Sv11.f; V(0,1) = Sv21.f; V(0,2) = Sv31.f; // note the transpose
			V(1,0) = Sv12.f; V(1,1) = Sv22.f; V(1,2) = Sv32.f;
			V(2,0) = Sv13.f; V(2,1) = Sv23.f; V(2,2) = Sv33.f;
			f(Um, Sm, Vm); // call method
		}
	} else {
		for (size_t i = 0; i < n_svds; ++i, c += 3){
			#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"
			ENABLE_SCALAR_IMPLEMENTATION(Sa11.f = A(0,c+0);) ENABLE_SCALAR_IMPLEMENTATION(Sa21.f = A(1,c+0);) ENABLE_SCALAR_IMPLEMENTATION(Sa31.f = A(2,c+0);)
			ENABLE_SCALAR_IMPLEMENTATION(Sa12.f = A(0,c+1);) ENABLE_SCALAR_IMPLEMENTATION(Sa22.f = A(1,c+1);) ENABLE_SCALAR_IMPLEMENTATION(Sa32.f = A(2,c+1);)
			ENABLE_SCALAR_IMPLEMENTATION(Sa13.f = A(0,c+2);) ENABLE_SCALAR_IMPLEMENTATION(Sa23.f = A(1,c+2);) ENABLE_SCALAR_IMPLEMENTATION(Sa33.f = A(2,c+2);)
			#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp" 
			S(0) = Sa11.f; S(1) = Sa22.f; S(2) = Sa33.f;  
			f(Sm); // call method
		}
	}
}