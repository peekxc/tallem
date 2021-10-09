// cppimport
#include <carma>
#include <armadillo>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

template< bool compute_uv, typename Lambda >
inline void svd_3x3_carma(np_array_t& x, Lambda&& f) noexcept {
	
	// Input matrix
	auto A = x.mutable_unchecked< 2 >();
	
	// Singular values to report
	arma::fmat::fixed< 3, 1 > S;

	size_t c = 0; // current column 
	const size_t n_svds = x.shape()[1] / 3;
		
	// Prepare singular vectors
	arma::fmat::fixed< 3, 3 > U;
	arma::fmat::fixed< 3, 3 > V;

	for (size_t i = 0; i < n_svds; ++i, c += 3){
		#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"
		ENABLE_SCALAR_IMPLEMENTATION(Sa11.f = A(0,c+0);) ENABLE_SCALAR_IMPLEMENTATION(Sa21.f = A(1,c+0);) ENABLE_SCALAR_IMPLEMENTATION(Sa31.f = A(2,c+0);)
		ENABLE_SCALAR_IMPLEMENTATION(Sa12.f = A(0,c+1);) ENABLE_SCALAR_IMPLEMENTATION(Sa22.f = A(1,c+1);) ENABLE_SCALAR_IMPLEMENTATION(Sa32.f = A(2,c+1);)
		ENABLE_SCALAR_IMPLEMENTATION(Sa13.f = A(0,c+2);) ENABLE_SCALAR_IMPLEMENTATION(Sa23.f = A(1,c+2);) ENABLE_SCALAR_IMPLEMENTATION(Sa33.f = A(2,c+2);)
		#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp" 
		U.at(0,0) = Su11.f; U.at(0,1) = Su12.f; U.at(0,2) = Su13.f;
		U.at(1,0) = Su21.f; U.at(1,1) = Su22.f; U.at(1,2) = Su23.f;
		U.at(2,0) = Su31.f; U.at(2,1) = Su32.f; U.at(2,2) = Su33.f;
		S.at(0,0) = Sa11.f; S.at(1,0) = Sa22.f; S.at(2,0) = Sa33.f;  
		V.at(0,0) = Sv11.f; V.at(0,1) = Sv21.f; V.at(0,2) = Sv31.f; // note the transpose
		V.at(1,0) = Sv12.f; V.at(1,1) = Sv22.f; V.at(1,2) = Sv32.f;
		V.at(2,0) = Sv13.f; V.at(2,1) = Sv23.f; V.at(2,2) = Sv33.f;
	}

	if constexpr(compute_uv){ f(U, S, V); } else { f(S); }
}
/* <% setup_pybind11(cfg) %> */