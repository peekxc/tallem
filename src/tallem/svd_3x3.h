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

using namespace pybind11::literals;
namespace py = pybind11;

// Stream SVD where one of the dimensions is 2
template<  bool compute_uv, size_t nr, size_t nc, typename Lambda >
void svd2_stream(np_array_t& x, Lambda f) {
	static_assert(nr <= 2 && nc <= 2, "Streaming svd only applicable to svds w/ max dimension 3.");
	static_assert(nr == 2 || nc == 2, "Streaming svd only applicable to 3x* or *x3 svds.");
	if (x.ndim() != 2){ throw std::runtime_error("Input should be 2-D NumPy array"); }
  if (x.shape()[0] != nr){ throw std::runtime_error("Input should have size [3,m]"); }
	if (x.shape()[1] % nc != 0){ throw std::runtime_error("Input column size should divide by 3."); }
	
	// Input matrix
	auto A = x.mutable_unchecked< 2 >();
	
	// Singular values to report
	np_array_t Sm(std::min(nr, nc));
	auto S = Sm.mutable_unchecked< 1 >();

	size_t cc = 0; // current column 
	const size_t n_svds = x.shape()[1] / nc;
		
	// Prepare singular vectors
	np_array_t Um({ nr, nr });
	np_array_t Vm({ nc, nc });
	auto V = Vm.mutable_unchecked< 2 >();
	auto U = Um.mutable_unchecked< 2 >();

	// Temporary variables 
	float a,b,c,d;
	float phi, theta; 
	float ct,st,cp,sp;
	float s1, s2, sgn1, sgn2; 

	for (size_t i = 0; i < n_svds; ++i, cc += nc){
		if constexpr(nr == 2 && nc == 2){
			a = A(0,cc+0), b = A(0,cc+1), c = A(1,cc+0), d = A(1,cc+1);
		} else if constexpr(nr == 2 && nc == 1){
			a = A(0,cc+0), b = 0, c = A(1,cc+0), d = 0;
		} else {
			a = A(0,cc+0), b = A(0,cc+1), c = 0, d = 0;
		}
		
		theta = 0.5*atan2(2*a*c + 2*b*d, a*a + b*b - c*c - d*d);
		phi = 0.5*atan2(2*a*b + 2*c*d, a*a - b*b + c*c - d*d); 
		ct = cos(theta), st = sin(theta), cp = cos(phi), sp = sin(phi);
		s1 = a*a + b*b + c*c + d*d;
		s2 = std::sqrt(std::pow(a*a + b*b - c*c - d*d, 2) + 4*std::pow(a*c + b*d, 2));
		sgn1 = ((a*ct+c*st)*cp + (b*ct + d*st)*sp) >= 0 ? 1 : -1; 
		sgn2 = ((a*st-c*ct)*sp + (-b*st + d*ct)*cp) >= 0 ? 1 : -1;

		if constexpr(nr == 2 && nc == 2){
			U(0,0) = ct, U(0,1) = -st, U(1,0) = st, U(1,1) = ct;
			V(0,0) = sgn1*cp, V(0,1) = -sgn2*sp, V(1,0) = sgn1*sp, V(1,1) = sgn2*cp;
			S(0) = std::sqrt((s1 + s2)/2);
			S(1) = std::sqrt((s1 - s2)/2);
		} else if constexpr(nr == 2 && nc == 1){
			U(0,0) = ct, U(0,1) = -st, U(1,0) = st, U(1,1) = ct;
			V(0,0) = sgn1*cp;
			S(0) = std::sqrt((s1 + s2)/2);
		} else {
			U(0,0) = ct;
			V(0,0) = sgn1*cp, V(0,1) = -sgn2*sp, V(1,0) = sgn1*sp, V(1,1) = sgn2*cp;
			S(0) = std::sqrt((s1 + s2)/2);
		}

		// Call the function 
		if constexpr(compute_uv){ f(Um, Sm, Vm); } else { f(Sm); }
	}

}

// Stream SVD where one of the dimensions is 3
template<  bool compute_uv, size_t nr, size_t nc, typename Lambda >
void svd3_stream(np_array_t& x, Lambda f) {
	static_assert(nr <= 3 && nc <= 3, "Streaming svd only applicable to svds w/ max dimension 3.");
	static_assert(nr == 3 || nc == 3, "Streaming svd only applicable to 3x* or *x3 svds.");
	if (x.ndim() != 2){ throw std::runtime_error("Input should be 2-D NumPy array"); }
  if (x.shape()[0] != nr){ throw std::runtime_error("Input should have size [3,m]"); }
	if (x.shape()[1] % nc != 0){ throw std::runtime_error("Input column size should divide by 3."); }
	
	// Input matrix
	auto A = x.mutable_unchecked< 2 >();
	
	// Singular values to report
	np_array_t Sm(std::min(nr, nc));
	auto S = Sm.mutable_unchecked< 1 >();

	size_t c = 0; // current column 
	const size_t n_svds = x.shape()[1] / nc;
		
	// Prepare singular vectors
	np_array_t Um({ nr, nr });
	np_array_t Vm({ nc, nc });
	auto V = Vm.mutable_unchecked< 2 >();
	auto U = Um.mutable_unchecked< 2 >();

	for (size_t i = 0; i < n_svds; ++i, c += nc){
		if constexpr(nr == 3 && nc == 3){
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
		} else if constexpr (nr == 3 && nc == 2){
			#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"
			ENABLE_SCALAR_IMPLEMENTATION(Sa11.f = A(0,c+0);) ENABLE_SCALAR_IMPLEMENTATION(Sa21.f = A(1,c+0);) ENABLE_SCALAR_IMPLEMENTATION(Sa31.f = A(2,c+0);)
			ENABLE_SCALAR_IMPLEMENTATION(Sa12.f = A(0,c+1);) ENABLE_SCALAR_IMPLEMENTATION(Sa22.f = A(1,c+1);) ENABLE_SCALAR_IMPLEMENTATION(Sa32.f = A(2,c+1);)
			ENABLE_SCALAR_IMPLEMENTATION(Sa13.f = 0;) ENABLE_SCALAR_IMPLEMENTATION(Sa23.f = 0;) ENABLE_SCALAR_IMPLEMENTATION(Sa33.f = 0;)
			#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp" 
			U(0,0) = Su11.f; U(0,1) = Su12.f; U(0,2) = Su13.f;
			U(1,0) = Su21.f; U(1,1) = Su22.f; U(1,2) = Su23.f;
			U(2,0) = Su31.f; U(2,1) = Su32.f; U(2,2) = Su33.f;
			S(0) = Sa11.f; S(1) = Sa22.f; 
			V(0,0) = Sv11.f; V(0,1) = Sv21.f; // note the transpose
			V(1,0) = Sv12.f; V(1,1) = Sv22.f;
		} else if constexpr (nr == 2 && nc == 3){
			#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"
			ENABLE_SCALAR_IMPLEMENTATION(Sa11.f = A(0,c+0);) ENABLE_SCALAR_IMPLEMENTATION(Sa21.f = A(1,c+0);) ENABLE_SCALAR_IMPLEMENTATION(Sa31.f = 0;)
			ENABLE_SCALAR_IMPLEMENTATION(Sa12.f = A(0,c+1);) ENABLE_SCALAR_IMPLEMENTATION(Sa22.f = A(1,c+1);) ENABLE_SCALAR_IMPLEMENTATION(Sa32.f = 0;)
			ENABLE_SCALAR_IMPLEMENTATION(Sa13.f = A(0,c+2);) ENABLE_SCALAR_IMPLEMENTATION(Sa23.f = A(2,c+1);) ENABLE_SCALAR_IMPLEMENTATION(Sa33.f = 0;)
			#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp" 
			U(0,0) = Su11.f; U(0,1) = Su12.f;
			U(1,0) = Su21.f; U(1,1) = Su22.f; 
			S(0) = Sa11.f; S(1) = Sa22.f; 
			V(0,0) = Sv11.f; V(0,1) = Sv21.f; V(0,2) = Sv31.f; // note the transpose
			V(1,0) = Sv12.f; V(1,1) = Sv22.f; V(1,2) = Sv32.f;
			V(2,0) = Sv13.f; V(2,1) = Sv23.f; V(2,2) = Sv33.f;
		} else if constexpr (nr == 3 && nc == 1){
			#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"
			ENABLE_SCALAR_IMPLEMENTATION(Sa11.f = A(0,c+0);) ENABLE_SCALAR_IMPLEMENTATION(Sa21.f = A(1,c+0);) ENABLE_SCALAR_IMPLEMENTATION(Sa31.f = A(2,c+0);)
			ENABLE_SCALAR_IMPLEMENTATION(Sa12.f = 0;) ENABLE_SCALAR_IMPLEMENTATION(Sa22.f = 0;) ENABLE_SCALAR_IMPLEMENTATION(Sa32.f = 0;)
			ENABLE_SCALAR_IMPLEMENTATION(Sa13.f = 0;) ENABLE_SCALAR_IMPLEMENTATION(Sa23.f = 0;) ENABLE_SCALAR_IMPLEMENTATION(Sa33.f = 0;)
			#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp" 
			U(0,0) = Su11.f; U(0,1) = Su12.f; U(0,2) = Su13.f;
			U(1,0) = Su21.f; U(1,1) = Su22.f; U(1,2) = Su23.f;
			U(2,0) = Su31.f; U(2,1) = Su32.f; U(2,2) = Su33.f;
			S(0) = Sa11.f; 
			V(0,0) = Sv11.f; 
		} else if constexpr (nr == 1 && nc == 3){
			#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"
			ENABLE_SCALAR_IMPLEMENTATION(Sa11.f = A(0,c+0);) ENABLE_SCALAR_IMPLEMENTATION(Sa21.f = 0;) ENABLE_SCALAR_IMPLEMENTATION(Sa31.f = 0;)
			ENABLE_SCALAR_IMPLEMENTATION(Sa12.f = A(0,c+1);) ENABLE_SCALAR_IMPLEMENTATION(Sa22.f = 0;) ENABLE_SCALAR_IMPLEMENTATION(Sa32.f = 0;)
			ENABLE_SCALAR_IMPLEMENTATION(Sa13.f = A(0,c+2);) ENABLE_SCALAR_IMPLEMENTATION(Sa23.f = 0;) ENABLE_SCALAR_IMPLEMENTATION(Sa33.f = 0;)
			#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp" 
			U(0,0) = Su11.f; U(0,1) = Su12.f; U(0,2) = Su13.f;
			U(1,0) = Su21.f; U(1,1) = Su22.f; U(1,2) = Su23.f;
			U(2,0) = Su31.f; U(2,1) = Su32.f; U(2,2) = Su33.f;
			S(0) = Sa11.f; 
			V(0,0) = Sv11.f; 
		} 
		
		// Call the function 
		if constexpr(compute_uv){ f(Um, Sm, Vm); } else { f(Sm); }

	} // end for loop 
}

// https://arxiv.org/pdf/1706.04129.pdf
constexpr size_t rosenburg_pair(const size_t x, const size_t y){
	return(x >= y ? x*(x+2) - y : y*y + x);
}

template< typename array_t = np_array_t >
array_t np_matrix(size_t width, size_t height, float* data_ptr  = nullptr){
 return array_t(
		py::buffer_info(
			data_ptr,
			sizeof(float), //itemsize
			py::format_descriptor< float >::format(),
			2, // ndim
			std::vector<size_t> { width, height }, // shape
			std::vector<size_t> { height * sizeof(float), sizeof(float)}  // strides
		)
	);
}


extern "C" {
	extern int sgesvd	(
		char* jobu, char* jobvt, int* m, int* n, float* a, int* lda, float* s, float* u, int* ldu, 
		float* vt, int* ldvt, float* work, int* lwork, int* info
	);	
}

//using np_carray_t = pybind11::array_t< float, pybind11::array::c_style | pybind11::array::forcecast >;

// Computes a stream of thin SVD's on every (r x n) submatrix of x
template< bool compute_uv, typename Lambda >
void svdn_stream(np_array_t& x, int n, Lambda f){
	if (x.ndim() != 2){ throw std::runtime_error("Input should be 2-D NumPy array"); }
	if (x.shape()[1] % n != 0){ throw std::runtime_error("Input column size should divide by 3."); }
	
	// Input matrix
	int m  =  x.shape()[0];
	int min_dim = std::min(m, n);
	int lda = m, ldu = m, ldvt = min_dim;
	float* A = x.mutable_data();
	
	// Singular values to report
	np_array_t Sm(min_dim);
	float* S = Sm.mutable_data();

	size_t c = 0; // current column 
	const size_t n_svds = x.shape()[1] / n;
		
	// Prepare singular vectors
	np_array_t Um({ ldu, min_dim });
	np_array_t Vm({ ldvt, n });
	float* U = Um.mutable_data();
	float* VT = Vm.mutable_data();
	int LWORK = std::max(3*min_dim+std::max(m,n),5*min_dim);
	std::vector< float > WORK(LWORK); 
	
	// option 'S' => the first min(m,n) columns of U (the left singular vectors) are returned in the array U;
	// option 'O' => the first min(m,n) columns of U (the left singular vectors) are overwritten on the array A
	auto write_str = std::string("S"); 
	char* overwrite = (char*) write_str.c_str();

	// Execute all the SVD computations
	int status = 0; 
	for (size_t i = 0; i < n_svds; ++i, c += n){
		sgesvd(overwrite, overwrite, &m, &n, A+(c*m), &lda, S, U, &ldu, VT, &ldvt, WORK.data(), &LWORK, &status);
		if constexpr (compute_uv){ f(Um, Sm, Vm); } else { f(Sm); }
	}
}

std::list< np_array_t > lapack_svd(np_array_t& x){
	int m = x.shape()[0], n = x.shape()[1];
	int min_dim = std::min(m, n);
	int lda = m, ldu = m, ldvt = min_dim;
	
	// Create the output arrays 
	np_array_t sm = np_array_t(min_dim);
	np_array_t um = np_matrix(ldu, min_dim);
	np_array_t vt = np_matrix(ldvt, n);

	// Extract the pointers 
	float* A = x.mutable_data();
	float* U = um.mutable_data();
	float* S = sm.mutable_data();
	float* VT = vt.mutable_data();
	int LWORK = std::max(3*min_dim+std::max(m,n),5*min_dim);
	std::vector< float > WORK(LWORK); 
	int status = 0; 
	
	// option 'O' => the first min(m,n) columns of U (the left singular vectors) are overwritten on the array A
	auto write_str = std::string("S"); 
	char* overwrite = (char*) write_str.c_str();
	
	// Call the LAPACK svd
	sgesvd(overwrite, overwrite, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, WORK.data(), &LWORK, &status);
	// py::print("status: ", status);

	// Record the outputs
	std::list< np_array_t > out; 
	out.push_back(um);
	out.push_back(sm);
	out.push_back(vt);
	return(out);
}


// Fast SVD streaming 
template< typename Lambda >
void fast_svd_stream(np_array_t& x, const size_t d, Lambda f){
	size_t cc = 0; // current column 
	const size_t n_svds = x.shape()[1] / d;
	const size_t nc = d; 
	switch(rosenburg_pair(x.shape()[0], d)){
		case 0: { // (0 x 0)
			np_array_t u = np_matrix(0, 0);
			np_array_t v = np_matrix(0, 0);
			np_array_t s = np_array_t();
			for (size_t i = 0; i < n_svds; ++i, cc += nc){ f(u,s,v); }
			break;
		}
		case 1: { // (0 x 1)
			np_array_t u = np_matrix(0, 0);
			np_array_t v = np_matrix(1, 1);
			np_array_t s = np_array_t();
			v.mutable_data()[0] = 1;
			for (size_t i = 0; i < n_svds; ++i, cc += nc){ f(u,s,v); }
			break; 
		}
		case 3:  { // (1 x 0) {
			np_array_t u = np_matrix(1, 1);
			np_array_t v = np_matrix(0, 0);
			np_array_t s = np_array_t();
			u.mutable_data()[0] = 1;
			for (size_t i = 0; i < n_svds; ++i, cc += nc){ f(u,s,v); }
			break; 
		}
		case 2: { // (1 x 1)
			np_array_t u = np_matrix(1, 1);
			np_array_t v = np_matrix(0, 0);
			np_array_t s(1, &x.data()[0]);
			u.mutable_data()[0] = 1;
			v.mutable_data()[0] = 1;
			for (size_t i = 0; i < n_svds; ++i, cc += nc){ f(u,s,v); }
			break;
		}
		case 4: { // (0 x 2)
			np_array_t u = np_matrix(0, 0);
			np_array_t v({ 2, 2 });
			np_array_t s = np_array_t();
			v.mutable_data()[0] = 1;
			v.mutable_data()[1] = 0; 
			v.mutable_data()[2] = 0;
			v.mutable_data()[3] = 1; 
			for (size_t i = 0; i < n_svds; ++i, cc += nc){ f(u,s,v); }
			break;
		}
		case 8: { // (2 x 0)
			np_array_t u({ 2, 2 });
			np_array_t v = np_matrix(0, 0);
			np_array_t s = np_array_t();
			u.mutable_data()[0] = 1;
			u.mutable_data()[1] = 0; 
			u.mutable_data()[2] = 0;
			u.mutable_data()[3] = 1; 
			for (size_t i = 0; i < n_svds; ++i, cc += nc){ f(u,s,v); }
			break;
		}
		case 9: { // (0 x 3)
			np_array_t u = np_matrix(0, 0);
			np_array_t v = np_matrix(3, 3);
			np_array_t s = np_array_t();
			v.mutable_data()[0] = 1;
			v.mutable_data()[1] = 0; 
			v.mutable_data()[2] = 0;
			v.mutable_data()[3] = 0; 
			v.mutable_data()[4] = 1; 
			v.mutable_data()[5] = 0; 
			v.mutable_data()[6] = 0; 
			v.mutable_data()[7] = 0; 
			v.mutable_data()[8] = 1; 
			for (size_t i = 0; i < n_svds; ++i, cc += nc){ f(u,s,v); }
			break;
		}
		case 15: { // (3 x 0)
			np_array_t u = np_matrix(3, 3);
			np_array_t v = np_matrix(0,0);
			np_array_t s = np_array_t();
			u.mutable_data()[0] = 1;
			u.mutable_data()[1] = 0; 
			u.mutable_data()[2] = 0;
			u.mutable_data()[3] = 0; 
			u.mutable_data()[4] = 1; 
			u.mutable_data()[5] = 0; 
			u.mutable_data()[6] = 0; 
			u.mutable_data()[7] = 0; 
			u.mutable_data()[8] = 1; 
			for (size_t i = 0; i < n_svds; ++i, cc += nc){ f(u,s,v); }
			break;
		}
		case 5: 
			svd2_stream< true, 1, 2 >(x, f);
			break;
		case 6: 
			svd2_stream< true, 2, 2 >(x, f);
			break;
		case 7:
			svd2_stream< true, 2, 1 >(x, f);
			break;
		case 10: 
			svd3_stream< true, 1, 3 >(x, f);
			break;
		case 11: 
			svd3_stream< true, 2, 3 >(x, f);
			break;
		case 12: 
			svd3_stream< true, 3, 3 >(x, f);
			break;
		case 13: 
			svd3_stream< true, 3, 2 >(x, f);
			break;
		case 14: 
			svd3_stream< true, 3, 1 >(x, f);
			break;
		default: 
			svdn_stream< true >(x, d, f);
			break;
	}
}

// Fast SVD testing  
std::list< np_array_t > fast_svd(np_array_t& x) {
	if ( x.ndim()     != 2){ throw std::runtime_error("Input should be 2-D NumPy array"); }
	auto out = std::list< np_array_t >();
	const auto record_f = [&out](np_array_t U, np_array_t S, np_array_t V){
		out.push_back(U); out.push_back(S); out.push_back(V);
	};
	fast_svd_stream(x, x.shape()[1], record_f);
	return(out);
}
