// cppimport
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cmath> // fabs
#include "svd_3x3.h" // fast 3x3 svd

using namespace pybind11::literals;
namespace py = pybind11;

// Compile command: g++ -O3 -Wall -shared -std=c++17 -fPIC -Wl,-undefined,dynamic_lookup $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)

// Require column-major layout (AoS - Fortran style)
using np_array_t = py::array_t< float, py::array::f_style | py::array::forcecast >;

// Evaluations a cost function quickly 
struct BetaNuclearDense {
	const size_t d; // intrinsic dimension
	const size_t n; // number of points 
	const size_t D; // target dimension of coordinatization
	np_array_t output; // preallocated output (D x d*n)
	BetaNuclearDense(int n_points, int dim, int target_dim) : d(dim), n(n_points), D(target_dim) {
		output = np_array_t({ D, d*n });
	}

	double numpy_svd(){
		auto svd = py::module::import("numpy.linalg").attr("svd");
		py::buffer_info output_buffer = output.request();
		const size_t inc = D*d;
		double nuclear_norm = 0.0;
		for (int j = 0; j < n; ++j){
			np_array_t inp = np_array_t({ D, d }, output.data()+(j*inc));
			np_array_t sv = svd(inp, false, false, false);
			auto r = sv.unchecked< 1 >();
			for (size_t i = 0; i < sv.shape(0); i++){ 
				nuclear_norm += r(i); 
			}
		}
		return(nuclear_norm);
	}

	double three_svd(){
		double nuclear_norm = 0.0;
		svd_3x3_stream< false >(output, [&nuclear_norm](np_array_t& S){
			nuclear_norm += std::fabs(S.data()[0]);
			nuclear_norm += std::fabs(S.data()[1]);
			nuclear_norm += std::fabs(S.data()[2]);	
		});
		return(nuclear_norm);
	}
};

PYBIND11_MODULE(example, m) {
	m.def("svd_3x3", &svd_3x3, "Yields the svd of a 3x3 matrix");
	py::class_<BetaNuclearDense>(m, "BetaNuclearDense")
		.def(py::init< int, int, int >())
		.def_readwrite("output", &BetaNuclearDense::output)
		.def("numpy_svd", &BetaNuclearDense::numpy_svd)
		.def("three_svd", &BetaNuclearDense::three_svd);
}

/* 
<% 
cfg['extra_compile_args'] = ['-std=c++17']
setup_pybind11(cfg) 
%> 
*/