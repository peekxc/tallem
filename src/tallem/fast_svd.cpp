
#include <carma>
#include <armadillo>

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
struct StiefelLoss {
	const size_t n; // number of points
	const size_t d; // intrinsic dimension 
	const size_t D; // target dimension of coordinatization
	np_array_t output; // preallocated output (D x d*n)

	StiefelLoss(int n_points, int dim, int target_dim) : n(n_points), d(dim), D(target_dim) {
		output = np_array_t({ D, d*n });
	}

	// Assuming output contains the result of (A^* x Phi)
	auto loss_gradient(const np_array_t& frames) -> std::list< np_array_t > {
		if (frames.shape()[0] % d != 0 || frames.shape()[1] != d*n){
			throw std::invalid_argument(
				"Frames must be a ( dJ x dn ) matrix. (received a " + std::to_string(frames.shape()[0]) +
				" x " + std::to_string(frames.shape()[1]) + " matrix)"
			);
		}
		const size_t J = frames.shape()[0] / d;
		auto nuclear_norm = (float) 0.0;
		auto G = arma::Mat< float >(D, d*n);
		auto i = size_t(0);
		fast_svd_stream(output, d, [this, &i, &G, &nuclear_norm](np_array_t& u, np_array_t& s, np_array_t& vt){
			py::buffer_info info = u.request();
    	float* data = carma::details::steal_copy_array<float>(u.ptr());
			arma::Mat< float > U { carma::details::arr_to_mat(info, data, true, false) };
    // return ;
			// auto u_copy = py::array_t< float >(u);
			auto v_copy = py::array_t< float >(vt);
			auto s_copy = py::array_t< float >(s);
			// arma::Mat< float > U { carma::arr_to_mat< float >(u, false) };
			arma::Mat< float > S { arma::diagmat(carma::arr_to_col< float >(s_copy)) };
			arma::Mat< float > V { carma::arr_to_mat< float >(v_copy) };
			G(arma::span::all, arma::span(i, i+d-1)) = U * S * V;
			nuclear_norm -= arma::trace(S);
			i += d;
		});
		
		// Multiply all the frames by the adjusted subgradients 
		const arma::Mat< float > Phi = carma::arr_to_mat< float >(frames);
		arma::Mat< float > GF = -(Phi * G.t()); // (dJ x dn)*(dn x D) => (dJ x D)
		// arma::Mat< float > GF = arma::zeros(d*J, D);
		auto out = std::list< np_array_t >();
		out.push_back(carma::mat_to_arr(arma::Mat< float >(&nuclear_norm, 1, 1)));
		out.push_back(carma::mat_to_arr(GF));
		return(out);
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
		svd3_stream< false, 3, 3 >(output, [&nuclear_norm](np_array_t& S){
			nuclear_norm += std::fabs(S.data()[0]);
			nuclear_norm += std::fabs(S.data()[1]);
			nuclear_norm += std::fabs(S.data()[2]);	
		});
		return(nuclear_norm);
	}
};

PYBIND11_MODULE(fast_svd, m) {
	//m.def("svd_2", &svd_2, "Yields the svd of a dimension 2 matrix");
	//m.def("svd_3", &svd_3, "Yields the svd of a dimension 3 matrix");
	m.def("fast_svd", &fast_svd, "Yields the svd of a matrix of low dimension");
	m.def("lapack_svd", &lapack_svd, "Yields the svd of a matrix of low dimension");
	py::class_<StiefelLoss>(m, "StiefelLoss")
		.def(py::init< int, int, int >())
		.def_readonly("d", &StiefelLoss::d)
		.def_readonly("n", &StiefelLoss::n)
		.def_readonly("D", &StiefelLoss::D)
		.def_readwrite("output", &StiefelLoss::output)
		.def("numpy_svd", &StiefelLoss::numpy_svd)
		.def("gradient", &StiefelLoss::loss_gradient)
		.def("three_svd", &StiefelLoss::three_svd)
		.def("__repr__",[](const StiefelLoss &stf) {
			return("Stiefel Loss w/ parameters n="+std::to_string(stf.n)+",d="+std::to_string(stf.d)+",D="+std::to_string(stf.D));
  	});
}

/* 
<% 
cfg['extra_compile_args'] = ['-std=c++17']
setup_pybind11(cfg) 
%> 
*/