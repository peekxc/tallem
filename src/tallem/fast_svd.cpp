
#include <carma>
#include <armadillo>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cmath> // fabs
#include "svd_3x3.h" // fast 3x3 svd
#include "carma_svd.h" // fast 3x3 svd using carma

#include <chrono>
using namespace std::chrono;

using namespace pybind11::literals;
namespace py = pybind11;

// Compile command: g++ -O3 -Wall -shared -std=c++17 -fPIC -Wl,-undefined,dynamic_lookup $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)

constexpr auto rank_comb2(size_t i, size_t j, size_t n) noexcept -> size_t { 
  if (j < i){ std::swap(i,j); }
  return(size_t(n*i - i*(i+1)/2 + j - i - 1));
}

inline std::array< size_t, 2 > unrank_comb2(const size_t x, const size_t n) noexcept {
	auto i = static_cast< size_t >( (n - 2 - floor(sqrt(-8*x + 4*n*(n-1)-7)/2.0 - 0.5)) );
	auto j = static_cast< size_t >( x + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 );
	return (std::array< size_t, 2 >{ i, j });
}

// Require column-major layout (AoS - Fortran style)
using np_array_t = py::array_t< float, py::array::f_style | py::array::forcecast >;

// Evaluations a cost function quickly 
struct StiefelLoss {
	using index_mat_map = std::unordered_map< size_t, arma::mat >;
	const size_t n; // number of points
	const size_t d; // intrinsic dimension 
	const size_t D; // target dimension of coordinatization
	index_mat_map rotations; // stores the rotation matrices (Omega)
	
	// TODO: test sparse implementation
	arma::mat frames; // all column-stacked frames (dJ x dn) for some choice of iota, weighted by PoU
	arma::sp_mat frames_sparse; 

	np_array_t output; // preallocated output for (A^T x Phi) => (D x dn)


	StiefelLoss(int n_points, int dim, int target_dim) : n(n_points), d(dim), D(target_dim) {
		output = np_array_t({ D, d*n });
	}

	// Assuming output contains the result of (A^* x Phi)
	auto gradient(const py::array_t< double >& At, bool normalize=false) -> std::list< np_array_t > {
		
		// First project the alignment to a D-dimensional space
		arma::Mat< double > A_star { carma::arr_to_mat< double >(At) };
		output = carma::mat_to_arr< double >(A_star * frames);
		
		// Calculate all the SVDs
		const size_t J = frames.n_rows / d;
		auto nuclear_norm = (float) 0.0;
		auto G = arma::mat(D, d*n);
		auto i = size_t(0);
		arma::mat S = arma::zeros(D, d);
		fast_svd_stream(output, d, [this, &i, &G, &S, &nuclear_norm](np_array_t& u, np_array_t& s, np_array_t& vt){
			auto u_copy = py::array_t< double >(u);
			auto v_copy = py::array_t< double >(vt);
			auto s_copy = py::array_t< double >(s);
			arma::mat U { carma::arr_to_mat< double >(u_copy, true) };
			S.diag() = carma::arr_to_col< double >(s_copy, true);
			arma::mat V { carma::arr_to_mat< double >(v_copy, true) };
			// py::print(U.n_rows, U.n_cols, S.n_rows, S.n_cols, V.n_rows, V.n_cols);
			G(arma::span::all, arma::span(i, i+d-1)) = U * S * V;
			nuclear_norm += arma::trace(S);
			i += d;
		});
		// Calculate gradient
		arma::mat GF = frames * G.t(); // (dJ x dn)*(dn x D) => (dJ x D)
		if (normalize){
			GF /= n;
			nuclear_norm /= n;
		}
		auto out = std::list< np_array_t >();
		out.push_back(carma::mat_to_arr(arma::Mat< float >(&nuclear_norm, 1, 1)));
		out.push_back(carma::mat_to_arr(GF));
		return(out);
	}

	// auto gradient(const py::array_t< double >& At) -> std::list< np_array_t >{
	// 	arma::Mat< double > A_star { carma::arr_to_mat< double >(At) };
	// 	output = carma::mat_to_arr< double >(A_star * frames);
	// 	return(gradient());
	// }

	// Assuming output contains the result of (A^* x Phi)
	auto benchmark_gradient(const np_array_t& frames) -> std::list< np_array_t > {
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

		auto start1 = high_resolution_clock::now();
		fast_svd_stream(output, d, [this, &i, &G, &nuclear_norm](np_array_t& u, np_array_t& s, np_array_t& vt){
		// 	py::buffer_info info = u.request();
    // 	float* data = carma::details::steal_copy_array<float>(u.ptr());
		// 	arma::Mat< float > U { carma::details::arr_to_mat(info, data, true, false) };
    // // return ;
		// 	// auto u_copy = py::array_t< float >(u);
		// 	auto v_copy = py::array_t< float >(vt);
		// 	auto s_copy = py::array_t< float >(s);
		// 	// arma::Mat< float > U { carma::arr_to_mat< float >(u, false) };
		// 	arma::Mat< float > S { arma::diagmat(carma::arr_to_col< float >(s_copy)) };
		// 	arma::Mat< float > V { carma::arr_to_mat< float >(v_copy) };
		// 	G(arma::span::all, arma::span(i, i+d-1)) = U * S * V;
		// 	nuclear_norm -= arma::trace(S);
			i += d;
		});
		auto stop1 = high_resolution_clock::now();
		auto duration1 = duration_cast< milliseconds >(stop1 - start1);
		py::print("SVD1 ms: ", duration1.count());

		i = 0; 
		auto start = high_resolution_clock::now();
		fast_svd_stream(output, d, [this, &i, &G, &nuclear_norm](np_array_t& u, np_array_t& s, np_array_t& vt){
			py::buffer_info info = u.request();
    	float* data = carma::details::steal_copy_array<float>(u.ptr());
			arma::Mat< float > U { carma::details::arr_to_mat(info, data, true, false) };
    // return ;
			// auto u_copy = py::array_t< float >(u);
			auto v_copy = py::array_t< float >(vt);
			auto s_copy = py::array_t< float >(s);
			// arma::Mat< float > U { carma::arr_to_mat< float >(u, false) };
			arma::Mat< float > S { arma::diagmat(carma::arr_to_col< float >(s_copy, true)) };
			arma::Mat< float > V { carma::arr_to_mat< float >(v_copy, true) };
			G(arma::span::all, arma::span(i, i+d-1)) = U * S * V;
			nuclear_norm -= arma::trace(S);
			i += d;
		});
		auto stop = high_resolution_clock::now();
		
		auto duration = duration_cast< milliseconds >(stop - start);
		py::print("SVD ms: ", duration.count());

		// Multiply all the frames by the adjusted subgradients 
		start = high_resolution_clock::now();
		const arma::Mat< float > Phi = carma::arr_to_mat< float >(frames);
		arma::Mat< float > GF = -(Phi * G.t()); // (dJ x dn)*(dn x D) => (dJ x D)
		stop = high_resolution_clock::now();
		duration = duration_cast< milliseconds >(stop - start);
		py::print("Gradient ms: ", duration.count());

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

	double three_svd_carma(){
		double nuclear_norm = 0.0;
		svd_3x3_carma< false >(output, [&nuclear_norm](arma::fmat::fixed< 3, 1 >& S){
			nuclear_norm += std::fabs(S[0]) + std::fabs(S[1]) + std::fabs(S[2]);
		});
		return(nuclear_norm);
	}

	
	// const size_t jb = J*(J-1)/2;
	
	// Populate the O(J^2)-sized hashmap mapping index pairs i,j \in J -> rotation matrices
	void init_rotations(py::list I_ind, py::list J_ind, py::array_t< double > omega_, const size_t J){
		std::vector< size_t > I1 = py::cast< std::vector< size_t > >(I_ind);
		std::vector< size_t > I2 = py::cast< std::vector< size_t > >(J_ind);
		// np_array_t O = static_cast< np_array_t >(omega_);
		arma::mat omega = carma::arr_to_mat< double >(omega_, true);
		for (size_t j = 0; j < I1.size(); ++j){
			size_t ii = I1[j], jj = I2[j];
			size_t key = rank_comb2(ii,jj,J);
			arma::mat R = omega(arma::span(j*d, (j+1)*d-1), arma::span::all);
			rotations.emplace(key, R);
		}
	}

	auto get_rotation(const size_t i, const size_t j, const size_t J) -> py::array_t< double > {
		size_t key = rank_comb2(i,j,J);
		if (rotations.find(key) != rotations.end()){
			return(carma::mat_to_arr(rotations[key], true));
		}
		throw std::invalid_argument("Invalid key given");
	}
	
	// Generates a weighted (dJ x d) frame relative to some origin subset 
	// This is equivalent to Phi_{origin}(x) where 'weights' are specific to 'x'
	auto generate_frame(const size_t origin, py::array_t< double > weights) -> py::array_t< double > {
		const size_t J = weights.size();
		auto w = weights.unchecked< 1 >();		
		arma::mat d_frame(d*J, d); // output 
		arma::mat I = arma::eye(d, d);
		for (size_t j = 0; j < J; ++j){
			auto r_rng = arma::span(j*d,(j+1)*d-1); 
			double w_j = static_cast< double >(w(j));
			if (j == origin || w_j == 0.0){ 
				d_frame(r_rng, arma::span::all) = w_j * I; 
				continue;
			} else {
				// If pair exists, load it up, otherwise use identity
				const size_t key = rank_comb2(origin, j, J); 
				bool key_exists = rotations.find(key) != rotations.end();
				d_frame(r_rng, arma::span::all) = double(w(j))*(key_exists ? (origin < j ? rotations[key] : rotations[key].t()) : I);
			}
		}
		return(carma::mat_to_arr< double >(d_frame, true));
	}

	// Using the rotations from the omega map, initialize the phi matrix representing the concatenation 
	// of the weighted frames for some choice of iota 
	// py::array_t< double > iota, bool sparse = false
	void populate_frame(const size_t i, py::array_t< double > weights, bool sparse = false){
		// if (iota.size() != n || weights.size() != ){
		// 	throw std::invalid_argument("Invalid input. Must have one weight for each cover element.")
		// }
		const size_t J = weights.size();
		auto w = weights.unchecked< 1 >();
		
		// Find the reference frame
		size_t k = 0; 
		double max_weight = 0.0; 
		for (size_t j = 0; j < J; ++j){
			if (w(j) > max_weight){
				max_weight = w(j);
				k = j; 
			}
		}

		if (sparse && frames_sparse.is_empty()){
			frames_sparse.resize(d*J, d*n);
		} else if (!sparse && frames.is_empty()){
			frames.resize(d*J, d*n);
		}	 

		py::array_t< double > _d_frame = generate_frame(k, weights);
		arma::mat d_frame = carma::arr_to_mat(_d_frame, true);

		// arma::mat d_frame(d*J, d); // output 
		// arma::mat I = arma::eye(d, d);
		// for (size_t j = 0; j < J; ++j){
		// 	auto r_rng = arma::span(j*d,(j+1)*d-1); 
		// 	if (j == k){ d_frame(r_rng, arma::span::all) = I; }
			
		// 	// If pair exists, load it up, otherwise use identity
		// 	size_t key = rank_comb2(k, j, J); 
		// 	bool key_exists = rotations.find(key) != rotations.end();
		// 	d_frame(r_rng, arma::span::all) = double(w(j))*(key_exists ? (k < j ? rotations[key] : rotations[key].t()) : I);
		// }

		// Assign the frame to right position in the frames matrix
		if (sparse){
			frames_sparse(arma::span::all, arma::span(i*d, (i+1)*d-1)) = d_frame; 
		} else {
			frames(arma::span::all, arma::span(i*d, (i+1)*d-1)) = d_frame; 
		}
	}


	// Returns the i'th frame of the matrix
	auto get_frame(const size_t i) -> py::array_t< double > {
		if (i >= n){ throw std::invalid_argument("Invalid index supplied."); }
		arma::mat d_frame = frames(arma::span::all, arma::span(i*d, (i+1)*d-1));
		py::array_t< double > res = carma::mat_to_arr< double >(d_frame, true);
		return(res);
	}

	// Returns all the frames 
	auto all_frames() -> py::array_t< double > {
		py::array_t< double > res = carma::mat_to_arr< double >(frames, true);
		return(res);
	}

	void embed(const py::array_t< double >& At){
		arma::Mat< double > A_star { carma::arr_to_mat< double >(At) };
		output = carma::mat_to_arr< double >(A_star * frames);
	}

	void benchmark_embedding(py::array_t< double >& At, const size_t m){
		if (frames_sparse.is_empty()){ frames_sparse = arma::sp_mat(frames); }
		arma::Mat< double > A_star { carma::arr_to_mat< double >(At, true) };
		
		size_t ms_dense = 0, ms_sparse = 0;
		
		auto start = high_resolution_clock::now();
		for (size_t i = 0; i < m; ++i){
			output = carma::mat_to_arr< double >(A_star * frames);
		}
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast< milliseconds >(stop - start);
		ms_dense = duration.count(); 

		start = high_resolution_clock::now();
		for (size_t i = 0; i < m; ++i){
			output = carma::mat_to_arr< double >(A_star * frames_sparse);
		}
		stop = high_resolution_clock::now();
		duration = duration_cast< milliseconds >(stop - start);
		ms_sparse = duration.count(); 

		py::print("ms dense: ", ms_dense, "ms sparse: ", ms_sparse);
	}

};

// to_sparse(, py::array_t<int> & locations){}


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
		// .def_readwrite("rotations", &StiefelLoss::rotations)
		.def("numpy_svd", &StiefelLoss::numpy_svd)
		//.def("benchmark_gradient", &StiefelLoss::benchmark_gradient)
		.def("gradient", &StiefelLoss::gradient)
		.def("three_svd", &StiefelLoss::three_svd)
		.def("three_svd_carma", &StiefelLoss::three_svd_carma)
		.def("init_rotations", &StiefelLoss::init_rotations)
		.def("get_rotation", &StiefelLoss::get_rotation)		
		.def("populate_frame", &StiefelLoss::populate_frame)
		.def("generate_frame", &StiefelLoss::generate_frame)
		.def("get_frame", &StiefelLoss::get_frame)
		.def("all_frames", &StiefelLoss::all_frames)
		.def("embed", &StiefelLoss::embed)
		.def("benchmark_embedding", &StiefelLoss::benchmark_embedding)
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