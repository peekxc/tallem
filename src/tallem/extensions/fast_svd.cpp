
#include <carma>
#include <armadillo>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cmath> // fabs
#include <chrono>
#include <limits>
#include "svd_3x3.h" // fast 3x3 svd
#include "carma_svd.h" // fast 3x3 svd using carma


using namespace std::chrono;
using std::vector; 

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
	
	arma::mat frames; // all column-stacked frames (dJ x dn) for some choice of iota, weighted by PoU
	arma::sp_mat frames_sparse; // same as frames, but as sp_mat. Only one should be used

	arma::sp_mat pou; // (J x n) partition of unity
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
		// const size_t J = frames.n_rows / d;
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
			G(arma::span::all, arma::span(i, i+d-1)) = U * S * V; // TODO: transpose?
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

	// Populate the O(J^2)-sized hashmap mapping index pairs i,j \in J -> rotation matrices
	// omega_ is expected to contain the vertically-stacked d-d rotation matrices corresponding to each (i,j) pair
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
		arma::mat d_frame(d*J, d, arma::fill::zeros); // output 
		vector< double > weights_vec = weights.cast< vector< double > >();
		generate_frame_(origin, weights_vec, d_frame);
		return(carma::mat_to_arr< double >(d_frame, true));
	}

	// This generates a given dense (dJ x d) frame with the weighted rotation matrices given in the 'rotations' table
	// Note: this applies sqrt to the phi weights! 
	// TODO: should the weights be sqrt'ed on identity matrices? Assume yes.
	// TODO: consider filling a (d x dJ) matrix instead to be more cache-friendly
	void generate_frame_(const size_t origin, const vector< double >& weights, arma::mat& d_frame) {
		const size_t J = weights.size();	
		arma::mat I = arma::eye(d, d);

		// This fills up the (dJ x d) frame one (d x d) matrix at a time 
		for (size_t j = 0; j < J; ++j){
			auto r_rng = arma::span(j*d,(j+1)*d-1); 
			if (j == origin){ 
				d_frame(r_rng, arma::span::all) = std::sqrt(weights[j]) * I; 
				continue;
			} else if (weights[j] == 0.0){
				d_frame(r_rng, arma::span::all) = std::sqrt(weights[j]) * I; 
				continue; 
			} else {
				// If pair exists, load it up, otherwise use identity
				const size_t key = rank_comb2(origin, j, J); 
				bool key_exists = rotations.find(key) != rotations.end();
				d_frame(r_rng, arma::span::all) = std::sqrt(weights[j])*(key_exists ? (origin < j ? rotations[key] : rotations[key].t()) : I);
			}
		}
	}

	// This generates a given dense (d x dJ) frame with the weighted rotation matrices given in the 'rotations' table
	// Note: this applies sqrt to the phi weights! 
	void generate_frame_T(const size_t origin, const vector< double >& weights, arma::mat& d_frame) {
		const size_t J = weights.size();	
		arma::mat I = arma::eye(d, d);

		// This fills up the (d x dJ) frame one (d x d) matrix at a time (column-wise)
		for (size_t j = 0; j < J; ++j){
			auto c_rng = arma::span(j*d,(j+1)*d-1); 
			auto weight = std::sqrt(weights[j]);
			if (j == origin || weights[j] == 0.0){ 
				d_frame(arma::span::all, c_rng) = std::sqrt(weights[j]) * I; 
				continue;
			} else {
				// If pair exists, load it up, otherwise use identity
				const size_t key = rank_comb2(origin, j, J); 
				bool key_exists = rotations.find(key) != rotations.end();
				d_frame(arma::span::all, c_rng) = weight *  (key_exists ? (origin < j ? rotations[key] : rotations[key].t()) : I);
			}
		}
	}

	// TODO: come back to this to fix the populate_frames functions efficiency
	template< typename OutputIt1, typename OutputIt2 >
	inline void generate_frame_ijx(const size_t origin, const size_t j, const size_t J, const double weight, const size_t col_offset, OutputIt1 RC, OutputIt2 X){
		
		// For each cover set, detect whether there exists rotation matrices to transform between them
		if (j == origin){ 
			const size_t row_offset = (j*d);
			auto w = std::sqrt(weight);
			for (size_t ri = 0; ri < d; ++ri){
				// py::print("row: ", row_offset + ri, ", col: ", col_offset + ri, ", val: ", weight);
				*RC++ = row_offset + ri; // row index of non-zero element
				*RC++ = col_offset + ri; // column index of non-zero element
				*X++ = w;
			}
		} else { // j != origin and weights[j] > 0
			const size_t row_offset = (j*d), key = rank_comb2(origin, j, J); 
			auto w = std::sqrt(weight);
			
			// If key exists, there's a non-empty intersection between the cover sets (i,j)
			// otherwise, just use the (weighted) identity matrix
			bool key_exists = rotations.find(key) != rotations.end();
			if (!key_exists){
				for (size_t ri = 0; ri < d; ++ri){
					// py::print("row: ", row_offset + ri, ", col: ", col_offset + ri, ", val: ", weight);
					*RC++ = row_offset + ri; // row index of non-zero element
					*RC++ = col_offset + ri; // column index of non-zero element
					*X++ = w;
				}
			} else {
				const arma::mat& omega = origin < j ? rotations[key] : rotations[key].t();
				arma::umat nz_ind = arma::ind2sub(arma::size(omega), arma::find(omega != 0.0));
				nz_ind.each_col([&](arma::uvec& a){ 
					// py::print("| row: ", row_offset + a[0], ", col: ", col_offset + a[1], ", val: ", omega(a[0], a[1]));
					*RC++ = row_offset + a[0];
					*RC++ = col_offset + a[1];
					*X++ = w*omega(a[0], a[1]);
				});  
			}
		}
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

	void setup_pou(py::object& P_csc){
		to_sparse(P_csc, pou);
	}
		
	auto extract_iota(){
		if (pou.is_empty()){ throw std::invalid_argument("Partition of unity matrix not populated."); }
		// weights.assign(J, 0.0);
		// auto ci = pou_.begin_col(i);
		// for (; ci != pou_.end_col(i); ++ci){ weights[ci.row()] = *ci; }
		// size_t iota_i = std::distance(weights.begin(), std::max_element(weights.begin(), weights.end()));
		arma::urowvec u = arma::index_max(pou,0);
		return(carma::row_to_arr< unsigned long long >(std::move(u)));
	}

	// Populate the 'frames_sparse' arma::sp_mat member variable using the iota bijection
	// Postcondition: self.frames_sparse is a sparse (dJ x dn) matrix representing the horizontal concatenation of all the Phi's for each x \in X
	void populate_frames_sparse(py::array_t< arma::uword >& iota){
		if (pou.n_cols != n){ throw std::invalid_argument("Invalid input. Must have one weight for each cover element."); }
		const size_t J = pou.n_rows;
		arma::uvec I = carma::arr_to_col(iota, true);

		// Prepare the vectors needed to construct the sparse matrix using COO-input 
		vector< arma::uword > RC; 
		vector< double > X;
		RC.reserve(2*d*d*n);
		X.reserve(d*d*n);

		// Output iterators
		auto rc_out = std::back_inserter(RC);
		auto x_out = std::back_inserter(X);

		// Generate all the frames using iota
		for (size_t i = 0; i < n; ++i){
			
			// Iterate through sparse columns
			auto ci = pou.begin_col(i);
			for (; ci != pou.end_col(i); ++ci){ 
				generate_frame_ijx(I[i], ci.row(), J, *ci, i*d, rc_out, x_out);
			}
		}
		
		// Assign to frames_sparse
		auto locations = arma::umat(RC.data(), 2, RC.size()/2, false, true);
		frames_sparse = arma::sp_mat(std::move(locations), arma::vec(std::move(X)), d*J, d*n);
	} // populate_frames_sparse

	// Using the rotations from the omega map, initialize the phi matrix representing the concatenation 
	// of the weighted frames for some *fixed* choice of iota 
	// iota := n-length vector of indices each in [0, J) indicating the most similar cover set
	// pou := (J x n) sparse csc_matrix representing the partition of unity
	// Note the transpose! 
	// TODO: change this to construct the sparse matrix *not* using block assignments
	void populate_frames(const py::array_t< size_t >& iota, py::object& pou, bool sparse = false){
		if (iota.size() != size_t(n)){
			throw std::invalid_argument("Invalid input. Must have one weight for each cover element.");
		}

		// Convert partition of unity to arma
		arma::sp_mat pou_;
		to_sparse(pou, pou_);
		const size_t J = pou_.n_rows;

		if (sparse && frames_sparse.is_empty()){
			frames_sparse.resize(d*J, d*n);
		} else if (!sparse && frames.is_empty()){
			frames.resize(d*J, d*n);
		}	 

		// Generate all the frames using iota
		vector< double > weights(J, 0.0);
		arma::mat d_frame(d*J, d, arma::fill::zeros);
		for (size_t i = 0; i < n; ++i){
			
			// Fill the weight vector
			weights.assign(J, 0.0);
			auto ci = pou_.begin_col(i);
			for (; ci != pou_.end_col(i); ++ci){ weights[ci.row()] = *ci; }

			// Generate the current frame using iota to specify the origin 
			generate_frame_(iota.at(i), weights, d_frame);

			// Assign the frame to right position in the frames matrix
			if (sparse){
				frames_sparse(arma::span::all, arma::span(i*d, (i+1)*d-1)) = d_frame; 
			} else {
				frames(arma::span::all, arma::span(i*d, (i+1)*d-1)) = d_frame; 
			}
		}
		
		// If sparse, make sure to clean up 
		if (sparse){ frames_sparse.clean(std::numeric_limits< double >::epsilon()); }
	} // populate_frames

	py::tuple initial_guess(const size_t D, bool sparse=true){
		if (sparse){
			if (frames_sparse.empty()) { throw std::invalid_argument("Frames sparse matrix has not been populated."); } 
			arma::vec eigval;
			arma::mat eigvec;
			arma::eigs_sym(eigval, eigvec, frames_sparse * frames_sparse.t(), D, "lm"); // largest first
			eigval = arma::reverse(eigval);
			eigvec = arma::fliplr(eigvec);
			return py::make_tuple(carma::col_to_arr(eigval), carma::mat_to_arr(eigvec));
		} else {
			if (frames.empty()) { throw std::invalid_argument("Frames matrix has not been populated."); } 
			arma::vec eigval;
			arma::mat eigvec;
			arma::eig_sym(eigval, eigvec, frames * frames.t()); // largest first
			eigval = arma::reverse(eigval);
			eigvec = arma::fliplr(eigvec);
			return py::make_tuple(carma::col_to_arr(eigval), carma::mat_to_arr(eigvec));
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

	auto all_frames_sparse() -> py::tuple {
		// py::tuple shape = S.attr("shape").cast< py::tuple >();
		// const size_t nr = shape[0].cast< size_t >(), nc = shape[1].cast< size_t >();
		// S.attr("indices") = py::array_t< arma::uword >();
		// S.attr("indptr") = py::array_t< arma::uword >()
		// arma::uvec ind_ptr = carma::arr_to_col(.cast< py::array_t< arma::uword > >());
		// arma::vec data = carma::arr_to_col(S.attr("data").cast< py::array_t< double > >());
		// out = arma::sp_mat(std::move(ind), std::move(ind_ptr), std::move(data), nr, nc);

		// py::array_t< double > res = carma::mat_to_arr< double >(frames, true);
		// return(res);
		frames_sparse.sync();
		//const ptr_aux_mem, number_of_elements
		auto ri = carma::col_to_arr(arma::uvec(frames_sparse.row_indices, frames_sparse.n_nonzero));
		auto cp = carma::col_to_arr(arma::uvec(frames_sparse.col_ptrs, frames_sparse.n_cols+1));
		auto x = carma::col_to_arr(arma::vec(frames_sparse.values, frames_sparse.n_nonzero));
		// auto ri = carma::col_to_arr< const arma::uword >(*frames_sparse.row_indices, true);
		// auto out = py::dict(
		// 	"indices"= py::array(ri), 
		// 	"indptr"=cp, "values"=x);
		return(py::make_tuple(ri, cp, x));
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


	using index_list = vector< vector< size_t > >;
	using vec_mats = vector< arma::mat >;

	// Given sorted range [b,e), finds 'element' in log time, or throws an exception if not found
	template< typename Iter, typename T = typename Iter::value_type >
	auto find_index(Iter b, const Iter e, T element) -> size_t {
		auto lb = std::lower_bound(b, e, element);
		if (lb != e && (*lb == element)){
			return(std::distance(b, lb));
		}
		throw std::logic_error("Unable to find element in array.");
	}
	void to_sparse(const py::object& S, arma::sp_mat& out){
		py::tuple shape = S.attr("shape").cast< py::tuple >();
		const size_t nr = shape[0].cast< size_t >(), nc = shape[1].cast< size_t >();
		arma::uvec ind = carma::arr_to_col(S.attr("indices").cast< py::array_t< arma::uword > >());
		arma::uvec ind_ptr = carma::arr_to_col(S.attr("indptr").cast< py::array_t< arma::uword > >());
		arma::vec data = carma::arr_to_col(S.attr("data").cast< py::array_t< double > >());
		out = arma::sp_mat(std::move(ind), std::move(ind_ptr), std::move(data), nr, nc);
	}


	// A := (dJ x D) dense orthonormal matrix 
	// pou := (J x n) sparse matrix representing the transpose of the PoU
	// cover_subsets := (J)-length vector of sorted cover sets 
	// local_models := (J)-length vector of column-oriented euclidean coordinate models (point for each column)
	// T := (D x n) matrix of translation vectors
	// Note: with armadillo's csc impmenentations, (dense x sparse) is faster than (sparse x dense)
	// Output => (D x n) matrix of the assembled coordinates
	void fast_assembly2(const arma::mat& A, const arma::sp_mat& pou, const index_list& cover_subsets, const vec_mats& local_models, const arma::mat& T, arma::mat& assembly){
		// arma::mat assembly = arma::zeros(D, n);
		arma::vec coords = arma::zeros(D);
		const size_t J = pou.n_rows;
		
		// Variables to re-use/cache in the loop 
		vector< double > phi_i(J); // the partition of unity weights for x_i 
		arma::mat d_frame(d*J, d); // the current frame to populate 
		arma::mat U, V;
		arma::vec s;

		// Build the assembly
		for (size_t i = 0; i < n; ++i){
			phi_i.assign(J, 0.0);
			coords.zeros();

			// Fill phi weight vector 
			arma::sp_mat::const_col_iterator ci = pou.begin_col(i);
			for (; ci != pou.end_col(i); ++ci){ phi_i[ci.row()] = *ci; }

			// Compute the weighted average of the Fj's using the partition of unity
			ci = pou.begin_col(i);
			for (; ci != pou.end_col(i); ++ci){
				size_t j = ci.row(); 
				generate_frame_(j, phi_i, d_frame); 			// populates the (dJ x d) frame in d_frame
				svd_econ(U, s, V, A * (A.t() * d_frame)); // compute SVD of A A^T phi_j (U := dj x r, V := r x d, r <= d)
				
				// In cover set U_j, find the index of the local coordinate for point x_i, add the translation vector
				size_t jj = find_index(cover_subsets[j].begin(), cover_subsets[j].end(), i); // todo: remove this
				arma::vec local_coord = local_models[j].col(jj) + T.col(j); // should be column vector
				
				// Add to the current coordinate
				coords += ((*ci) * A.t() * U * V.t()) * local_coord; // lhs := (D x d), local coords := (d x 1)
			}
			assembly.col(i) = coords; 
		}
	}

	// High dimensional (dJ) assembly
	// assembly := output (dJ x n) matrix 
	void fast_assembly_high(const arma::sp_mat& pou, const index_list& cover_subsets, const vec_mats& local_models, const arma::mat& T, arma::mat& assembly){
		
		const size_t J = pou.n_rows;
		arma::vec coords = arma::zeros(d*J);
		
		// Variables to re-use/cache in the loop 
		vector< double > phi_i(J); // the partition of unity weights for x_i 
		arma::mat d_frame(d*J, d); // the current frame to populate 

		// Build the assembly
		for (size_t i = 0; i < n; ++i){
			phi_i.assign(J, 0.0);
			coords.zeros();

			// Fill phi weight vector 
			arma::sp_mat::const_col_iterator ci = pou.begin_col(i);
			for (; ci != pou.end_col(i); ++ci){ phi_i[ci.row()] = *ci; }

			// Compute the weighted average of the Fj's using the partition of unity
			ci = pou.begin_col(i);
			for (; ci != pou.end_col(i); ++ci){
				size_t j = ci.row(); 
				generate_frame_(j, phi_i, d_frame); 			// populates the (dJ x d) frame in d_frame
				size_t jj = find_index(cover_subsets[j].begin(), cover_subsets[j].end(), i); // todo: remove this
				arma::vec local_coord = local_models[j].col(jj) + T.col(j); // should be (d)-length column vector
				coords += (*ci) * d_frame * local_coord; // lhs := (dJ x 1), local coords := (d x 1)
			}
			assembly.col(i) = coords; 
		}
	}

	// Uses the fast_assembly2() function. 
	// All inputs as passed as-is to fast_assembly2; do not transpose anything here
	auto assemble_frames2(const py::array_t< double >& A, const py::object& pou, const py::list& cover_subsets, const py::list& local_models, const py::array_t< double >& T, bool high) -> py::array_t< double > {
		arma::mat A_ = carma::arr_to_mat(A);
		
		// Partition of unity
		arma::sp_mat pou_;
		to_sparse(pou, pou_);
		const size_t J = pou_.n_rows;

		// Convert cover subsets to C++ versions
		auto subsets = vector< vector< size_t > >();
		for (auto ind: cover_subsets){ subsets.push_back(ind.cast< vector< size_t > >()); }

		// Copy the local euclidean models (transposed)	
		auto models = vector< arma::mat >();
		for (auto pts: local_models){
			py::array_t< double > pts_ = pts.cast< py::array_t< double > >();
			models.push_back(carma::arr_to_mat(pts_, true));
		}

		// Copy/move the translations
		arma::mat translations = carma::arr_to_mat(T);
		
		// Output assembly
		if (high){
			arma::mat assembly = arma::zeros(d*J, n);
			fast_assembly_high(pou_, subsets, models, translations, assembly);
			return(carma::mat_to_arr(assembly));	
		}
		else {
			arma::mat assembly = arma::zeros(D, n);
			fast_assembly2(A_, pou_, subsets, models, translations, assembly);
			return(carma::mat_to_arr(assembly));
		}
	}

	// TODO: make pou and local_models transposed to access by column
	arma::mat fast_assembly(const arma::mat& A, const arma::sp_mat& pou, const index_list& cover_subsets, const vec_mats& local_models, const arma::mat& T){
		//py::array_t< double > phi_i = generate_frame(const size_t origin, py::array_t< double > weights);
		arma::mat assembly = arma::zeros(n, D);
		arma::vec coords = arma::zeros(D);
		const size_t J = pou.n_cols;
		
		// Variables to re-use/cache in the loop 
		vector< double > phi_i(J); // the partition of unity weights for x_i 
		arma::mat d_frame(d*J, d); // the current frame to populate 
		arma::mat U, V;
		arma::vec s;

		// Build the assembly
		for (size_t i = 0; i < n; ++i){
			phi_i.assign(J, 0.0);
			coords.zeros();

			// Fill phi weight vector 
			arma::sp_mat::const_row_iterator ri = pou.begin_row(i);
			for (; ri != pou.end_row(i); ++ri){ phi_i[ri.col()] = *ri; }

			// Compute the weighted average of the Fj's using the partition of unity
			ri = pou.begin_row(i);
			for (; ri != pou.end_row(i); ++ri){
				size_t j = ri.col(); 
				size_t jj = find_index(cover_subsets[j].begin(), cover_subsets[j].end(), i);
				generate_frame_(j, phi_i, d_frame); 			// populates d_frame
				svd_econ(U, s, V, A * (A.t() * d_frame)); // compute SVD of A A^T phi_j
				arma::rowvec local_coord = local_models[j].row(jj) + T.row(j); // should be column vector
				coords += ((*ri) * A.t() * U * V.t()) * local_coord.t(); // left-side should be (D x d)
			}
			assembly.row(i) = coords.t(); 
		}
		return(assembly);
	}

	// Wrapper for the fast_assembly above
	auto assemble_frames(py::array_t< double >& A, py::object& pou, py::list& cover_subsets, const py::list& local_models, py::array_t< double >& T ) -> py::array_t< double > {
		arma::mat A_ = carma::arr_to_mat(A, true);
		arma::sp_mat pou_;
		to_sparse(pou, pou_);
		auto subsets = vector< vector< size_t > >();
		for (auto ind: cover_subsets){
			subsets.push_back(ind.cast< vector< size_t > >());
		}	
		auto models = vector< arma::mat >();
		for (auto pts: local_models){
			py::array_t< double > pts_ = pts.cast< py::array_t< double > >();
			models.push_back(carma::arr_to_mat(pts_, true));
		}
		arma::mat translations = carma::arr_to_mat(T, true);
		arma::mat assembly = fast_assembly(A_, pou_, subsets, models, translations);
		return(carma::mat_to_arr(assembly));
	}

};

// if len(translations) != len(cover): raise ValueError("There should be a translation vector for each subset of the cover.")
// assembly = np.zeros((stf.n, stf.D), dtype=np.float64)
// coords = np.zeros((1,stf.D), dtype=np.float64)
// index_set = list(local_models.keys())
// for i in range(stf.n):
// 	w_i = np.ravel(pou[i,:].todense())
// 	nz_ind = np.where(w_i > 0)[0]
// 	coords.fill(0)
// 	## Construct assembly functions F_j(x) for x_i
// 	for j in nz_ind: 
// 		subset_j = cover[index_set[j]]
// 		relative_index = find_where(i, subset_j, True) ## This should always be true!
// 		u, s, vt = np.linalg.svd((A @ (A.T @ stf.generate_frame(j, w_i))), full_matrices=False, compute_uv=True) 
// 		d_coords = local_models[index_set[j]][relative_index,:]
// 		coords += (w_i[j]*A.T @ (u @ vt) @ (d_coords + translations[j]).T).T
// 	assembly[i,:] = coords

// extern void slaed1(int* N, float* D, float* Q, int* LDQ, int* INDXQ, float* RHO, int* CUTPNT, float* WORK, int* IWORK, int* INFO);	

// Computes U A U^T + sigma (u * u^T)
// SLAED1 computes the updated eigensystem of a diagonal matrix after modification by a rank-one symmetric matrix.
// void dpr1(py::array_t< float > D, py::array_t< float > V, float sigma, py::array_t< float > u){
// 	arma::fvec d = carma::arr_to_col(D, true);
// 	arma::fvec v = carma::arr_to_col(u, true);
// 	arma::fmat Q = carma::arr_to_mat(V, true);
// 	int N = d.size();
// 	int LDQ = Q.n_rows;
// 	vector< int > indxq(N);
// 	std::iota(indxq.begin(), indxq.end(), 0);
// 	int CUTPNT = N/2;
// 	vector< float > workspace(4*N + N*N);
// 	vector< int > iworkspace(4*N);
// 	int info = 0;
// 	slaed1(&N, d.memptr(), Q.memptr(), &LDQ, indxq.data(), &sigma, &CUTPNT, workspace.data(), iworkspace.data(), &info);
// 	py::print("Info: ", info);
// }

// diag( D ) +  RHO *  Z * Z_transpose.
// void dpr1(py::array_t< float > D, float rho, py::array_t< float > Z, int I){
// 	arma::fvec d = carma::arr_to_col(D, true);
// 	arma::fvec z = carma::arr_to_col(Z, true);
// 	int N = D.size(), info = 0;
// 	arma::fvec delta(N); // used for reconstructing eiegnvectors
// 	float lambda = 0; // output eigenvalue
// 	slaed4(&N, &I, d.memptr(), z.memptr(), delta.memptr(), &rho, &lambda, &info);
// 	py::print("Info: ", info, "updated ev: ", lambda);
// }

// slaed9(int* K, int* KSTART, int* KSTOP, int* N, float* D, float* Q, int* LDQ, float* rho, float* dlambda, float* W, float* S, int& lds, int& info); 	
// auto dpr1_ev(py::array_t< float > Q, py::array_t< float > D, float rho, py::array_t< float > Z) -> py::dict {
// 	arma::fmat q = carma::arr_to_mat(Q, true); // eigenvectors
// 	arma::fvec d = carma::arr_to_col(D, true); // diagonal entries / poles 
// 	arma::fvec z = carma::arr_to_col(Z, true); // perturbation vector
// 	int K = d.size(), N = q.n_rows, info = 0;
// 	int KSTART = 1, KEND = K;
// 	arma::fvec lambda(K); 
// 	arma::fmat S(q.n_rows, q.n_cols);
// 	slaed9(&K, &KSTART, &KEND, &N, lambda.memptr(), q.memptr(), &N, &rho, d.memptr(), z.memptr(), S.memptr(), &N, &info); 	
// 	// py::print("Info: ", info);
// 	py::dict output; 
// 	output["info"] = info;
// 	output["eval"] = carma::col_to_arr(lambda, true);
// 	output["evec"] = carma::mat_to_arr(S, true);
// 	return(output);
// }

// T = Q(in) ( D(in) + RHO * Z*Z**T ) Q**T(in) = Q(out) * D(out) * Q**T(out)


PYBIND11_MODULE(fast_svd, m) {
	m.def("fast_svd", &fast_svd, "Yields the svd of a matrix of low dimension");
	//m.def("lapack_svd", &lapack_svd, "Yields the svd of a matrix of low dimension");
	//m.def("test_sparse", &test_sparse, "Test conversion to sparse matrix");
	//m.def("dpr1", &dpr1, "Diagonal + rank-1 matrix eigenvalue update");
	//m.def("dpr1_ev", &dpr1_ev, "Diagonal + rank-1 matrix eigenvalue update");
	py::class_<StiefelLoss>(m, "StiefelLoss")
		.def(py::init< int, int, int >())
		.def_readonly("d", &StiefelLoss::d)
		.def_readonly("n", &StiefelLoss::n)
		.def_readonly("D", &StiefelLoss::D)
		.def_readwrite("output", &StiefelLoss::output)
		// .def_readwrite("rotations", &StiefelLoss::rotations)
		//.def("benchmark_gradient", &StiefelLoss::benchmark_gradient)
		.def("gradient", &StiefelLoss::gradient)
		.def("init_rotations", &StiefelLoss::init_rotations)
		.def("get_rotation", &StiefelLoss::get_rotation)		
		.def("populate_frame", &StiefelLoss::populate_frame)
		.def("populate_frames", &StiefelLoss::populate_frames)
		.def("populate_frames_sparse", &StiefelLoss::populate_frames_sparse)
		.def("generate_frame", &StiefelLoss::generate_frame)
		.def("get_frame", &StiefelLoss::get_frame)
		.def("all_frames", &StiefelLoss::all_frames)
		.def("all_frames_sparse", &StiefelLoss::all_frames_sparse)
		.def("embed", &StiefelLoss::embed)
		.def("extract_iota", &StiefelLoss::extract_iota)
		.def("setup_pou", &StiefelLoss::setup_pou)
		.def("benchmark_embedding", &StiefelLoss::benchmark_embedding)
		.def("assemble_frames", &StiefelLoss::assemble_frames)
		.def("assemble_frames2", &StiefelLoss::assemble_frames2)
		.def("initial_guess", &StiefelLoss::initial_guess)
		.def("__repr__",[](const StiefelLoss &stf) {
			return("Stiefel Loss w/ parameters n="+std::to_string(stf.n)+",d="+std::to_string(stf.d)+",D="+std::to_string(stf.D));
  	});
}





// double numpy_svd(){
// 	auto svd = py::module::import("numpy.linalg").attr("svd");
// 	py::buffer_info output_buffer = output.request();
// 	const size_t inc = D*d;
// 	double nuclear_norm = 0.0;
// 	for (int j = 0; j < n; ++j){
// 		np_array_t inp = np_array_t({ D, d }, output.data()+(j*inc));
// 		np_array_t sv = svd(inp, false, false, false);
// 		auto r = sv.unchecked< 1 >();
// 		for (size_t i = 0; i < sv.shape(0); i++){ 
// 			nuclear_norm += r(i); 
// 		}
// 	}
// 	return(nuclear_norm);
// }