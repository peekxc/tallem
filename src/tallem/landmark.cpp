// landmark.cpp
// Based on landmarks_maxmin.cpp in the 'landmark' R package by Matt Piekenbrock, Jason Cory Brunson, Yara Skaf
#include <carma>
#include <armadillo>

#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>

using std::vector; 
using std::size_t;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace pybind11::literals;
namespace py = pybind11;

template< typename InputIt >
inline double sq_euc_dist(InputIt x, InputIt y, const size_t d){
	double res = 0.0; 
	for(size_t i = 0; i < d; ++i, ++x, ++y){
		res += ((*x) - (*y)) * ((*x) - (*y));
	}
	return(res);
}
// a Distance Function here just returns the distance between two points, given their *indices*
using DistFunction = typename std::function<double(size_t, size_t)>;

// dist_f  := distance function between two (indexed) points
// n_pts   := number of points in the data set
// eps     := distance threshold used as a stopping criterion for the maxmin procedure
// n       := cardinality threshold used as a stopping criterion for the maxmin procedure
// metric  := metric to use. If 0, uses `dist_f`, otherwise picks one of the available metrics.
// seed    := initial point (default is point at index 0)
// pick    := criterion to break ties. Possible values include 0 (first), 1 (random), or 2 (last).
// cover   := whether to report set membership for each point
void maxmin_f(DistFunction dist_f, const size_t n_pts,
              const double eps, const size_t n,
              const size_t seed, const size_t pick,
							vector< size_t >& indices, 
							vector< double >& radii
							) {
  if (eps == -1.0 && n == 0){ throw std::invalid_argument("Must supply either positive 'eps' or positive 'n'."); }
  if (pick > 2){ throw std::invalid_argument("tiebreaker 'pick' choice must be in { 0, 1, 2 }."); }
  if (seed >= n_pts){ throw std::invalid_argument("Invalid seed index given."); }
	if (indices.size() == 0 || radii.size() == 0){ throw std::invalid_argument("Indices and radii must have at least one element to begin with."); }

  // Make a function that acts as a sentinel
  enum CRITERION { NUM, EPS, NUM_OR_EPS }; // These are the only ones that make sense
  const CRITERION stopping_criterion = (eps == -1.0) ? NUM : ((n == 0) ? EPS : NUM_OR_EPS);
  const auto is_finished = [stopping_criterion, eps, n](size_t n_landmarks, double c_eps){
    switch(stopping_criterion){
      case NUM: return(n_landmarks >= n);
      case EPS: return(c_eps <= eps);
      case NUM_OR_EPS: return(n_landmarks >= n || c_eps <= eps);
    }
  };

  // Indices of possible candidate landmarks
  vector< size_t > candidate_pts(n_pts, 0);
  std::iota(begin(candidate_pts), end(candidate_pts), 0);
  candidate_pts.erase(begin(candidate_pts) + seed);

  // Preallocate distance vector for landmarks; one for each point
	double cover_radius = std::numeric_limits<double>::infinity();
  vector< double > lm_dist(n_pts, cover_radius);

  // Generate the landmarks
  bool stop_reached = false;
  while (!stop_reached){
    const size_t c_lm = indices.back(); // update current landmark

    // Update non-landmark points with distance to nearest landmark
    for (auto idx: candidate_pts){
      double c_dist = dist_f(c_lm, idx);
      if (c_dist < lm_dist[idx]){
        lm_dist[idx] = c_dist; // update minimum landmark distance
      }
    }

    // Of the remaining candidate points, find the one with the maximum landmark distance
    auto max_landmark = std::max_element(begin(candidate_pts), end(candidate_pts), [&lm_dist](size_t ii, size_t jj){
      return lm_dist[ii] < lm_dist[jj];
    });

    // If not greedily picking the first candidate point, partition the candidate points, then use corresponding strategy
    if (pick > 0 && max_landmark != end(candidate_pts)){
      double max_lm_dist = lm_dist[(*max_landmark)];
      auto it = std::partition(begin(candidate_pts), end(candidate_pts), [max_lm_dist, &lm_dist](size_t j){
        return lm_dist[j] == max_lm_dist;
      });
			max_landmark = 
				pick == 1 ? (it != begin(candidate_pts) ? std::prev(it) : begin(candidate_pts)) :
				begin(candidate_pts) + (rand() % std::distance(begin(candidate_pts), it));
    }

    // If the iterator is valid, we have a new landmark, otherwise we're finished
    if (max_landmark != end(candidate_pts)){
      cover_radius = lm_dist[(*max_landmark)];
      stop_reached = is_finished(indices.size(), cover_radius);
      if (!stop_reached){
        indices.push_back(*max_landmark);
				radii.push_back(cover_radius);
				candidate_pts.erase(max_landmark);
      }
    } else {
      cover_radius = 0.0;
      stop_reached = true;
    }
  } // while(!finished())
}

// Point cloud wrapper - See 'maxmin_f' below for implementation
py::tuple maxmin_pc(
	const arma::mat& X, 
	const double eps, const size_t n,
	const size_t metric = 1, const size_t seed = 0, const size_t pick = 0
){
  const size_t n_pts = X.n_cols, d = X.n_rows;
  if (seed < 0 || seed >= n_pts){ throw std::invalid_argument("Invalid seed point."); }

	// Initial covering radius == Inf 
	vector< double > cover_radii{ std::numeric_limits<double>::infinity() };

  // Choose the initial landmark
  vector< size_t > lm { 0 };
  lm.reserve(n != 0 ? n : size_t(n_pts*0.15));

  // Choose the distance function
  DistFunction dist = [&X, d](size_t i, size_t j) { 
		return(sq_euc_dist(X.begin_col(i), X.begin_col(j), d));
	};

  // Call the generalized procedure
  maxmin_f(dist, n_pts, eps, n, seed, pick, lm, cover_radii);
	
	return(py::make_tuple(lm, cover_radii));
}

// Converts (i,j) indices in the range { 0, 1, ..., n - 1 } to its 0-based position
// in a lexicographical ordering of the (n choose 2) combinations.
constexpr size_t to_nat_2(size_t i, size_t j, size_t n) noexcept {
  return i < j ? (n*i - i*(i+1)/2 + j - i - 1) : (n*j - j*(j+1)/2 + i - j - 1);
}

// n_pts := number of points in X
py::tuple maxmin_dist(
	const arma::vec& X, const size_t n_pts,
	const double eps, const size_t n,
	const size_t seed = 0, const size_t pick = 0){
  if (seed < 0 || seed >= n_pts){ throw std::invalid_argument("Invalid seed point."); }

  // Parameterize the distance function
  DistFunction dist = [&X, n_pts](size_t i, size_t j) -> double {
    return X[to_nat_2(i,j,n_pts)];
  };
	
	// Initial covering radius == Inf 
	vector< double > cover_radii{ std::numeric_limits<double>::infinity() };

  // Choose the initial landmark
  vector< size_t > lm { 0 };
  lm.reserve(n != 0 ? n : size_t(n_pts*0.15));

  // Call the generalized procedure
  maxmin_f(dist, n_pts, eps, n, seed, pick, lm, cover_radii);
	return(py::make_tuple(lm, cover_radii));
}

// Maxmin procedure O(n^2)
py::tuple maxmin(const py::array_t<double>& x, const double eps, const size_t n, bool pairwise_dist){
	if (pairwise_dist){
		const arma::vec dx = carma::arr_to_col< double >(x);
		const size_t N = dx.size();

		// Find n such that choose(n, 2) == N
		size_t lb = std::sqrt(2*N); 
		size_t n_pts = size_t(floor(lb));
		for (; n_pts <= size_t(std::ceil(lb+2)); ++n_pts){
			if (N == ((n_pts * (n_pts - 1))/2)){ break; }
		}
		return(maxmin_dist(dx, n_pts, eps, n, 0, 0));
	} else {
		const arma::mat X = carma::arr_to_mat< double >(x);
		return(maxmin_pc(X, eps, n, 1, 0, 0));
	}
}

#include <thread>
using std::thread; 

void doSomething(int thread_id, vector< double >& output) {
	output[thread_id] = std::sqrt(static_cast< double >(thread_id));
}

// Spawns n threads
auto spawnThreads(int n) -> vector< double > {
	vector< thread > threads(n);
	vector< double > output(n, 0.0); 
	for (int i = 0; i < n; i++) {
		threads[i] = thread(doSomething, i, std::ref(output));
	}
	for (auto& th : threads) { th.join(); } // each thread blocks until it's finished, sequentially 
	return(output);
}

// Classical MDS 
// D := distance matrix
void cmds(const arma::mat& D, const size_t d, arma::vec& w, arma::mat& v){
	const size_t n = D.n_rows;
	arma::mat H(n, n, arma::fill::none);
	H.fill(-1/n);
	H.diag().fill(1-(1/n)); 
	arma::eig_sym(w, v, -0.5 * H * D * H);
	// arma::eigs_sym(w, v, -0.5 * H * D * H, d);
}

PYBIND11_MODULE(landmark, m) {
	m.def("maxmin", &maxmin, "finds maxmin landmarks");
	m.def("do_parallel", [](size_t n_threads) -> py::array_t< double > {
		/* Release GIL before calling into (potentially long-running) C++ code */
		py::gil_scoped_release release;
		vector< double > res = spawnThreads(n_threads);
		py::gil_scoped_acquire acquire;
		return(carma::col_to_arr(arma::vec(res)));
	});
	m.def("cmds", [](py::array_t< double >& X, const size_t d) -> py::tuple {
		arma::mat v; 
		arma::vec w; 
		const arma::mat D = carma::arr_to_mat< double >(X);
		cmds(D, d, w, v);
		return py::make_tuple(carma::col_to_arr< double >(w), carma::mat_to_arr< double >(v));
	});
};

