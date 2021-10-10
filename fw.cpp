using uw = arma::uword; 
arma::mat D = carma::arr_to_mat< double >(d, false);


void floyd_warshall(arma::mat& D){
	const uw n = D.nrows;
	for (uw k = 0; k < n; ++k) {
		for (uw i = 0; i < n; ++i) {
			for (uw j = 0; j < n; ++j) {
				D(i,j) = std::min(D(i,j), D(i,k) + D(k,j)); 
			}
		}
	}
}
