for (int i = 1; i < M; ++i) {
  for (int k = 0; k < i; ++k) {
	for (int j = 0; j < i; ++j) {
	  r[i] = x[i][j] * y[j][k];
	}
  }
}
