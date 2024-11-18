for (int i = 1; i < M; ++i) {
  for (int j = 1, j < M; ++j) {
    for (int k = 1; k < N; ++k) {
      for (int l = 1; l < N; ++k) {
		r[i][k] = x[i][l] * y[l][j] * s[j][k];
      }
    }
  }
}
