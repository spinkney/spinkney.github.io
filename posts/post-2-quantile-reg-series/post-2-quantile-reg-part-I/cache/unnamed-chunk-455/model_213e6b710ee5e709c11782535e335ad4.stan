functions{
  vector q_loss(real q, vector u){
      return 0.5 * (abs(u) + (2 * q - 1) * u);
  }
    
  vector score(real q, vector y, matrix x, vector beta) {
    int N = num_elements(y);
    int P = num_elements(beta);
 
    return x' * q_loss(q, y - x * beta);
  }
    
  matrix make_w (matrix x, real q) {
    int N = rows(x);
    int P = cols(x);
    real alpha = inv(q * (1 - q));
    matrix[P, P] out = rep_matrix(0., P, P);
      
    for (n in 1:N)
      out += x[n]' * x[n];
        
    return alpha * inverse( out ) * N;
  }
}
data {
  int N;                 // Number of observation
  int P;                 // Number of predictors
  real q;
  vector[N] y;           // Response variable sorted
  matrix[N, P] x;
}
transformed data {
  matrix[P, P] W = make_w(x, q);
}
parameters {
  vector[P] beta;
}
model {
  vector[P] score_vec = score(q, y, x, beta);

  beta ~ normal(0, 4);
 
 // target += -score_vec' * W * score_vec * inv(2 * N);
  target += -square(score_vec) * W * inv(2 * N);
}
