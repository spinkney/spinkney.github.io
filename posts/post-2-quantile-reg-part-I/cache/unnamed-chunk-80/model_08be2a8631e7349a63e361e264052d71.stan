functions{
    real q_loss(real q, vector u){
        return 0.5 * sum(fabs(u) + (2 * q - 1) * u);
    }

    real ald_lpdf(real q, vector y, real tau, vector q_est){
        int N = num_elements(y);

        return N * (log(tau) + log(q) + log1m(q)) - tau * q_loss(q, y - q_est);
    }
}
data {
    int N;                 // Number of observation
    int M;                 // Number of predictors
    real q;
    vector[N] y;           // Response variable sorted
    matrix[N, M] x;
}
parameters {
   vector[M] beta;
   real<lower=0> tau;
}
model {
  beta ~ normal(0, 4);
  tau ~ gamma(1., 1.);
  q ~ ald(y, tau, x * beta);
}
