functions {
   vasicek_quantile_lpdf(vector y, real theta, real mu, real tau) {
    int N = num_elements(y);
    real lpdf = 0.5 * N * log1m(theta) - log(theta);
    
    if (tau <= 0 || tau >= 1) 
      reject("tau must be between 0 and 1 found tau = ", tau);
    
    if (theta <= 0 || theta >= 1) 
      reject("theta must be between 0 and 1 found theta = ", theta);
    
    if (mu <= 0 || mu >= 1) 
      reject("mu must be between 0 and 1 found mu = ", mu);
    
    return lpdf - 0.5 / theta * 
            sum( (sqrt(1 - theta) * (inv_Phi(y) - inv_Phi(mu)) + 
                  sqrt(theta) * inv_Phi(tau))^2 );
}