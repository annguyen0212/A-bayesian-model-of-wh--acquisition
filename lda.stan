data {
  // training data
  int<lower=1> K;               // num topics
  int<lower=1> S_total;         // num s = 2
  int<lower=1> N;               // total observation = 88
  int<lower=1,upper=2> s[N];    // surface structure of each observation
  real dur[N];                   // duration of each observation
  real delta[N];                    // pitch contour of each observation
  vector[K] alpha;
  vector[S_total] beta;
  
}
parameters {
  simplex[K] theta;             // topic prevalence
  simplex[S_total] phi[K];      // structure dist for topic k
  vector<lower=0>[K] mu_dur;            //mixture component means of duration
  real sigma_dur[K];
  ordered[K] mu_delta;            //mixture component means of duration
  real sigma_delta[K];
}

transformed parameters{
  matrix[N,K] gamma; 
   for (n in 1:N){
    for (k in 1:K){
      gamma[n, k] = categorical_lpmf(k| theta) + categorical_lpmf(s[n] | phi[k]) + normal_lpdf(dur[n]|mu_dur[k],exp(sigma_dur[k])) + normal_lpdf(delta[n]|mu_delta[k],exp(sigma_delta[k]));
      }
      }
}

model {
  theta ~ dirichlet(alpha);
  sigma_dur ~ lognormal(0, 2); //uninformative
  mu_dur ~ normal(1, 1); //uninformative prior
  sigma_delta ~ lognormal(0, 2); //uninformative
  mu_delta ~ normal(10, 50);
  for (k in 1:K){
    phi[k] ~ dirichlet(beta);}
  for (n in 1:N){
    target += log_sum_exp(gamma[n]);
        }
}

generated quantities {
  matrix[N,K]p;
  for (n in 1:N){
    for (k in 1:K){
      p[n,k]= exp(gamma[n,k]-log_sum_exp(gamma[n]));
    }
}}