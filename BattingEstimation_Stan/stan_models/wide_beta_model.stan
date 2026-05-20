
data {
  int<lower=1> n_players;                            // number of players observed
  array[n_players] int<lower=0> hits;                // number of hits - needs to be integer type because of binomial call
  array[n_players] int<lower=0> atbat;               // number of at-bats - needs to be integer type because of binomial call
  real<lower=0, upper=1> mean_batting_avg;             // mean player batting average
}
parameters {
  vector<lower=0, upper=1>[n_players] gamma;           // unobserved "true" batting abilities
}
model {
  // wide priors
  gamma ~ beta(mean_batting_avg, 1 - mean_batting_avg); 
  hits ~ binomial(atbat, gamma);                    // relation of hits to per-player ability
}
