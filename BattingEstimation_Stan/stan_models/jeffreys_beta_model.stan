
data {
  int<lower=1> n_players;                            // number of players observed
  array[n_players] int<lower=0> hits;                // number of hits - needs to be integer type because of binomial call
  array[n_players] int<lower=0> atbat;               // number of at-bats - needs to be integer type because of binomial call
}
parameters {
  vector<lower=0, upper=1>[n_players] gamma;           // unobserved "true" batting abilities
}
model {
  // relations between parameters and data
  gamma ~ beta(0.5, 0.5);                           // distribution of unobservable batting ability - Jeffreys prior
  hits ~ binomial(atbat, gamma);                    // relation of hits to per-player ability
}
