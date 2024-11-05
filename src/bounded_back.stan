functions {
  real bounded(real x, real lower_bound, real upper_bound) {
    return lower_bound + (upper_bound - lower_bound) * inv_logit(x);
  }
}

data {
  int<lower=0> N;                         // Number of time points
  array[N] int<lower=0> D;                // Number of observed positive tests
  array[N] int<lower=0> H;                // Number of people tested
  array[N] real t;                        // Time points
  real<lower=0,upper=1> Se;               // Sensitivity
}

parameters {
  real<lower=0> fp_influx_rate;           // Rate of false positive influx
  real logit_init_fp_rate;                // Logit of initial false positive rate
  real<lower=0> fp_influx_rate_alfa;
  real<lower=0> fp_influx_rate_beta;
}

transformed parameters {
  vector<lower=0,upper=1>[N] tested_prev;
  vector<lower=0,upper=1>[N] inflated_fp_rate;
  vector<lower=0,upper=1>[N] Sp;                    // Time-varying specificity
  vector<lower=0,upper=1>[N] rogan_gladen_prev;
  vector<lower=0,upper=1>[N] expected_positive_rate;

  real init_fp_rate = bounded(logit_init_fp_rate, 0, 1); // Initial false positive rate

  for (i in 1:N) {
    // Cap tested_prev to 1 using fmin to ensure it stays within [0, 1]
    tested_prev[i] = fmin(D[i] * 1.0 / H[i], 1.0);

    inflated_fp_rate[i] = bounded(init_fp_rate + fp_influx_rate * t[i] / N, 0, 0.3);

    Sp[i] = 1 - inflated_fp_rate[i]; // Time-varying specificity

    // Rogan-Gladen adjusted prevalence, bounded to [0, 1]
    rogan_gladen_prev[i] = bounded((tested_prev[i] + Sp[i] - 1) / (Se + Sp[i] - 1), 0, 1);

    expected_positive_rate[i] = rogan_gladen_prev[i] * Se + (1 - rogan_gladen_prev[i]) * (1 - Sp[i]);
  }
}

model {
  // Priors
  fp_influx_rate ~ gamma(fp_influx_rate_alfa, fp_influx_rate_alfa);
  logit_init_fp_rate ~ normal(0, 1);

  // Likelihood
  D ~ binomial(H, expected_positive_rate);
}

generated quantities {
  array[N] real<lower=0,upper=1> tested_prev_gen;
  array[N] real<lower=0,upper=1> ppv;
  array[N] real<lower=0> true_positives;
  array[N] real<lower=0> false_positives;

  for (i in 1:N) {
    // Include tested_prev into generated quantities
    tested_prev_gen[i] = D[i] * 1.0 / H[i];

    // Compute expected number of true positives
    real expected_tp = H[i] * rogan_gladen_prev[i] * Se;

    // Ensure true positives do not exceed observed positives
    true_positives[i] = fmin(expected_tp, D[i]);

    // Calculate false positives as the remainder
    false_positives[i] = D[i] - true_positives[i];

    // Positive Predictive Value
    ppv[i] = true_positives[i] / fmax(D[i], 1e-10);
  }
}
