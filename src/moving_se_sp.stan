data {
  int<lower=0> N;                               // Number of observations/time points
  array[N] int<lower=0> D;                      // Number of diagnosed cases at each time point
  array[N] int<lower=0> H;                      // Number of help-seeking cases at each time point
  array[N] real t;                              // Time points
  real<lower=0,upper=1> Se;                     // Sensitivity of the diagnostic test
  real<lower=0,upper=1> Sp;                     // Specificity of the diagnostic test

  // Hyperparameters for trend priors 
  real<lower=0> trend_scale_se;                 // Scale parameter for sensitivity trend prior
  real<lower=0> trend_scale_sp;                 // Scale parameter for specificity trend prior
}

parameters {
  real<lower=0,upper=1> true_prev;              // True prevalence of the condition

  real<lower=0,upper=1> eff_se_init_raw;        // Initial raw sensitivity
  real eff_se_trend;                            // Sensitivity trend over time

  real<lower=0,upper=1> eff_sp_init_raw;        // Initial raw specificity
  real eff_sp_trend;                            // Specificity trend over time

  // Latent Variables Representing Drift in Test Characteristics
  real<lower=0,upper=1> theta_se;               // Drift factor for sensitivity
  real<lower=0,upper=1> theta_sp;               // Drift factor for specificity
}

transformed parameters {
  real<lower=0,upper=1> eff_se_init;            // Effective initial sensitivity
  real<lower=0,upper=1> eff_sp_init;            // Effective initial specificity

  // Calculate Effective Initial Sensitivity and Specificity
  eff_se_init = Se * theta_se + (1 - Se) * eff_se_init_raw;
  eff_sp_init = Sp * theta_sp + (1 - Sp) * eff_sp_init_raw;
}

model {
  array[N] real eff_se;                         // Effective sensitivity over time
  array[N] real eff_sp;                         // Effective specificity over time
  array[N] real apparent_prev;                  // Apparent prevalence over time

  //------------------------------------------------------------
  // Priors
  //------------------------------------------------------------
  true_prev ~ beta(7, 3);                       // Prior for true prevalence

  eff_se_init_raw ~ beta(7, 3);                 // Prior for initial raw sensitivity
  eff_se_trend ~ normal(0, trend_scale_se);     // Prior for sensitivity trend

  eff_sp_init_raw ~ beta(7, 3);                 // Prior for initial raw specificity
  eff_sp_trend ~ normal(0, trend_scale_sp);     // Prior for specificity trend

  theta_se ~ beta(7, 3);                        // Prior for sensitivity drift factor
  theta_sp ~ beta(7, 3);                        // Prior for specificity drift factor

  //------------------------------------------------------------
  // Likelihood
  //------------------------------------------------------------
  for (i in 1:N) {
    // Update Sensitivity and Specificity Over Time Using Logistic Transformation
    eff_se[i] = inv_logit(logit(eff_se_init) + eff_se_trend * t[i]);
    eff_sp[i] = inv_logit(logit(eff_sp_init) + eff_sp_trend * t[i]);

    // Calculate Apparent Prevalence Based on Test Characteristics
    apparent_prev[i] = true_prev * eff_se[i] + (1 - true_prev) * (1 - eff_sp[i]);

    
    // Observation Model: Diagnosed Cases Follow a Binomial Distribution
    D[i] ~ binomial(H[i], fmin(apparent_prev[i], 1.0));

  }
}

generated quantities {
  array[N] real diagnosed_cases;
  array[N] real eff_se;
  array[N] real eff_sp;
  array[N] real apparent_prev;
  array[N] real true_positives;
  array[N] real false_positives;
  array[N] real ppv;

  for (i in 1:N) {
    // Recompute Effective Sensitivity and Specificity
    eff_se[i] = inv_logit(logit(eff_se_init) + eff_se_trend * t[i]);
    eff_sp[i] = inv_logit(logit(eff_sp_init) + eff_sp_trend * t[i]);

    // Recompute Apparent Prevalence
    apparent_prev[i] = true_prev * eff_se[i] + (1 - true_prev) * (1 - eff_sp[i]);

    // Diagnosed cases as apparent prevalence times help-seeking individuals
    diagnosed_cases[i] = H[i] * apparent_prev[i];

    // Calculate True Positives and False Positives
    true_positives[i] = H[i] * true_prev * eff_se[i];
    false_positives[i] = H[i] * (1 - true_prev) * (1 - eff_sp[i]);

    // Calculate Positive Predictive Value (PPV)
    ppv[i] = (true_prev * eff_se[i]) / apparent_prev[i];
  }
}
