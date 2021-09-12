#!/usr/bin/env Rscript
# Solves for the threshold that will spend exactly alpha
# in a sequential test
# Arg 1: csv file for covariance matrix
# Arg 2: alpha to spend
# Arg 3 and more: critical values for prior normalized statistics
# Stdout: the critical value for desired alpha spending

library(mvtnorm)

args = commandArgs(trailingOnly=TRUE)

corr_mat_df = read.csv(args[1], sep=",", header = F)
corr_mat = data.matrix(corr_mat_df)
corr_mat = unname(corr_mat)
corr_mat = (corr_mat + t(corr_mat))/2
alpha_spend = as.numeric(args[2])
lower_prior = as.numeric(args[seq(3, length(args))])

corr_prior = corr_mat[seq(length(lower_prior)), seq(length(lower_prior))]
surv_prior = pmvnorm(lower=lower_prior, sigma=corr_prior)

get_sequential_spend <- function(thres) {
  lower_all = c(lower_prior, thres)
  if (thres < -100) {
    # When threshold is super super small, you might as well set this equal to the same as the original calculation
    surv_all = surv_prior
  } else {
    surv_all = pmvnorm(lower=lower_all, sigma=corr_mat, maxpts=50000, abseps =0.00001)
  }
  surv_prior - surv_all
}


get_sequential_spend_diff <- function(thres) {
  reject_prob = get_sequential_spend(thres)
  reject_prob - alpha_spend
}

#print(min(lower_prior) - 1000)
#print(get_sequential_spend_diff(-1000))
#print(get_sequential_spend_diff(0))
res = uniroot(get_sequential_spend_diff, c(min(lower_prior) - 100, 0))
print(res$root)
