#!/usr/bin/env Rscript
# Solves for the threshold that will spend close to the desired alpha
# in a sequential test. The test creates upper bounds for the test statistics
# Arg 1: csv file for covariance matrix
# Arg 2: alpha to spend
# Arg 3 and more: critical values for prior normal statistics
# Stdout: the critical value for desired alpha spending

library(mvtnorm)

args = commandArgs(trailingOnly=TRUE)

corr_mat_df = read.csv(args[1], sep=",", header = F)
corr_mat = data.matrix(corr_mat_df)
corr_mat = unname(corr_mat)
corr_mat = (corr_mat + t(corr_mat))/2

bounds_df = read.csv(args[2], sep=",", header = F)
bounds_mat = data.matrix(bounds_df)

alpha_spend = 10^as.numeric(args[3])
alt_greater = as.numeric(args[4])

get_sequential_spend <- function(thres) {
  test_lower = ifelse(alt_greater == 1, thres, -Inf)
  test_upper = ifelse(alt_greater == 1, Inf, thres)
  lower_all = c(bounds_df[,1], test_lower)
  upper_all = c(bounds_df[,2], test_upper)
  res = pmvnorm(lower=lower_all, upper = upper_all, sigma=corr_mat, maxpts=50000, abseps =0.00001)
  res
}

get_sequential_spend_diff <- function(thres) {
  reject_prob = get_sequential_spend(thres)
  reject_prob - alpha_spend
}

if (alt_greater == 1){
    boundaries = c(0, max(bounds_df[,2]) + 100)
} else {
    boundaries = c(min(bounds_df[,2]) - 100, 0)
}
res = uniroot(get_sequential_spend_diff, boundaries)
print(res$root)
