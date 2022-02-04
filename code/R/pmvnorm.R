#!/usr/bin/env Rscript
# Solves for the threshold that will spend close to the desired alpha
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
alpha_spend = 10^as.numeric(args[2])
#lower_prior = as.numeric(args[seq(3, length(args))])
upper_prior = as.numeric(args[seq(3, length(args))])

get_sequential_spend <- function(thres) {
  #lower_all = c(lower_prior, -Inf)
  #upper_all = c(rep(Inf, length(lower_prior)), thres)
  lower_all = c(rep(-Inf, length(upper_prior)), thres)
  upper_all = c(upper_prior, Inf)
  res = pmvnorm(lower=lower_all, upper = upper_all, sigma=corr_mat, maxpts=50000, abseps =0.00001)
  res
}


get_sequential_spend_diff <- function(thres) {
  reject_prob = get_sequential_spend(thres)
  reject_prob - alpha_spend
}

res = uniroot(get_sequential_spend_diff, c(0, max(upper_prior) + 100))
print(res$root)
