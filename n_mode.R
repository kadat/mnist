d <- 2 # 特徴写像の次元数

mode2_product <- function(x, y) {
  left_dim <- length(x[,1,1]) # xのモード1の次元数
  right_dim <- length(x[1,1,]) # xのモード2の次元数
  a <- array(seq(0, 0, length = left_dim * right_dim), dim=c(left_dim, right_dim))
  
  for (i in 1 : left_dim) {
    for (j in 1 : right_dim) {
      for (k in 1 : d) {
        a[i, j] <- a[i, j] + x[i, k, j] * y[k]
      }
    }
  }
  return(a)
}
