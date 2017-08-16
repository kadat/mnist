library(rTensor)

# working directoryの設定
setwd("/Users/kada/Documents/szk/mnist")

# 正解数のカウント
count = 0

# データのテスト
for (a in 10001 : 10100) {
  # エンコード
  x <- data.matrix(comp_train_label[a,-1])
  for (i in 1 : N) {
    phi1[i] <- cos(pi * x[i] / (2 * 1020)) # ここで割る mean(...)/255
    phi2[i] <- sin(pi * x[i] / (2 * 1020))
  }
  
  tilde <- list() # B_lによって分離されたテンソル組を定義
  
  # 予測値f_lの計算
  f_product <- list()
  for (i in 1 : N) { # w_tnsrとphiの積
    temp <- ttm(w_tnsr[[i]], matrix(c(phi1[i], phi2[i]), 1, 2), m = 2)
    f_product <- c(f_product, temp)
  }
  
  temp <- seq(0, 0, length = m)
  for (i in 2 : (l-1)) { # lより左の計算, l-1には括弧が必要
    left_v <- seq(0, 0, length = m)
    if (i == 2) {
      for (j in 1 : m) { # dim(f_product[[i]][1,1,]) = m
        for (k in 1 : m) { # dim(f_product[[i+1]][1,1,]) = m
          left_v[j] <- left_v[j] + f_product[[1]][1,1,k]@data * f_product[[2]][k,1,j]@data
        }
      }
    } else {
      for (j in 1 : m) {
        for (k in 1 : m) {
          left_v[j] <- left_v[j] + temp[k] * f_product[[i]][k,1,j]@data
        }
      }
    }
    temp <- left_v
    #print(left_v)
  }
  tilde <- c(tilde, list(left_v))
  tilde <- c(tilde, list(c(phi1[l], phi2[l])))
  tilde <- c(tilde, list(c(phi1[l+1], phi2[l+1])))
  
  temp <- seq(0, 0, length = m)
  for (i in ((N-2) : (l+2))) { # l+1より右の計算
    right_v <- seq(0, 0, length = m)
    if (i == (N-2)) {
      for (j in 1 : m) { # dim(f_product[[i-1]][,1,1]) = m
        for (k in 1 : m) { # dim(f_product[[i]][,1,1]) = m
          right_v[j] <- right_v[j] + f_product[[N]][k,1,1]@data * f_product[[N-1]][j,1,k]@data
        }
      }
    } else {
      for (j in 1 : m) {
        for (k in 1 : m) {
          right_v[j] <- right_v[j] + temp[k] * f_product[[i]][j,1,k]@data
        }
      }
    }
    temp <- right_v
  }
  tilde <- c(tilde, list(right_v))
  
  # f_lの計算
  f_l <- seq(0, 0, length = N_L)
  for (i1 in 1 : N_L) {
    for (i2 in 1 : m) {
      for (i3 in 1 : 2) {
        for (i4 in 1 : 2) {
          for (i5 in 1 : m) {
            f_l[i1] <- f_l[i1] + B[i2,i3,i4,i5,i1]@data * tilde[[1]][i2] * tilde[[2]][i3] * tilde[[3]][i4] * tilde[[4]][i5]
          }
        }
      }
    }
  }
  
  if (train[a,-2][1] == which.max(abs(f_l)) - 1) {
    count = count + 1
  }
  #paste(a, train[a,-2][1], which.max(abs(f_l)) - 1)
  print(c(a, train[a,-2][1], which.max(f_l) - 1))
}
