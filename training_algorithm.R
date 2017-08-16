library(rTensor)

# working directoryの設定
setwd("/Users/kada/Documents/szk/mnist")

N_T <- 42000 # データ数
N_L <- 10 # ラベル数(0~9の10通り)

N <- 14*14 # 画像のピクセル数
phi1 <- seq(0, 0, length = N) # x1の特徴写像
phi2 <- seq(0, 0, length = N) # x2の特徴写像

eta <- 0.02 # 学習率


# テンソルトレインを定義(初期化)
m <- 10 # band dimension
l <- 50 # 番号のラベル
w_tnsr <- list()
for (i in 1 : N) {
  if (i == 1) {
    temp =  rand_tensor(c(1,2,m)) / 3
  } else if (i == N) {
    temp =  rand_tensor(c(m,2,1)) / 3
  } else if (i == l) {
    temp =  rand_tensor(c(m,2,m,N_L)) / 3
  } else {
    temp =  rand_tensor(c(m,2,m)) / 3
  }
  w_tnsr <- c(w_tnsr, temp)
}


for (times in 1 : 100) {

# テンソルの結合
B <- as.tensor(array(seq(0, 0, length = m*2*2*m*N_L), dim = c(m,2,2,m,N_L)))
delta <- B # テンソル微分(初期化)
for (i1 in 1 : dim(B[,1,1,1,1])) {
  for (i2 in 1 : dim(B[1,,1,1,1])) {
    for (i3 in 1 : dim(B[1,1,,1,1])) {
      for (i4 in 1 : dim(B[1,1,1,,1])) {
        for (i5 in 1 : dim(B[1,1,1,1,])) {
          for (j in 1 : m) {
            B[i1,i2,i3,i4,i5] <- B[i1,i2,i3,i4,i5] + w_tnsr[[l]][i1,i2,j,i5] * w_tnsr[[l+1]][j,i3,i4]
          }
        }
      }
    }
  }
}

# データの訓練
for (a in (100*times-99) : (100*times)) { # 1 : N_L
  # エンコード
  x <- data.matrix(comp_train_label[a,-1])
  for (i in 1 : N) {
    phi1[i] <- cos(pi * x[i] / (2 * 1020)) # ここで割る mean(...)/255
    phi2[i] <- sin(pi * x[i] / (2 * 1020))
  }
  
  # ラベルy_lの計算
  y_l <- seq(0, 0, length = N_L)
  for (i in 0 : 9) {
    if (i == train[a,-2][1]) {
      y_l[i] <- 1
    } else {
      y_l[i] <- 0
    }
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
  
  # シグマ計算
  delta <- delta + tilde[[1]] %o% tilde[[2]] %o% tilde[[3]] %o% tilde[[4]] %o% (y_l - f_l)
  print(a)
}

# コスト関数の更新
B <- B + eta * delta

# B_lの行列化
C <- matrix(seq(0, 0, length = m*2*2*m*N_L), nrow=m*2, ncol=2*m*N_L)
for (i1 in 1 : m) {
  for (i2 in 1 : 2) {
    for (i3 in 1 : 2) {
      for (i4 in 1 : m) {
        for (i5 in 1 : N_L) {
          C[(i1-1)*2+i2, (i3-1)*m*N_L+(i4-1)*N_L+i5] <- B[i1,i2,i3,i4,i5]@data
        }
      }
    }
  }
}

# 行列化されたB_lの特異値分解(TT-format)
U <- svd(C)$u
V <- svd(C)$v
S <- diag(sqrt(svd(C)$d))

# テンソルトレインの再構成
w_tnsr[[l]] <- as.tensor(array(seq(0, 0, length = m*2*m), dim = c(m,2,m)))
for (i1 in 1 : m) {
  for (i2 in 1 : 2) {
    for (i3 in 1 : m) {
      w_tnsr[[l]][i1,i2,i3] <- U[(i1-1)*2+i2,i3]
    }
  }
}

w_tnsr[[l+1]] <- as.tensor(array(seq(0, 0, length = m*2*m*N_L), dim = c(m,2,m,N_L)))
for (i1 in 1 : m) {
  for (i2 in 1 : 2) {
    for (i3 in 1 : m) {
      for (i4 in 1 : N_L) {
        for (j in 1 : 2*m) {
          w_tnsr[[l+1]][i1,i2,i3,i4] <- w_tnsr[[l+1]][i1,i2,i3,i4] + S[i1,j] * V[(i2-1)*m*N_L+(i3-1)*N_L+i4,j]
        }
      }
    }
  }
}
l <- l + 1

}
