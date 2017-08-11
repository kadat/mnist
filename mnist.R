# MNISTのトレーニングデータ読み込み
# train <- read.csv("MNIST/train.csv")

# MNISTのトレーニングデータを画像表示する
view_train <- function(train, range = 1:20) {
  par(mfrow=c(length(range)/4, 5))
  par(mar=c(0,0,0,0))
  for (i in range) {
    m <- matrix(data.matrix(train[i,-1]), 28, 28)
    image(m[,28:1])
  }
}

# ラベルを表示
view_label <- function(train, range = 1:20) {
  matrix(train[range,"label"], 4, 5, byrow = TRUE)
}

range <- 1:20
view_train(train, range)
view_label(train, range)
