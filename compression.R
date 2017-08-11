# MNISTのトレーニングデータを画像表示する
setwd("/Users/kada/Documents/szk/mnist")

view_train <- function(train, range) {
  par(mfrow=c(length(range)/4, 5))
  par(mar=c(0,0,0,0))
  x <- seq(0, 0, length = 14*14)
  for (a in range) {
    digit <- data.matrix(train[a,-1])
    for (i in 0:(14-1)) {
      for (j in 0:(14-1)) {
        # あえて割らずに整数値にしておく mean(...)/255
        x[14*i+j] <- digit[28*2*i+2*j] + digit[28*2*i+2*j+1] + digit[28*(2*i+1)+2*j] + digit[28*(2*i+1)+2*j+1]
      }
    }
    data.A <- matrix(x, nrow=1, ncol=196)
    write.table(data.A, file="comp_test.csv", sep=",", col.names=F, row.names=F, quote=F, append = TRUE)
    #m <- matrix(x, 14, 14)
    #image(m[,14:1])
  }
}

range <- 1:42000
view_train(train, range)
