EP <- read_excel("EnglishProsody.xlsx")
alpha <- c(1, 1)
beta <- c(0.5, 0.5)
data <- list(K=2, S_total=2, N=88, s=EP$Sur, delta=EP$Delta, dur=EP$Dur, alpha=alpha,beta=beta)
data$s <- as.numeric(as.factor(data$s)) 
data$dur <- data$dur * 1000

fit<-stan("lda.stan",data=data, warmup=800,thin=2,iter=2000,chains=4,control = list(adapt_delta = 0.99,max_treedepth=15))
sample <- extract(fit)
predicted <- colMeans(sample$p)
write.csv(predicted,"predicted.csv")