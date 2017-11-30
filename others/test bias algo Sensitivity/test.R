# install.packages("gtools")
# install.packages("mvtnorm")
# install.packages("condMVNorm")
# install.packages("sensitivity")

library(gtools)
library(mvtnorm) # Multivariate Gaussian variables
library(condMVNorm) # Conditional multivariate Gaussian variables
library(sensitivity)

# Definition of the model 

modlin <- function(X) apply(X,1,sum)
d <- 3
mu <- rep(0,d)
sig <- c(1,1,2)
ro <- -0.6
Cormat <- matrix(c(1,0,0,0,1,ro,0,ro,1),d,d)
Covmat <- (sig %*% t(sig)) * Cormat

Xall <- function(n) mvtnorm::rmvnorm(n,mu,Covmat)

Xset <- function(n, Sj, Sjc, xjc){
  if (is.null(Sjc)){
    if (length(Sj) == 1){ rnorm(n,mu[Sj],sqrt(Covmat[Sj,Sj]))
    } else{ mvtnorm::rmvnorm(n,mu[Sj],Covmat[Sj,Sj])}
  } else{ condMVNorm::rcmvnorm(n, mu, Covmat, dependent.ind=Sj, given.ind=Sjc, X.given=xjc)}}

# Analytical values for the indices 
# true_First = c(0.104167,0.816667,0.876042)
# true_Total = c(0.104167,0.019792,0.079167)
# true_Shapley = c(0.104167,0.418229,0.477604)

true_First = c(0.27777778,0.01111111,0.54444444)
true_Total = c(0.27777778,0.17777778,0.71111111)
true_Shapley = c(0.27777778,0.09444444,0.62777778)

#################
# Coverage rate #
#################

# n_run = 50
# n_perms = c(100,1000,5000,10^4)
# 
# cov_First <- array(0,dim = c(d,length(n_perms),n_run))
# cov_Total <- array(0,dim = c(d,length(n_perms),n_run))
# cov_Shapley <- array(0,dim = c(d,length(n_perms),n_run))
# 
# cat("Coverage rate \n")
# 
# for(i in 1:length(n_perms)){
#   cat("n_perms", i, "\n")
#   
#   n_perms_temp = n_perms[i]
#   
#   for(j in 1:n_run){
#     x <- shapleyPermRand(model = modlin, Xall=Xall, Xset=Xset, d=d, Nv=1e4, m = n_perms_temp, No = 1, Ni = 3)
#     
#     cov_First[,i,j] = (x$SobolS[,3] < true_First) & (true_First < x$SobolS[,4])
#     cov_Total[,i,j] = (x$SobolT[,3] < true_Total) & (true_Total < x$SobolT[,4])
#     cov_Shapley[,i,j] = (x$Shapley[,3] < true_Shapley) & (true_Shapley < x$Shapley[,4])
#     
#   }
# }
# 
# cov_First_mean <- apply(cov_First,c(1,2),mean)
# cov_Total_mean <- apply(cov_Total,c(1,2),mean)
# cov_Shapley_mean <- apply(cov_Shapley,c(1,2),mean)
# 
# pdf("random_coverage_rate.pdf", width = 8, height = 6)
# 
# par(mfrow=c(2,3))
# ylim = c(0,1)
# matplot(n_perms,t(cov_First_mean),t='l',lty = rep("solid",3), lwd = 2, col=c("red","blue","green"),las=1,main = "Full First",ylab = "",xlab = "",ylim = ylim)
# matplot(n_perms,t(cov_Total_mean),t='l',lty = rep("solid",3),lwd = 2, col=c("red","blue","green"),las=1,main = "Ind Total",ylab = "",ylim = ylim)
# matplot(n_perms,t(cov_Shapley_mean),t='l',lty = rep("solid",3),lwd = 2, col=c("red","blue","green"),las=1,main = "Shapley",ylab = "",xlab = "", ylim = ylim)
# 
# legend("topright", legend = c("X1", "X2","X3"),bty = "n", pt.cex = 0, cex = 1.5, text.col = c("red", "blue","green"), horiz = F, inset = c(0.01, 0.01))
# 
# dev.off()

################
# Exact method #
################

cat("Exact method \n")

n_run = 5
values <- as.data.frame(matrix(0, ncol = n_run, nrow = d*3))
row.names(values) = paste(c(rep("First_",3),rep("Total_",3),rep("Shapley_",3)),rep(1:d,3),sep="")
values_col_names <- c()
  
for(i in 1:n_run){
  cat("loop", i, "\n")
 
  values_col_names <- c(values_col_names,paste("run_",i,sep=""))
  x <- shapleyPermEx(model = modlin, Xall=Xall, Xset=Xset, d=d, Nv=1e4, No = 50000, Ni = 5)
  values[1:3,i] = x$SobolS
  values[4:6,i] = x$SobolT
  values[7:9,i] = x$Shapley
  
}

colnames(values) <- values_col_names
write.csv2(values,"exact_method.csv")

#################
# Random method #
#################

cat("Random method \n")

cat("x \n")
x <- shapleyPermRand(model = modlin, Xall=Xall, Xset=Xset, d=d, Nv=1e4, m = 50000, No = 5, Ni = 5)

cat("y \n")
y <- shapleyPermRand(model = modlin, Xall=Xall, Xset=Xset, d=d, Nv=1e4, m = 50000, No = 5, Ni = 5)

cat("z \n")
z <- shapleyPermRand(model = modlin, Xall=Xall, Xset=Xset, d=d, Nv=1e4, m = 50000, No = 5, Ni = 5)

save(x,y,z,file = "random_method.RData")