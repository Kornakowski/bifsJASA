
# Contains frst, second, and third order simulations

library(INLA)
# inla.setOption(pardiso.license = "~/sys/licenses/pardiso.lic")
# inla.pardiso.check()

## set up Q matrix per Eric

n <- 128 #128 # 256 # x and y dimensions of larger image on torus to subsample from
# nsub <- 4 # 128 dimension of subsample to take so not wrapped on torus
nsq <- n^2
itrs <- 1000 # number of images to simulate

# For 1st order intrinsic GMRF
tau <- 1 # the scale parameter
s1st <- 4 * tau
c1 <- -1 / s1st

# For 3rd order intrinsic GMRF
eta <- 1 # the scale parameter
s3rd <- 20 * eta # precision of x_ij
d1 <- -8 / s3rd # for 1st order neighbors
d2 <- 2 / s3rd # for 2nd order
d3 <- 1 / s3rd # 3rd order

# only compute this once
dv1st <- list(rep(1, nsq), 
              c1 * rep(1, nsq - 1), 
              c1 * rep(1, nsq - n)
              )

dv3rd <- list(rep(1, nsq), 
              d1 * rep(1, nsq - 1), 
              d1 * rep(1, nsq - n)
              )
               

Q1st <- bandSparse(n = nsq, k = c(0, 1, n), diagonals = dv1st, symmetric = TRUE)
Q3rd <- bandSparse(n = nsq, k = c(0, 1, n), diagonals = dv3rd, symmetric = TRUE)

# fix boundary conditions

# Note that x[i, j] = x[i + n(j - 1)] -- second index represents the locations on Q matrix

for (i in 1:n) { # neighbors j = 1 and j = n
  Q1st[i, i + n*(n - 1)] <- c1  # top
  Q1st[i + n*(n - 1), i] <- c1  # bottom
}
for (j in 1:n) { # neighbors i = 1 and i = n
  Q1st[1 + n*(j - 1), n + n*(j - 1)] <- c1  # left
  Q1st[n + n*(j - 1), 1 + n*(j - 1)] <- c1  # right
}

for (i in seq(n + 1, nsq, n)) Q1st[i, i-1] <- 0  # drop neighbors wrapping on left side
for (i in seq(n, nsq - n, n)) Q1st[i, i+1] <- 0  # drop neighbors wrapping on right side


for (i in 1:n) { # neighbors j = 1 and j = n
  Q3rd[i, i + n*(n - 1)] <- d1  # top
  Q3rd[i + n*(n - 1), i] <- d1  # bottom
}
for (j in 1:n) { # neighbors i = 1 and i = n
  Q3rd[1 + n*(j - 1), n + n*(j - 1)] <- d1  # left
  Q3rd[n + n*(j - 1), 1 + n*(j - 1)] <- d1  # right
}

for (i in seq(n + 1, nsq, n)) Q3rd[i, i-1] <- 0  # drop neighbors wrapping on left side
for (i in seq(n, nsq - n, n)) Q3rd[i, i+1] <- 0  # drop neighbors wrapping on right side

# Now add in 2nd order neighbors

for (i in 0:(n-1)) {
  for (j in 0:(n-1)) {
    Q3rd[i + n * j + 1, (i - 1) %% n + n * ((j - 1) %% n) + 1] <- d2  ## bottom left diag. neighbor wrapped
    Q3rd[i + n * j + 1, (i - 1) %% n + n * ((j + 1) %% n) + 1] <- d2  ## top left diag. neighbor wrapped
    Q3rd[i + n * j + 1, (i + 1) %% n + n * ((j - 1) %% n) + 1] <- d2  ## bottom right diag. neighbor wrapped
    Q3rd[i + n * j + 1, (i + 1) %% n + n * (j + 1) %% n + 1] <- d2  ## top right diag. neighbor wrapped
  }
}


# Now add in 3rd order neighbors

for (i in 0:(n-1)) {
  for (j in 0:(n-1)) {
    Q3rd[i + n * j + 1, (i - 2) %% n + n * j + 1] <- d3  ## 2 to left neighbor wrapped
    Q3rd[i + n * j + 1, (i + 2) %% n + n * j + 1] <- d3  ## 2 to right neighbor wrapped
    Q3rd[i + n * j + 1, i + n * ((j - 2) %% n) + 1] <- d3  ## bottom right diag. neighbor wrapped
    Q3rd[i + n * j + 1, i + n * ((j + 2) %% n) + 1] <- d3  ## top right diag. neighbor wrapped
  }
}

# for(i in 1:nsq) print(sum(Q3rd[i, ]))
# for(i in 1:nsq) print(sum(Q3rd[, i]))

constraint <- list(A=matrix(rep(1,nsq), 1, nsq), e = 0)

dextra <- list(rep(1e-7, nsq))

Q4th <- Q3rd + bandSparse(n = nsq, k = c(0), diagonals = dextra, symmetric = TRUE)

x1 <- inla.qsample(itrs, Q = Q1st, constr = constraint, seed = 0)
x2 <- inla.qsample(itrs, Q = Q4th, constr = constraint, seed = 0)

x1array <- array(x1, dim = c(n, n, itrs))
x2array <- array(x2, dim = c(n, n, itrs))

par(mfrow=c(2,2))
image(x1array[ , , 1])
image(x1array[c((n/2 + 1):n, 1:(n/2)), c((n/2 + 1):n, 1:(n/2)), 1])
image(x2array[ , , 1])
image(x2array[c((n/2 + 1):n, 1:(n/2)), c((n/2 + 1):n, 1:(n/2)), 1])


# x1sub <- x1array[1:nsub, 1:nsub, 1:itrs]
# x2sub <- x2array[1:nsub, 1:nsub, 1:itrs]

# x1subv <- as.vector(x1sub)
# x2subv <- as.vector(x2sub)

x1v <- as.vector(x1array)
x2v <- as.vector(x2array)

#Q1file <- file("q1gmrf.dat", "wb")
#writeBin(x1subv, Q1file)
#close(Q1file)

#Q4file <- file("q4gmrf.dat", "wb")
#writeBin(x2subv, Q4file)
#close(Q4file)

iQ1file <- file("SimData/iq1gmrf.dat", "wb") # prefix with i for intrinsic
writeBin(x1v, iQ1file)
close(iQ1file)

iQ4file <- file("SimData/iq4gmrf.dat", "wb")
writeBin(x2v, iQ4file)
close(iQ4file)


iq1sd <- iq4sd <- matrix(0, n, n)
for(i in 1:n){
  for(j in 1:n){
    iq1sd[i, j] <- sd(x1array[i, j, ])
    iq4sd[i, j] <- sd(x2array[i, j, ])
  }
}

iq1var <- iq4var <- matrix(0, n, n)
for(i in 1:n){
  for(j in 1:n){
    iq1var[i, j] <- var(x1array[i, j, ])
    iq4var[i, j] <- var(x2array[i, j, ])
  }
}

mean(as.vector(iq1sd))
mean(as.vector(iq4sd))

exp(0.5 * mean(as.vector(log(iq1var))))
exp(0.5 * mean(as.vector(log(iq4var))))

ratioAdjust <- exp(0.5 * mean(as.vector(log(iq1var)))) / exp(0.5 * mean(as.vector(log(iq4var))))

x2arrayAdjust <- ratioAdjust * x2array


### Create adjusted version of 3rd order prior to match marginal SD of 1st order prior
iq4varAdjust <- matrix(0, n, n)
for(i in 1:n) for(j in 1:n) iq4varAdjust[i, j] <- var(x2arrayAdjust[i, j, ])

exp(0.5 * mean(as.vector(log(iq1var))))
exp(0.5 * mean(as.vector(log(iq4varAdjust))))

x2vAdjust <- as.vector(x2arrayAdjust)

iQ4fileAdjust <- file("SimData/iq4gmrfAdjust.dat", "wb")
writeBin(x2vAdjust, iQ4fileAdjust)
close(iQ4fileAdjust)
