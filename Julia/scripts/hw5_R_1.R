# Question 1 - NN Basics

# I. Fitting Data

# ===========================================================
# INSTALAR E IMPORTAR TORCH
# ===========================================================
# install.packages("torch")
library(torch)

# Instalar backend si es primera vez
torch::install_torch()

torch_manual_seed(123)

# ===========================================================
# GENERAR LOS DATOS
# ===========================================================
n <- 1000
X <- runif(n, 0, 2*pi)
eps <- rnorm(n, 0, 0.15)
y <- sin(X) + eps

# Tensores (n,1)
X_t <- torch_tensor(X)$view(c(n,1))
y_t <- torch_tensor(y)$view(c(n,1))

# ===========================================================
# DEFINIR FUNCIONES PARA CREAR MODELOS
# ===========================================================

# Modelo genérico de 3 capas escondidas
crear_modelo <- function(activacion) {
  
  fn_act <- switch(
    activacion,
    "logistic" = torch_sigmoid,
    "tanh"     = torch_tanh,
    "relu"     = torch_relu,
    stop("Activación no soportada.")
  )
  
  nn_module(
    classname = paste0("MLP_", activacion),
    
    initialize = function() {
      self$l1 <- nn_linear(1, 50)
      self$l2 <- nn_linear(50, 50)
      self$l3 <- nn_linear(50, 50)
      self$out <- nn_linear(50, 1)
      self$act <- fn_act
    },
    
    forward = function(x) {
      x %>%
        self$l1() %>% self$act() %>%
        self$l2() %>% self$act() %>%
        self$l3() %>% self$act() %>%
        self$out()
    }
  )
}

# Mixed activations model
MixedNN <- nn_module(
  classname = "MixedNN",
  
  initialize = function() {
    self$l1 <- nn_linear(1, 50)
    self$l2 <- nn_linear(50, 50)
    self$l3 <- nn_linear(50, 50)
    self$out <- nn_linear(50, 1)
  },
  
  forward = function(x) {
    x %>%
      self$l1() %>% torch_tanh() %>%
      self$l2() %>% torch_relu() %>%
      self$l3() %>% torch_sigmoid() %>%
      self$out()
  }
)

# ===========================================================
# FUNCIÓN PARA ENTRENAR UN MODELO
# ===========================================================
entrenar <- function(modelo, X_t, y_t, lr=0.01, epocas=5000) {
  
  optim <- optim_adam(modelo$parameters, lr=lr)
  loss_fn <- nn_mse_loss()
  
  for (epoch in 1:epocas) {
    optim$zero_grad()
    y_pred <- modelo(X_t)
    loss <- loss_fn(y_pred, y_t)
    loss$backward()
    optim$step()
    
    if (epoch %% 1000 == 0)
      cat("Epoch", epoch, "| loss:", loss$item(), "\n")
  }
  
  return(modelo)
}

# ===========================================================
# ENTRENAR LOS 4 MODELOS
# ===========================================================
modelos <- list()

cat("Entrenando logistic…\n")
modelos$logistic <- entrenar(crear_modelo("logistic")(), X_t, y_t)

cat("Entrenando tanh…\n")
modelos$tanh <- entrenar(crear_modelo("tanh")(), X_t, y_t)

cat("Entrenando relu…\n")
modelos$relu <- entrenar(crear_modelo("relu")(), X_t, y_t)

cat("Entrenando mixto…\n")
modelos$mixto <- entrenar(MixedNN(), X_t, y_t)

# ===========================================================
# PREDECIR EN UNA GRILLA
# ===========================================================
x_grid <- seq(0, 2*pi, length.out = 500)
y_true <- sin(x_grid)

x_grid_t <- torch_tensor(x_grid)$view(c(500,1))

predicciones <- list()

for (nombre in names(modelos)) {
  modelos[[nombre]]$eval()
  with_no_grad({
    pred <- modelos[[nombre]](x_grid_t)
  })
  predicciones[[nombre]] <- as.numeric(pred)
}

# ===========================================================
# CALCULAR MSE
# ===========================================================
mse <- sapply(predicciones, function(p) mean((p - y_true)^2))

print("MSE de cada modelo:")
print(mse)

cat("\nMejor modelo:", names(which.min(mse)), 
    "| MSE =", min(mse), "\n")

# ===========================================================
# GRAFICAR
# ===========================================================
par(mfrow=c(2,2))

for (nombre in names(predicciones)) {
  plot(X, y, col=rgb(0,0,0,0.4), pch=16,
       main=paste("NN activación:", nombre),
       xlab="x", ylab="y")
  
  lines(x_grid, y_true, col="green", lwd=2)
  lines(x_grid, predicciones[[nombre]], col="red", lwd=2)
}

# II. Learning-rate

# Briefly explain what the learning-rate is in the context of Neural Networks.

# The learning rate in a neural network is basically the speed at which the model learns.
# During training, the network adjusts its internal parameters (weights) to reduce prediction errors — kind of like how an economist updates expectations when new data comes in. The learning rate determines how big those adjustments are each time the model learns from the data.
# If the learning rate is too high, the model makes big jumps and might miss the best solution.
# If it’s too low, it moves very slowly and can get stuck before reaching the optimal point.

#############################################################
# II. Learning-rate
#############################################################

#############################################################
# Briefly explain what the learning-rate is in the context of NNs
#############################################################

# The learning rate in a neural network determines how fast the model 
# updates its weights during training. A high learning rate makes large 
# steps and may overshoot the optimal solution, while a very small learning 
# rate makes learning slow and may get stuck in suboptimal points.

#############################################################
# Use the same data simulation and the activation that performed best
# (logistic in your case). Simulación: y = sin(x) + ruido
#############################################################

library(torch)
library(ggplot2)

set.seed(123)

n <- 1000
X <- runif(n, 0, 2*pi)
eps <- rnorm(n, 0, 0.15)
y <- sin(X) + eps

X_t <- torch_tensor(X)$view(c(n,1))
y_t <- torch_tensor(y)$view(c(n,1))

x_grid <- seq(0, 2*pi, length.out = 500)
x_grid_t <- torch_tensor(x_grid)$view(c(500,1))
y_true <- sin(x_grid)

#############################################################
# Función general para MLP con activación logistic (sigmoid)
#############################################################

build_mlp <- function(layer_sizes) {
  nn_module(
    classname = "MLP",
    
    initialize = function() {
      self$layers <- nn_module_list()
      input_size <- 1
      
      for (h in layer_sizes) {
        self$layers$append(nn_linear(input_size, h))
        input_size <- h
      }
      self$out <- nn_linear(input_size, 1)
    },
    
    forward = function(x) {
      for (i in seq_len(length(self$layers))) {
        layer <- self$layers[[i]]
        x <- torch_sigmoid(layer(x))
      }
      x <- self$out(x)
      x
    }
  )
}

#############################################################
# Función de entrenamiento con Adam
#############################################################

train_mlp <- function(model, lr, epochs=5000) {
  opt <- optim_adam(model$parameters, lr = lr)
  loss_fn <- nn_mse_loss()
  
  for (epoch in 1:epochs) {
    opt$zero_grad()
    y_pred <- model(X_t)
    loss <- loss_fn(y_pred, y_t)
    loss$backward()
    opt$step()
  }
  return(model)
}

#############################################################
# Learning rates a evaluar
#############################################################

learning_rates <- c(0.0001, 0.001, 0.01, 0.1)

#############################################################
# 1) PRIMERA PARTE DE LA INSTRUCCIÓN:
#    Entrenar SOLO la red de 1 hidden layer (50)
#############################################################

cat("\n==================== 1 hidden layer ====================\n")

pred_1 <- list()
mse_1 <- c()

for (lr in learning_rates) {
  cat("Training lr =", lr, "\n")
  model <- build_mlp(c(50))()
  model <- train_mlp(model, lr)
  
  model$eval()
  with_no_grad({ pred <- model(x_grid_t) })
  
  pred_vec <- as.numeric(pred)
  pred_1[[as.character(lr)]] <- pred_vec
  mse_1[as.character(lr)] <- mean((pred_vec - y_true)^2)
}

# Graficar
df_1 <- data.frame(
  x = rep(x_grid, length(learning_rates)),
  y = unlist(pred_1),
  lr = rep(names(pred_1), each = length(x_grid))
)

p1 <- ggplot() +
  geom_point(aes(X, y), data = data.frame(X = X, y = y), alpha = 0.25) +
  geom_line(aes(x_grid, y_true), color = "green", linewidth = 1.3) +
  geom_line(data = df_1, aes(x, y, color = lr), linewidth = 1.1) +
  labs(
    title = paste0("1 Hidden Layer (50)\nBest lr = ", names(which.min(mse_1)),
                   " — MSE = ", round(min(mse_1),6)),
    x = "x", y = "y"
  ) +
  theme_minimal()

print(p1)
cat("\nBest LR (1 layer):", names(which.min(mse_1)), "\n")

#############################################################
# 2) SEGUNDA PARTE:
#    Repetir para 2 hidden layers (50, 50)
#############################################################

cat("\n==================== 2 hidden layers ====================\n")

pred_2 <- list()
mse_2 <- c()

for (lr in learning_rates) {
  cat("Training lr =", lr, "\n")
  model <- build_mlp(c(50,50))()
  model <- train_mlp(model, lr)
  
  model$eval()
  with_no_grad({ pred <- model(x_grid_t) })
  
  pred_vec <- as.numeric(pred)
  pred_2[[as.character(lr)]] <- pred_vec
  mse_2[as.character(lr)] <- mean((pred_vec - y_true)^2)
}

df_2 <- data.frame(
  x = rep(x_grid, length(learning_rates)),
  y = unlist(pred_2),
  lr = rep(names(pred_2), each = length(x_grid))
)

p2 <- ggplot() +
  geom_point(aes(X, y), data = data.frame(X = X, y = y), alpha = 0.25) +
  geom_line(aes(x_grid, y_true), color = "green", linewidth = 1.3) +
  geom_line(data = df_2, aes(x, y, color = lr), linewidth = 1.1) +
  labs(
    title = paste0("2 Hidden Layers (50,50)\nBest lr = ", names(which.min(mse_2)),
                   " — MSE = ", round(min(mse_2),6)),
    x = "x", y = "y"
  ) +
  theme_minimal()

print(p2)
cat("\nBest LR (2 layers):", names(which.min(mse_2)), "\n")

#############################################################
# 3) TERCERA PARTE:
#    Repetir para 3 hidden layers (50, 50, 50)
#############################################################

cat("\n==================== 3 hidden layers ====================\n")

pred_3 <- list()
mse_3 <- c()

for (lr in learning_rates) {
  cat("Training lr =", lr, "\n")
  model <- build_mlp(c(50,50,50))()
  model <- train_mlp(model, lr)
  
  model$eval()
  with_no_grad({ pred <- model(x_grid_t) })
  
  pred_vec <- as.numeric(pred)
  pred_3[[as.character(lr)]] <- pred_vec
  mse_3[as.character(lr)] <- mean((pred_vec - y_true)^2)
}

df_3 <- data.frame(
  x = rep(x_grid, length(learning_rates)),
  y = unlist(pred_3),
  lr = rep(names(pred_3), each = length(x_grid))
)

p3 <- ggplot() +
  geom_point(aes(X, y), data = data.frame(X = X, y = y), alpha = 0.25) +
  geom_line(aes(x_grid, y_true), color = "green", linewidth = 1.3) +
  geom_line(data = df_3, aes(x, y, color = lr), linewidth = 1.1) +
  labs(
    title = paste0("3 Hidden Layers (50,50,50)\nBest lr = ", names(which.min(mse_3)),
                   " — MSE = ", round(min(mse_3),6)),
    x = "x", y = "y"
  ) +
  theme_minimal()

print(p3)
cat("\nBest LR (3 layers):", names(which.min(mse_3)), "\n")

#############################################################
# Conclusion: relationship between LR and network depth
#############################################################

#Shallow networks (1–2 layers) learned best with intermediate learning rates (0.001–0.01).
#For the 3-layer network, the best learning rate was 0.1, which is unusual but occurs because sigmoid activations saturate in deeper networks, making small learning rates ineffective.
#In general, deeper networks require more careful learning-rate selection, and activation functions strongly affect this relationship.
