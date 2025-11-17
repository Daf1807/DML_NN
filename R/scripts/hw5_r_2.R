# Cargar las librerías necesarias
library(readr)

file_path <- "C:/Users/FERNANDO/Documents/GitHub/DML_NN/Data/penn_jae.dat"

# Abrir las primeras líneas para ver cómo está separado
con <- file(file_path, "r")

# Leer las primeras 5 líneas
for (i in 1:5) {
  line <- readLines(con, n = 1)
  print(line)
}

# Cerrar la conexión
close(con)



# =========================================================
# Cleaning and Set-up
# =========================================================

# --- Crear carpeta de salida si no existe ---
OUTPUT_DIR <- "../output"
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

# --- 1. Cargar datos desde tu ruta local ---
library(readr)
library(dplyr)
library(fastDummies)



file_path <- "C:/Users/FERNANDO/Documents/GitHub/DML_NN/Data/penn_jae.dat"

# Leer el archivo .dat separado por espacios
rm(list=ls())
# 1. Definir URL
url <- "https://raw.githubusercontent.com/VC2015/DMLonGitHub/master/penn_jae.dat"

# 2. Definir nombre del archivo local
destfile <- "penn_jae.dat"

# 3. Descargar archivo
download.file(url, destfile, mode = "wb")

# 4. Leer el archivo .dat (separado por espacios)
df <- read.table(destfile, header = TRUE)



cat("✅ Datos cargados correctamente\n")
cat("Shape original:", dim(df), "\n")
print(head(df))

# --- 2. Filtrar sólo observaciones donde 'tg' == 0 o 4 ---
df <- df %>% filter(tg %in% c(0, 4))

# --- 3. Definir variable de tratamiento ---
df <- df %>% mutate(T4 = ifelse(tg == 4, 1, 0))

# --- 4. Definir variable de resultado ---
df <- df %>% mutate(y = log(inuidur1))

# --- 5. Crear variables dummy para 'dep' ---
df <- fastDummies::dummy_cols(df, select_columns = "dep", remove_first_dummy = FALSE)

# Mostrar qué dummies se generaron
dep_dummies <- names(df)[grepl("^dep_", names(df))]
cat("Columnas dummies de 'dep':", dep_dummies, "\n")

# --- 6. Definir covariables (X) ---
x_vars <- c(
  'female','black','othrace',
  'dep_1','dep_2',
  'q2','q3','q4','q5','q6',
  'recall','agelt35','agegt54',
  'durable','nondurable','lusd','husd'
)

# Comprobar si falta alguna variable
missing <- x_vars[!x_vars %in% names(df)]
if (length(missing) > 0) {
  cat("⚠️ Variables faltantes:", missing, "\n")
}

# --- 7. Definir conjuntos finales ---
y <- df$y
d <- df$T4
X <- df[, x_vars]

cat("\n✅ Preparación completa\n")
cat("y:", length(y), "\n")
cat("d:", length(d), "\n")
cat("X:", dim(X), "\n")




# =========================================================
# Función DML (Debiased Machine Learning) con Cross-Fitting
# =========================================================

library(glmnet)
library(randomForest)
library(nnet)

dml <- function(y, d, X, ml_method = "lasso", n_splits = 2, seed = 42) {
  
  set.seed(seed)
  n <- length(y)
  
  # Convertir X a matriz si es necesario
  X_mat <- as.matrix(X)
  
  # ============================
  # Seleccionar modelo ML base
  # ============================
  
  fit_model <- function(method) {
    if (method == "ols") {
      return(list(
        fit = function(X, y) lm(y ~ X),
        pred = function(model, X) predict(model, newdata = data.frame(X = X))
      ))
      
    } else if (method == "lasso") {
      return(list(
        fit = function(X, y) cv.glmnet(X, y, alpha = 1),
        pred = function(model, X) predict(model, X, s = "lambda.min")
      ))
      
    } else if (method == "rf") {
      return(list(
        fit = function(X, y) randomForest(x = X, y = y, ntree = 200),
        pred = function(model, X) predict(model, X)
      ))
      
    } else if (method == "nn") {
      return(list(
        fit = function(X, y) nnet(X, y, size = 3, linout = TRUE, maxit = 1000, trace = FALSE),
        pred = function(model, X) predict(model, X)
      ))
      
    } else {
      stop("Método no reconocido: usa 'ols', 'lasso', 'rf' o 'nn'.")
    }
  }
  
  model_y <- fit_model(ml_method)
  model_d <- fit_model(ml_method)
  
  # ============================
  # Cross-fitting
  # ============================
  
  folds <- sample(rep(1:n_splits, length.out = n))
  
  y_tilde <- rep(NA, n)
  d_tilde <- rep(NA, n)
  
  for (k in 1:n_splits) {
    
    test_idx <- which(folds == k)
    train_idx <- setdiff(1:n, test_idx)
    
    # Entrenamiento
    fit_y <- model_y$fit(X_mat[train_idx, , drop = FALSE], y[train_idx])
    fit_d <- model_d$fit(X_mat[train_idx, , drop = FALSE], d[train_idx])
    
    # Predicciones fuera del fold
    y_hat <- model_y$pred(fit_y, X_mat[test_idx, , drop = FALSE])
    d_hat <- model_d$pred(fit_d, X_mat[test_idx, , drop = FALSE])
    
    # Residuos
    y_tilde[test_idx] <- y[test_idx] - y_hat
    d_tilde[test_idx] <- d[test_idx] - d_hat
  }
  
  # ============================
  # Estimar theta (coeficiente DML)
  # ============================
  theta_hat <- sum(d_tilde * y_tilde) / sum(d_tilde^2)
  
  # Error estándar aproximado
  resid <- y_tilde - theta_hat * d_tilde
  sigma2 <- mean(resid^2)
  se <- sqrt(sigma2 / sum(d_tilde^2))
  
  return(list(theta = theta_hat, se = se, method = ml_method))
}


# =========================================================
# Estimar y comparar modelos
# =========================================================

modelos <- c("ols", "lasso", "rf", "nn")
results <- list()

for (m in modelos) {
  cat("Estimando con", toupper(m), "...\n")
  r <- dml(y, d, X, ml_method = m, n_splits = 2)
  results[[m]] <- r
}

# Convertir resultados a data frame
results_df <- do.call(rbind, lapply(results, function(x) {
  data.frame(
    method = x$method,
    theta = x$theta,
    se = x$se,
    t_stat = x$theta / x$se
  )
}))

# Mostrar tabla final
results_df

