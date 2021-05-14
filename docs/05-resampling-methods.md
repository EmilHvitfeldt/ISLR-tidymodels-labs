# Resampling Methods


```r
library(tidymodels)
```

```
## Registered S3 method overwritten by 'tune':
##   method                   from   
##   required_pkgs.model_spec parsnip
```

```
## ── Attaching packages ────────────────────────────────────── tidymodels 0.1.3 ──
```

```
## ✓ broom        0.7.6          ✓ recipes      0.1.16    
## ✓ dials        0.0.9          ✓ rsample      0.1.0.9000
## ✓ dplyr        1.0.6          ✓ tibble       3.1.1     
## ✓ ggplot2      3.3.3          ✓ tidyr        1.1.3     
## ✓ infer        0.5.4          ✓ tune         0.1.5     
## ✓ modeldata    0.1.0          ✓ workflows    0.2.2     
## ✓ parsnip      0.1.5.9002     ✓ workflowsets 0.0.2     
## ✓ purrr        0.3.4          ✓ yardstick    0.0.8
```

```
## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
## x purrr::discard() masks scales::discard()
## x dplyr::filter()  masks stats::filter()
## x dplyr::lag()     masks stats::lag()
## x recipes::step()  masks stats::step()
## • Use tidymodels_prefer() to resolve common conflicts.
```

```r
library(ISLR)

Auto <- as_tibble(Auto)
Portfolio <- as_tibble(Portfolio)
```


## The Validation Set Approach


```r
set.seed(1)
Auto_split <- initial_split(Auto)

Auto_train <- training(Auto_split)
Auto_test <- testing(Auto_split)
```


```r
lm_spec <- linear_reg() %>%
  set_engine("lm")
```


```r
lm_fit <- lm_spec %>% 
  fit(mpg ~ horsepower, data = Auto_train)
```


```r
augment(lm_fit, new_data = Auto_test) %>%
  rmse(truth = mpg, estimate = .pred)
```

```
## # A tibble: 1 x 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 rmse    standard        4.93
```


```r
poly_fit <- lm_spec %>% 
  fit(mpg ~ poly(horsepower, 2), data = Auto_train)
```


```r
augment(poly_fit, new_data = Auto_test) %>%
  rmse(truth = mpg, estimate = .pred)
```

```
## # A tibble: 1 x 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 rmse    standard        4.25
```


```r
poly_rec <- recipe(mpg ~ horsepower, data = Auto_train) %>%
  step_poly(horsepower, degree = 2)

poly_wf <- workflow() %>%
  add_recipe(poly_rec) %>%
  add_model(lm_spec)

poly_fit <- fit(poly_wf, data = Auto_train)
```


```r
augment(poly_fit, new_data = Auto_test) %>%
  rmse(truth = mpg, estimate = .pred)
```

```
## # A tibble: 1 x 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 rmse    standard        4.25
```


```r
set.seed(2)
Auto_split <- initial_split(Auto)

Auto_train <- training(Auto_split)
Auto_test <- testing(Auto_split)
```


```r
poly_fit <- fit(poly_wf, data = Auto_train)

augment(poly_fit, new_data = Auto_test) %>%
  rmse(truth = mpg, estimate = .pred)
```

```
## # A tibble: 1 x 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 rmse    standard        4.35
```

## Leave-One-Out Cross-Validation

Leave-One-Out Cross-Validation is not integrated into the broader tidymodels framework. For more information read [here](https://www.tmwr.org/resampling.html#leave-one-out-cross-validation).

## k-Fold Cross-Validation


```r
poly_rec <- recipe(mpg ~ horsepower, data = Auto_train) %>%
  step_poly(horsepower, degree = tune())
```


```r
lm_spec <- linear_reg() %>%
  set_engine("lm")

poly_wf <- workflow() %>%
  add_recipe(poly_rec) %>%
  add_model(lm_spec)

Auto_folds <- vfold_cv(Auto_train, v = 10)

degree_grid <- grid_regular(degree(range = c(1, 10)), levels = 10)

tune_res <- tune_grid(
  object = poly_wf, 
  resamples = Auto_folds, 
  grid = degree_grid
)
```

It can be helpful to add `control = control_grid(verbose = TRUE)` 


```r
autoplot(tune_res)
```

<img src="05-resampling-methods_files/figure-html/unnamed-chunk-14-1.png" width="672" />


```r
collect_metrics(tune_res)
```

```
## # A tibble: 20 x 7
##    degree .metric .estimator  mean     n std_err .config              
##     <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>                
##  1      1 rmse    standard   4.81     10  0.172  Preprocessor01_Model1
##  2      1 rsq     standard   0.621    10  0.0316 Preprocessor01_Model1
##  3      2 rmse    standard   4.37     10  0.209  Preprocessor02_Model1
##  4      2 rsq     standard   0.677    10  0.0436 Preprocessor02_Model1
##  5      3 rmse    standard   4.40     10  0.217  Preprocessor03_Model1
##  6      3 rsq     standard   0.675    10  0.0446 Preprocessor03_Model1
##  7      4 rmse    standard   4.43     10  0.218  Preprocessor04_Model1
##  8      4 rsq     standard   0.670    10  0.0453 Preprocessor04_Model1
##  9      5 rmse    standard   4.42     10  0.203  Preprocessor05_Model1
## 10      5 rsq     standard   0.674    10  0.0436 Preprocessor05_Model1
## 11      6 rmse    standard   4.41     10  0.189  Preprocessor06_Model1
## 12      6 rsq     standard   0.670    10  0.0423 Preprocessor06_Model1
## 13      7 rmse    standard   4.40     10  0.176  Preprocessor07_Model1
## 14      7 rsq     standard   0.670    10  0.0420 Preprocessor07_Model1
## 15      8 rmse    standard   4.41     10  0.175  Preprocessor08_Model1
## 16      8 rsq     standard   0.670    10  0.0420 Preprocessor08_Model1
## 17      9 rmse    standard   4.47     10  0.207  Preprocessor09_Model1
## 18      9 rsq     standard   0.663    10  0.0445 Preprocessor09_Model1
## 19     10 rmse    standard   4.50     10  0.227  Preprocessor10_Model1
## 20     10 rsq     standard   0.658    10  0.0465 Preprocessor10_Model1
```

```r
show_best(tune_res, metric = "rmse")
```

```
## # A tibble: 5 x 7
##   degree .metric .estimator  mean     n std_err .config              
##    <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>                
## 1      2 rmse    standard    4.37    10   0.209 Preprocessor02_Model1
## 2      3 rmse    standard    4.40    10   0.217 Preprocessor03_Model1
## 3      7 rmse    standard    4.40    10   0.176 Preprocessor07_Model1
## 4      6 rmse    standard    4.41    10   0.189 Preprocessor06_Model1
## 5      8 rmse    standard    4.41    10   0.175 Preprocessor08_Model1
```


```r
best_degree <- select_best(tune_res, metric = "rmse")
```


```r
final_wf <- finalize_workflow(poly_wf, best_degree)

final_wf
```

```
## ══ Workflow ════════════════════════════════════════════════════════════════════
## Preprocessor: Recipe
## Model: linear_reg()
## 
## ── Preprocessor ────────────────────────────────────────────────────────────────
## 1 Recipe Step
## 
## • step_poly()
## 
## ── Model ───────────────────────────────────────────────────────────────────────
## Linear Regression Model Specification (regression)
## 
## Computational engine: lm
```


```r
final_fit <- fit(final_wf, Auto_train)

final_fit
```

```
## ══ Workflow [trained] ══════════════════════════════════════════════════════════
## Preprocessor: Recipe
## Model: linear_reg()
## 
## ── Preprocessor ────────────────────────────────────────────────────────────────
## 1 Recipe Step
## 
## • step_poly()
## 
## ── Model ───────────────────────────────────────────────────────────────────────
## 
## Call:
## stats::lm(formula = ..y ~ ., data = data)
## 
## Coefficients:
##       (Intercept)  horsepower_poly_1  horsepower_poly_2  
##             23.34            -104.85              34.39
```

## The Bootstrap


```r
Portfolio_boots <- bootstraps(Portfolio, times = 1000)

alpha.fn <- function(split) {
  data <- analysis(split)
  X <- data$X
  Y <- data$Y
  
  (var(Y) - cov(X, Y)) / (var(X) + var(Y) - 2 * cov(X, Y))
}

alpha_res <- Portfolio_boots %>%
  mutate(alpha = map_dbl(splits, alpha.fn))

alpha_res
```

```
## # Bootstrap sampling 
## # A tibble: 1,000 x 3
##    splits           id            alpha
##    <list>           <chr>         <dbl>
##  1 <split [100/38]> Bootstrap0001 0.674
##  2 <split [100/41]> Bootstrap0002 0.586
##  3 <split [100/31]> Bootstrap0003 0.701
##  4 <split [100/33]> Bootstrap0004 0.572
##  5 <split [100/35]> Bootstrap0005 0.684
##  6 <split [100/36]> Bootstrap0006 0.530
##  7 <split [100/35]> Bootstrap0007 0.609
##  8 <split [100/37]> Bootstrap0008 0.530
##  9 <split [100/36]> Bootstrap0009 0.619
## 10 <split [100/39]> Bootstrap0010 0.475
## # … with 990 more rows
```


```r
Auto_boots <- bootstraps(Auto)

boot.fn <- function(split) {
  lm_fit <- lm_spec %>% fit(mpg ~ horsepower, data = analysis(split))
  tidy(lm_fit)
}

boot_res <- Auto_boots %>%
  mutate(models = map(splits, boot.fn))

boot_res %>%
  unnest(cols = c(models)) %>%
  group_by(term) %>%
  summarise(mean = mean(estimate),
            sd = sd(estimate))
```

```
## # A tibble: 2 x 3
##   term          mean      sd
##   <chr>        <dbl>   <dbl>
## 1 (Intercept) 39.7   0.691  
## 2 horsepower  -0.156 0.00614
```
