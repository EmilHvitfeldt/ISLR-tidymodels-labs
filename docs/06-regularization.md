# Linear Model Selection and Regularization


```r
library(tidymodels)
```

```
## Registered S3 method overwritten by 'tune':
##   method                   from   
##   required_pkgs.model_spec parsnip
```

```
## ── Attaching packages ────────────────────────────────────── tidymodels 0.1.2 ──
```

```
## ✓ broom     0.7.5           ✓ recipes   0.1.15.9000
## ✓ dials     0.0.9           ✓ rsample   0.0.9      
## ✓ dplyr     1.0.5           ✓ tibble    3.1.0      
## ✓ ggplot2   3.3.3           ✓ tidyr     1.1.3      
## ✓ infer     0.5.4           ✓ tune      0.1.3      
## ✓ modeldata 0.1.0           ✓ workflows 0.2.2      
## ✓ parsnip   0.1.5.9002      ✓ yardstick 0.0.8      
## ✓ purrr     0.3.4
```

```
## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
## x purrr::discard() masks scales::discard()
## x dplyr::filter()  masks stats::filter()
## x dplyr::lag()     masks stats::lag()
## x recipes::step()  masks stats::step()
```

```r
library(ISLR)

Hitters <- as_tibble(Hitters) %>%
  filter(!is.na(Salary))
```


## Best Subset Selection

tidymodels does not currently support subset selection methods, and it unlikely to include it in the [near future](https://stackoverflow.com/questions/66651033/stepwise-algorithm-in-tidymodels#comment117845482_66651033).

## Forward and Backward Stepwise Selection

tidymodels does not currently support forward and backward stepwise selection methods, and it unlikely to include it in the [near future](https://stackoverflow.com/questions/66651033/stepwise-algorithm-in-tidymodels#comment117845482_66651033).

## Ridge Regression


```r
ridge_spec <- linear_reg(mixture = 0) %>%
  set_engine("glmnet")
```


```r
ridge_fit <- fit(ridge_spec, Salary ~ ., data = Hitters)
```


```r
tidy(ridge_fit, penalty = 11498)
```

```
## Loading required package: Matrix
```

```
## 
## Attaching package: 'Matrix'
```

```
## The following objects are masked from 'package:tidyr':
## 
##     expand, pack, unpack
```

```
## Loaded glmnet 4.1-1
```

```
## # A tibble: 20 x 3
##    term         estimate penalty
##    <chr>           <dbl>   <dbl>
##  1 (Intercept) 407.        11498
##  2 AtBat         0.0370    11498
##  3 Hits          0.138     11498
##  4 HmRun         0.525     11498
##  5 Runs          0.231     11498
##  6 RBI           0.240     11498
##  7 Walks         0.290     11498
##  8 Years         1.11      11498
##  9 CAtBat        0.00314   11498
## 10 CHits         0.0117    11498
## 11 CHmRun        0.0876    11498
## 12 CRuns         0.0234    11498
## 13 CRBI          0.0242    11498
## 14 CWalks        0.0250    11498
## 15 LeagueN       0.0866    11498
## 16 DivisionW    -6.23      11498
## 17 PutOuts       0.0165    11498
## 18 Assists       0.00262   11498
## 19 Errors       -0.0206    11498
## 20 NewLeagueN    0.303     11498
```



```r
tidy(ridge_fit, penalty = 705)
```

```
## # A tibble: 20 x 3
##    term        estimate penalty
##    <chr>          <dbl>   <dbl>
##  1 (Intercept)  54.4        705
##  2 AtBat         0.112      705
##  3 Hits          0.656      705
##  4 HmRun         1.18       705
##  5 Runs          0.937      705
##  6 RBI           0.847      705
##  7 Walks         1.32       705
##  8 Years         2.58       705
##  9 CAtBat        0.0108     705
## 10 CHits         0.0468     705
## 11 CHmRun        0.338      705
## 12 CRuns         0.0937     705
## 13 CRBI          0.0979     705
## 14 CWalks        0.0718     705
## 15 LeagueN      13.7        705
## 16 DivisionW   -54.7        705
## 17 PutOuts       0.119      705
## 18 Assists       0.0161     705
## 19 Errors       -0.704      705
## 20 NewLeagueN    8.61       705
```


```r
tidy(ridge_fit, penalty = 50)
```

```
## # A tibble: 20 x 3
##    term          estimate penalty
##    <chr>            <dbl>   <dbl>
##  1 (Intercept)   48.2          50
##  2 AtBat         -0.354        50
##  3 Hits           1.95         50
##  4 HmRun         -1.29         50
##  5 Runs           1.16         50
##  6 RBI            0.809        50
##  7 Walks          2.71         50
##  8 Years         -6.20         50
##  9 CAtBat         0.00609      50
## 10 CHits          0.107        50
## 11 CHmRun         0.629        50
## 12 CRuns          0.217        50
## 13 CRBI           0.215        50
## 14 CWalks        -0.149        50
## 15 LeagueN       45.9          50
## 16 DivisionW   -118.           50
## 17 PutOuts        0.250        50
## 18 Assists        0.121        50
## 19 Errors        -3.28         50
## 20 NewLeagueN    -9.42         50
```


```r
predict(ridge_fit, new_data = Hitters, penalty = 50)
```

```
## # A tibble: 263 x 1
##     .pred
##     <dbl>
##  1  469. 
##  2  663. 
##  3 1023. 
##  4  505. 
##  5  550. 
##  6  200. 
##  7   79.4
##  8  105. 
##  9  836. 
## 10  865. 
## # … with 253 more rows
```


```r
Hitters_split <- initial_split(Hitters)

Hitters_train <- training(Hitters_split)
Hitters_test <- testing(Hitters_split)

Hitters_fold <- vfold_cv(Hitters_train, v = 10)
```



```r
ridge_recipe <- 
  recipe(formula = Salary ~ ., data = Hitters_train) %>% 
  step_novel(all_nominal(), -all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors(), -all_nominal()) 

ridge_spec <- 
  linear_reg(penalty = tune(), mixture = 0) %>% 
  set_mode("regression") %>% 
  set_engine("glmnet") 

ridge_workflow <- 
  workflow() %>% 
  add_recipe(ridge_recipe) %>% 
  add_model(ridge_spec) 

penalty_grid <- grid_regular(penalty(range = c(-5, 5)), levels = 50)

tune_res <- tune_grid(
  ridge_workflow,
  resamples = Hitters_fold, 
  grid = penalty_grid
)

autoplot(tune_res)
```

<img src="06-regularization_files/figure-html/unnamed-chunk-9-1.png" width="672" />


```r
best_penalty <- select_best(tune_res, metric = "rsq")
```


```r
ridge_final <- finalize_workflow(ridge_workflow, best_penalty)

ridge_final_fit <- fit(ridge_final, data = Hitters_train)

tidy(ridge_final_fit)
```

```
## # A tibble: 20 x 3
##    term        estimate penalty
##    <chr>          <dbl>   <dbl>
##  1 (Intercept)  534.       356.
##  2 AtBat         21.4      356.
##  3 Hits          47.0      356.
##  4 HmRun         -1.36     356.
##  5 Runs          31.0      356.
##  6 RBI           31.2      356.
##  7 Walks         40.4      356.
##  8 Years          9.79     356.
##  9 CAtBat        28.2      356.
## 10 CHits         39.2      356.
## 11 CHmRun        25.8      356.
## 12 CRuns         36.8      356.
## 13 CRBI          41.5      356.
## 14 CWalks        12.8      356.
## 15 PutOuts       40.9      356.
## 16 Assists        2.62     356.
## 17 Errors        -2.32     356.
## 18 League_N       4.25     356.
## 19 Division_W   -37.2      356.
## 20 NewLeague_N    0.991    356.
```

## The Lasso


```r
lasso_recipe <- 
  recipe(formula = Salary ~ ., data = Hitters_train) %>% 
  step_novel(all_nominal(), -all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors(), -all_nominal()) 

lasso_spec <- 
  linear_reg(penalty = tune(), mixture = 1) %>% 
  set_mode("regression") %>% 
  set_engine("glmnet") 

lasso_workflow <- 
  workflow() %>% 
  add_recipe(lasso_recipe) %>% 
  add_model(lasso_spec) 

penalty_grid <- grid_regular(penalty(range = c(-2, 2)), levels = 50)

tune_res <- tune_grid(
  lasso_workflow,
  resamples = Hitters_fold, 
  grid = penalty_grid
)

autoplot(tune_res)
```

<img src="06-regularization_files/figure-html/unnamed-chunk-12-1.png" width="672" />


```r
best_penalty <- select_best(tune_res, metric = "rsq")
```


```r
lasso_final <- finalize_workflow(lasso_workflow, best_penalty)

lasso_final_fit <- fit(lasso_final, data = Hitters_train)

tidy(lasso_final_fit)
```

```
## # A tibble: 20 x 3
##    term        estimate penalty
##    <chr>          <dbl>   <dbl>
##  1 (Intercept)   534.      3.39
##  2 AtBat        -162.      3.39
##  3 Hits          235.      3.39
##  4 HmRun         -14.9     3.39
##  5 Runs            0       3.39
##  6 RBI            11.4     3.39
##  7 Walks         103.      3.39
##  8 Years         -28.8     3.39
##  9 CAtBat          0       3.39
## 10 CHits           0       3.39
## 11 CHmRun          0       3.39
## 12 CRuns         156.      3.39
## 13 CRBI          190.      3.39
## 14 CWalks       -122.      3.39
## 15 PutOuts        63.2     3.39
## 16 Assists         3.09    3.39
## 17 Errors          0       3.39
## 18 League_N        3.16    3.39
## 19 Division_W    -60.9     3.39
## 20 NewLeague_N     0       3.39
```
## Principal Components Regression


```r
pca_recipe <- 
  recipe(formula = Salary ~ ., data = Hitters_train) %>% 
  step_novel(all_nominal(), -all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors(), -all_nominal()) %>%
  step_pca(all_predictors(), threshold = tune())

lm_spec <- 
  linear_reg() %>% 
  set_mode("regression") %>% 
  set_engine("lm") 

pca_workflow <- 
  workflow() %>% 
  add_recipe(pca_recipe) %>% 
  add_model(lm_spec) 

threshold_grid <- grid_regular(threshold(), levels = 10)

tune_res <- tune_grid(
  pca_workflow,
  resamples = Hitters_fold, 
  grid = threshold_grid
)

autoplot(tune_res)
```

<img src="06-regularization_files/figure-html/unnamed-chunk-15-1.png" width="672" />

```r
best_threshold <- select_best(tune_res, metric = "rmse")
```


```r
pca_final <- finalize_workflow(pca_workflow, best_threshold)

pca_final_fit <- fit(pca_final, data = Hitters_train)

tidy(pca_final_fit)
```

```
## # A tibble: 8 x 5
##   term        estimate std.error statistic  p.value
##   <chr>          <dbl>     <dbl>     <dbl>    <dbl>
## 1 (Intercept)    534.      22.3     24.0   2.68e-59
## 2 PC1           -115.       8.36   -13.7   3.69e-30
## 3 PC2            -35.4     10.9     -3.25  1.37e- 3
## 4 PC3             22.9     15.5      1.48  1.41e- 1
## 5 PC4             26.2     18.3      1.43  1.53e- 1
## 6 PC5             57.8     22.1      2.61  9.85e- 3
## 7 PC6             63.3     25.2      2.52  1.27e- 2
## 8 PC7             22.8     26.4      0.864 3.89e- 1
```

## Partial Least Squares


```r
pls_recipe <- 
  recipe(formula = Salary ~ ., data = Hitters_train) %>% 
  step_novel(all_nominal(), -all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors(), -all_nominal()) %>%
  step_pls(all_predictors(), num_comp = tune(), outcome = "Salary")

lm_spec <- 
  linear_reg() %>% 
  set_mode("regression") %>% 
  set_engine("lm") 

pls_workflow <- 
  workflow() %>% 
  add_recipe(pls_recipe) %>% 
  add_model(lm_spec) 

num_comp_grid <- grid_regular(num_comp(c(1, 20)), levels = 10)

tune_res <- tune_grid(
  pls_workflow,
  resamples = Hitters_fold, 
  grid = num_comp_grid
)

autoplot(tune_res)
```

<img src="06-regularization_files/figure-html/unnamed-chunk-17-1.png" width="672" />

```r
best_threshold <- select_best(tune_res, metric = "rmse")
```


```r
pls_final <- finalize_workflow(pls_workflow, best_threshold)

pls_final_fit <- fit(pls_final, data = Hitters_train)

tidy(pls_final_fit)
```

```
## # A tibble: 4 x 5
##   term        estimate std.error statistic  p.value
##   <chr>          <dbl>     <dbl>     <dbl>    <dbl>
## 1 (Intercept)    534.      21.8      24.5  2.13e-61
## 2 PLS1           121.       8.27     14.6  5.47e-33
## 3 PLS2            51.9     15.1       3.44 7.01e- 4
## 4 PLS3            48.3     16.8       2.88 4.44e- 3
```
