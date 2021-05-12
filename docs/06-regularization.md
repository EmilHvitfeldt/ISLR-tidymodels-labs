# Linear Model Selection and Regularization


```r
library(tidymodels)
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



```r
tidy(ridge_fit, penalty = 705)
```


```r
tidy(ridge_fit, penalty = 50)
```


```r
predict(ridge_fit, new_data = Hitters, penalty = 50)
```

```
## Error in predict(ridge_fit, new_data = Hitters, penalty = 50): object 'ridge_fit' not found
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


```r
best_penalty <- select_best(tune_res, metric = "rsq")
```


```r
ridge_final <- finalize_workflow(ridge_workflow, best_penalty)

ridge_final_fit <- fit(ridge_final, data = Hitters_train)

tidy(ridge_final_fit)
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


```r
best_penalty <- select_best(tune_res, metric = "rsq")
```


```r
lasso_final <- finalize_workflow(lasso_workflow, best_penalty)

lasso_final_fit <- fit(lasso_final, data = Hitters_train)

tidy(lasso_final_fit)
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

best_threshold <- select_best(tune_res, metric = "rmse")
```


```r
pca_final <- finalize_workflow(pca_workflow, best_threshold)

pca_final_fit <- fit(pca_final, data = Hitters_train)

tidy(pca_final_fit)
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

best_threshold <- select_best(tune_res, metric = "rmse")
```


```r
pls_final <- finalize_workflow(pls_workflow, best_threshold)

pls_final_fit <- fit(pls_final, data = Hitters_train)

tidy(pls_final_fit)
```
