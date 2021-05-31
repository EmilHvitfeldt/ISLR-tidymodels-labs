# Support Vector Machines


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
## ✓ broom        0.7.6           ✓ recipes      0.1.16.9000
## ✓ dials        0.0.9           ✓ rsample      0.1.0      
## ✓ dplyr        1.0.6           ✓ tibble       3.1.2      
## ✓ ggplot2      3.3.3           ✓ tidyr        1.1.3      
## ✓ infer        0.5.4           ✓ tune         0.1.5      
## ✓ modeldata    0.1.0           ✓ workflows    0.2.2      
## ✓ parsnip      0.1.6           ✓ workflowsets 0.0.2      
## ✓ purrr        0.3.4           ✓ yardstick    0.0.8
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
```

## Support Vector Classifier


```r
set.seed(1)
sim_data <- tibble(
  x1 = rnorm(20),
  x2 = rnorm(20),
  y  = factor(rep(c(-1, 1), 10))
) %>%
  mutate(x1 = ifelse(y == 1, x1 + 1, x1),
         x2 = ifelse(y == 1, x2 + 1, x2))

ggplot(sim_data, aes(x1, x2, color = y)) +
  geom_point()
```

<img src="09-support-vector-machines_files/figure-html/unnamed-chunk-2-1.png" width="672" />


```r
svm_linear_spec <- svm_poly(degree = 1) %>%
  set_mode("classification") %>%
  set_engine("kernlab", scaled = FALSE)

svm_linear_fit <- svm_linear_spec %>% 
  set_args(cost = 10) %>%
  fit(y ~ ., data = sim_data)

svm_linear_fit
```

```
## parsnip model object
## 
## Fit time:  732ms 
## Support Vector Machine object of class "ksvm" 
## 
## SV type: C-svc  (classification) 
##  parameter : cost C = 10 
## 
## Polynomial kernel function. 
##  Hyperparameters : degree =  1  scale =  1  offset =  1 
## 
## Number of Support Vectors : 12 
## 
## Objective Function Value : -108.1122 
## Training error : 0.2 
## Probability model included.
```

```r
library(kernlab)
```

```
## 
## Attaching package: 'kernlab'
```

```
## The following object is masked from 'package:purrr':
## 
##     cross
```

```
## The following object is masked from 'package:ggplot2':
## 
##     alpha
```

```
## The following object is masked from 'package:scales':
## 
##     alpha
```

```r
plot(svm_linear_fit$fit)
```

<img src="09-support-vector-machines_files/figure-html/unnamed-chunk-3-1.png" width="672" />


```r
svm_linear_fit <- svm_linear_spec %>% 
  set_args(cost = 0.1) %>%
  fit(y ~ ., data = sim_data)

svm_linear_fit
```

```
## parsnip model object
## 
## Fit time:  26ms 
## Support Vector Machine object of class "ksvm" 
## 
## SV type: C-svc  (classification) 
##  parameter : cost C = 0.1 
## 
## Polynomial kernel function. 
##  Hyperparameters : degree =  1  scale =  1  offset =  1 
## 
## Number of Support Vectors : 18 
## 
## Objective Function Value : -1.5443 
## Training error : 0.2 
## Probability model included.
```


```r
svm_linear_wf <- workflow() %>%
  add_model(svm_linear_spec %>% set_args(cost = tune())) %>%
  add_formula(y ~ .)

set.seed(1234)
sim_data_fold <- vfold_cv(sim_data, strata = y)

param_grid <- grid_regular(cost(), levels = 10)

tune_res <- tune_grid(
  svm_linear_wf, 
  resamples = sim_data_fold, 
  grid = param_grid
)

autoplot(tune_res)
```

<img src="09-support-vector-machines_files/figure-html/unnamed-chunk-5-1.png" width="672" />


```r
best_cost <- select_best(tune_res, metric = "accuracy")
```


```r
svm_linear_final <- finalize_workflow(svm_linear_wf, best_cost)

svm_linear_fit <- svm_linear_final %>%
  fit(sim_data)
```


```r
set.seed(2)
sim_data_test <- tibble(
  x1 = rnorm(20),
  x2 = rnorm(20),
  y  = factor(rep(c(-1, 1), 10))
) %>%
  mutate(x1 = ifelse(y == 1, x1 + 1, x1),
         x2 = ifelse(y == 1, x2 + 1, x2))
```


```r
augment(svm_linear_fit, new_data = sim_data_test) %>%
  conf_mat(truth = y, estimate = .pred_class)
```

```
##           Truth
## Prediction -1 1
##         -1  8 2
##         1   2 8
```



## Support Vector Machine



## ROC Curves

## SVM with Multiple Classes

## Application to Gene Expression Data
