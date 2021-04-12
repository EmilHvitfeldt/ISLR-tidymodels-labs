# Classification

## The Stock Market Data


```r
library(tidymodels)
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
## ✓ parsnip   0.1.5           ✓ yardstick 0.0.8      
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

Smarket <- as_tibble(Smarket)
Caravan <- as_tibble(Caravan)
```


```r
library(corrr)
cor_Smarket <- correlate(Smarket[-9])
```

```
## 
## Correlation method: 'pearson'
## Missing treated using: 'pairwise.complete.obs'
```

```r
rplot(cor_Smarket, colours = c("indianred2", "black", "skyblue1"))
```

```
## Don't know how to automatically pick scale for object of type noquote. Defaulting to continuous.
```

<img src="04-classification_files/figure-html/unnamed-chunk-2-1.png" width="672" />


```r
library(paletteer)
cor_Smarket %>%
  stretch() %>%
  ggplot(aes(x, y, fill = r)) +
  geom_tile() +
  geom_text(aes(label = as.character(fashion(r)))) +
  scale_fill_paletteer_c("scico::roma", limits = c(-1, 1), direction = -1) +
  theme_minimal()
```

<img src="04-classification_files/figure-html/unnamed-chunk-3-1.png" width="672" />



## Logistic Regression


```r
lr_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")
```


```r
lr_fit <- lr_spec %>%
  fit(
    Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
    data = Smarket
    )

lr_fit
```

```
## parsnip model object
## 
## Fit time:  9ms 
## 
## Call:  stats::glm(formula = Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + 
##     Lag5 + Volume, family = stats::binomial, data = data)
## 
## Coefficients:
## (Intercept)         Lag1         Lag2         Lag3         Lag4         Lag5  
##   -0.126000    -0.073074    -0.042301     0.011085     0.009359     0.010313  
##      Volume  
##    0.135441  
## 
## Degrees of Freedom: 1249 Total (i.e. Null);  1243 Residual
## Null Deviance:	    1731 
## Residual Deviance: 1728 	AIC: 1742
```


```r
summary(lr_fit$fit)
```

```
## 
## Call:
## stats::glm(formula = Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + 
##     Lag5 + Volume, family = stats::binomial, data = data)
## 
## Deviance Residuals: 
##    Min      1Q  Median      3Q     Max  
## -1.446  -1.203   1.065   1.145   1.326  
## 
## Coefficients:
##              Estimate Std. Error z value Pr(>|z|)
## (Intercept) -0.126000   0.240736  -0.523    0.601
## Lag1        -0.073074   0.050167  -1.457    0.145
## Lag2        -0.042301   0.050086  -0.845    0.398
## Lag3         0.011085   0.049939   0.222    0.824
## Lag4         0.009359   0.049974   0.187    0.851
## Lag5         0.010313   0.049511   0.208    0.835
## Volume       0.135441   0.158360   0.855    0.392
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1731.2  on 1249  degrees of freedom
## Residual deviance: 1727.6  on 1243  degrees of freedom
## AIC: 1741.6
## 
## Number of Fisher Scoring iterations: 3
```


```r
tidy(lr_fit)
```

```
## # A tibble: 7 x 5
##   term        estimate std.error statistic p.value
##   <chr>          <dbl>     <dbl>     <dbl>   <dbl>
## 1 (Intercept) -0.126      0.241     -0.523   0.601
## 2 Lag1        -0.0731     0.0502    -1.46    0.145
## 3 Lag2        -0.0423     0.0501    -0.845   0.398
## 4 Lag3         0.0111     0.0499     0.222   0.824
## 5 Lag4         0.00936    0.0500     0.187   0.851
## 6 Lag5         0.0103     0.0495     0.208   0.835
## 7 Volume       0.135      0.158      0.855   0.392
```


```r
predict(lr_fit, new_data = Smarket)
```

```
## # A tibble: 1,250 x 1
##    .pred_class
##    <fct>      
##  1 Up         
##  2 Down       
##  3 Down       
##  4 Up         
##  5 Up         
##  6 Up         
##  7 Down       
##  8 Up         
##  9 Up         
## 10 Down       
## # … with 1,240 more rows
```


```r
predict(lr_fit, new_data = Smarket, type = "prob")
```

```
## # A tibble: 1,250 x 2
##    .pred_Down .pred_Up
##         <dbl>    <dbl>
##  1      0.493    0.507
##  2      0.519    0.481
##  3      0.519    0.481
##  4      0.485    0.515
##  5      0.489    0.511
##  6      0.493    0.507
##  7      0.507    0.493
##  8      0.491    0.509
##  9      0.482    0.518
## 10      0.511    0.489
## # … with 1,240 more rows
```


```r
augment(lr_fit, new_data = Smarket) %>%
  conf_mat(truth = Direction, estimate = .pred_class)
```

```
##           Truth
## Prediction Down  Up
##       Down  145 141
##       Up    457 507
```


```r
augment(lr_fit, new_data = Smarket) %>%
  conf_mat(truth = Direction, estimate = .pred_class) %>%
  autoplot(type = "heatmap")
```

<img src="04-classification_files/figure-html/unnamed-chunk-11-1.png" width="672" />


```r
augment(lr_fit, new_data = Smarket) %>%
  accuracy(truth = Direction, estimate = .pred_class)
```

```
## # A tibble: 1 x 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary         0.522
```


```r
Smarket_train <- Smarket %>%
  filter(Year != 2005)

Smarket_test <- Smarket %>%
  filter(Year == 2005)
```


```r
lr_fit <- lr_spec %>%
  fit(
    Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
    data = Smarket_train
    )
```


```r
augment(lr_fit, new_data = Smarket_test) %>%
  conf_mat(truth = Direction, estimate = .pred_class) 
```

```
##           Truth
## Prediction Down Up
##       Down   77 97
##       Up     34 44
```

```r
augment(lr_fit, new_data = Smarket_test) %>%
  accuracy(truth = Direction, estimate = .pred_class) 
```

```
## # A tibble: 1 x 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary         0.480
```


```r
lr_fit <- lr_spec %>%
  fit(
    Direction ~ Lag1 + Lag2,
    data = Smarket_train
    )
```


```r
augment(lr_fit, new_data = Smarket_test) %>%
  conf_mat(truth = Direction, estimate = .pred_class) 
```

```
##           Truth
## Prediction Down  Up
##       Down   35  35
##       Up     76 106
```

```r
augment(lr_fit, new_data = Smarket_test) %>%
  accuracy(truth = Direction, estimate = .pred_class) 
```

```
## # A tibble: 1 x 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary         0.560
```


```r
predict(
  lr_fit,
  new_data = tibble(Lag1 = c(1.2, 1.5), Lag2 = c(1.1, -0.8)), 
  type = "prob"
)
```

```
## # A tibble: 2 x 2
##   .pred_Down .pred_Up
##        <dbl>    <dbl>
## 1      0.521    0.479
## 2      0.504    0.496
```


## Linear Discriminant Analysis


```r
library(discrim)
```

```
## 
## Attaching package: 'discrim'
```

```
## The following object is masked from 'package:dials':
## 
##     smoothness
```

```r
lda_spec <- discrim_linear() %>%
  set_engine("MASS")

lda_fit <- lda_spec %>%
  fit(Direction ~ Lag1 + Lag2, data = Smarket_train)

lda_fit
```

```
## parsnip model object
## 
## Fit time:  5ms 
## Call:
## lda(Direction ~ Lag1 + Lag2, data = data)
## 
## Prior probabilities of groups:
##     Down       Up 
## 0.491984 0.508016 
## 
## Group means:
##             Lag1        Lag2
## Down  0.04279022  0.03389409
## Up   -0.03954635 -0.03132544
## 
## Coefficients of linear discriminants:
##             LD1
## Lag1 -0.6420190
## Lag2 -0.5135293
```


```r
predict(lda_fit, new_data = Smarket_test)
```

```
## # A tibble: 252 x 1
##    .pred_class
##    <fct>      
##  1 Up         
##  2 Up         
##  3 Up         
##  4 Up         
##  5 Up         
##  6 Up         
##  7 Up         
##  8 Up         
##  9 Up         
## 10 Up         
## # … with 242 more rows
```

```r
predict(lda_fit, new_data = Smarket_test, type = "prob")
```

```
## # A tibble: 252 x 2
##    .pred_Down .pred_Up
##         <dbl>    <dbl>
##  1      0.490    0.510
##  2      0.479    0.521
##  3      0.467    0.533
##  4      0.474    0.526
##  5      0.493    0.507
##  6      0.494    0.506
##  7      0.495    0.505
##  8      0.487    0.513
##  9      0.491    0.509
## 10      0.484    0.516
## # … with 242 more rows
```


```r
augment(lda_fit, new_data = Smarket_test) %>%
  conf_mat(truth = Direction, estimate = .pred_class) 
```

```
##           Truth
## Prediction Down  Up
##       Down   35  35
##       Up     76 106
```

```r
augment(lda_fit, new_data = Smarket_test) %>%
  accuracy(truth = Direction, estimate = .pred_class) 
```

```
## # A tibble: 1 x 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary         0.560
```

## Quadratic Discriminant Analysis


```r
library(discrim)
qda_spec <- discrim_regularized() %>%
  set_args(frac_common_cov = 0, frac_identity = 0) %>%
  set_engine("klaR")

qda_fit <- qda_spec %>%
  fit(Direction ~ Lag1 + Lag2, data = Smarket_train)

qda_fit
```

```
## parsnip model object
## 
## Fit time:  44ms 
## Call: 
## rda(formula = Direction ~ Lag1 + Lag2, data = data, lambda = ~0, 
##     gamma = ~0)
## 
## Regularization parameters: 
##  gamma lambda 
##      0      0 
## 
## Prior probabilities of groups: 
##     Down       Up 
## 0.491984 0.508016 
## 
## Misclassification rate: 
##        apparent: 48.597 %
```


```r
augment(qda_fit, new_data = Smarket_test) %>%
  conf_mat(truth = Direction, estimate = .pred_class) 
```

```
##           Truth
## Prediction Down  Up
##       Down   30  20
##       Up     81 121
```

```r
augment(qda_fit, new_data = Smarket_test) %>%
  accuracy(truth = Direction, estimate = .pred_class) 
```

```
## # A tibble: 1 x 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary         0.599
```

## K-Nearest Neighbors


```r
knn_spec <- nearest_neighbor(neighbors = 3) %>%
  set_engine("kknn") %>%
  set_mode("classification")

knn_fit <- knn_spec %>%
  fit(Direction ~ Lag1 + Lag2, data = Smarket_train)

knn_fit
```

```
## parsnip model object
## 
## Fit time:  36ms 
## 
## Call:
## kknn::train.kknn(formula = Direction ~ Lag1 + Lag2, data = data,     ks = min_rows(3, data, 5))
## 
## Type of response variable: nominal
## Minimal misclassification: 0.492986
## Best kernel: optimal
## Best k: 3
```


```r
augment(knn_fit, new_data = Smarket_test) %>%
  conf_mat(truth = Direction, estimate = .pred_class) 
```

```
##           Truth
## Prediction Down Up
##       Down   43 58
##       Up     68 83
```

```r
augment(knn_fit, new_data = Smarket_test) %>%
  accuracy(truth = Direction, estimate = .pred_class) 
```

```
## # A tibble: 1 x 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary           0.5
```

## An Application to Caravan Insurance Data


```r
Caravan_test <- Caravan[seq_len(1000), ]
Caravan_train <- Caravan[-seq_len(1000), ]
```


```r
rec_spec <- recipe(Purchase ~ ., data = Caravan_train) %>%
  step_normalize(all_numeric_predictors())
```


```r
Caravan_wf <- workflow() %>%
  add_recipe(rec_spec)
```


```r
knn_spec <- nearest_neighbor() %>%
  set_engine("kknn") %>%
  set_mode("classification")
```


```r
knn1_wf <- Caravan_wf %>%
  add_model(knn_spec %>% set_args(neighbors = 1))
```


```r
knn1_fit <- fit(knn1_wf, data = Caravan_train)
```


```r
augment(knn1_fit, new_data = Caravan_test) %>%
  conf_mat(truth = Purchase, estimate = .pred_class)
```

```
##           Truth
## Prediction  No Yes
##        No  874  50
##        Yes  67   9
```


```r
augment(knn1_fit, new_data = Caravan_test) %>%
  accuracy(truth = Purchase, estimate = .pred_class)
```

```
## # A tibble: 1 x 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary         0.883
```


```r
knn3_wf <- Caravan_wf %>%
  add_model(knn_spec %>% set_args(neighbors = 3))

knn3_fit <- fit(knn3_wf, Caravan_train)
```


```r
augment(knn3_fit, new_data = Caravan_test) %>%
  conf_mat(truth = Purchase, estimate = .pred_class)
```

```
##           Truth
## Prediction  No Yes
##        No  875  50
##        Yes  66   9
```



```r
knn5_wf <- Caravan_wf %>%
  add_model(knn_spec %>% set_args(neighbors = 5))

knn5_fit <- fit(knn5_wf, Caravan_train)
```


```r
augment(knn5_fit, new_data = Caravan_test) %>%
  conf_mat(truth = Purchase, estimate = .pred_class)
```

```
##           Truth
## Prediction  No Yes
##        No  874  50
##        Yes  67   9
```
