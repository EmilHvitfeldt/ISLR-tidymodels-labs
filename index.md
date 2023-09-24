# Introduction

This book aims to be a complement to the 2nd edition [An Introduction to Statistical Learning](https://www.statlearning.com/) book with translations of the labs into using the [tidymodels](https://www.tidymodels.org/) set of packages.

The labs will be mirrored quite closely to stay true to the original material.

## Edition Differences {-}

All listed changes will be relative to the 1st edition.

- [Naive Bayes](04-classification.qmd#naive-bayes) has been added to chapter 4 on [Classification](04-classification.qmd)
- [Poisson Regression](04-classification.qmd#poisson-regression) has been added to chapter 4 on [Classification](04-classification.qmd)
- "Application to Caravan Insurance Data" section is no longer treated as its own section and is now part of the [K-Nearest Neighbors](04-classification.qmd#k-nearest-neighbors) section
- [Bayesian Additive Regression Trees](08-tree-based-methods.qmd#bayesian-additive-regression-trees) has been added to chapter 8 on [Tree-Based Methods](08-tree-based-methods.qmd)
- chapter on [Unsupervised Learning](12-unsupervised-learning.qmd) has been renumbered to chapter 12 instead of 10
- [Matrix Completion](12-unsupervised-learning.qmd#matrix-completion) has been added to chapter 12 on [Unsupervised Learning](12-unsupervised-learning.qmd)
- chapter on [Deep learning](10-deep-learning.qmd) has been added as chapter 10
- chapter on [Survival Analysis and Censored Data](11-survival-analysis.qmd) as been added as chapter 11
- chapter on [Multiple Testing](13-multiple-testing.qmd) as been added as chapter 13

## Colophon {-}

This book was written in [RStudio](https://posit.co/products/open-source/rstudio/) using [**Quarto**](https://quarto.org/). The [website](https://emilhvitfeldt.github.io/ISLR-tidymodels-labs/index.html) is hosted via [GitHub Pages](https://pages.github.com/), and the complete source is available on [GitHub](https://github.com/EmilHvitfeldt/ISLR-tidymodels-labs).

This version of the book was built with R version 4.3.1 (2023-06-16) and Quarto version 1.4.104 and the following packages:


::: {.cell-output-display}
|package      |version |source                                                 |
|:------------|:-------|:------------------------------------------------------|
|broom        |1.0.5   |CRAN (R 4.3.0)                                         |
|corrr        |0.4.4   |CRAN (R 4.3.0)                                         |
|dials        |1.2.0   |CRAN (R 4.3.0)                                         |
|discrim      |1.0.1   |CRAN (R 4.3.0)                                         |
|downlit      |0.4.3   |CRAN (R 4.3.0)                                         |
|dplyr        |1.1.3   |CRAN (R 4.3.0)                                         |
|factoextra   |1.0.7   |CRAN (R 4.3.0)                                         |
|ggplot2      |3.4.3   |CRAN (R 4.3.0)                                         |
|glmnet       |4.1-8   |CRAN (R 4.3.0)                                         |
|infer        |1.0.5   |CRAN (R 4.3.0)                                         |
|ISLR         |1.4     |CRAN (R 4.3.0)                                         |
|ISLR2        |1.3-2   |CRAN (R 4.3.0)                                         |
|kernlab      |0.9-32  |CRAN (R 4.3.0)                                         |
|kknn         |1.3.1   |CRAN (R 4.3.0)                                         |
|klaR         |1.7-2   |CRAN (R 4.3.0)                                         |
|MASS         |7.3-60  |CRAN (R 4.3.0)                                         |
|mclust       |6.0.0   |CRAN (R 4.3.0)                                         |
|mixOmics     |6.24.0  |bioc_xgit (\@e8cf785f5d4c9fd3691ed3ec8bd81c6f3d966bb3) |
|modeldata    |1.2.0   |CRAN (R 4.3.0)                                         |
|paletteer    |1.5.0   |CRAN (R 4.3.0)                                         |
|parsnip      |1.1.1   |CRAN (R 4.3.0)                                         |
|patchwork    |1.1.3   |CRAN (R 4.3.0)                                         |
|poissonreg   |1.0.1   |CRAN (R 4.3.0)                                         |
|proxy        |0.4-27  |CRAN (R 4.3.0)                                         |
|purrr        |1.0.2   |CRAN (R 4.3.0)                                         |
|randomForest |4.7-1.1 |CRAN (R 4.3.0)                                         |
|readr        |2.1.4   |CRAN (R 4.3.0)                                         |
|recipes      |1.0.8   |CRAN (R 4.3.0)                                         |
|rpart        |4.1.19  |CRAN (R 4.3.0)                                         |
|rpart.plot   |3.1.1   |CRAN (R 4.3.0)                                         |
|rsample      |1.2.0   |CRAN (R 4.3.0)                                         |
|scico        |1.5.0   |CRAN (R 4.3.0)                                         |
|tibble       |3.2.1   |CRAN (R 4.3.0)                                         |
|tidyclust    |0.1.2   |CRAN (R 4.3.0)                                         |
|tidymodels   |1.1.1   |CRAN (R 4.3.0)                                         |
|tidyr        |1.3.0   |CRAN (R 4.3.0)                                         |
|tune         |1.1.2   |CRAN (R 4.3.0)                                         |
|vip          |0.4.1   |CRAN (R 4.3.0)                                         |
|workflows    |1.1.3   |CRAN (R 4.3.0)                                         |
|workflowsets |1.0.1   |CRAN (R 4.3.0)                                         |
|xgboost      |1.7.5.1 |CRAN (R 4.3.0)                                         |
|yardstick    |1.2.0   |CRAN (R 4.3.0)                                         |
:::
