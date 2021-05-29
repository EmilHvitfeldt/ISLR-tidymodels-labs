# Statistical learning

The original labs introduce you to the basics of R. We will not repeat that endeavor here as it was done well. You will notice that the labs in this book will look slightly different since we are following the [tidyverse style guide](https://style.tidyverse.org/).

The primary purpose of this book is to rewrite the labs of ISLR using the tidymodels packages. A great introduction to tidymodels can be found on the tidymodels [website](https://www.tidymodels.org/).

Proper introductions to the various tidymodels packages will not be present in these labs since they aim to mirror the labs in ISLR. The [getting started](https://www.tidymodels.org/start/) page has good introductions to

- [parsnip](https://www.tidymodels.org/start/models/)
- [recipes and workflows](https://www.tidymodels.org/start/recipes/)
- [rsample](https://www.tidymodels.org/start/resampling/)
- [tune](https://www.tidymodels.org/start/tuning/)

The charts whenever possible will be created using ggplot2, if you are unfamiliar with ggplot2 I would recommend you start by reading the [Data Visualisation](https://r4ds.had.co.nz/data-visualisation.html) chapter of the [R for Data Science](https://r4ds.had.co.nz/index.html) book.

The tidymodels packages can be installed as a whole using `install.packages("tidymodels")` and the ISLR package which contains many of the data set we will be using can be installed using `install.packages("ISLR")`.
