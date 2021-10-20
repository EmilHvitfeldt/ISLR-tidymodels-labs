print.model_fit <- print.model_fit <- function(x, ...) {
  cat("parsnip model object\n\n")

  if (inherits(x$fit, "try-error")) {
    cat("Model fit failed with error:\n", x$fit, "\n")
  } else {
    print(x$fit, ...)
  }
  invisible(x)
}

