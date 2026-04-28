args <- commandArgs(trailingOnly=TRUE)

in_csv  <- args[[1]]
out_csv <- args[[2]]

suppressPackageStartupMessages(library(ScottKnottESD))

df <- read.csv(in_csv, check.names = FALSE)

# coerce numeric safely
for (j in seq_len(ncol(df))) {
  df[[j]] <- suppressWarnings(as.numeric(df[[j]]))
}

# drop all-NA columns
df <- df[, colSums(!is.na(df)) > 0, drop = FALSE]

# if only 0/1 method, trivial grouping
if (ncol(df) < 2) {
  out <- data.frame(Method = colnames(df), Group = rep(1L, ncol(df)))
  write.csv(out, out_csv, row.names = FALSE)
  quit(status = 0)
}

res <- tryCatch(
  sk_esd(df, version = "p"),
  error = function(e) NULL
)

if (is.null(res)) {
  out <- data.frame(Method = colnames(df), Group = rep(1L, ncol(df)))
  write.csv(out, out_csv, row.names = FALSE)
  quit(status = 0)
}

# ---- robust extraction across ScottKnottESD versions ----
groups_raw <- NULL

# Some versions expose $groups, others put it in [[1]]
if (!is.null(res$groups)) {
  groups_raw <- res$groups
} else {
  groups_raw <- res[[1]]
}

groups <- suppressWarnings(as.integer(unlist(groups_raw, use.names = FALSE)))

# sanity check: must match number of methods (columns)
if (length(groups) != ncol(df) || any(is.na(groups))) {
  # fallback: all in one group (better than crashing)
  groups <- rep(1L, ncol(df))
}

out <- data.frame(Method = colnames(df), Group = groups)
write.csv(out, out_csv, row.names = FALSE)
