# Custom ggplot2 theme matching my-light.theme
# Colors derived from the blog's syntax highlighting theme

library(ggplot2)

# Color palette from my-light.theme
blog_colors <- list(
  # Plot background - subtle contrast with page (#fafbfc)
  background = "#f0f2f5",
  text = "#24292e",
  text_light = "#6a737d",
  teal = "#0d7d6c",
  blue = "#0366d6",
  purple = "#6f42c1",
  gold = "#b08800",
  red = "#cb2431",
  navy = "#032f62",
  grid = "#e1e4e8"
)

#' Blog Theme for ggplot2
#'
#' A clean theme matching the blog's visual style
#'
#' @param base_size Base font size (default 12)
#' @param base_family Base font family (default "sans")
#' @export
theme_blog <- function(base_size = 12, base_family = "sans") {
  theme_minimal(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # Background
      plot.background = element_rect(fill = blog_colors$background, color = NA),
      panel.background = element_rect(fill = blog_colors$background, color = NA),

      # Grid
      panel.grid.major = element_line(color = blog_colors$grid, linewidth = 0.4),
      panel.grid.minor = element_line(color = blog_colors$grid, linewidth = 0.2),

      # Axes
      axis.text = element_text(color = blog_colors$text_light, size = rel(0.9)),
      axis.title = element_text(color = blog_colors$text, size = rel(1), face = "bold"),
      axis.line = element_line(color = blog_colors$text_light, linewidth = 0.4),
      axis.ticks = element_line(color = blog_colors$text_light, linewidth = 0.4),

      # Title and labels
      plot.title = element_text(
        color = blog_colors$text,
        size = rel(1.3),
        face = "bold",
        hjust = 0,
        margin = margin(b = 10)
      ),
      plot.subtitle = element_text(
        color = blog_colors$text_light,
        size = rel(1),
        hjust = 0,
        margin = margin(b = 15)
      ),
      plot.caption = element_text(
        color = blog_colors$text_light,
        size = rel(0.8),
        hjust = 1,
        margin = margin(t = 10)
      ),

      # Legend
      legend.background = element_rect(fill = blog_colors$background, color = NA),
      legend.key = element_rect(fill = blog_colors$background, color = NA),
      legend.text = element_text(color = blog_colors$text_light, size = rel(0.9)),
      legend.title = element_text(color = blog_colors$text, size = rel(0.95), face = "bold"),
      legend.position = "bottom",

      # Facets
      strip.background = element_rect(fill = blog_colors$grid, color = NA),
      strip.text = element_text(
        color = blog_colors$text,
        size = rel(0.95),
        face = "bold",
        margin = margin(5, 5, 5, 5)
      ),

      # Margins
      plot.margin = margin(15, 15, 15, 15)
    )
}

#' Blog color scale for discrete variables
#'
#' @param ... Arguments passed to discrete_scale
#' @export
scale_color_blog <- function(...) {
  discrete_scale(
    "colour", "blog",
    palette = function(n) {
      colors <- c(
        blog_colors$teal,
        blog_colors$blue,
        blog_colors$purple,
        blog_colors$gold,
        blog_colors$red,
        blog_colors$navy
      )
      colors[seq_len(min(n, length(colors)))]
    },
    ...
  )
}

#' Blog fill scale for discrete variables
#'
#' @param ... Arguments passed to discrete_scale
#' @export
scale_fill_blog <- function(...) {
  discrete_scale(
    "fill", "blog",
    palette = function(n) {
      colors <- c(
        blog_colors$teal,
        blog_colors$blue,
        blog_colors$purple,
        blog_colors$gold,
        blog_colors$red,
        blog_colors$navy
      )
      colors[seq_len(min(n, length(colors)))]
    },
    ...
  )
}

# Blog color scheme for bayesplot
mix_hex <- function(foreground, background, weight_foreground = 0.5) {
  fg <- grDevices::col2rgb(foreground)
  bg <- grDevices::col2rgb(background)
  grDevices::rgb(t(weight_foreground * fg + (1 - weight_foreground) * bg), maxColorValue = 255)
}

# bayesplot expects 6 ordered shades:
# light, light_highlight, mid, mid_highlight, dark, dark_highlight
#
# Note: bayesplot's internal hex-color validation doesn't accept named
# character vectors, so keep this unnamed.
blog_scheme <- unname(c(
  mix_hex(blog_colors$teal, "#ffffff", 0.15),
  mix_hex(blog_colors$teal, "#ffffff", 0.3),
  blog_colors$teal,
  mix_hex(blog_colors$teal, "#000000", 0.8),
  blog_colors$teal,
  blog_colors$navy
))

#' Set up bayesplot to use blog theme
#'
#' Call this after loading bayesplot to apply blog colors and theme
#' @export
use_blog_bayesplot <- function(base_size = 12, base_family = "sans") {
  if (!requireNamespace("bayesplot", quietly = TRUE)) {
    stop("Package 'bayesplot' is required but not installed.", call. = FALSE)
  }

  bayesplot::color_scheme_set(blog_scheme)
  bayesplot::bayesplot_theme_set(theme_blog(base_size = base_size, base_family = base_family))
}

#' Register bayesplot hooks for blog theme
#'
#' Ensures bayesplot defaults are replaced even if bayesplot is loaded after
#' sourcing this file.
register_blog_bayesplot_hook <- function(base_size = 12, base_family = "sans") {
  if (isTRUE(getOption("blog.theme_blog.bayesplot_hook_set", FALSE))) {
    return(invisible(NULL))
  }

  setHook(
    packageEvent("bayesplot", "onLoad"),
    function(...) try(use_blog_bayesplot(base_size = base_size, base_family = base_family), silent = TRUE),
    action = "append"
  )
  setHook(
    packageEvent("bayesplot", "attach"),
    function(...) try(use_blog_bayesplot(base_size = base_size, base_family = base_family), silent = TRUE),
    action = "append"
  )

  options(blog.theme_blog.bayesplot_hook_set = TRUE)
  invisible(NULL)
}

# Set as default theme when sourced
theme_set(theme_blog())

# Auto-setup bayesplot if already loaded, and ensure future loads use it too
register_blog_bayesplot_hook()
if ("bayesplot" %in% loadedNamespaces()) use_blog_bayesplot()
