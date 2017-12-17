library(blogdown)
install_hugo()

# create site
new_site(theme = 'jbub/ghostwriter',
         format = 'toml')

# create new post
new_post(title = 'first_post.Rmd')

serve_site()


new_post(title = "test_post.Rmd",
         tags = "test")
