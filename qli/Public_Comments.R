# Public Comments Project
# Read in excel
setwd("~/Downloads/Study/Public_Comments")
getwd()
excerpts <- read.csv("comment_excerpts.csv")
dim(excerpts)
names(excerpts)
x<-excerpts$Comment_Excerpt
x
