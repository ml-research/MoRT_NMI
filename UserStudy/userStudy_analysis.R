# This code presents the statistical analysis conducted for this paper.
# R version: 3.5.2
# 2021

data <- read.csv(file="userStudy_diff_score.csv",header=TRUE,sep=",") # upload the data
diff.score = abs(datasetSub$score) # obtain the diff_score in absolute value

# run the Shapiro-Wilk test to test the normality of the data
check.normality <- shapiro.test(diff.score)
check.normality
# since p-value < 0.001, normality violated.
# apply non-parametric Wilcoxon’s signed-rank test
difference.test <- wilcox.test(diff.score, mu=0, paired=FALSE, alternative='greater')
difference.test
# since p-value < 0.001, true mean is greater than 0.