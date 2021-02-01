library(lme4)
library(lmerTest)
library(effects)

#load the data for the first two model
myDataSet <- read.csv(file="userStudy_data.csv",header=TRUE,sep=",")

myDataSet$subid <- as.factor(myDataSet$subid)
myDataSet$blockid <- as.factor(myDataSet$blockid)
myDataSet$answer <- as.factor(myDataSet$answer)
myDataSet$answer.interaction <- paste(myDataSet$answer, "_", myDataSet$bias_atomic)
myDataSet$answer.interaction <- as.character(myDataSet$answer.interaction)

myDataSet <- subset(myDataSet,answer == "no" | answer == "yes")

#MODEL 1
model.1 <- lmer(difference.in.rt ~ difference.in.moral.score + (1|subid) + (1|blockid), REML=FALSE, data = myDataSet)
summary(model.1)

#MODEL 2
model.2 <- lmer(difference.in.rt ~ difference.in.moral.score + answer.interaction + (1|subid) + (1|blockid), REML=FALSE, data = myDataSet)
summary(model.2)

#comparison of the models
anova(model.1,model.2)

#visualizing the effects
e <- allEffects(model.2)
print(e)
plot(e)

#load the data for the last model
myDataSet2 <- read.csv(file="userStudy_data_block.csv",header=TRUE,sep=",")

myDataSet2$subid <- as.factor(myDataSet2$subid)
myDataSet2$blockid <- as.factor(myDataSet2$blockid)

#MODEL 3
model.3 <- lmer(difference.in.rt.aa ~ mean.difference.in.rt + mean.absolute.difference.in.moral.score + (1|subid) + (1|blockid), data = myDataSet2)
summary(model.3)

#visualizing the effects
e <- allEffects(model.3)
print(e)
plot(e)
