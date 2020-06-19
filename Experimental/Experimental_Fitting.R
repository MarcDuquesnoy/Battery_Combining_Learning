Y = read.csv("./data",header= TRUE,sep=",")
subset(Y, select = c(X.NMC, calender.gap.um, porosite_before, thickness_before, porosity))