library(MASS)
library(lmtest)
library(car)

### path of the dataset

path_to_experimental_data = ""

Y = read.csv(path_to_experimental_data,header= TRUE,sep=",")
subset(Y, select = c(X.NMC, calender.gap.um, porosite_before, thickness_before, porosity))


### experimental fitting with three steps

model = lm(porosity ~ -1 + X.NMC + calender.gap.um + porosite_before + thickness_before +
             I(X.NMC**2) + I(calender.gap.um**2) + I(porosite_before**2) + I(thickness_before**2) +
             X.NMC*calender.gap.um + X.NMC*porosite_before + X.NMC*thickness_before +
             calender.gap.um*porosite_before + calender.gap.um*thickness_before +
             porosite_before*thickness_before, data=Y)
model_aic = stepAIC(model, direction = "both")


model = lm(porosity ~ -1 + X.NMC + calender.gap.um + porosite_before + thickness_before +
             I(calender.gap.um^2) +
             X.NMC*calender.gap.um + calender.gap.um*porosite_before, data=Y)
summary(model)


model_fit = lm(porosity ~ -1 + X.NMC + porosite_before + thickness_before +
                 I(calender.gap.um^2) +
                 + calender.gap.um*porosite_before, data=Y)
summary(model_fit)


### Validation of the fitting

plot(model_fit)

shapiro.test(resid(model_fit))

acf(residuals(model_fit))
    
dwtest(model_fit)

