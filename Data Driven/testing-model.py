from sklearn.metrics import r2_score
import pandas as pd


# Loading of the SISSO.out and testing set files
# You need to adapt with the desired output feature you want to test

Test = pd.read_csv('./data/path_you_need_for_testing_output-feature.dat', sep=' ')
Validation = pd.read_csv('./data/path_you_need_for_validation_output-feature.dat', sep=' ')


def Find_model(path):
    """

    :param path: a string with the path of the SISSO.out
    :return: the place into the file with the best model
    """

    X = pd.DataFrame(pd.read_csv(path, sep=" "))

    typeof = X.iloc[:, 0]
    RMSE = []
    idx = RMSE.copy()

    for i in range(len(typeof)):
        if typeof[i].find('(model)') != -1:
            RMSE.append(float(X.iloc[i + 1, 1].split(" ")[3]))
            idx.append(i + 3)

    return min(RMSE), idx[RMSE.index(min(RMSE))], RMSE.index(min(RMSE))


def Features(path):
    """

    :param path: a string with the path of SISSO.out
    :return: a list with all the new descriptors in the new subspace of initial features, and also the dimension
    used
    """

    X = pd.DataFrame(pd.read_csv(path, sep=" "))
    rmse, idx, rg = Find_model(path)

    dictionnary = ["log", "exp", "sin", "cos", "tan", "sqrt", "cbrt"]
    symbol = ["^"]

    l_ = [idx + k for k in range(rg + 1)]
    descriptors = []

    for i in l_:

        test = X.iloc[i, 0].split("[")[1].split("]")[0]

        for j in dictionnary:
            if j in test:
                test = test.replace(j, "np." + j)

        for j in symbol:
            if j in test:
                test = test.replace(j, "**")

        descriptors.append(test)

    return descriptors, len(descriptors)


def Coefficients(path):
    """

    :param path: a string with the path of SISSO.out
    :return: a list with the coefficient of the model, and also the intercept of the regression
    """

    X = pd.DataFrame(pd.read_csv(path, sep=" "))
    rmse, idx, rg = Find_model(path)
    descriptors, n = Features(path=path)

    test = X.iloc[idx + n, 0].split(":")[1]
    intercept = X.iloc[idx + n + 1, 0].split(":")[1]
    test = test.replace("   ", ",").split(",")[1:]

    return test, intercept


def Prediction(path, XPred, true_descriptors, true_features):
    """

    :param path: a string with the path of SISSO.out
    :param true_descriptors: a list with the new descriptors of subspace
    :param true_features: a list with the names of features used for the model
    :param XPred: a dataframe with new data to be predict
    :return: a dataframe with all new predictions
    """

    predictions_data = pd.DataFrame(columns=["Values"])
    coefs, intercept = Coefficients(path=path)
    coefs = [float(i) for i in coefs]

    for i in range(XPred.shape[0]):

        new_desc = []
        for desc in true_descriptors:
            for feat in true_features:

                if desc.find(feat) != -1:
                    desc = desc.replace(feat, str(list(XPred[feat])[i]))

            new_desc.append(eval(desc))

        predictions_data.loc[len(predictions_data)] = sum([float(i) * j for i, j in
                                                           zip(new_desc, coefs)]) + float(intercept)

    return predictions_data


# #### Testing

path_to_sisso_out = ''
sisso_descriptors, N = Features(path_to_sisso_out)
predictions = Prediction(path_to_sisso_out, Test[["Gap", "AM", "CBD", "porous"]], sisso_descriptors,
                         ["Gap", "AM", "CBD", "porous"])
YTest = pd.DataFrame(Test['tliq'])
YTest.columns, predictions.columns = ['output_feature'], ['output_feature']

# YTest and predictions are the 'true' and 'predicted' values for the SISSO model
# Finally you calculate the goodness of fitting with the R2 score

print(r2_score(list(YTest[0]), list(predictions[0])))

# #### Validation
# The code below uses a dataset never used for the training and testing steps

predictions = Prediction(path_to_sisso_out, Validation[["Gap", "AM", "CBD", "porous"]], sisso_descriptors,
                         ["Gap", "AM", "CBD", "porous"])
YValidation = pd.DataFrame(Validation['tliq'])
YValidation.columns, predictions.columns = ['output_feature'], ['output_feature']

print(r2_score(list(YValidation[0]), list(predictions[0])))