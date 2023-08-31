import numpy as np
from scipy.special import boxcox
import math
from scipy.stats import norm


class ALM(object):
    def __init__(
        self,
        distribution="dnorm",
        loss="likelihood",
        occurrence="none",
        orders=(0, 0, 0),
        parameters=None,
        fast=False,
        *args,
        **kwargs
    ):
        """
        Initialize the ALM model.

        Parameters:
        distribution (str): The distribution of the model. Defaults to 'normal'.
        loss (str): The loss function. Defaults to 'L2'.
        occurrence (str): The occurrence to be used. Defaults to 'none'.
        orders (tuple): The orders of the terms in the model. Defaults to (0,0,0).
        parameters (array-like): Additional parameters for the model. Defaults to None.
        fast (bool): Whether to use a faster, possibly less accurate, method for fitting the model.
        """

        # ARIMA orders
        self.arOrder = abs(self.orders[0])  # Ensure arOrder is positive
        self.iOrder = abs(self.orders[1])  # Ensure iOrder is positive
        self.maOrder = abs(self.orders[2])  # Ensure maOrder is positive
        self.ariOrder = self.arOrder + self.iOrder
        self.ariModel = self.ariOrder > 0
        self.args = args["nu"]

        # Create polynomials for the AR, I, and MA orders
        if self.arOrder > 0:
            self.poly1 = np.ones(self.arOrder + 1)
        else:
            self.poly1 = np.array([1, 1])

        if self.iOrder > 0:
            self.poly2 = np.array([1, -1])
            if self.iOrder > 1:
                for _ in range(1, self.iOrder):
                    self.poly2 = np.polymul(self.poly2, np.array([1, -1]))
        else:
            self.poly2 = np.array([1, -1])

        if self.maOrder > 0:
            self.poly3 = np.ones(self.maOrder + 1)
        else:
            self.poly3 = np.array([1, 1])

        # If we are provided with distribution-specific parameters

        # Initialize a flag that will be turned of if not parameter is provided
        self.aParameterProvided = False

        if ("alpha" in args) & (distribution is "dalaplace"):
            self.distribution_parameter = args["alpha"]
        elif ("size" in args) & (distribution is "dchisq"):
            self.distribution_parameter = {"size"}
        elif ("nu" in args) & (distribution is "dnbinom"):
            self.distribution_parameter = args["nu"]
        elif ("sigma" in args) & ((distribution is "dfnorm") | (distribution is "drectnorm")):
            self.distribution_parameter = args["sigma"]
        elif ("shape" in args) & ((distribution is "dgnorm") | (distribution is "dlgnorm")):
            self.distribution_parameter = args["sigma"]
        elif ("lambdaBC" in args) & (distribution is "dbcnorm"):
            self.distribution_parameter = args["lambdaBC"]
        elif ("nu" in args) & (distribution is "dt"):
            self.distribution_parameter = args["lambdaBC"]
        else:
            self.distribution_parameter = None
            self.aParameterProvided = True

    def fit(self, data, subset=None, na_action=None):
        """
        Fit the ALM model to the data.

        Parameters:
        data (DataFrame): The data to fit the model to.
        subset (array-like): An array-like object of booleans, integers, or index values that
                            specify a subset of the data to be used in the model. Defaults to None.
        na_action (str): What should happen when the data contain NAs. Defaults to None.
        """

        self.data = data  # Store the data
        self.subset = subset
        self.na_action = na_action

    def optimize(self):
        """
        Perform optimization.
        """
        pass

    def regularize(self):
        """
        Perform regularization.
        """
        pass

    def predict(self, new_data):
        """
        Use the fitted model to make predictions on new data.

        Parameters:
        new_data (DataFrame): The new data to make predictions on.
        """

        pass

    def fitter(self, B, matrixXreg):
        """
        Fit the model based on the selected distribution

        Parameters:
        B (array-like): The parameters of the model.
        distribution (str): The distribution of the model.
        y (DataFrame): The data. -> I am passing self.data from fit here. If I dont use fitter on fit then I need to pass y here.
        matrixXreg (DataFrame): The matrix of regressors.
        """
        # Deal with additional parameters
        if self.distribution in [
            "dalaplace",
            "dnbinom",
            "dchisq",
            "dfnorm",
            "drectnorm",
            "dgnorm",
            "dlgnorm",
            "dbcnorm",
            "dt",
        ]:
            # If we are not given the parameters for the distribution but the starting values
            if not self.aParameterProvided:
                other = B[0]
                B = B[1:]
            else:
                # distribution_parameter parameter is provided at the initialization
                other = self.distribution_parameter
        else:
            # raise warning for the distribution
            # Maybe this message should be given on initialiation
            raise ValueError("The distribution is not supported")

        # If there is ARI, then calculate polynomials
        if self.arOrder > 0 and self.iOrder > 0:
            self.poly1[-1] = -B[-self.arOrder :]
            # This condition is needed for cases of only ARI models
            if self.nVariables > self.arOrder:
                B = np.append(B[: (len(B) - self.arOrder)], -np.polymul(self.poly2, self.poly1)[-1])
            else:
                B = -np.polymul(self.poly2, self.poly1)[-1]
        elif self.iOrder > 0:
            B = np.append(B, -self.poly2[-1])
        elif self.arOrder > 0:
            self.poly1[-1] = -B[-self.arOrder :]

        # This is a hack. If lambda=1, then we only need the mean of the data
        if self.loss in ["LASSO", "RIDGE"] and self.lambda_ == 1:
            B[1:] = 0

        # Define mu based on the distribution
        mu_calculations = {
            "dnorm": np.dot(matrixXreg, B),
            "dlaplace": np.dot(matrixXreg, B),
            "dnbinom": np.exp(np.dot(matrixXreg, B)),
            # Add other distributions...
        }

        # Get the mean from the distribution
        # Should this be self.mu?
        mu = mu_calculations.get(self.distribution)

        # Get the scale value
        scale = self.scalerInternal(B, self.distribution, matrixXreg, mu, other)

        # return form here should be a dictionary or something?
        out = {"mu": mu, "scale": scale, "other": other}
        return out  # Return the estimated parameters

    def scalerInternal(self, B, distribution, matrixXreg, mu, other, otU):
        """
        Calculate the scale of the residuals for the selected distribution.

        Parameters:
            B (array-like): The parameters of the model.
            distribution (str): The distribution of the model.
            matrixXreg (DataFrame): The matrix of regressors.
            mu (array-like): The fitted values of the model.
            other (float): Additional parameter for certain distributions.
        otU (array-like): The vector of {0,1} for the occurrence model.

        """

        # Make a temp copy here -> will remove it later
        y = self.data.copy()

        # Should the dictionary be moved to the utils file?
        # Maybe this function is not needed at all.
        # When we get the data we can calculate the scale directly from self.distribution given on init.
        scale_calculations = {
            "dbeta": np.exp(np.dot(matrixXreg, B[int(len(B) / 2) :])),
            "dnorm": np.sqrt(np.sum((y.loc[otU] - mu.loc[otU]) ** 2) / len(otU)),
            "dlaplace": np.sum(np.abs(y.loc[otU] - mu.loc[otU])) / len(otU),
            "ds": np.sum(np.sqrt(np.abs(y.loc[otU] - mu.loc[otU]))) / (len(otU) * 2),
            "dgnorm": (other * np.sum(np.abs(y.loc[otU] - mu.loc[otU]) ** other) / len(otU)) ** (1 / other),
            "dlogis": np.sqrt(np.sum((y.loc[otU] - mu.loc[otU]) ** 2) / len(otU) * 3 / np.pi**2),
            "dalaplace": np.sum((y.loc[otU] - mu.loc[otU]) * (other - (y.loc[otU] <= mu.loc[otU]) * 1)) / len(otU),
            "dlnorm": np.sqrt(np.sum((np.log(y.loc[otU]) - mu.loc[otU]) ** 2) / len(otU)),
            "dllaplace": np.sum(np.abs(np.log(y.loc[otU]) - mu.loc[otU])) / len(otU),
            "dls": np.sum(np.sqrt(np.abs(np.log(y.loc[otU]) - mu.loc[otU]))) / (len(otU) * 2),
            "dlgnorm": (other * np.sum(np.abs(np.log(y.loc[otU]) - mu.loc[otU]) ** other) / len(otU)) ** (1 / other),
            # 'dbcnorm': np.sqrt(np.sum((special.boxcox(y.loc[otU], other) - mu.loc[otU]) ** 2) / len(otU)),
            # 'dinvgauss': np.sum(((y.loc[otU] / mu.loc[otU] - 1) ** 2) / (y.loc[otU] / mu.loc[otU])) / len(otU),
            # 'dgamma': np.sum(((y.loc[otU] / mu.loc[otU] - 1) ** 2)) / len(otU),
            # 'dlogitnorm': np.sqrt(np.sum((np.log(y.loc[otU] / (1 - y.loc[otU])) - mu.loc[otU]) ** 2) / len(otU)),
            # 'dfnorm': ,
            # 'drectnorm': ,
            # 'dt': ,
            # 'dchisq': ,
            "dnbinom": np.abs(other),
            "dpois": mu.loc[otU],
            # 'pnorm': np.sqrt(np.mean(special.ndtr((y - special.ndtr(mu)) / 2) ** 2)),
            # 'plogis': np.sqrt(np.mean(np.log((1 + y * (1 + np.exp(mu))) / (1 + np.exp(mu) * (2 - y) - y)) ** 2)),
            # 1 is the default and the value for the dexp
            "default": 1,
        }

        # Get the values
        scale = scale_calculations.get(distribution)

        return scale

    def fitterRecursive(self, B, matrixXreg):
        """
        Fit dynamic models.

        Parameters:
        B (array-like): The parameters of the model. -> Isnt this given on .fit()

        matrixXreg (DataFrame): The matrix of regressors.
        """

        fitterReturn = self.fitter(B, matrixXreg)

        # Fill in the first ariOrder elements with fitted values
        for i in range(self.ariOrder):
            if self.distribution == "dbcnorm":
                matrixXreg.loc[~self.ariZeroes[:, i], self.nVariablesExo + i] = boxcox(
                    self.ariElementsOriginal[~self.ariZeroes[:, i], i], fitterReturn["other"]
                )
            else:
                mu_values = {"dbeta": np.log(fitterReturn["mu"])}.get(self.distribution, fitterReturn["mu"])
                matrixXreg.loc[self.ariZeroes[:, i], self.nVariablesExo + i] = mu_values[~self.otU][
                    : self.ariZeroesLengths[i]
                ]

        # matrixXreg = self.fitterRecursion(matrixXreg, B, y, self.ariZeroes, self.nVariablesExo, distribution)
        fitterReturn = self.fitter(B, matrixXreg)
        fitterReturn["matrixXreg"] = matrixXreg

        return fitterReturn

    def extractorFitted(self, mu, scale):
        # comment -> if this is given

        if self.distribution == "dfnorm":
            result = math.sqrt(2 / math.pi) * scale * np.exp(-(mu**2) / (2 * scale**2)) + mu * (
                1 - 2 * norm.cdf(-mu / scale)
            )
        elif self.distribution == "drectnorm":
            result = mu * (1 - norm.cdf(0, loc=mu, scale=scale)) + scale * norm.pdf(0, loc=mu, scale=scale)
        elif self.distribution == "dnorm":
            result = None  # Placeholder; add the equation here
        elif self.distribution == "dgnorm":
            result = None  # Placeholder
        elif self.distribution == "dinvgauss":
            result = None  # Placeholder
        elif self.distribution == "dgamma":
            result = None  # Placeholder
        elif self.distribution == "dexp":
            result = None  # Placeholder
        elif self.distribution == "dlaplace":
            result = None  # Placeholder
        elif self.distribution == "dalaplace":
            result = None  # Placeholder
        elif self.distribution == "dlogis":
            result = None  # Placeholder
        elif self.distribution == "dt":
            result = None  # Placeholder
        elif self.distribution == "ds":
            result = None  # Placeholder
        elif self.distribution == "dpois":
            result = None  # Placeholder
        elif self.distribution == "dnbinom":
            result = mu  # Placeholder
        elif self.distribution == "dchisq":
            result = mu + self.args["nu"]  # Placeholder, 'nu' not defined
        elif self.distribution == "dlnorm":
            result = None  # Placeholder
        elif self.distribution == "dllaplace":
            result = None  # Placeholder
        elif self.distribution == "dls":
            result = None  # Placeholder
        elif self.distribution == "dlgnorm":
            result = np.exp(mu)
        elif self.distribution == "dlogitnorm":
            result = np.exp(mu) / (1 + np.exp(mu))
        elif self.distribution == "dbcnorm":
            result = None  # Placeholder; bcTransformInv function not defined
        elif self.distribution == "dbeta":
            result = mu / (mu + scale)
        elif self.distribution == "pnorm":
            result = norm.cdf(mu, loc=0, scale=1)
        elif self.distribution == "plogis":
            result = 1 / (1 + np.exp(-mu))
        else:
            result = None  # Placeholder for any other cases

        return result

    def extractorResiduals(self, yFitted, lambdaBC=None):
        """
        Calculate the error term in the transformed scale based on the distribution,
        mu, yFitted, and optionally lambdaBC for Box-Cox transformation.
        """

        if self.distribution == "dbeta":
            result = self.data - yFitted
        elif self.distribution == "dfnorm":
            result = None  # Placeholder
        elif self.distribution == "drectnorm":
            result = None  # Placeholder
        elif self.distribution == "dnorm":
            result = None  # Placeholder
        elif self.distribution == "dlaplace":
            result = None  # Placeholder
        elif self.distribution == "ds":
            result = None  # Placeholder
        elif self.distribution == "dgnorm":
            result = None  # Placeholder
        elif self.distribution == "dalaplace":
            result = None  # Placeholder
        elif self.distribution == "dlogis":
            result = None  # Placeholder
        elif self.distribution == "dt":
            result = None  # Placeholder
        elif self.distribution == "dnbinom":
            result = None  # Placeholder
        elif self.distribution == "dpois":
            result = self.data - self.mu
        elif self.distribution == "dinvgauss":
            result = None  # Placeholder
        elif self.distribution == "dgamma":
            result = None  # Placeholder
        elif self.distribution == "dexp":
            result = self.data / self.mu
        elif self.distribution == "dchisq":
            result = math.sqrt(self.data) - math.sqrt(self.mu)
        elif self.distribution == "dlnorm":
            result = None  # Placeholder
        elif self.distribution == "dllaplace":
            result = None  # Placeholder
        elif self.distribution == "dls":
            result = None  # Placeholder
        elif self.distribution == "dlgnorm":
            result = np.log(self.data) - self.mu
        elif self.distribution == "dbcnorm":
            # Placeholder; bcTransform function not defined
            result = None if lambdaBC is None else boxcox(self.data, lambdaBC) - self.mu
        elif self.distribution == "dlogitnorm":
            result = np.log(self.data / (1 - y)) - self.mu
        elif self.distribution == "pnorm":
            result = norm.ppf((y - norm.cdf(self.mu, loc=0, scale=1) + 1) / 2, loc=0, scale=1)
        elif self.distribution == "plogis":
            result = np.log((1 + y * (1 + np.exp(self.mu))) / (1 + np.exp(self.mu) * (2 - y) - y))
        else:
            result = None  # Placeholder for any other cases

        return result

    # Not ready yet!!!
    # I stopped here!

    def CF(
        self,
        B,
        loss,
        matrixXreg,
        recursiveModel,
        denominator,
        lambda_value=0,
        interceptIsNeeded=False,
        occurrenceModel=False,
        arOrder=0,
    ):
        if recursiveModel:
            fitterReturn = self.fitterRecursive(B, matrixXreg)
        else:
            fitterReturn = self.fitter(B, matrixXreg)

        if loss == "likelihood":
            # The original log-likelihood (placeholder)
            # You'd replace this with your actual log likelihood computation
            CFValue = -np.sum(0)
        else:
            # Assuming that extractorFitted is defined somewhere
            yFitted = self.extractorFitted(fitterReturn["mu"], fitterReturn["scale"])

            if loss == "MSE":
                CFValue = meanFast((y - yFitted) ** 2)
            elif loss == "MAE":
                CFValue = meanFast(np.abs(y - yFitted))
            elif loss == "HAM":
                CFValue = meanFast(np.sqrt(np.abs(y - yFitted)))
            elif loss == "LASSO":
                CFValue = (1 - lambda_value) * meanFast((y - yFitted) ** 2) + lambda_value * np.sum(np.abs(B))
            elif loss == "RIDGE":
                B *= denominator

                if interceptIsNeeded:
                    CFValue = (1 - lambda_value) * meanFast((y - yFitted) ** 2) + lambda_value * np.sqrt(
                        np.sum(B[1:] ** 2)
                    )
                else:
                    CFValue = (1 - lambda_value) * meanFast((y - yFitted) ** 2) + lambda_value * np.sqrt(np.sum(B**2))

                if lambda_value == 1:
                    CFValue = meanFast((y - yFitted) ** 2)
            elif loss == "custom":
                CFValue = lossFunction(actual=y, fitted=yFitted, B=B, xreg=matrixXreg)

        if np.isnan(CFValue) or np.isinf(CFValue):
            CFValue = 1e300

        # Check the roots of polynomials (Placeholder)
        if arOrder > 0 and any_condition:  # Replace any_condition with your condition
            CFValue /= np.min(some_value)  # Replace some_value with your values

        return CFValue
