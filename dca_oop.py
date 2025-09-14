# only one class to deal with all methods
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


class WrongModelName(Exception): pass


class SingleModel:
    def __init__(self, Q, T):
        self.Q = Q
        self.T = T
        self.q_max = max(Q)
        self.T_max = max(T)
        self.models_functions = {
            "ex": self.exposential,
            "hr": self.harmonic,
            "hp": self.hyperbolic
        }

    def normalize(self):
        """ Normalize the the input [ production and time ] for faster and better fitting"""
        self.T = self.T / max(self.T)
        self.Q = self.Q / max(self.Q)

    def RMSE(self, y, y_fit):
        N = len(y)
        return np.sqrt(np.sum((y - y_fit) ** 2) / N)

    def hyperbolic(self, t, qi, di, b):
        return qi / ((1 + b * di * t) ** (1 / b))

    def exposential(self, t, qi, di):
        return qi * np.exp(-di * t)

    def harmonic(self, t, qi, di):
        return qi / (1 + di * t)

    def fit(self, model):
        if model not in ("ex", "hr", "hp"):
            raise WrongModelName("The model name is worng choose between [ ex , hr , hp ]")

        self.normalize()

        # ---- initial guesses in normalized space ----
        # qi_n ~ 1 (since Q was normalized), di_n ~ small, b ~ 0.5 as a safe start
        if model == "ex":
            p0 = [1.0, 0.1]  # [qi_n, di_n]
            bounds = ([0.0, 0.0],  # qi_n >= 0, di_n >= 0
                      [np.inf, np.inf])
            func = self.exposential

        elif model == "hr":
            p0 = [1.0, 0.1]  # [qi_n, di_n] ; b is fixed to 1.0 later
            bounds = ([0.0, 0.0],
                      [np.inf, np.inf])
            func = self.harmonic

        else:  # "hp"
            p0 = [1.0, 0.1, 0.5]  # [qi_n, di_n, b]
            #                 qi_n  di_n    b
            bounds = ([0.0, 0.0, 1e-6],  # enforce b > 0
                      [np.inf, np.inf, np.inf])
            func = self.hyperbolic

        # ---- fit with bounds ----
        params, pcov = curve_fit(
            func, self.T, self.Q,
            p0=p0, bounds=bounds, method="trf", maxfev=20000
        )

        # 1-sigma (normalized-parameter space)
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(len(params), np.nan)

        if model == "ex":
            qi_n, di_n = params
            b = 0.0
            qi_n_std, di_n_std = perr
            b_std = 0.0
        elif model == "hr":
            qi_n, di_n = params
            b = 1.0
            qi_n_std, di_n_std = perr
            b_std = 0.0
        else:  # "hp"
            qi_n, di_n, b = params
            qi_n_std, di_n_std, b_std = perr

        # Fit curve (normalized)
        qfit_normalized = func(self.T, *params)

        # De-normalize to original units
        q_fitted = qfit_normalized * self.q_max
        RMSE = self.RMSE(self.Q, qfit_normalized)

        qi = qi_n * self.q_max
        di = di_n / self.T_max
        qi_std = qi_n_std * self.q_max
        di_std = di_n_std / self.T_max
        b_std = float(b_std)

        parameters = [qi, di, b, RMSE, model]
        param_stds = [qi_std, di_std, b_std]
        return parameters, q_fitted, param_stds


def hyperbolic(t, qi, di, b):
    return qi / (np.abs((1 + b * di * t)) ** (1 / b))


def exposential(t, qi, di):
    return qi * np.exp(-di * t)


def harmonic(t, qi, di):
    return qi / (1 + di * t)


# ------------------------- start the whole arps model ------------------------------
class ARPS:
    """
    Takes the productions and date columns and provide ARP's paramters for the 3 models
    """

    def __init__(self, dataframe: pd.DataFrame, production_col: str, date_column: str):
        """
        :param dataframe: the pandas.DataFrame object the contain the data
        :param production_col: name of the production column
        :param date_column: name of the data column
        """
        self.q = production_col
        self.date = date_column
        self.df = dataframe[[date_column, production_col]]
        self.Q = 0
        self.T = 0
        self.dd = dataframe[[date_column, production_col]]
        self.ex_params = None
        self.hp_params = None
        self.Hr_params = None

    def smooth(self, window_size: int, stds: int, trim: bool):
        """Smooting the data using moving average method with removing the outliers.
        window_size : int -> the size of the window for the moving average
        stds : int -> number of standard deviations to remove data after [ for the outliers ]
        trim : bool -> whether to trim the data in which the production increase or not
        return :
          df : pd.DataFrame -> the data frame after smoothing
        """
        # smoothing using moving average
        self.df[self.q + '_rol_Av'] = self.df[self.q].rolling(window=window_size, center=True).mean()
        # identify the outliers and remove them
        self.df[self.q + '_rol_Std'] = self.df[self.q].rolling(window=window_size, center=True).std()
        self.df[self.q + '_is_Outlier'] = (
                    abs(self.df[self.q] - self.df[self.q + '_rol_Av']) > (stds * self.df[self.q + '_rol_Std']))
        result = self.df.drop(self.df[self.df[self.q + '_is_Outlier'] == True].index).reset_index(drop=True)
        # Remove rows where "_rol_Av" has NaNs
        result = result[result[self.q + '_rol_Av'].notna()]
        # remove the increasing part of the curve we only concern about the decline part
        if trim == True:
            # Trim initial buildup
            maxi = result[self.q + '_rol_Av'].max()
            maxi_index = (result[result[self.q + '_rol_Av'] == maxi].index.values)[0]
            result = result.iloc[maxi_index:, :].reset_index(drop=True)

        self.Q = result[self.q + '_rol_Av']
        self.df = result
        return self.df

    def prepocess_date_col(self, frequency="Daily", df=None):
        """
        Convert the date column into number so that it can be used for fitting
        frequency : the frequency of the production data taken [ Daily , Monthly , Yearly ]
        """
        self.df[self.date] = pd.to_datetime(self.df[self.date])
        self.df[f"Time [{frequency}]"] = (self.df[self.date] - self.df[self.date].iloc[0])
        if frequency == "Daily":
            self.df[f"Time [{frequency}]"] = self.df[f"Time [{frequency}]"] / np.timedelta64(1, 'D')
            self.df[f"Time [{frequency}]"] = self.df[f"Time [{frequency}]"].astype(int)
        elif frequency == "Monthly":
            self.df[f"Time [{frequency}]"] = self.df[f"Time [{frequency}]"] / np.timedelta64(1, 'M')
            self.df[f"Time [{frequency}]"] = self.df[f"Time [{frequency}]"].astype(int)
        elif frequency == "Yearly":
            self.df[f"Time [{frequency}]"] = self.df[f"Time [{frequency}]"] / np.timedelta64(1, 'Y')
            self.df[f"Time [{frequency}]"] = self.df[f"Time [{frequency}]"].astype(int)
        self.T = self.df[f"Time [{frequency}]"]
        self.Q = self.df[self.q + '_rol_Av']
        # the separation time
        self.last_time = self.T.iloc[-1]

        if df is not None:
            df[self.date] = pd.to_datetime(df[self.date])
            df[f"Time [{frequency}]"] = (df[self.date] - df[self.date].iloc[0])
            if frequency == "Daily":
                df[f"Time [{frequency}]"] = df[f"Time [{frequency}]"] / np.timedelta64(1, 'D')
                df[f"Time [{frequency}]"] = df[f"Time [{frequency}]"].astype(int)
            elif frequency == "Monthly":
                df[f"Time [{frequency}]"] = df[f"Time [{frequency}]"] / np.timedelta64(1, 'M')
                df[f"Time [{frequency}]"] = df[f"Time [{frequency}]"].astype(int)
            elif frequency == "Yearly":
                df[f"Time [{frequency}]"] = df[f"Time [{frequency}]"] / np.timedelta64(1, 'Y')
                df[f"Time [{frequency}]"] = df[f"Time [{frequency}]"].astype(int)

            return df[f"Time [{frequency}]"], df[self.q]

    def fit_exponential(self):
        """
        Fit the data only for the Exponential model
        return : parameters , values of the fitted line to draw , stds
        """
        parameters, q_fitted, stds = SingleModel(self.Q, self.T).fit("ex")
        self.ex_params = parameters
        return parameters, q_fitted, stds

    def fit_hyperbolic(self):
        """
        Fit the data only for the Hyperbolic model
        return : parameters , values of the fitted line to draw , stds
        """
        parameters, q_fitted, stds = SingleModel(self.Q, self.T).fit("hp")
        self.hp_params = parameters
        return parameters, q_fitted, stds

    def fit_harmonic(self):
        """
        Fit the data only for the Harmonic model
        return : parameters , values of the fitted line to draw , stds
        """
        parameters, q_fitted, stds = SingleModel(self.Q, self.T).fit("hr")
        self.Hr_params = parameters
        return parameters, q_fitted, stds

    def fit_all_models(self):
        """
        Fit the data for the three model once [ exponential , harmonic , hyperbolic ]
        return :
         data_parameters : Dict -> a dictionary of all the information and parameters of each model [ qi, di , b , rmse ]
         Qs : pandas.DataFrame -> dataframe for these columns [ Time, originalSmoothed_Q , Exponential_fitted_Q , Hyperbolic_fiited_Q , Harmonic_fitted_Q ]
        """
        ex, qex, ex_stds = self.fit_exponential()
        hp, qhp, hp_stds = self.fit_hyperbolic()
        hr, qhr, hr_stds = self.fit_harmonic()

        all_params = [ex, hp, hr]
        all_stds = [ex_stds, hp_stds, hr_stds]

        data_info = pd.DataFrame({
            "Model": [i[-1] for i in all_params],
            "Qi":    [i[0] for i in all_params],
            "Qi_std":[s[0] for s in all_stds],
            "Di":    [i[1] for i in all_params],
            "Di_std":[s[1] for s in all_stds],
            "b":     [i[2] for i in all_params],
            "b_std": [s[2] for s in all_stds],
            "Normalized RMSE": [i[3] for i in all_params],
        })

        Qs = pd.DataFrame({
            "Time": self.T,
            "Original_Smoothed": self.Q,
            "Exponential": qex,
            "Hyperbolic": qhp,
            "Harmonic": qhr,
            "Date": self.df[self.date]
        }).set_index("Time", drop=True)

        self.model_params = data_info
        # ensure a scalar (not a Series)
        best_model = data_info.loc[data_info["Normalized RMSE"].idxmin(), "Model"]
        return data_info, Qs, best_model

    def forecast_hyperbolic(self, Q_limit):
        t = self.T.iloc[-1]
        qi, d, b = self.model_params["Qi"].iloc[1], self.model_params["Di"].iloc[1], self.model_params["b"].iloc[1]
        qs = []
        ts = []
        while True:
            ts.append(t)
            q = hyperbolic(t, qi, d, b)
            qs.append(q)

            if q < Q_limit:
                break
            t += 5
        return ts, qs

    def forecast_harmonic(self, Q_limit):
        t = self.T.iloc[-1]
        qi, d = self.model_params["Qi"].iloc[2], self.model_params["Di"].iloc[2]
        qs = []
        ts = []
        while True:
            ts.append(t)
            q = harmonic(t, qi, d)
            qs.append(q)

            if q < Q_limit:
                break
            t += 5
        return ts, qs

    def forecast_exponential(self, Q_limit):
        t = self.T.iloc[-1]
        qi, d = self.model_params["Qi"].iloc[0], self.model_params["Di"].iloc[0]
        qs = []
        ts = []
        while True:
            ts.append(t)
            q = exposential(t, qi, d)
            qs.append(q)

            if q < Q_limit:
                break
            t += 5
        return ts, qs

    def forecast(self, best_model, Q_limit):
        q_cum_last = [self.dd[self.q].cumsum().iloc[-1]]
        v_line = self.T.iloc[-1]
        if best_model == "ex":
            ts, qs = self.forecast_exponential(Q_limit)
            q_cum_last.extend(qs)
            q_cum = np.cumsum(q_cum_last)
        elif best_model == "hr":
            ts, qs = self.forecast_harmonic(Q_limit)
            q_cum_last.extend(qs)
            q_cum = np.cumsum(q_cum_last)
        elif best_model == "hp":
            ts, qs = self.forecast_hyperbolic(Q_limit)
            q_cum_last.extend(qs)
            q_cum = np.cumsum(q_cum_last)

        return ts, qs, q_cum, v_line

    def total_cum_production(self, freq, ts, qs):
        date, q = self.prepocess_date_col(frequency=freq, df=self.dd)
        v_line = date.iloc[-1]
        date._append(pd.Series(ts[1:]))
        q._append(pd.Series(qs[1:]))
        Q_cum = q.cumsum()

        return date, Q_cum, v_line,
