{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "9f709ceb",
   "metadata": {},
   "source": [
    "## 1. Hidden Markov Model\n",
    "\n",
    "Markets exhibit periods with disticnt behaviour and characteristics, these patterns are commonly refered to as market regimes. Market regimes are not directly defined or observable, however identifying these persistent changes in market behaviour can be useful for investment and risk-management decisions. For example, characteristics of a \"crisis\" market regime could be characterized by extreme volatility, large  absolute returns and persistant negative trends. Identifying such regimes could provide additional information about the current level of market risk.\n",
    "\n",
    "Market regimes can be described as latent states that can't be observed directly. A Hidden Markov Model (HMM) provides a probabilistic framework that can be used to find hidden states through observable market data. The implied regimes depend strongly on the information provided to the model. The initial specification in this notebook uses three features: daily log returns, momentum, and conditional volatility estimated by the GARCH(1,1) model. \n",
    "\n",
    "The objective of this notebook is to evaluate whether the implemented HMM can identify persistent and economically interpretable market regimes, and then investigate how the model specification affects the resulting regime classification."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f1f30eb8",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from scipy.stats import t, norm\n",
    "import statsmodels.api as smapi\n",
    "import statsmodels as sm\n",
    "from statsmodels.stats.diagnostic import acorr_ljungbox\n",
    "import seaborn as sns\n",
    "\n",
    "project_root = Path.cwd().parent\n",
    "\n",
    "\n",
    "if str(project_root) not in sys.path:\n",
    "    sys.path.append(str(project_root))\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "18359be3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from src import data_loader as dl\n",
    "from src import preprocessing as pp\n",
    "from src import garch\n",
    "from src import features\n",
    "from src.markov_model.hmm import HiddenMarkovModel as hmm\n",
    "from src.markov_model.forward import forward_filtering\n",
    "from src.markov_model.backward import backward_filtering\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# Download S&P 500 data from 2000-01-01 and compute daily log returns.\n",
    "raw_data = dl.load_market_data()\n",
    "market_data = pp.preprocess_market_data(raw_data)\n",
    "vix = dl.load_market_data(ticker = \"^VIX\")\n",
    "garch_output = garch.fit_garch(market_data[\"Return\"]*100)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9900f558-7554-4a2c-aa83-cd83c7ef61a6",
   "metadata": {},
   "source": [
    "## 2. Data & baseline features \n",
    "\n",
    "The initial dataset is the S&P 500 market/price data spaning from 2000-01-01 through present date. This dataset is utilized to construct the baseline features, which include daily log returns, a 20-day momentum window, and conditional volatility extracted from a GARCH(1,1) model. To prevent overfitting and the introduction of unnecessary noise, it is preferable to train the Hidden Markov Model (HMM) on a limited set of distinct features initially. Starting with a parsimonious feature set limits model complexity and provides an interpretable baseline from which additional features can later be evaluated.\n",
    "\n",
    "These baseline features capture distinct market dynamics: daily log returns capture the direction and magnitude of short-term price changes, momentum captures trend behavior, and conditional volatility serves as a proxy for current market risk. Finally, the dataset is partitioned chronologically, reserving the first 70% of the observations for in-sample training and the remaining 30% for out-of-sample evaluation."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5eaa63b9-bb1b-4c49-a9dd-df58b61f972f",
   "metadata": {},
   "source": [
    "### 2.1 Descriptive Statistics of Training Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fd9e93eb-690c-4468-b7f3-0ad5da76c935",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Returns</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "      <th>Momentum_20</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>4652.000000</td>\n",
       "      <td>4652.000000</td>\n",
       "      <td>4652.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.000142</td>\n",
       "      <td>0.010622</td>\n",
       "      <td>0.003013</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.011979</td>\n",
       "      <td>0.006157</td>\n",
       "      <td>0.045542</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>-0.094695</td>\n",
       "      <td>0.003931</td>\n",
       "      <td>-0.330730</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>-0.004794</td>\n",
       "      <td>0.006730</td>\n",
       "      <td>-0.017200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.000538</td>\n",
       "      <td>0.008919</td>\n",
       "      <td>0.010230</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>0.005613</td>\n",
       "      <td>0.012481</td>\n",
       "      <td>0.029198</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>0.109572</td>\n",
       "      <td>0.058699</td>\n",
       "      <td>0.211030</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Returns  Conditional_Volatility  Momentum_20\n",
       "count  4652.000000             4652.000000  4652.000000\n",
       "mean      0.000142                0.010622     0.003013\n",
       "std       0.011979                0.006157     0.045542\n",
       "min      -0.094695                0.003931    -0.330730\n",
       "25%      -0.004794                0.006730    -0.017200\n",
       "50%       0.000538                0.008919     0.010230\n",
       "75%       0.005613                0.012481     0.029198\n",
       "max       0.109572                0.058699     0.211030"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "hmm_features = features.build_hmm_features(market_data,garch_output)\n",
    "\n",
    "hmm_features[\"VIX\"] = vix[\"Close\"]\n",
    "hmm_features = hmm_features.dropna()\n",
    "\n",
    "split_idx = int(len(hmm_features) * 0.70)\n",
    "\n",
    "train_df = hmm_features.iloc[:split_idx].copy()\n",
    "test_df = hmm_features.iloc[split_idx:].copy()\n",
    "\n",
    "sp_price_train = market_data.loc[train_df.index,\"Close\"]\n",
    "sp_price_test = market_data.loc[test_df.index,\"Close\"]\n",
    "\n",
    "baseline_features = [\"Returns\", \"Conditional_Volatility\", \"Momentum_20\"]\n",
    "\n",
    "baseline_X_train = train_df[baseline_features]\n",
    "baseline_X_test = test_df[baseline_features]\n",
    "dates = train_df.index\n",
    "\n",
    "baseline_X_train.describe()\n",
    "#X_train.corr()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "91b51f0d-7025-4616-838e-8f19a110d4b0",
   "metadata": {},
   "source": [
    "#### Discussion \n",
    "\n",
    "The baseline features exhibit noticeably different statistical properties and scales. Momentum has the largest variance, with a standard deviation of approximately 0.046, compared with 0.012 for daily log returns and 0.006 for conditional volatility. Both returns and momentum take positive and negative values and are centered relatively close to zero, whereas conditional volatility is strictly positive and centered around 0.011. Conditional volatility also exhibits a smaller variation relative to its mean. These differences suggest that the features capture different aspects of market behaviour, however the differences in scale should be considered when specifying the HMM."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f2d1b5bf-98e8-432e-9951-3912f6a7b799",
   "metadata": {},
   "source": [
    "### 2.2 Correlation Matrix of Training Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8e0a0945-fab9-4668-a0f1-d186eea2801a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Returns</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "      <th>Momentum_20</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Returns</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.001087</td>\n",
       "      <td>0.214938</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Conditional_Volatility</th>\n",
       "      <td>-0.001087</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.452712</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Momentum_20</th>\n",
       "      <td>0.214938</td>\n",
       "      <td>-0.452712</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                         Returns  Conditional_Volatility  Momentum_20\n",
       "Returns                 1.000000               -0.001087     0.214938\n",
       "Conditional_Volatility -0.001087                1.000000    -0.452712\n",
       "Momentum_20             0.214938               -0.452712     1.000000"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "baseline_X_train.corr()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b2a5aa82-5e2d-4a11-8f46-50b9ecd57fbd",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "The correlation matrix exhbits no linnear correlation (0.001) between daily returns and conditional volatility, not out of character since volatility describes uncertainty in the market rather than direction. Momentum and returns have a weak positive correlation (0.216), which is reasonable since the daily return is part of the current momentum window. The strongest relationship is observed between momentum and conditional volatility, with a correlation of -0.445. This indicates that periods of elevated conditional volatility tend to exhibit weaker momentum, demonstrating the combination of negative trends and elevated volatility often observed during periods of market stress."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "419d7cb1-d972-45c3-8fa4-18795be0c049",
   "metadata": {},
   "source": [
    "### 2.3 Feature Scaling\n",
    "\n",
    "The baseline features have different numerical scales, which can affect the KMeans initialization of the HMM. The features are standardized to have zero mean and unit variance. To prevent data leakage, the scaler is fitted exclusively on the training data and then applied to the test data using the same transformation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e4aea51c-c00d-4479-a1d7-1e4792d3f8e9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Means: [ 9.16435171e-18 -4.88765424e-17  0.00000000e+00]\n",
      "Standard deviations: [1. 1. 1.]\n"
     ]
    }
   ],
   "source": [
    "scaler = StandardScaler()\n",
    "baseline_X_train = scaler.fit_transform(baseline_X_train)\n",
    "baseline_X_test = scaler.transform(baseline_X_test)\n",
    "\n",
    "print(\"Means:\", baseline_X_train.mean(axis=0))\n",
    "print(\"Standard deviations:\", baseline_X_train.std(axis=0))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4738cbab-b8f7-4409-8fe1-2e25f45782f3",
   "metadata": {},
   "source": [
    "## 3. Baseline HMM\n",
    "\n",
    "The baseline Hidden Markov Model is specified with three hidden states, creating a model that can distinguish between different market conditions while keeping it initialy relatively simple. A full covariance structure is used, this to allow the model to capture dependencies between the standardized log returns, 20-day momentum, and conditional volatility features.\n",
    "\n",
    "The model is trained on the first 70% of the observations using a chronological split, while the remaining 30% is reserved for out-of-sample evaluation. Model parameters are initialized using KMeans clustering before being iteratively estimated using the Baum–Welch algorithm. Training continues until the change in log-likelihood falls below the convergence tolerance of $10^{−5}$, or until the maximum of 100 iterations is reached.\n",
    "\n",
    "This specification serves as the baseline model. The number of states, covariance structure, and feature specification will be evaluated and compared in later sections rather than assumed to be optimal."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c8d2cfdb-e9f2-49a7-af62-e4038af4e602",
   "metadata": {},
   "outputs": [],
   "source": [
    "HMM = hmm(feature_matrix = baseline_X_train, states = 3, covariance_type = \"full\").fit()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "245c8e25-9628-46b0-bbc1-7ce48c45db06",
   "metadata": {},
   "source": [
    "### 3.1 Convergence"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "f79474ee-5ada-4cbb-97c3-676b61c7ff11",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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bF9OeYyRd9iEvMMWZB02Ie+wCXl3F0UpxOyNXlBQQFEqykoqquTO1+u77WWkJsuFB+1spjhFbJF32IS8wxZkHTYh77AJeXcXRSnAR+55e0FDt71zpu0keVlpbu8SAAAAAABhJYtPlJeXy7fffit5eXkybFjTs4ytWrVKioqKZPDgwZKRkRH28RUrVjRaP2rUKMnJyWm0fvPmzbJkyRIzXlL//v2bfN3Wbge01yc/rZf3f9jebW9oD/nVzn04mQAAAACAuOX5llKLFi2SM888UwYNGiT77befXH/99WG3u/vuu00havz48XLyySdL9+7d5aKLLpKqqqqg7R555BHZf//95fLLLw+6zZ8/v9Fz3nDDDdK7d28566yzZOzYsXLAAQdISUlJu7fzO7qRRU9JeZXc9fo8s9wlLVkuOmgM59tl5B02Ie+wBVmHTcg7bELe3eP5opQWi3bffXdTnNKiU1OuueYaOf/882X16tXy/fffy4cffij333+/XHfddY227dOnj3z00UdBt7322itom+eee06uvfZaefPNN+Wrr76Sn3/+WZYtWybnnntuu7azYUaDnj17MvtelNz99nwpKvtft73cTLrtuYm8wybkHbYg67AJeYdNyLu7PF+UOuKII0xLqaysrGa3e+yxx0zLqKSkJPOzFrIOPPBAeeWVV9r1uv/85z9l2rRpss8++5iftWvfJZdcIk8++aTk5+e3eTsbBo+rrKxkoPMo+Hjhepk5b61ZnjC0h+w3hm57biPvsAl5hy3IOmxC3mET8u4uzxelWku7zIXSsaW6du3aaH1tba3MmzfPtKjSsapC1dTUyGeffSYTJ04MWj9p0iTz2Oeff96m7Wx5o+v51ntETrF223ujfra9Lmmd5AK67cUF8g6bkHfYgqzDJuQdNiHv7vLNQOdtpd33Zs6cKbfcckujx5YvXy5HHnmkCacun3HGGXLbbbdJly5dzOMFBQWm1U+vXr2Cfs/5ec2aNW3aLhz9Pb05nDGotGCmN6ffq950PwOLPS2td36/veu1eWPoc7d2fUf3PR6PqaP73pFj+veb82RzWf24aOdMGyndMlIaHvfqMTW33ivHpALz7odj8uN14pgic52ayzvXiez56TNCl53f9csxNbeeY7L7OgXm3S/H1JH1HJO/r1NzeffqMcXLdfJcUWrlypXm1py0tDTT9a4j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J+9kj/dTaz7HvPq5p/TzPXDGSa8fkx+vE8fU8eukv6dfKjt16tSh68F14v3khc+IwLw7P3v9mDqynmPy93XS59JilDOOtB+OKR6ukxa0u3btau6dGkbct5S68MILg4o34bRnLCidAU8/VA866KCGdU5ro759+8r06dPlb3/7m/k5tCClJk6cKI899pgpMvXq1ctUrrt06SKrVq0K2m716tXmfsCAAea+tduFoxVdp7VFaEhCx9FyLnqoptY3NQ5XW9a39TWVFtS0SBf4fM6YUnmZ9f8S46VjivZ6jsm710n/YxAu714+prau55jsuU6ad/2XyXB59+oxRXI9x+Sf69Rc1r16TC2t55jsvU6heffDMXVkPcfk7+sU+re7H44pXq5Ta8RVUUqb3IV2dYuEE044QQ4++OCgdffee69ceeWV8t1335nqXXN+/PFH05pKm6w5J3efffaRmTNnBm33wQcfmO0mTJjQpu1sVlBa2VCUAgAAAAAA9vD87HutoS2OtNVS4M0Z10n7kzrLSltNaaFKW1Zp95tHHnlE/vOf/8hZZ50V1HLpsssuk08++UT++9//NrTGuu222+T88883raPaup2NqmtqZXNZfVEql6IUAAAAAABW8XxR6uOPP24oNC1cuFCee+65hp8DB+FqrSOPPNIUoHTALh2c64YbbjBd+0LHutp7773NIOg6GLkWtXSsKm2RddNNN7VrOxuENt8r3FIpTs/TvCxaSsFfWttcFfAD8g5bkHXYhLzDJuTdPXE10Hl76OCqOnBWOFqYaoqOKaWjxze1jT6vBlMHKm5JaWmpKTg11a+zrduFowOda6GspUHCvGTB6iK5+KFPzPL1x+8uewzt6fYuAQAAAACADmptDSOuxpRqD50RorniU1N0TKfAWffCPW9raYuqSG7nR1r71O6QOqOBU4UuKKkf5FwxphT8nnfAr8g7bEHWYRPyDpuQd3d5vvsevPNG1xkNAhvmbdo+857Ky0p3ac+A2OQd8CvyDluQddiEvMMm5N1dFKXgmoLtRalOSYmSld76lmkAAAAAAMD7KErBNfnbu+/lZqbSxQkAAAAAAMtQlELMhA4a77SUys1k5j34T2smSQD8grzDFmQdNiHvsAl5dw/fmhATOuNg6ID0+duLUgxyDhvyDvgVeYctyDpsQt5hE/LuLlpKIWaDx5WXlzcM/Kz3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gtZPnjzZtHqYO3duVLazic6+BwAAAAAA4KuBzluydOlSM85LTU2NGdvpoosukj/84Q+SnFx/GvLz86WsrEz69esX9Ht9+/Y19ytWrIjKduFo0UpvDqeQplVcvTmDselNi1+BDd5aWu/8fnvXa5/b0Odu7fqO7ns8HlNH951j8t91UoF598Mx8X7iOjWVvebyTvZ4P/npc0+Xnd/1yzE1t55jsvs6BebdL8fUkfUck7+vU3N59+oxxct18lxRSgsvTbVicmgBaYcddmjzcw8ZMkTeeecd00pJT/Rzzz0np556qhlX6p///KfZZuvW+hniQmfHS01NDXo80tuFc/PNN8t1113XaL12+XN+T2ca1OZwes60i6AjIyPDdAssKioKGlBdm8x17txZCgsLpbq6umG9Fuh0n/S5A4OTm5srSUlJphtSIO2SpIU9HSPLoaHTgp++nr5u4PXSGQ51/3QGRH1+Pf96TnJyckzLNS3cObx2TIF55Zi4Tk72OnXqZLLt5J3s8X7y82dE165dZdu2beb5nbx7/Zj8eJ04po5fJ/1DXzOu9/oPj1wnsufnzwjNuX62K78ckx+vE8cUmeukedfz2r17d7Md16kiYtnz3JhSt912m5klrzna6ujTTz9t85hS4VxyySVy9913mz6OGlY9aXqy//Of/5gZ8hzr1q2T3r17y0MPPSTTp0+P+HatbSmlx67P6fTHtLXayjFxncge7yc+I/gs579P/DeXvyP424i/Yfm7nO8afH/ie278fnfXOov+42VLY0rFVUupSy+91NxiZdiwYabwoxXVXr16mUqqVqV//vnnRt3+1NChQ819pLcLR4tkTouq0JA4/xIdetFDNbU+9Pfbs76tr6m0hZdWUwMfb+vzxNMxRXs9x+Td66QfyuHy7uVjaut6jsme66R5138dC5d3rx5TJNdzTP65ToFZ98sxtbSeY7L3OoV+tvvhmDqynmPy93XqSN7j9ZgCuXlMrWHNQOehFUP14YcfmsqdNtNzHHDAAfLGG28EVfpee+0108Rv/PjxUdvO7/T4tbVXHDXMA6KGvMMm5B22IOuwCXmHTci7uzxflNKWTjoulN60/2h5eXnDz4EFkIkTJ8ojjzwiP/zwg3z11Vdy8cUXy5NPPil/+ctfGgY6V1deeaWsXLlSLrzwQlm8eLHZRrsUXn/99WacmGhtBwAAAAAAYJO4GlOqPXR8qWOOOSbsY4sWLTIDnintQqdjVjnjUWnXvQsuuED22muvRr/39ddfy7XXXivz5883g9KdddZZYcd+ivR2zdFWRjqIWEv9MeO5pZoOJKfH31RTQ8AvyDtsQt5hC7IOm5B32IS8R0draxieL0rZwutFKY2ZDtKu42y1tm8p4FXkHTYh77AFWYdNyDtsQt7drWHE1UDn8C8tROlMhIANyDtsQt5hC7IOm5B32IS8u4t+VIhZ9bm0tJSBzmEF8g6bkHfYgqzDJuQdNiHv7qIohZi90cvKyihKwQrkHTYh77AFWYdNyDtsQt7dRVEKAAAAAAAAMUdRCgAAAAAAADFHUQoxGzwuPT2dmfdgBfIOm5B32IKswybkHTYh7+5i9j3E7I2u00ECNiDvsAl5hy3IOmxC3mET8u4uWkohZoPHFRcXM9A5rEDeYRPyDluQddiEvMMm5N1dFKUQszd6RUUFRSlYgbzDJuQdtiDrsAl5h03Iu7soSgEAAAAAACDmGFPKQ9VbVVJSIl5UW1srpaWlkpaWJomJ1ELhb+QdNiHvsAVZh03IO2xC3qPDqV04tYymUJTyCC3oqH79+rm9KwAAAAAAAK2qZTQ36VlCXUtlK8RN9Xbt2rWSmZlpZgfwYpVUC2qrVq2SrKwst3cHiCryDpuQd9iCrMMm5B02Ie/RoaUmLUj17t272d5StJTyCL2Iffv2Fa/TghRFKdiCvMMm5B22IOuwCXmHTch75DXXQsrB4D4AAAAAAACIOYpSAAAAAAAAiDmKUoiJ1NRUueaaa8w94HfkHTYh77AFWYdNyDtsQt7dxUDnAAAAAAAAiDlaSgEAAAAAACDmKEoBAAAAAAAg5ihKAQAAAAAAIOaSY/+SsM2PP/4oL7zwgpSVlcmECRPk0EMPdXuXgA6rq6uTOXPmyGuvvSbdu3eXSy+9tMltX3nlFZk7d65kZGTIkUceKSNHjuQKwFNZnzVrlslwdXW17LLLLnLQQQdJQkJCo203b94sTz75pCxfvlwGDRokJ5xwgmRlZbmy30Bbbdy4UW666SazrPnu2bOn+btl8uTJYbefN2+evPTSS1JRUSETJ0407wvAq5/zN9xwgxQUFMjll18uO+ywQ6PHNetffPGFZGZmylFHHSXDhg1zbX+Btrr//vtl/vz5jdb/5S9/MZkOlJ+fL0899ZSsWrVKhg4dav6W0b/hET20lEJUPfPMM+YLzMqVK82sBmeffbYcf/zx5j9ugFd9//335o+xK6+8Ut544w15/PHHw26nOde8a+41//o+0PeDvi8AL9Ai05gxY+S4446ToqIiqayslLPOOksmTZokJSUlQdtqvnXbp59+Wrp16yaPPfaYjB07VtauXeva/gNt0alTJxk4cKC59e/f32RX/yEt3N8tmu/dd99d1q1bJ8nJyfLb3/5WfvOb33DC4Ul333233HjjjXLnnXeaL+SBamtrzT+oXXDBBZKWliY///yz7LzzzvLyyy+7tr9AW2leP/roo4bPeOeWlJQUtN3ixYtl9OjR5h+U9W+Z++67T3bbbbdG7wtEWB0QJSUlJXXdunWr+/Of/9yw7rvvvqtLSEioe/755znv8KyVK1fWLVmyxCwfd9xxdWPHjg273XPPPWfyrrl36PshJyenrrS0NGb7C7TXqlWr6iZNmlS3adOmoPynp6fXXXfddUHbHn300XU777xzXXV1tfm5qqqqbsSIEXWnnHIKFwCe9dZbb2k1ytw7CgsL6zIzM+uuv/76hnWff/652e711193aU+B9lm0aFFdVlZW3bXXXmsy/MMPPwQ9PmPGjLqkpKS6H3/8sWHdJZdcUtejR4+6iooKTjs84aCDDqo76aSTWtzugAMOqJswYUJdbW2t+bm8vLxuwIABdeecc04M9tJetJRC1Lz99tvmX9bPOOOMhnX6Lyv6L4vavQPwqn79+sngwYNb3E5z/otf/MLk3nH66adLYWGheX8A8S4nJ0fefPNNycvLC8r/kCFD5IcffmhYp92z9V8Vp0+f3vCvjtrq5JRTTpHnn39eqqqqXNl/oKO0+57zr+eO119/XUpLS4P+vtHPem0pyN838JKamhrTwu+cc84xrUHC0Uxr69gRI0YE/S2j3V3ff//9GO4tEF3affWdd94xLV+dIQrS09PlxBNPNN356OkTPRSlEDXffvut6X+74447Bq3XP9q+++47zjyseA9o3gPpODudO3fmPQBP0KyGjrWgRdUlS5aYcRYCxw7UwlNo3vXn8vJysz3gRR988IH5cqJfygM/23Nzc6VXr15B2/L3Dbzmr3/9q/kifu2117bpbxktUOk/PPD3PLxk0aJF8uc//9mMHaj/uBBaZNJ/bNPuquH+ltGGFjrGFKKDgc4RNfrF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",
      "text/plain": [
       "<Figure size 1200x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iterations: 53\n",
      "Initial LL: -14965.288219274345\n",
      "Final LL: -11448.860846663947\n",
      "Final improvement: 8.03523107606452e-06\n",
      "Number of decreases: 0\n"
     ]
    }
   ],
   "source": [
    "plt.figure(figsize=(12, 6))\n",
    "\n",
    "plt.plot(\n",
    "    HMM.log_likelihood_history,\n",
    "    linewidth=1.8,\n",
    "    color=\"steelblue\"\n",
    ")\n",
    "\n",
    "plt.title(\n",
    "    \"HMM Log-Likelihood Convergence\",\n",
    "    fontsize=18,\n",
    "    fontweight=\"bold\"\n",
    ")\n",
    "\n",
    "plt.xlabel(\"Iteration\", fontsize=13)\n",
    "plt.ylabel(\"Log-Likelihood\", fontsize=13)\n",
    "\n",
    "plt.grid(True, linestyle=\"--\", alpha=0.3)\n",
    "plt.tick_params(axis=\"both\", labelsize=11)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "ll = np.array(HMM.log_likelihood_history)\n",
    "\n",
    "print(\"Iterations:\", len(ll))\n",
    "print(\"Initial LL:\", ll[0])\n",
    "print(\"Final LL:\", ll[-1])\n",
    "print(\"Final improvement:\", ll[-1] - ll[-2])\n",
    "print(\"Number of decreases:\", np.sum(np.diff(ll) < 0))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f126d9c5-b903-4701-9c1c-585bc99440b6",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "The HMM shows significant improvement in log-likelihood during training, increasing from approximately -14,965 to -11,449. The largest improvements happens during the first iterations, after which the rate of improvement decreases and the log-likelihood begins to level off after approximately 10 iterations. The final improvement is only $8.035×10^{−6}$, this indicates that further parameter updates have a extremly small effect on the objective function.\n",
    "\n",
    "No decreases in log-likelihood are observed during the training process, which is consistent with the expected behaviour of the Baum–Welch algorithm. Training terminates after 53 iterations when the improvement in log-likelihood falls below the convergence tolerance of $10^{−5}$. These results concludes stable numerical convergence of the baseline HMM however they do not prove that the selected model specification or identified regimes are economically meaningful."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c24dd278-0a64-4038-83fc-d00f711c8a11",
   "metadata": {},
   "source": [
    "### 3.2 State Means"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "b47660e0-1e27-4980-9174-22c32f02c2e8",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "Shape of passed values is (3, 3), indices imply (3, 7)",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mValueError\u001b[39m                                Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[15]\u001b[39m\u001b[32m, line 1\u001b[39m\n\u001b[32m----> \u001b[39m\u001b[32m1\u001b[39m state_means_scaled = pd.DataFrame(\n\u001b[32m      2\u001b[39m     HMM.means,\n\u001b[32m      3\u001b[39m     columns=train_df.columns,\n\u001b[32m      4\u001b[39m     index=[f\"State {i}\" \u001b[38;5;28;01mfor\u001b[39;00m i \u001b[38;5;28;01min\u001b[39;00m range(HMM.states)]\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~/regime-var-engine/regime-switching-var/.venv/lib/python3.13/site-packages/pandas/core/frame.py:814\u001b[39m, in \u001b[36mDataFrame.__init__\u001b[39m\u001b[34m(self, data, index, columns, dtype, copy)\u001b[39m\n\u001b[32m    810\u001b[39m                     dtype=dtype,\n\u001b[32m    811\u001b[39m                     copy=copy,\n\u001b[32m    812\u001b[39m                 )\n\u001b[32m    813\u001b[39m             \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[32m--> \u001b[39m\u001b[32m814\u001b[39m                 mgr = ndarray_to_mgr(\n\u001b[32m    815\u001b[39m                     data,\n\u001b[32m    816\u001b[39m                     index,\n\u001b[32m    817\u001b[39m                     columns,\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~/regime-var-engine/regime-switching-var/.venv/lib/python3.13/site-packages/pandas/core/internals/construction.py:299\u001b[39m, in \u001b[36mndarray_to_mgr\u001b[39m\u001b[34m(values, index, columns, dtype, copy)\u001b[39m\n\u001b[32m    294\u001b[39m \u001b[38;5;66;03m# _prep_ndarraylike ensures that values.ndim == 2 at this point\u001b[39;00m\n\u001b[32m    295\u001b[39m index, columns = _get_axes(\n\u001b[32m    296\u001b[39m     values.shape[\u001b[32m0\u001b[39m], values.shape[\u001b[32m1\u001b[39m], index=index, columns=columns\n\u001b[32m    297\u001b[39m )\n\u001b[32m--> \u001b[39m\u001b[32m299\u001b[39m \u001b[30;43m_check_values_indices_shape_match\u001b[39;49m\u001b[30;43m(\u001b[39;49m\u001b[30;43mvalues\u001b[39;49m\u001b[30;43m,\u001b[39;49m\u001b[30;43m \u001b[39;49m\u001b[30;43mindex\u001b[39;49m\u001b[30;43m,\u001b[39;49m\u001b[30;43m \u001b[39;49m\u001b[30;43mcolumns\u001b[39;49m\u001b[30;43m)\u001b[39;49m\n\u001b[32m    301\u001b[39m values = values.T\n\u001b[32m    303\u001b[39m \u001b[38;5;66;03m# if we don't have a dtype specified, then try to convert objects\u001b[39;00m\n\u001b[32m    304\u001b[39m \u001b[38;5;66;03m# on the entire block; this is to convert if we have datetimelike's\u001b[39;00m\n\u001b[32m    305\u001b[39m \u001b[38;5;66;03m# embedded in an object type\u001b[39;00m\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~/regime-var-engine/regime-switching-var/.venv/lib/python3.13/site-packages/pandas/core/internals/construction.py:372\u001b[39m, in \u001b[36m_check_values_indices_shape_match\u001b[39m\u001b[34m(values, index, columns)\u001b[39m\n\u001b[32m    370\u001b[39m passed = values.shape\n\u001b[32m    371\u001b[39m implied = (\u001b[38;5;28mlen\u001b[39m(index), \u001b[38;5;28mlen\u001b[39m(columns))\n\u001b[32m--> \u001b[39m\u001b[32m372\u001b[39m \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[33mf\u001b[39m\u001b[33m\"\u001b[39m\u001b[33mShape of passed values is \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mpassed\u001b[38;5;132;01m}\u001b[39;00m\u001b[33m, indices imply \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mimplied\u001b[38;5;132;01m}\u001b[39;00m\u001b[33m\"\u001b[39m)\n",
      "\u001b[31mValueError\u001b[39m: Shape of passed values is (3, 3), indices imply (3, 7)"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "state_means_scaled = pd.DataFrame(\n",
    "    HMM.means,\n",
    "    columns=hmm_features.columns,\n",
    "    index=[f\"State {i}\" for i in range(HMM.states)]\n",
    ")\n",
    "\n",
    "plt.figure(figsize=(10, 4))\n",
    "\n",
    "sns.heatmap(\n",
    "    state_means_scaled,\n",
    "    annot=True,\n",
    "    fmt=\".2f\",\n",
    "    cmap=\"RdBu_r\",\n",
    "    center=0,\n",
    "    linewidths=0.5\n",
    ")\n",
    "\n",
    "plt.title(\"Standardized Feature Means by Hidden State\",\n",
    "          fontsize=16,\n",
    "          fontweight=\"bold\")\n",
    "plt.xlabel(\"Feature\")\n",
    "plt.ylabel(\"Hidden State\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bba7bd39-5089-424b-b264-3d10a44ecfdc",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "State 0 exhibits returns close to the training mean, moderately positive momentum, and substantially lower conditional volatility. This suggests that State 0 is associated with periods of relatively low volatility and positive medium-term price trends. State 1 also exhibits daily returns close to the mean, but is characterized by negative momentum and substantially elevated conditional volatility. Interestingly, the relatively small difference in mean daily returns between States 0 and 1 suggests that daily returns alone provide limited separation between the regimes, whereas their behaviour over the 20-day momentum window differs more clearly.\n",
    "\n",
    "State 2 is less distinctive, with all three feature means relatively close to the training average, potentially representing more intermediate market conditions. Overall conditional volatility exhibits the greatest separation across state means, then momentum and daily returns differ only slightly. Based on these characteristics, State 0 may represent a low-volatility positive-trend regime, State 1 a high-volatility negative-trend regime, and State 2 an intermediate regime. These interpretations remain preliminary until the states are examined over time."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e538168e-1d96-41ce-acdf-4f9ffd107456",
   "metadata": {},
   "source": [
    "### 3.3 Original-scale State Means"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 193,
   "id": "692c2dd8-fa6f-40ee-bb8b-5b5c1d226743",
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "</style>\n",
       "<table id=\"T_bc404\">\n",
       "  <caption>Original-scale State Means</caption>\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th id=\"T_bc404_level0_col0\" class=\"col_heading level0 col0\" >Returns</th>\n",
       "      <th id=\"T_bc404_level0_col1\" class=\"col_heading level0 col1\" >Momentum_20</th>\n",
       "      <th id=\"T_bc404_level0_col2\" class=\"col_heading level0 col2\" >Conditional_Volatility</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_bc404_level0_row0\" class=\"row_heading level0 row0\" >State 0</th>\n",
       "      <td id=\"T_bc404_row0_col0\" class=\"data row0 col0\" >0.000608</td>\n",
       "      <td id=\"T_bc404_row0_col1\" class=\"data row0 col1\" >0.018611</td>\n",
       "      <td id=\"T_bc404_row0_col2\" class=\"data row0 col2\" >0.006641</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_bc404_level0_row1\" class=\"row_heading level0 row1\" >State 1</th>\n",
       "      <td id=\"T_bc404_row1_col0\" class=\"data row1 col0\" >-0.000353</td>\n",
       "      <td id=\"T_bc404_row1_col1\" class=\"data row1 col1\" >-0.025709</td>\n",
       "      <td id=\"T_bc404_row1_col2\" class=\"data row1 col2\" >0.021962</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_bc404_level0_row2\" class=\"row_heading level0 row2\" >State 2</th>\n",
       "      <td id=\"T_bc404_row2_col0\" class=\"data row2 col0\" >-0.000172</td>\n",
       "      <td id=\"T_bc404_row2_col1\" class=\"data row2 col1\" >-0.004369</td>\n",
       "      <td id=\"T_bc404_row2_col2\" class=\"data row2 col2\" >0.011247</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x146754cd0>"
      ]
     },
     "execution_count": 193,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "state_means_original = pd.DataFrame(\n",
    "    scaler.inverse_transform(HMM.means),\n",
    "    columns=hmm_features.columns,\n",
    "    index=[f\"State {i}\" for i in range(HMM.states)]\n",
    ")\n",
    "\n",
    "state_means_original.style.set_caption(\"Original-scale State Means\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "62cdc93c-73f8-4e14-8c67-e279088d9e27",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "Transforming the state means back to their original scale improves their economic interpretability. The original-scale feature means are consistent with the observations in the standardized heatmap. State 0 exhibits small positive daily returns, positive 20-day momentum, and relatively low conditional volatility. State 1 exhibits small negative daily returns, more distinct negative 20-day momentum, and substantially higher conditional volatility. State 2 exhibits daily returns close to zero and relatively weak momentum, while conditional volatility is higher than in State 0 but substantially lower than in State 1.\n",
    "\n",
    "Overall, the original-scale representation provides a more intuitive economic interpretation of the differences between the identified regimes, while supporting the same general patterns observed in the standardized state means."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9ca49de8-9596-451a-bac0-1dda7125982c",
   "metadata": {},
   "source": [
    "### 3.4 Transition Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "id": "ff4e93e3-1175-486e-b878-78654f416224",
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>To State 0</th>\n",
       "      <th>To State 1</th>\n",
       "      <th>To State 2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>From State 0</th>\n",
       "      <td>0.9812</td>\n",
       "      <td>0.0001</td>\n",
       "      <td>0.0187</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>From State 1</th>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.9732</td>\n",
       "      <td>0.0268</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>From State 2</th>\n",
       "      <td>0.0214</td>\n",
       "      <td>0.0092</td>\n",
       "      <td>0.9694</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              To State 0  To State 1  To State 2\n",
       "From State 0      0.9812      0.0001      0.0187\n",
       "From State 1      0.0000      0.9732      0.0268\n",
       "From State 2      0.0214      0.0092      0.9694"
      ]
     },
     "execution_count": 194,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "transition_matrix = pd.DataFrame(\n",
    "    HMM.transition_matrix,\n",
    "    index=[f\"From State {i}\" for i in range(HMM.states)],\n",
    "    columns=[f\"To State {i}\" for i in range(HMM.states)]\n",
    ")\n",
    "\n",
    "transition_matrix.style.set_caption(\"Transition Matrix\")\n",
    "transition_matrix.round(4)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "24b20a49-8993-440e-80f2-63e75e5f4f9a",
   "metadata": {},
   "source": [
    "#### Discussion \n",
    "\n",
    "The transition matrix indicates that all three states are highly persist, with relativly low probabilities of switching regime. Both State 0 and State 1 have substanially higher probabilities of switching to State 2 than directly to each other. The probabilities of direct transitions bewteen State 1 and State 0 is extremely low. This supports the initial hypothesis that State 2 acts as an intermediate regime between the low-volatility, positive-trend conditions of State 0 and the high-volatility, negative-trend conditions of State 1.  When leaving State 2, the model is more likely to transition to State 0 than to State 1. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "40f75703-e011-4549-8799-72227df8d269",
   "metadata": {},
   "source": [
    "### 3.5 Expected State Duration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "id": "49ee7246-a20e-41bf-b825-00bdbc37ef2b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Expected Duration (Trading Days)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>State 0</th>\n",
       "      <td>53.17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State 1</th>\n",
       "      <td>37.25</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State 2</th>\n",
       "      <td>32.73</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Expected Duration (Trading Days)\n",
       "State 0                             53.17\n",
       "State 1                             37.25\n",
       "State 2                             32.73"
      ]
     },
     "execution_count": 195,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "expected_durations = 1 / (1 - np.diag(HMM.transition_matrix))\n",
    "\n",
    "duration_df = pd.DataFrame(\n",
    "    {\n",
    "        \"Expected Duration (Trading Days)\": expected_durations\n",
    "    },\n",
    "    index=[f\"State {i}\" for i in range(HMM.states)]\n",
    ")\n",
    "\n",
    "duration_df.round(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cc6789c2-60f7-448f-ad40-7efc3a39234d",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "The expected state durations further support the high regime persistence observed in the transition matrix. State 0 has the longest expected duration at approximately 53 trading days, followed by State 1 at 37 days and State 2 at 33 days. State 0 therefore appears substantially more persistent, while States 1 and 2 have relatively similar expected durations. The shorter duration of State 2, combined with the transition patterns observed previously, provides further support for its preliminary interpretation as an intermediate regime between States 0 and 1."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a411643c-4e5b-4fc2-9544-a315d8d759c4",
   "metadata": {},
   "source": [
    "### 3.6 State Occupancy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "id": "8f447030-a38a-4414-bf3e-6f1e22984083",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Occupancy</th>\n",
       "      <th>Occupancy (%)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>State 0</th>\n",
       "      <td>0.45</td>\n",
       "      <td>45.16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State 1</th>\n",
       "      <td>0.14</td>\n",
       "      <td>14.15</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State 2</th>\n",
       "      <td>0.41</td>\n",
       "      <td>40.70</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Occupancy  Occupancy (%)\n",
       "State 0       0.45          45.16\n",
       "State 1       0.14          14.15\n",
       "State 2       0.41          40.70"
      ]
     },
     "execution_count": 196,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "smooth_probs = HMM.smooth_proba(baseline_X_train)\n",
    "\n",
    "state_occupancy = smooth_probs.mean(axis=0)\n",
    "\n",
    "occupancy_df = pd.DataFrame(\n",
    "    {\"Occupancy\": state_occupancy},\n",
    "    index=[f\"State {i}\" for i in range(HMM.states)]\n",
    ")\n",
    "\n",
    "occupancy_df[\"Occupancy (%)\"] = occupancy_df[\"Occupancy\"] * 100\n",
    "\n",
    "occupancy_df.round(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6a776543-5b7e-4da8-a7f7-4b41b93ff502",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "The smoothed state probabilities indicate that State 0 has the highest occupancy with 45.15%,  followed by State 2 with 40.70%. State 1 has a lower occupancy of 14,15%. From an economical perspective, the lower frequency of State 1 is consistent with its previously identified characteristics of elevated volatility and negative momentum, which may represent less common periods of market stress.\n",
    "\n",
    "The relatively high occupancy of State 2 challenges the initial interpretation of it as simply an intermediate regime between States 0 and 1. Although the transition matrix suggests that State 2 facilitates transitions between the two more distinct regimes, its high occupancy indicates that it may also represent a persistent and economically relevant market environment itself."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "db1b06eb-bc28-41e8-a3cc-ca6b84c48bac",
   "metadata": {},
   "source": [
    "## 4. Regime Analysis Over Time\n",
    "\n",
    "The previous analysis characterized the hidden states through their feature means, transition probabilities, expected durations, and state occupancy. These statistics do not reveal when the identified regimes occur or how they evolve through time. This section will examine the estimated state probabilities and regime classifications across the historical market data. The objective is to evaluate whether the identified states correspond to economically meaningful market periods and to examine how confidently the model distinguishes between regimes."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a01ed837-6bd0-425b-b3f6-0d49e7a31567",
   "metadata": {},
   "source": [
    "### 4.1 Smoothed State Probabilities"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 197,
   "id": "b317c0a6-06b3-497f-88cc-c57bee37d07e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABW4AAAHqCAYAAACUWtfDAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlcelbwAAAAlwSFlzAAAPYQAAD2EBqD+naQABAABJREFUeJzsfQecHbXx/+y96+e7c+82YDAd03sPnSSUhEBIIISSAumV8APCLwlJSKOGNMovJOQPAZKASULvvfeOjSvuZ9/Z5+v3/p/Rrna1Wu2utO3tvtM3Me/ePq00kkaj0Wg0Msrlchk0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDRyg5pKE6ChoaGhoaGhoaGhoaGhoaGhoaGhoeGGNtxqaGhoaGhoaGhoaGhoaGhoaGhoaOQM2nCroaGhoaGhoaGhoaGhoaGhoaGhoZEzaMOthoaGhoaGhoaGhoaGhoaGhoaGhkbOoA23GhoaGhoaGhoaGhoaGhoaGhoaGho5gzbcamhoaGhoaGhoaGhoaGhoaGhoaGjkDNpwq6GhoaGhoaGhoaGhoaGhoaGhoaGRM2jDrYaGhoaGhoaGhoaGhoaGhoaGhoZGzqANtxoaGhoaGhoaGhoaGhoaGhoaGhoaOYM23GpoaGhoaGhULZYvXw6GYbj+3X333TAS69fX1weXXnop7L333jBmzBgolUr2O6NHj4a3337bk9fDDz8MRUa193+1IUse3WuvvVz5nHXWWYnXR0MjLWjZpqGhoTFyoA23GhoaGhoE06dPdy1id9ppJ9+Wuf/++z2L58svvzw0T/rv9ttvD2z1XXfdVfjetddeW5EyVPDII4/AGWecAdtttx20t7dDXV0djBs3DrbYYgtiKDjzzDPhiiuugGeeeUb4/nHHHeeiB7+njUqUGQVPPPEEfO9734N99tkHpk2bBo2NjeTflClT4MADD4TzzjsPnnrqqUqTmUsMDw/DUUcdBd/5znfg6aefhnXr1pFnRURR+DVLVMPYqCYe1QDo7++HG2+8EU499VTYeuutyTxYX18PEyZMgB122AG+8IUvwL/+9a8R1cfI0yK9Q+XfoYceWulqaGhoaGhkjNqsC9TQ0NDQ0PjNb37ja2x58MEH4cUXXyxEGSwGBgbg9NNPh7/97W+e3zo6Osi/efPm2QZbNBw///zzidJQrXjttdfgi1/8IjHm+Hke4b9HH30ULrnkEjj//PPh4osvzpzOPAO9TB966KFKk6GRMKppbGgerR784x//gG984xuwdOlSz2+rV68m/15//XWyUTp79mzyecABB1SEVg0NDQ0NjbxDG241NDQ0NDLH448/Ds899xzsvvvuQoNrUcpg8e1vf1totNWIh1tuuQVOO+006O3tlX5nw4YNI6bZJ0+eDOVyOTTdSy+95Pre1NREnqHRpKbGOYAlk1c1tk8RUW1jY6TyaLXhxz/+MVx00UXS6d977z34yEc+Atdccw3Z/NSQQzXLNg0NDQ0NN3SoBA0NDQ2NigDjGPJ466234K677ipUGYg1a9bA73//e9czPPKLR5PxaOTGjRtJuVdffbXQkKwhBnoJ4jFb3jB15JFHEu88bHeMiblo0SL4f//v/8Ehhxyim9IHPT09ru9jx46FrbbaymUQ0ygOqnFsaB4tPm666SaP0XbUqFHw61//mvAi8iQaai+88EISNoFiaGgIvvSlL8Fjjz0G1QyM04zGVv4fntjhceyxxwrTYqgqDQ0NDY0RhrKGhoaGhka5XJ42bRq6btj/dtxxR992ue+++1xp8d9ll10Wmmd9fb39d21tbXnhwoWu9GeeeaYwLf13zTXXVKSMMMydO9dDQ3d3t2/6Bx98kNDBYvPNN/fQ4vfvpZdest9bvHhx+YYbbih/7nOfK8+ZM6c8evTocqlUKjc3N5enT59ePvLII8tXXXVVubOz00NH1DIp3njjjfI3vvGN8s4771weM2ZMua6urjx+/PjyfvvtV/7Zz35W7ujoKEfF4OBgeeutt/bQccEFFwS+d88995R/9atf2d+XLVvmyeOuu+7yvBenHSl6enrKV199dfnwww8nfNnQ0FBuamoqz5o1i7TJV7/61fI//vGPcldXV2LvhtXvpJNOku5j0b+HHnrIt77z588v/8///E953333LU+YMIH0/8SJE8u77757+bzzziu/+eabibZzFH6V7X+KF198sXzOOefYtKEMGTduXHnPPfckdeLlSVg/rF69unzuuecSXm5sbCy3tbWVDzjgANKXlR4beal3Wjy6fv368oUXXljeaqutyHgaO3Zs+ZBDDinffPPN5HesG5sPL5OTkHVp8EWUcRe3HjLYuHFjecqUKa66Ih89/vjjwvS33Xabp22QruHhYfL75z//eddv2Hd+QL5l037r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/9ZbbzXbrsbkiCOOkOeff77KfAsWLJCTTjrJxFb3F7fffnuV/Xci80ybNs28hx4H2rdvL+eff74sW7Ys7euJ+L7//nsZMWKE7LffftK2bVuzfep2Gm3mzJnSo0cPs4/Xe/1/pPLychPfPn36SGlpqUyePDnm+9X0Osgs3Xc/++yzJm66D+jatavcdNNNsmXLlkrzbd682RwfNEdatWol5513nqxatarSPIsXLzbn+no+0bRp05TfC5m1YcMGueOOO+Tggw82++bDDjtMnn766Srz6TFdj+26j9djvR7zowf0/uijj+QPf/iDNG7cWM4444xq31evA/UaVM8H9ToR2fPpp5/K73//e9lrr73Mtfo555wjixYtqjKf7tf3339/s3336tXLxDvSpk2bzDy6b9e4vvnmm3Fjf+6550qbNm3M+cKdd95JDmSRnqu98MILcuyxx5p9s8Z4zJgxJp6R9Hper/H1Wl/n02t/rQFE+vnnn81+XXNJcyDe9afuH/R19HhyzDHHyFtvvZXWdUQCdHAyeM/nn38e6tevX+ill14K/fzzz6EPPvgg1KVLl9AxxxxTab7Ro0eHmjdvHnrjjTdC3377beiEE04I/epXvwpt3brV9XmQWRMmTAj96U9/Cr377ruhZcuWhZ5//vlQSUlJ6Pbbbw/Ps2PHjlDPnj1Dv/nNb0Jff/116L333gvtueeeoQsuuCCpeRJ5L2Sebu+PPvpoaMGCBaGFCxeGhgwZEmrSpEnof//7X3iet956K5Sfnx+67777Qj/99FPo1ltvDdWpUyf02WefJTWPQ3OkTZs2oYMPPjjUv3//jK0rqpo/f37o2GOPDf373/82Mf/4449D3bp1Cx111FGV5vvLX/4Saty4ceg///lP6LvvvgsNGDAgtPfee4c2b96c1Dz6+rrdX3HFFaEff/wx9P3334fOO++80JIlSwhPluix+frrrw998sknZt/80EMPhXJzc0PTpk0Lz7NhwwazzQ4ePNjETM8b6tevX2n/ncg8//d//xfKyckJ/e1vfwstX7489MUXX4R69OhRJd+QWccff3xo4sSJoW+++cZsl3/4wx9CDRs2DC1atCg8z/vvv2/28XfffbfZx991113m/zrdoa8xcODA0Kuvvmq2c82laIm8DjJr1qxZoTPPPDP0+uuvm32Anqe3bdvW7Jsj6flBhw4d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",
      "text/plain": [
       "<Figure size 1400x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(14, 5))\n",
    "\n",
    "plt.plot(\n",
    "    train_features.index,\n",
    "    max_probs,\n",
    "    color=\"steelblue\",\n",
    "    linewidth=0.9\n",
    ")\n",
    "\n",
    "plt.axhline(\n",
    "    0.90,\n",
    "    color=\"darkred\",\n",
    "    linestyle=\"--\",\n",
    "    alpha=0.7,\n",
    "    label=\"90% confidence\"\n",
    ")\n",
    "\n",
    "plt.title(\n",
    "    \"HMM State Classification Confidence Over Time\",\n",
    "    fontsize=18,\n",
    "    fontweight=\"bold\"\n",
    ")\n",
    "\n",
    "plt.xlabel(\"Date\", fontsize=13)\n",
    "plt.ylabel(\"Maximum Smoothed State Probability\", fontsize=13)\n",
    "\n",
    "plt.ylim(0, 1.02)\n",
    "plt.grid(True, linestyle=\"--\", alpha=0.3)\n",
    "plt.legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d1a67b3c-aa1a-4a18-8f31-4932f6fe079a",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "The model is generally confident in its regime classifications with a mean of 98 percent condfidence. The maximum probability exceeds 99% for approximately 87% of the observations, indicating that the model typically distinguishes clearly between the identified regimes. However, the mean confidence on regime-switching days decreases considerably to approximately 73%. This suggests more uncertainty around regime transitions, where observations may share characteristics with multiple states. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3a69ac93-2361-4d20-a1b0-1292e244e943",
   "metadata": {},
   "source": [
    "### 4.2 Historical Regime Classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 198,
   "id": "212ba678-fe76-4c3d-af9f-e0b5616956ff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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/fbbpW3bth5lUlJSZM6cOeb/f5EiRWTAgAEyePBgv8to29AyP/74o+l79Lz36NEj12PMrW4dT2mflh3tj/XvQmjGWvr6v/fee/J///d/0qBBAxkzZozUq1fPo5zep2Odb775RipUqCAjRoyQjh07ZhkP6XVJ+6VevXqZ/sibk3qyE6jnR/7Ytm2bLFmyxFxjKlWqJDfffLNcf/31Wdres88+K++//775vXv37mZ85D7u1viFtk+9zhUqVMiM37w5qSc7uT3u888/l/Hjx2f7WH2cd3wCUcyKYNddd5115513Wm+88Yb14YcfWl27drUqVapkHTp0yFXm4MGDVvny5a0BAwZYq1evtkaOHGkVLVrU+vbbb/0q88MPP1hffPGF62vDhg1WkSJFrPHjx/s8Rid1//vf/7bq1atnTZ8+3ZTV7wiNuXPnWh07drReeOEF69NPP7UeffRRq2DBgtY777zjUa5bt25WgwYNrLffftt67rnnrPj4eGvWrFl+lTly5IhHm9Kvq6++2rSFc+fO+TzOYNaN4Bk0aJB1yy23WK+//rr18ccfWzfeeKNVokQJa9euXa4yp06dsmrVqmX6sw8++MB66KGHrEKFClnr16/3q8y6deusmJgY68EHHzQ/L1myxCpTpox17733+jxGJ3Vre6pZs6Y1adIkq379+taYMWMC/lrBOe2nunfvbr300kvW2rVrzTn2PmeZmZlW69atrcsvv9x69913raefftqKjY01j/GnzP/+9z+rWLFi1q233mqea/ny5VadOnWsHj16+DxGJ3Xv378/S7/VtGlTq1WrVjSHENLrzfDhw81Ya82aNaatVahQwTpw4ICrzOHDh62KFSta/fr1M2Odu+++24yRtm3b5leZhQsXmttmz55t2td9991nlS5d2vrll198HqOTurU/q1u3rjVt2jQzVnziiScC/lrBmWeeecZq3769tXjxYnOeJ0+ebK5X2sbc9erVy1xj3nzzTev555+3ihcvnmWMnFuZ3bt3WwkJCdaQIUOsTz75xJTRa6F735OT3Oo+evRolj5Lx5B6DWWsFTrDhg2zbrrpJuu1114zY63Bgweb69bOnTtdZc6ePWtddNFFVqdOnaz333/f1Qa1j7Np/1G9enUzDrrkkkvMe05vTurJTqCeH/lDxy0tWrSwFixYYPqsp556ylxjdCzjTt/n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BhcsWGD2JLSTjujegh988IGZGZgTDTJq0pIXXnhBihcvzlkDAAAIoIKBrAwAAADwNnnyZElNTZWhQ4ea348cOSJFixb1mDGoy4e9aSAwJSXF9bvud6gzCZ9//nmPcpmZmXL06FFeeAAAgPPATEIAAAAEzO7du2X58uWSkZEhhw4dknfffVd27dplMhzXrl3blGndurUJAPbv31+6d+8uW7dulTfffDPbZCcvvvii2aNQg4qa3ViTouhSZa2zVatWcuDAAZMVWROfaF0AAADIG7IbAwAAICAefPBBs2+giouLM3sNNm/eXHr27CkJCQkeZX/99VdZuHChmVV48cUXS69evUwm4/nz50v58uVNmePHj8uzzz5rliXrMuSlS5ea27dv326yJB8+fFjq1KljEpZoHQAAAMg7goQAAAAAAABAlGNPQgAAAAAAACDKESQEAAAAAAAAohxBQgAAAAAAACDKESQEAAAAAAAAohxBQgAAAAAAACDKESQEAAAAAAAAohxBQgAAAAAAACDKESQEAAAAAAAAohxBQgAAAAAAACDKESQEAAAAAAAAohxBQgAAAAAAACDKESQEAAAAAAAAJLr9P6A2g2VLjvvFAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1300x800 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from matplotlib.patches import Patch\n",
    "\n",
    "smoothed_states = np.argmax(HMM.smooth_proba(baseline_X_train),axis=1)\n",
    "\n",
    "state_colors = {0: \"seagreen\",1: \"firebrick\",2: \"steelblue\"}\n",
    "fig, axes = plt.subplots(2, 1, figsize=(13, 8))\n",
    "\n",
    "\n",
    "def add_regime_background(ax, dates, states):\n",
    "    current_state = states[0]\n",
    "    start = 0\n",
    "\n",
    "    for i in range(1, len(states)):\n",
    "        if states[i] != current_state:\n",
    "            ax.axvspan(\n",
    "                dates[start],\n",
    "                dates[i],\n",
    "                color=state_colors[current_state],\n",
    "                alpha=0.20\n",
    "            )\n",
    "\n",
    "            current_state = states[i]\n",
    "            start = i\n",
    "\n",
    "    ax.axvspan(\n",
    "        dates[start],\n",
    "        dates[-1],\n",
    "        color=state_colors[current_state],\n",
    "        alpha=0.20\n",
    "    )\n",
    "\n",
    "\n",
    "# --------------------------------------------------\n",
    "# Plot 1 — Full training period\n",
    "# --------------------------------------------------\n",
    "\n",
    "axes[0].plot(dates, sp_price_train, color=\"black\", linewidth=1.3)\n",
    "add_regime_background(axes[0], dates, smoothed_states)\n",
    "\n",
    "axes[0].set_title(\"S&P 500 with HMM Market Regimes — Training Period\",fontsize=14,fontweight=\"bold\")\n",
    "axes[0].set_ylabel(\"S&P 500 Index\")\n",
    "axes[0].grid(True, linestyle=\"--\", alpha=0.25)\n",
    "\n",
    "# --------------------------------------------------\n",
    "# Plot 2 — Financial Crisis\n",
    "# --------------------------------------------------\n",
    "\n",
    "axes[1].plot(dates, sp_price_train, color=\"black\", linewidth=1.3)\n",
    "add_regime_background(axes[1], dates, smoothed_states)\n",
    "axes[1].set_xlim(pd.Timestamp(\"2007-01-01\"), pd.Timestamp(\"2010-12-31\"))\n",
    "axes[1].set_title(\"HMM Regime Classification During the Global Financial Crisis\", fontsize=14, fontweight=\"bold\")\n",
    "\n",
    "axes[1].set_xlabel(\"Date\")\n",
    "axes[1].set_ylabel(\"S&P 500 Index\")\n",
    "axes[1].grid(True, linestyle=\"--\", alpha=0.25)\n",
    "\n",
    "plt.tight_layout()\n",
    "\n",
    "legend_elements = [\n",
    "    Patch(facecolor=\"seagreen\", alpha=0.3, label=\"State 0\"),\n",
    "    Patch(facecolor=\"firebrick\", alpha=0.3, label=\"State 1\"),\n",
    "    Patch(facecolor=\"steelblue\", alpha=0.3, label=\"State 2\")\n",
    "]\n",
    "\n",
    "axes[0].legend(\n",
    "    handles=legend_elements,\n",
    "    loc=\"upper left\",\n",
    "    frameon=True\n",
    ")\n",
    "\n",
    "axes[1].legend(\n",
    "    handles=legend_elements,\n",
    "    loc=\"upper left\",\n",
    "    frameon=True\n",
    ")\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "752de11d-8897-4062-b0fc-71428028998f",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "The historical classification supports the previous interpretation of the hidden states. State 0 primarily occurs during periods of stable market growth, while State 1 is associated with declining markets and elevated stress. State 2 appears more frequently during moderate or transitional market conditions.\n",
    "\n",
    "The 2008 Financial Crisis provides a clear example. During 2007 and early 2008, the model primarily switches between States 0 and 2. Later into 2008 and beginning of 2009, after the colapse of Bear Sterns and Lehman Brothers, State 1 becomes dominant. The model does not identify the crisis regime at its earliest stage, however this is reasonable since the HMM identifies changes in observed market characteristics rather than predicting future regimes. While the historical classification is economically credible an out-of-sample evaluation is required to decide whether these patterns generalize."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "10e9ba52-a0e3-4ca3-8625-06f5a2dd39cc",
   "metadata": {},
   "source": [
    "### 4.3 Smoothed vs Viterbi Classification\n",
    "\n",
    "The smoothed classification selects the most probable state for each individual day, while Viterbi identifies the most probable state sequence over the entire period. Comparing the two shows whether the choice of classification method significantly affects the identified market regimes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 199,
   "id": "54dc4af6-0ca8-44fa-a77b-4352c3d460ca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Agreement: 99.06%\n"
     ]
    }
   ],
   "source": [
    "smoothed_states = np.argmax(HMM.smooth_proba(baseline_X_train), axis=1)\n",
    "\n",
    "viterbi_states = HMM.decode_viterbi(baseline_X_train)\n",
    "agreement = np.mean(smoothed_states == viterbi_states)\n",
    "\n",
    "print(f\"Agreement: {agreement:.2%}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "81f1c82d-c2c0-4172-b9a3-5d2dfab91dde",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "The two methods agree on 99.06% of the state classifications, showing that they produce almost identical regime sequences. This is consistent with the previously observed high classification confidence. However, this does not show how well the model performs on unseen data, which will be evaluated out-of-sample."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e2440346-a744-4cd0-9024-9413f7ea17f9",
   "metadata": {},
   "source": [
    "## 5. Out-of-Sample Evaluation\n",
    "\n",
    "The model has so far been evaluated using the same data on which it was trained on. To examine whether the identified regimes generalize to unseen market data, the fitted HMM is now evaluated on the remaining 30% of the dataset. The model parameters will remain during the evaluation, allowing the out-of-sample regime classifications to be compared with the characteristics identified during training."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "591f9bfb-8dac-4949-9e76-e2af33e49af4",
   "metadata": {},
   "source": [
    "### 5.1 Out-of-Sample Regime Classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 200,
   "id": "83acb0c9-a3bd-4c41-9f03-ded45e81c252",
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 1300x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "test_smoothed_probs = HMM.smooth_proba(baseline_X_test)\n",
    "test_smoothed_states = np.argmax(test_smoothed_probs, axis=1)\n",
    "fig, ax = plt.subplots(figsize=(13, 5))\n",
    "\n",
    "# S&P 500 price\n",
    "ax.plot(\n",
    "    test_df.index,\n",
    "    sp_price_test,\n",
    "    color=\"black\",\n",
    "    linewidth=1.3\n",
    ")\n",
    "\n",
    "#HMM Regimes\n",
    "add_regime_background(ax, test_df.index, test_smoothed_states)\n",
    "\n",
    "ax.set_title(\"S&P 500 with Out-of-Sample HMM Regimes\", fontsize=14, fontweight=\"bold\")\n",
    "ax.set_xlabel(\"Date\")\n",
    "ax.set_ylabel(\"S&P 500 Index\")\n",
    "ax.grid(True, linestyle=\"--\", alpha=0.25)\n",
    "\n",
    "legend_elements = [\n",
    "    Patch(facecolor=\"seagreen\", alpha=0.3, label=\"State 0\"),\n",
    "    Patch(facecolor=\"firebrick\", alpha=0.3, label=\"State 1\"),\n",
    "    Patch(facecolor=\"steelblue\", alpha=0.3, label=\"State 2\")\n",
    "]\n",
    "\n",
    "ax.legend(\n",
    "    handles=legend_elements,\n",
    "    loc=\"upper left\",\n",
    "    frameon=True\n",
    ")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6f626f20-6716-403d-83a2-35744c0931f6",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "The out-of-sample classification further strengthens the initial interpretation of the hidden states. State 0 generally occurs during periods of stable market growth, while State 1 is associated with declining markets and elevated stress. State 2 appears more frequently during moderate or transitional market conditions.\n",
    "\n",
    "The model notably classifies the 2020 COVID-19 market crash as State 1, followed by transitions between States 0 and 2 during the following recovery. State 1 also reappears during periods of market stress in 2022 and in 2025. Overall, the results suggest that the economic interpretation of the regimes remains meaningful on previously unseen data. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b5f0c5fe-c0f6-46d3-a2f6-3842908a129b",
   "metadata": {},
   "source": [
    "### 5.2 In-Sample vs Out-of-Sample Behaviour"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 201,
   "id": "580c7979-e758-49e8-825d-7d41743df67d",
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr:last-of-type th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th colspan=\"3\" halign=\"left\">In-Sample</th>\n",
       "      <th colspan=\"3\" halign=\"left\">Out-of-Sample</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th>Returns</th>\n",
       "      <th>Momentum_20</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "      <th>Returns</th>\n",
       "      <th>Momentum_20</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0006</td>\n",
       "      <td>0.0187</td>\n",
       "      <td>0.0066</td>\n",
       "      <td>0.0009</td>\n",
       "      <td>0.0281</td>\n",
       "      <td>0.0070</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-0.0004</td>\n",
       "      <td>-0.0260</td>\n",
       "      <td>0.0221</td>\n",
       "      <td>-0.0004</td>\n",
       "      <td>-0.0236</td>\n",
       "      <td>0.0224</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-0.0002</td>\n",
       "      <td>-0.0045</td>\n",
       "      <td>0.0113</td>\n",
       "      <td>0.0004</td>\n",
       "      <td>0.0011</td>\n",
       "      <td>0.0111</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      In-Sample                                    Out-of-Sample              \\\n",
       "        Returns Momentum_20 Conditional_Volatility       Returns Momentum_20   \n",
       "State                                                                          \n",
       "0        0.0006      0.0187                 0.0066        0.0009      0.0281   \n",
       "1       -0.0004     -0.0260                 0.0221       -0.0004     -0.0236   \n",
       "2       -0.0002     -0.0045                 0.0113        0.0004      0.0011   \n",
       "\n",
       "                              \n",
       "      Conditional_Volatility  \n",
       "State                         \n",
       "0                     0.0070  \n",
       "1                     0.0224  \n",
       "2                     0.0111  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "train_states = np.argmax(HMM.smooth_proba(baseline_X_train), axis=1)\n",
    "test_states = np.argmax(HMM.smooth_proba(baseline_X_test), axis=1)\n",
    "\n",
    "train_regimes = train_df.copy()\n",
    "test_regimes = test_df.copy()\n",
    "\n",
    "train_regimes[\"State\"] = train_states\n",
    "test_regimes[\"State\"] = test_states\n",
    "\n",
    "train_means = train_regimes.groupby(\"State\").mean()\n",
    "test_means = test_regimes.groupby(\"State\").mean()\n",
    "\n",
    "comparison = pd.concat(\n",
    "    [train_means, test_means],\n",
    "    keys=[\"In-Sample\", \"Out-of-Sample\"],\n",
    "    axis=1\n",
    ")\n",
    "\n",
    "display(comparison.round(4))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f102499f-af23-4b9a-a4dd-f4d20b7e4a0c",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "The out-of-sample results show that the main economic characteristics of the regimes remain stable. State 0 continues to exhibit positive returns, positive momentum, and low volatility, while State 1 maintains negative returns, negative momentum, and substantially higher volatility. State 2 remains the intermediate-volatility regime, although its returns and momentum shift from slightly negative in-sample to slightly positive out-of-sample.\n",
    "\n",
    "Overall, the similarity between the in-sample and out-of-sample characteristics suggests that the HMM identifies economically consistent regimes beyond the data used for training."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f1f8296d-d6ff-453b-be05-52963013e1f9",
   "metadata": {},
   "source": [
    "## 6. Model Robustness & Selection\n",
    "\n",
    "The baseline HMM has showed economically interpretable regimes that remain consistent out-of-sample. However, the resulting regime structure may depend on modelling choices such as the number of hidden states and the selected input features. This section evaluates alternative model specifications to determine whether the baseline results are robust and whether alternative configurations provide a better representation of market regimes."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eb6a3cd3-bb90-4bd2-8ab9-07f2722be3ef",
   "metadata": {},
   "source": [
    "### 6.1 Number of Hidden States"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 215,
   "id": "fe1647ba-5d34-46b5-b26b-03d3b7bbdd1e",
   "metadata": {},
   "outputs": [],
   "source": [
    "state_range = range(2, 11)\n",
    "\n",
    "models = {}\n",
    "\n",
    "for n_states in state_range:\n",
    "    model = hmm(feature_matrix=baseline_X_train, states=n_states, covariance_type=\"full\")\n",
    "    model.fit(max_itr=100, tol=1e-5)\n",
    "    models[n_states] = model\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "00d2992f-206f-47ba-8694-cc6997e9f2e1",
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [],
   "source": [
    "\n",
    "results = []\n",
    "n = len(baseline_X_train)\n",
    "\n",
    "for K, model in models.items():\n",
    "    D = X_train.shape[1]\n",
    "    k = (\n",
    "        (K - 1)                    # Initial probabilities\n",
    "        + K * (K - 1)              # Transition matrix\n",
    "        + K * D                    # Means\n",
    "        + K * (D * (D + 1) / 2)    # Full covariance matrices\n",
    "    )\n",
    "    \n",
    "    log_likelihood = model.log_likelihood_history[-1]\n",
    "\n",
    "    # Calculate AIC\n",
    "    aic = 2*k - 2*log_likelihood\n",
    "\n",
    "    # Calculate BIC\n",
    "    bic = k*np.log(n) - 2*log_likelihood\n",
    "\n",
    "    results.append({\n",
    "        \"States\": K,\n",
    "        \"Log-Likelihood\": log_likelihood,\n",
    "        \"Parameters\": k,\n",
    "        \"AIC\": aic,\n",
    "        \"BIC\": bic\n",
    "    })\n",
    "\n",
    "results_df = pd.DataFrame(results)\n",
    "\n",
    "display(results_df.round(2))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fc776acd-e895-4710-ae5e-0c2d05877cc3",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "Both AIC and BIC decrease as the number of hidden states increases, suggesting that the improved model fit outweighs the penalty for additional complexity. However, more states do not necessarily result in a better model, as the regimes may become difficult to interpret economically. Therefore, AIC and BIC should be considered together with regime stability, occupancy and economic interpretation."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ec518697-fb18-4659-a033-92017fbbc6ee",
   "metadata": {},
   "source": [
    "### 6.2 Three-State vs Six-State Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 217,
   "id": "6622493c-649c-4c31-9004-d6d2d39f1608",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Occupancy (%)</th>\n",
       "      <th>Returns</th>\n",
       "      <th>Momentum_20</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>State 0</th>\n",
       "      <td>13.0026</td>\n",
       "      <td>0.0013</td>\n",
       "      <td>0.0381</td>\n",
       "      <td>0.0111</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State 1</th>\n",
       "      <td>13.7511</td>\n",
       "      <td>-0.0017</td>\n",
       "      <td>-0.0392</td>\n",
       "      <td>0.0133</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State 2</th>\n",
       "      <td>21.8563</td>\n",
       "      <td>-0.0002</td>\n",
       "      <td>-0.0085</td>\n",
       "      <td>0.0089</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State 3</th>\n",
       "      <td>4.1916</td>\n",
       "      <td>-0.0001</td>\n",
       "      <td>-0.0571</td>\n",
       "      <td>0.0316</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State 4</th>\n",
       "      <td>37.7459</td>\n",
       "      <td>0.0007</td>\n",
       "      <td>0.0223</td>\n",
       "      <td>0.0064</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State 5</th>\n",
       "      <td>9.4525</td>\n",
       "      <td>0.0002</td>\n",
       "      <td>-0.0079</td>\n",
       "      <td>0.0182</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Occupancy (%)  Returns  Momentum_20  Conditional_Volatility\n",
       "State 0        13.0026   0.0013       0.0381                  0.0111\n",
       "State 1        13.7511  -0.0017      -0.0392                  0.0133\n",
       "State 2        21.8563  -0.0002      -0.0085                  0.0089\n",
       "State 3         4.1916  -0.0001      -0.0571                  0.0316\n",
       "State 4        37.7459   0.0007       0.0223                  0.0064\n",
       "State 5         9.4525   0.0002      -0.0079                  0.0182"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model_6 = models[6]\n",
    "\n",
    "# Smoothed classification on training data\n",
    "probs_6 = model_6.smooth_proba(baseline_X_train)\n",
    "states_6 = np.argmax(probs_6, axis=1)\n",
    "\n",
    "occupancy_6 = np.bincount(states_6, minlength=model_6.states) / len(states_6)\n",
    "\n",
    "regimes_6 = train_df.copy()\n",
    "regimes_6[\"State\"] = states_6\n",
    "\n",
    "means_6 = regimes_6.groupby(\"State\").mean()\n",
    "\n",
    "comparison_6 = means_6.copy()\n",
    "comparison_6.insert(0, \"Occupancy (%)\", occupancy_6 * 100)\n",
    "\n",
    "comparison_6.index = [f\"State {i}\" for i in range(model_6.states)]\n",
    "\n",
    "display(comparison_6.round(4))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "89c881a9-d883-4ee2-83ca-9fe1c6422834",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "The six-state model provides a more specific representation of market conditions, separating calm, bullish, bearish and extreme-stress periods into regimes. However, several states represent relatively similar market conditions at different volatility or momentum levels, making the regime structure less straightforward to interpret than the three-state specification."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7b9c23ab-6288-4723-950c-5467d6378cfc",
   "metadata": {},
   "source": [
    "### 6.3 Regime Stability"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 220,
   "id": "640f219b-ef78-4fe3-b16d-f70d90126556",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr:last-of-type th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th colspan=\"3\" halign=\"left\">In-Sample</th>\n",
       "      <th colspan=\"3\" halign=\"left\">Out-of-Sample</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th>Returns</th>\n",
       "      <th>Momentum_20</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "      <th>Returns</th>\n",
       "      <th>Momentum_20</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0013</td>\n",
       "      <td>0.0381</td>\n",
       "      <td>0.0111</td>\n",
       "      <td>0.0022</td>\n",
       "      <td>0.0462</td>\n",
       "      <td>0.0110</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-0.0017</td>\n",
       "      <td>-0.0392</td>\n",
       "      <td>0.0133</td>\n",
       "      <td>-0.0013</td>\n",
       "      <td>-0.0364</td>\n",
       "      <td>0.0133</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-0.0002</td>\n",
       "      <td>-0.0085</td>\n",
       "      <td>0.0089</td>\n",
       "      <td>-0.0005</td>\n",
       "      <td>-0.0054</td>\n",
       "      <td>0.0090</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-0.0001</td>\n",
       "      <td>-0.0571</td>\n",
       "      <td>0.0316</td>\n",
       "      <td>0.0005</td>\n",
       "      <td>-0.0534</td>\n",
       "      <td>0.0370</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0007</td>\n",
       "      <td>0.0223</td>\n",
       "      <td>0.0064</td>\n",
       "      <td>0.0010</td>\n",
       "      <td>0.0313</td>\n",
       "      <td>0.0067</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0.0002</td>\n",
       "      <td>-0.0079</td>\n",
       "      <td>0.0182</td>\n",
       "      <td>0.0004</td>\n",
       "      <td>-0.0133</td>\n",
       "      <td>0.0177</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      In-Sample                                    Out-of-Sample              \\\n",
       "        Returns Momentum_20 Conditional_Volatility       Returns Momentum_20   \n",
       "State                                                                          \n",
       "0        0.0013      0.0381                 0.0111        0.0022      0.0462   \n",
       "1       -0.0017     -0.0392                 0.0133       -0.0013     -0.0364   \n",
       "2       -0.0002     -0.0085                 0.0089       -0.0005     -0.0054   \n",
       "3       -0.0001     -0.0571                 0.0316        0.0005     -0.0534   \n",
       "4        0.0007      0.0223                 0.0064        0.0010      0.0313   \n",
       "5        0.0002     -0.0079                 0.0182        0.0004     -0.0133   \n",
       "\n",
       "                              \n",
       "      Conditional_Volatility  \n",
       "State                         \n",
       "0                     0.0110  \n",
       "1                     0.0133  \n",
       "2                     0.0090  \n",
       "3                     0.0370  \n",
       "4                     0.0067  \n",
       "5                     0.0177  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>In-Sample (%)</th>\n",
       "      <th>Out-of-Sample (%)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>State 0</th>\n",
       "      <td>13.00</td>\n",
       "      <td>15.71</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State 1</th>\n",
       "      <td>13.75</td>\n",
       "      <td>13.17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State 2</th>\n",
       "      <td>21.86</td>\n",
       "      <td>22.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State 3</th>\n",
       "      <td>4.19</td>\n",
       "      <td>3.29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State 4</th>\n",
       "      <td>37.75</td>\n",
       "      <td>36.86</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State 5</th>\n",
       "      <td>9.45</td>\n",
       "      <td>8.98</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         In-Sample (%)  Out-of-Sample (%)\n",
       "State 0          13.00              15.71\n",
       "State 1          13.75              13.17\n",
       "State 2          21.86              22.00\n",
       "State 3           4.19               3.29\n",
       "State 4          37.75              36.86\n",
       "State 5           9.45               8.98"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "train_probs_6 = model_6.smooth_proba(baseline_X_train)\n",
    "test_probs_6 = model_6.smooth_proba(baseline_X_test)\n",
    "\n",
    "train_states_6 = np.argmax(train_probs_6, axis=1)\n",
    "test_states_6 = np.argmax(test_probs_6, axis=1)\n",
    "\n",
    "# Add state labels to original-scale feature data\n",
    "train_regimes_6 = train_df.copy()\n",
    "test_regimes_6 = test_df.copy()\n",
    "\n",
    "train_regimes_6[\"State\"] = train_states_6\n",
    "test_regimes_6[\"State\"] = test_states_6\n",
    "\n",
    "# Mean feature values within each state\n",
    "train_means_6 = train_regimes_6.groupby(\"State\").mean()\n",
    "test_means_6 = test_regimes_6.groupby(\"State\").mean()\n",
    "\n",
    "# Combine train vs test\n",
    "stability_6 = pd.concat(\n",
    "    [train_means_6, test_means_6],\n",
    "    keys=[\"In-Sample\", \"Out-of-Sample\"],\n",
    "    axis=1\n",
    ")\n",
    "\n",
    "train_occupancy_6 = np.bincount(train_states_6, minlength=model_6.states) / len(train_states_6)\n",
    "test_occupancy_6 = np.bincount(test_states_6, minlength=model_6.states) / len(test_states_6)\n",
    "occupancy_comparison = pd.DataFrame({\n",
    "    \"In-Sample (%)\": train_occupancy_6 * 100,\n",
    "    \"Out-of-Sample (%)\": test_occupancy_6 * 100\n",
    "})\n",
    "\n",
    "occupancy_comparison.index = [\n",
    "    f\"State {i}\" for i in range(model_6.states)\n",
    "]\n",
    "\n",
    "\n",
    "\n",
    "display(stability_6.round(4))\n",
    "\n",
    "display(occupancy_comparison.round(2))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "40a78d31-9d82-406c-9e9f-a24674c9cbad",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "The three-state specification is retained as the baseline model because it provides a simpler and more economically interpretable representation of market regimes. Although the six-state model achieves a better statistical fit and remains stable out-of-sample, the additional states mainly provide a more granular separation of similar market conditions. For the purpose of regime-based risk modelling, the three-state model therefore provides a better balance between model complexity and economic interpretability."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "03e1c353-6130-4aef-9ddd-a48f94b1cc9b",
   "metadata": {},
   "source": [
    "### 6.4 Ablation Feature Analysis\n",
    "\n",
    "The choice of input features directly affects how the HMM identifies and separates market regimes. To evaluate the importance of the selected features, an ablation analysis is performed by removing one feature at a time from the baseline model. The resulting models are compared to determine which features contribute most to the identified regime structure."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 225,
   "id": "f6ca6c5a-7380-49a5-8dd6-ae85a25781c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_sets = {\n",
    "    \"No Returns\": [\"Momentum_20\", \"Conditional_Volatility\"],\n",
    "    \"No Momentum\": [\"Returns\", \"Conditional_Volatility\"],\n",
    "    \"No Volatility\": [\"Returns\", \"Momentum_20\"]\n",
    "}\n",
    "\n",
    "feature_models = {}\n",
    "\n",
    "for name, columns in feature_sets.items():\n",
    "    X_temp = scaler.fit_transform(train_df[columns])\n",
    "    model = hmm(feature_matrix=X_temp, states=3, covariance_type=\"full\").fit()\n",
    "\n",
    "    probs = model.smooth_proba(X_temp)\n",
    "    states = np.argmax(probs, axis=1)\n",
    "\n",
    "    feature_models[name] = {\n",
    "        \"model\": model,\n",
    "        \"states\": states,\n",
    "        \"columns\": columns\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 226,
   "id": "68977f5c-8245-4cdf-8f4d-ee6557bb208f",
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Baseline\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Occupancy (%)</th>\n",
       "      <th>Returns</th>\n",
       "      <th>Momentum_20</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>45.2096</td>\n",
       "      <td>0.0006</td>\n",
       "      <td>0.0187</td>\n",
       "      <td>0.0066</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>13.9863</td>\n",
       "      <td>-0.0004</td>\n",
       "      <td>-0.0260</td>\n",
       "      <td>0.0221</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>40.8041</td>\n",
       "      <td>-0.0002</td>\n",
       "      <td>-0.0045</td>\n",
       "      <td>0.0113</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Occupancy (%)  Returns  Momentum_20  Conditional_Volatility\n",
       "State                                                             \n",
       "0            45.2096   0.0006       0.0187                  0.0066\n",
       "1            13.9863  -0.0004      -0.0260                  0.0221\n",
       "2            40.8041  -0.0002      -0.0045                  0.0113"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "No Returns\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Occupancy (%)</th>\n",
       "      <th>Momentum_20</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>45.2524</td>\n",
       "      <td>0.0180</td>\n",
       "      <td>0.0066</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>39.5637</td>\n",
       "      <td>-0.0051</td>\n",
       "      <td>0.0112</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>15.1839</td>\n",
       "      <td>-0.0208</td>\n",
       "      <td>0.0215</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Occupancy (%)  Momentum_20  Conditional_Volatility\n",
       "State                                                    \n",
       "0            45.2524       0.0180                  0.0066\n",
       "1            39.5637      -0.0051                  0.0112\n",
       "2            15.1839      -0.0208                  0.0215"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "No Momentum\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Occupancy (%)</th>\n",
       "      <th>Returns</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>45.7015</td>\n",
       "      <td>0.0006</td>\n",
       "      <td>0.0066</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>38.5586</td>\n",
       "      <td>-0.0002</td>\n",
       "      <td>0.0111</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>15.7399</td>\n",
       "      <td>-0.0002</td>\n",
       "      <td>0.0213</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Occupancy (%)  Returns  Conditional_Volatility\n",
       "State                                                \n",
       "0            45.7015   0.0006                  0.0066\n",
       "1            38.5586  -0.0002                  0.0111\n",
       "2            15.7399  -0.0002                  0.0213"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "No Volatility\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Occupancy (%)</th>\n",
       "      <th>Returns</th>\n",
       "      <th>Momentum_20</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>56.5441</td>\n",
       "      <td>0.0008</td>\n",
       "      <td>0.0243</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>14.8845</td>\n",
       "      <td>-0.0007</td>\n",
       "      <td>-0.0275</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>28.5714</td>\n",
       "      <td>-0.0007</td>\n",
       "      <td>-0.0234</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Occupancy (%)  Returns  Momentum_20\n",
       "State                                     \n",
       "0            56.5441   0.0008       0.0243\n",
       "1            14.8845  -0.0007      -0.0275\n",
       "2            28.5714  -0.0007      -0.0234"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "for name, result in feature_models.items():\n",
    "    columns = result[\"columns\"]\n",
    "    states = result[\"states\"]\n",
    "\n",
    "    temp = train_df[columns].copy()\n",
    "    temp[\"State\"] = states\n",
    "\n",
    "    means = temp.groupby(\"State\").mean()\n",
    "    occupancy = pd.Series(states).value_counts(normalize=True).sort_index() * 100\n",
    "\n",
    "    means.insert(0, \"Occupancy (%)\", occupancy.values)\n",
    "\n",
    "    print(f\"\\n{name}\")\n",
    "    display(means.round(4))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4e8809c0-8d50-4a24-a2f3-e560c57b271f",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "The ablation analysis shows that removing daily returns has a small effect on the identified regime structure. This suggests that most of the regime information is already captured by momentum and conditional volatility. Removing momentum still has the separation between low- and high-volatility states, but reduces the economic distinction between positive and negative market conditions. Removing conditional volatility changes the regime structure more substantially and removes the clear separation by volatility level. Overall the momentum and conditional volatility appear to be the main drivers of the identified regimes for the three state model."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "26bcbe03-7583-4e77-961d-031919bb713c",
   "metadata": {},
   "source": [
    "### 6.5 Alternative Feature Specifications\n",
    "\n",
    "The previous analysis suggests that momentum and conditional volatility contain most of the information used for regime identification. Alternative feature specifications are therefore tested to determine whether additional variables, particularly drawdown and longer-term momentum, can improve the economic separation of the identified regimes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "6de6290c-65b9-4014-a1cf-0f6da5da97d3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Baseline\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Occupancy (%)</th>\n",
       "      <th>Momentum_20</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>45.3998</td>\n",
       "      <td>0.0183</td>\n",
       "      <td>0.0066</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37.4248</td>\n",
       "      <td>-0.0019</td>\n",
       "      <td>0.0109</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>17.1754</td>\n",
       "      <td>-0.0265</td>\n",
       "      <td>0.0207</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Occupancy (%)  Momentum_20  Conditional_Volatility\n",
       "State                                                    \n",
       "0            45.3998       0.0183                  0.0066\n",
       "1            37.4248      -0.0019                  0.0109\n",
       "2            17.1754      -0.0265                  0.0207"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "With Drawdown\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Occupancy (%)</th>\n",
       "      <th>Momentum_20</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "      <th>Drawdown</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>28.7188</td>\n",
       "      <td>0.0125</td>\n",
       "      <td>0.0072</td>\n",
       "      <td>-0.0130</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>39.2089</td>\n",
       "      <td>0.0108</td>\n",
       "      <td>0.0081</td>\n",
       "      <td>-0.1952</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>32.0722</td>\n",
       "      <td>-0.0151</td>\n",
       "      <td>0.0169</td>\n",
       "      <td>-0.2485</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Occupancy (%)  Momentum_20  Conditional_Volatility  Drawdown\n",
       "State                                                              \n",
       "0            28.7188       0.0125                  0.0072   -0.0130\n",
       "1            39.2089       0.0108                  0.0081   -0.1952\n",
       "2            32.0722      -0.0151                  0.0169   -0.2485"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Long Momentum\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Occupancy (%)</th>\n",
       "      <th>Momentum_60</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>45.9372</td>\n",
       "      <td>0.0402</td>\n",
       "      <td>0.0066</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>18.6586</td>\n",
       "      <td>-0.0361</td>\n",
       "      <td>0.0198</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>35.4041</td>\n",
       "      <td>-0.0077</td>\n",
       "      <td>0.0110</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Occupancy (%)  Momentum_60  Conditional_Volatility\n",
       "State                                                    \n",
       "0            45.9372       0.0402                  0.0066\n",
       "1            18.6586      -0.0361                  0.0198\n",
       "2            35.4041      -0.0077                  0.0110"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "feature_sets = {\n",
    "    \"Baseline\": [\n",
    "        \"Momentum_20\",\n",
    "        \"Conditional_Volatility\"\n",
    "    ],\n",
    "\n",
    "    \"With Drawdown\": [\n",
    "        \"Momentum_20\",\n",
    "        \"Conditional_Volatility\",\n",
    "        \"Drawdown\"\n",
    "    ],\n",
    "\n",
    "    \"Long Momentum\": [\n",
    "        \"Momentum_60\",\n",
    "        \"Conditional_Volatility\"\n",
    "    ]\n",
    "}\n",
    "\n",
    "feature_models = {}\n",
    "\n",
    "for name, columns in feature_sets.items():\n",
    "    X_temp = scaler.fit_transform(train_df[columns])\n",
    "    model = hmm(feature_matrix=X_temp, states=3, covariance_type=\"full\").fit()\n",
    "\n",
    "    probs = model.smooth_proba(X_temp)\n",
    "    states = np.argmax(probs, axis=1)\n",
    "\n",
    "    temp = train_df[columns].copy()\n",
    "    temp[\"State\"] = states\n",
    "\n",
    "    means = temp.groupby(\"State\").mean()\n",
    "    occupancy = (pd.Series(states).value_counts(normalize=True).sort_index()* 100)\n",
    "\n",
    "    means.insert(0,\"Occupancy (%)\",occupancy.values)\n",
    "\n",
    "    feature_models[name] = {\n",
    "        \"model\": model,\n",
    "        \"states\": states,\n",
    "        \"means\": means\n",
    "    }\n",
    "\n",
    "    print(f\"\\n{name}\")\n",
    "    display(means.round(4))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1df6fbbb-1156-4456-b0eb-ff10fbd17537",
   "metadata": {},
   "source": [
    "#### Discussion\n",
    "\n",
    "The results show that longer-term momentum produces similar market regimes to the baseline model. Adding drawdown changes the regimes more, but does not make them easier to interpret. Therefore, momentum and conditional volatility remain the main features, while additional risk-related features are tested next."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a6aefd6d-c791-48b3-95c9-b5b6bd310679",
   "metadata": {},
   "source": [
    "### 6.6 Risk-Oriented Feature Extensions\n",
    "\n",
    "The analysis in 6.5 identifies momentum and GARCH conditional volatility as the pillar features of the baseline regime structure. To examine whether additional risk information improves regime identification, three risk-oriented features are considered: rolling volatility, downside volatility, and the VIX. Each feature is added separately to the baseline specification to evaluate whether it provides economically distinct information without unnecessarily increasing model complexity."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "66f7fff6-f274-4ece-80e6-5115053324d4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Baseline\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Occupancy (%)</th>\n",
       "      <th>Momentum_20</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>45.3998</td>\n",
       "      <td>0.0183</td>\n",
       "      <td>0.0066</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37.4248</td>\n",
       "      <td>-0.0019</td>\n",
       "      <td>0.0109</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>17.1754</td>\n",
       "      <td>-0.0265</td>\n",
       "      <td>0.0207</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Occupancy (%)  Momentum_20  Conditional_Volatility\n",
       "State                                                    \n",
       "0            45.3998       0.0183                  0.0066\n",
       "1            37.4248      -0.0019                  0.0109\n",
       "2            17.1754      -0.0265                  0.0207"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "With Downside Vol\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Occupancy (%)</th>\n",
       "      <th>Momentum_20</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "      <th>Downside Vol</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>44.4755</td>\n",
       "      <td>-0.0025</td>\n",
       "      <td>0.0109</td>\n",
       "      <td>0.0063</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>41.3801</td>\n",
       "      <td>0.0185</td>\n",
       "      <td>0.0065</td>\n",
       "      <td>0.0029</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>14.1445</td>\n",
       "      <td>-0.0248</td>\n",
       "      <td>0.0218</td>\n",
       "      <td>0.0132</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Occupancy (%)  Momentum_20  Conditional_Volatility  Downside Vol\n",
       "State                                                                  \n",
       "0            44.4755      -0.0025                  0.0109        0.0063\n",
       "1            41.3801       0.0185                  0.0065        0.0029\n",
       "2            14.1445      -0.0248                  0.0218        0.0132"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "With Rolling Vol\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Occupancy (%)</th>\n",
       "      <th>Momentum_20</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "      <th>Rolling_Volatility_20</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>39.1660</td>\n",
       "      <td>0.0044</td>\n",
       "      <td>0.0100</td>\n",
       "      <td>20.8854</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37.7902</td>\n",
       "      <td>0.0172</td>\n",
       "      <td>0.0066</td>\n",
       "      <td>13.7377</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>23.0439</td>\n",
       "      <td>-0.0225</td>\n",
       "      <td>0.0184</td>\n",
       "      <td>37.9875</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Occupancy (%)  Momentum_20  Conditional_Volatility  \\\n",
       "State                                                       \n",
       "0            39.1660       0.0044                  0.0100   \n",
       "1            37.7902       0.0172                  0.0066   \n",
       "2            23.0439      -0.0225                  0.0184   \n",
       "\n",
       "       Rolling_Volatility_20  \n",
       "State                         \n",
       "0                    20.8854  \n",
       "1                    13.7377  \n",
       "2                    37.9875  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "With VIX\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Occupancy (%)</th>\n",
       "      <th>Momentum_20</th>\n",
       "      <th>Conditional_Volatility</th>\n",
       "      <th>VIX</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>State</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>41.8745</td>\n",
       "      <td>0.0004</td>\n",
       "      <td>0.0108</td>\n",
       "      <td>20.6697</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>15.0473</td>\n",
       "      <td>-0.0204</td>\n",
       "      <td>0.0211</td>\n",
       "      <td>35.0849</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>43.0782</td>\n",
       "      <td>0.0137</td>\n",
       "      <td>0.0068</td>\n",
       "      <td>13.2581</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Occupancy (%)  Momentum_20  Conditional_Volatility      VIX\n",
       "State                                                             \n",
       "0            41.8745       0.0004                  0.0108  20.6697\n",
       "1            15.0473      -0.0204                  0.0211  35.0849\n",
       "2            43.0782       0.0137                  0.0068  13.2581"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "feature_sets = {\n",
    "    \"Baseline\": [\n",
    "        \"Momentum_20\",\n",
    "        \"Conditional_Volatility\"\n",
    "    ],\n",
    "\n",
    "    \"With Downside Vol\": [\n",
    "        \"Momentum_20\",\n",
    "        \"Conditional_Volatility\",\n",
    "        \"Downside Vol\"\n",
    "    ],\n",
    "    \n",
    "    \"With Rolling Vol\": [\n",
    "        \"Momentum_20\",\n",
    "        \"Conditional_Volatility\",\n",
    "        \"Rolling_Volatility_20\"\n",
    "    ],\n",
    "\n",
    "    \"With VIX\": [\n",
    "        \"Momentum_20\",\n",
    "        \"Conditional_Volatility\",\n",
    "        \"VIX\"\n",
    "    ]\n",
    "    \n",
    "}\n",
    "\n",
    "feature_models = {}\n",
    "\n",
    "for name, columns in feature_sets.items():\n",
    "    X_temp = scaler.fit_transform(train_df[columns])\n",
    "    model = hmm(feature_matrix=X_temp, states=3, covariance_type=\"full\").fit()\n",
    "\n",
    "    probs = model.smooth_proba(X_temp)\n",
    "    states = np.argmax(probs, axis=1)\n",
    "\n",
    "    temp = train_df[columns].copy()\n",
    "    temp[\"State\"] = states\n",
    "\n",
    "    means = temp.groupby(\"State\").mean()\n",
    "    occupancy = (pd.Series(states).value_counts(normalize=True).sort_index()* 100)\n",
    "\n",
    "    means.insert(0,\"Occupancy (%)\",occupancy.values)\n",
    "\n",
    "    feature_models[name] = {\n",
    "        \"model\": model,\n",
    "        \"states\": states,\n",
    "        \"means\": means\n",
    "    }\n",
    "\n",
    "    print(f\"\\n{name}\")\n",
    "    display(means.round(4))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "43253352-54d5-442b-b153-39e1ff54fb8f",
   "metadata": {},
   "source": [
    "## 7. Benchmark Against Established HMM"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fa46b9bb-2503-4b8b-8f53-f4604ce8380e",
   "metadata": {},
   "source": [
    "### 7.1 hmmlearn Specification"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bdff7f07-e6a2-430a-97fa-8494deb4fb47",
   "metadata": {},
   "source": [
    "### 7.2 Parameter / Regime Comparison"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab756dc1-6697-440b-94c3-79e50e5fd790",
   "metadata": {},
   "source": [
    "### 7.3 Runtime & Numerical Behaviour"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f52d6417-d9c4-4a9b-a826-e0d1370ffa3f",
   "metadata": {},
   "source": [
    "### 7.4 Discussion"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7fbdf4f4-fc38-4e64-98bc-8b4236bcea72",
   "metadata": {},
   "source": [
    "## 8. Final Discussion"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c59520b-09af-4b4a-ac99-4533a5765a1e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
