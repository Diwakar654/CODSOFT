{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "7fb4b26d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9fd4db71",
   "metadata": {},
   "outputs": [],
   "source": [
    "data=pd.read_csv(\"test.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "503f4cd7",
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Kelly, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330911</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
       "      <td>female</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>363272</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>Myles, Mr. Thomas Francis</td>\n",
       "      <td>male</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>240276</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Wirz, Mr. Albert</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>315154</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
       "      <td>female</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3101298</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>1305</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Spector, Mr. Woolf</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>A.5. 3236</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>414</th>\n",
       "      <td>1306</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Oliva y Ocana, Dona. Fermina</td>\n",
       "      <td>female</td>\n",
       "      <td>39.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17758</td>\n",
       "      <td>108.9000</td>\n",
       "      <td>C105</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>415</th>\n",
       "      <td>1307</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Saether, Mr. Simon Sivertsen</td>\n",
       "      <td>male</td>\n",
       "      <td>38.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>SOTON/O.Q. 3101262</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>416</th>\n",
       "      <td>1308</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Ware, Mr. Frederick</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>359309</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>417</th>\n",
       "      <td>1309</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Peter, Master. Michael J</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2668</td>\n",
       "      <td>22.3583</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>418 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass  \\\n",
       "0            892         0       3   \n",
       "1            893         1       3   \n",
       "2            894         0       2   \n",
       "3            895         0       3   \n",
       "4            896         1       3   \n",
       "..           ...       ...     ...   \n",
       "413         1305         0       3   \n",
       "414         1306         1       1   \n",
       "415         1307         0       3   \n",
       "416         1308         0       3   \n",
       "417         1309         0       3   \n",
       "\n",
       "                                             Name     Sex   Age  SibSp  Parch  \\\n",
       "0                                Kelly, Mr. James    male  34.5      0      0   \n",
       "1                Wilkes, Mrs. James (Ellen Needs)  female  47.0      1      0   \n",
       "2                       Myles, Mr. Thomas Francis    male  62.0      0      0   \n",
       "3                                Wirz, Mr. Albert    male  27.0      0      0   \n",
       "4    Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female  22.0      1      1   \n",
       "..                                            ...     ...   ...    ...    ...   \n",
       "413                            Spector, Mr. Woolf    male   NaN      0      0   \n",
       "414                  Oliva y Ocana, Dona. Fermina  female  39.0      0      0   \n",
       "415                  Saether, Mr. Simon Sivertsen    male  38.5      0      0   \n",
       "416                           Ware, Mr. Frederick    male   NaN      0      0   \n",
       "417                      Peter, Master. Michael J    male   NaN      1      1   \n",
       "\n",
       "                 Ticket      Fare Cabin Embarked  \n",
       "0                330911    7.8292   NaN        Q  \n",
       "1                363272    7.0000   NaN        S  \n",
       "2                240276    9.6875   NaN        Q  \n",
       "3                315154    8.6625   NaN        S  \n",
       "4               3101298   12.2875   NaN        S  \n",
       "..                  ...       ...   ...      ...  \n",
       "413           A.5. 3236    8.0500   NaN        S  \n",
       "414            PC 17758  108.9000  C105        C  \n",
       "415  SOTON/O.Q. 3101262    7.2500   NaN        S  \n",
       "416              359309    8.0500   NaN        S  \n",
       "417                2668   22.3583   NaN        C  \n",
       "\n",
       "[418 rows x 12 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0f82492e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(418, 12)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "ed8621dc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 418 entries, 0 to 417\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  418 non-null    int64  \n",
      " 1   Survived     418 non-null    int64  \n",
      " 2   Pclass       418 non-null    int64  \n",
      " 3   Name         418 non-null    object \n",
      " 4   Sex          418 non-null    object \n",
      " 5   Age          332 non-null    float64\n",
      " 6   SibSp        418 non-null    int64  \n",
      " 7   Parch        418 non-null    int64  \n",
      " 8   Ticket       418 non-null    object \n",
      " 9   Fare         417 non-null    float64\n",
      " 10  Cabin        91 non-null     object \n",
      " 11  Embarked     418 non-null    object \n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 39.3+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "fbeae25c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age             86\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             1\n",
       "Cabin          327\n",
       "Embarked         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "04525bed",
   "metadata": {},
   "outputs": [],
   "source": [
    "data=data.drop(columns='Cabin',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ce15ba75",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Age'].fillna(data['Age'].mean(),inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "8b3c4dc6",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "779e4fd3",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Fare'].fillna(data['Fare'].mode()[0],inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "5e9f717a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "6c9bd6cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Survived\n",
       "0    266\n",
       "1    152\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Survived'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "0873f2d0",
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1100.500000</td>\n",
       "      <td>0.363636</td>\n",
       "      <td>2.265550</td>\n",
       "      <td>30.272590</td>\n",
       "      <td>0.447368</td>\n",
       "      <td>0.392344</td>\n",
       "      <td>35.560497</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>120.810458</td>\n",
       "      <td>0.481622</td>\n",
       "      <td>0.841838</td>\n",
       "      <td>12.634534</td>\n",
       "      <td>0.896760</td>\n",
       "      <td>0.981429</td>\n",
       "      <td>55.857145</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>892.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.170000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>996.250000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>23.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.895800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1100.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>30.272590</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1204.750000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>35.750000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.471875</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1309.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>76.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   418.000000  418.000000  418.000000  418.000000  418.000000   \n",
       "mean   1100.500000    0.363636    2.265550   30.272590    0.447368   \n",
       "std     120.810458    0.481622    0.841838   12.634534    0.896760   \n",
       "min     892.000000    0.000000    1.000000    0.170000    0.000000   \n",
       "25%     996.250000    0.000000    1.000000   23.000000    0.000000   \n",
       "50%    1100.500000    0.000000    3.000000   30.272590    0.000000   \n",
       "75%    1204.750000    1.000000    3.000000   35.750000    1.000000   \n",
       "max    1309.000000    1.000000    3.000000   76.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  418.000000  418.000000  \n",
       "mean     0.392344   35.560497  \n",
       "std      0.981429   55.857145  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.895800  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.471875  \n",
       "max      9.000000  512.329200  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "1c93ef0f",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.set()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "edb3ca38",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n",
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n",
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Survived', ylabel='count'>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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8QMAAIxC/AAAAKMQPwAAwCjEDwAAMArxAwAAjEL8AAAAoxA/AADAKMQPAAAwCvEDAACMQvwAAACjED8AAMAoxA8AADAK8QMAAIxC/AAAAKMQPwAAwCjEDwAAMArxAwAAjGJ5/NTV1em3v/2tMjMzNWDAAN13332qqKjwrN+xY4f+/d//Xf3791dOTo42btzotX9TU5OefPJJpaen6/rrr9cjjzyi2trajn4bAAAgSFgePw8//LA++ugjLViwQK+99pquueYajR07VgcOHND+/fs1ceJEZWRkaP369brnnns0bdo07dixw7P/nDlz9P777+vZZ5/V6tWrdeDAAeXl5Vn4jgAAQCBzWnnwQ4cOafv27SotLdUNN9wgSZo1a5a2bdumDRs26Ntvv1VSUpKmTp0qSUpMTNSePXv04osvKj09XdXV1Xr99de1dOlSDRw4UJK0YMEC5eTk6KOPPtL1119v2XsDAACBydIrP9HR0Vq2bJmuu+46zzKbzSabzaaGhgZVVFQoPT3da59BgwZp165dcrvd2rVrl2fZWb1791ZcXJzKy8s75k0AAICgYmn8REVF6aabblJoaKhn2dtvv61Dhw4pIyNDVVVVio+P99onNjZWjY2NOn78uKqrqxUdHa2wsLA221RVVXXIewAAAMHF0tte/+jDDz/UjBkzNGTIEGVlZenUqVNeYSTJ8+fm5mY1Nja2WS9JYWFhampquqRZnE7/daHDYfmjVkDA4vwA4G8BEz9btmzRo48+qgEDBqikpETSmYhpbm722u7snyMiIhQeHt5mvXTmE2ARERE+z2K32xQd3dnn/QH4LirK93MXAC5EQMTPyy+/rIKCAuXk5Oh3v/ud52pOz549VVNT47VtTU2NOnXqpC5duig+Pl51dXVqbm72ugJUU1OjuLg4n+dxudxqaDjp8/7fx+Gw8w0eOI+Ghka1trqsHgNAkImKirjgK8eWx09paanmzZunUaNG6YknnpDNZvOsGzhwoD744AOv7Xfu3KkBAwbIbrfrhhtukMvl0q5duzwPRh88eFDV1dVKTU29pLlaWvjmC1ihtdXF+QfAryy9uX7w4EEVFhbqlltu0cSJE3Xs2DEdPXpUR48e1d///neNGjVKH3/8sUpKSrR//36tWLFC//3f/61x48ZJkuLi4jRs2DDl5+errKxMH3/8sR5++GGlpaUp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eH5vs9l0xx13aMuWLWpubpZ05ipOQkKC+vfvf87977jjDpWXl+vo0aOe7SMjIz3hsmPHDl1zzTWKi4tTS0uLWlpaZLfblZmZqb/+9a+SpIqKCoWEhHg9SN2pUyfddNNNF/1+APgH8QPAUl27dtXw4cNVUFCgLVu26E9/+pMSExNVXFys48ePX9Rrde7c2evPd955p+rr67Vt2zY1Nzdr8+bNuuOOO867f05OjpxOp9566y1JZ+Ln1ltvVXh4uCSprq5Ou3fv1k9+8hOvX2vXrtXf//53NTY2qr6+XpdddplsNpvXa8fExFzUewHgP9z2AtDhqqurNXLkSD300EOeW0xnJScna+rUqZo8ebKOHDkiSW0eWj558uQFHad3797q16+f3nrrLdntdjU0NPzT+OnSpYuys7P11ltvadCgQdq3b59mzZrltT4tLU3Tpk075/6hoaGKjo7W8ePH1draKofD4VlXV1d3QTMD8D+u/ADocD169JDT6VRpaamamprarD9w4IDCwsL04x//WJGRkaqqqvJav2vXrgs+1p133qlt27Zp48aNGjBggC6//PLv3X737t165ZVX9KMf/UhpaWmedWlpaTp48KB69+6t6667zvPrz3/+s/74xz/K4XAoPT1dLS0t2rJli2e/5uZmbd++/YJnBuBfxA+ADudwODRnzhzt3btXI0eO1CuvvKIPPvhAW7duVWFhoZ555hk9+OCD6tq1q/71X/9V5eXlev7557Vz504VFhZq586dF3ysoUOH6sSJE9q0adM5H3T+RxkZGbrsssv06quv6vbbb/e6fTV69Gi5XC6NHj1amzZt0o4dOzRr1iytWbNGvXv3lnTm4eazH7UvLS3V1q1bNWnSJNXW1l78FwqAX3DbC4AlsrKy9Ic//EHLly/X0qVLVVtbq9DQUCUnJ2vhwoUaMmSIJGnixImqra3V8uXLdfr0aWVlZamgoECTJk26oON069ZNgwcP1vbt25WTk/O92zudTg0bNkxr1qxpc4ssLi5O69at09NPP605c+aoqalJCQkJKigo0N133+3Z7rnnnlNJSYkWL16spqYmDR06VPfee6/efffdi/gKAfAXm/vsP6wBAABgAG57AQAAoxA/AADAKMQPAAAwCvEDAACMQvwAAACjED8AAMAoxA8AADAK8QMAAIxC/AAAAKMQPwAAwCjEDwAAMArxAwAAjPK/S7+wUG1z7LcAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='Survived',data=data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "3bf902f0",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n",
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n",
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Sex', ylabel='count'>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='Sex',data=data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "7d1a3c39",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n",
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n",
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n",
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Sex', ylabel='count'>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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4ou3XTYuLk4zZ850fj9p0iQtXLhQnTp10ogRIwqjXACFiPf5AVBkJScnq127dnrttdfUvn17l75q1appwIAB6tOnj+x2uyRp0KBBqlevXo7tlChRwvn14cOHtWzZMj388MP6+9//rqeeeko1a9Ys2AMBUKiY8AygyCpdurS8vLyUkJDgnO/ze8eOHZOvr68qV66skJAQnTlzRjVq1HD+K1u2rN555x3t379fkpSdna0hQ4aoXLlyWrFihapWrarBgwffctsAii5GfgAUWTabTaNHj1afPn3Url07dezYUZUqVVJ6erq2bNmijz76SK+99pqCg4M1YMAAjRw5UjabTc2aNVNaWpo++OADJScn65FHHpEkzZkzR/v371dCQoL8/Pw0btw4tW/fXtOnT9eQIUPcfLQA8gtzfgAUefv27dOCBQu0a9cuXbx4UT4+PqpWrZo6d+6s5s2bO5dbt26d5s+fr59++kkBAQGqXbu2+vfvrypVqujgwYN67rnn1L59e40aNcq5Tnx8vJYsWaLly5erTp067jg8APmM8AMAAEyFOT8AAMBUCD8AAMBUCD8AAMBUCD8AAMBUCD8AAMBUCD8AAMBUCD8AAMBUCD8AAMBUCD8AAMBUCD8AAMBUCD8AAMBUCD8AAMBU/h/NM+CC6moBmwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='Sex',hue='Survived',data=data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "e59a3c25",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n",
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n",
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Pclass', ylabel='count'>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='Pclass',data=data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "5e7cd01a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n",
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n",
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n",
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\seaborn\\_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead\n",
      "  if pd.api.types.is_categorical_dtype(vector):\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Pclass', ylabel='count'>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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VSf/sli5F84ZHccwgRExqtOlr6xWC+UHpkT5AWA6uRfOKTf9e6NjADAIF74BAICpUH4AAICpUH4AAICpUH4AAICpUH4AAICpUH4AAICpUH4AAICpUH4AAICpUH4AAICpUH4AAICpUH4AAICpUH4AAICpUH4AAICpUH4AAICpUH4AAICpUH4AAICpVIryc+nSJf3pT39Shw4ddO+99+qpp55Sampq8fyePXv0m9/8Rk2bNlXnzp21ZcsWA9MCAABvVinKz4gRI3To0CHNmDFDGzZsUP369dW7d2+dOHFC3333nfr376/27dtr48aNevLJJzV69Gjt2bPH6NgAAMAL+Rgd4JecPn1au3fv1urVq9W8eXNJ0iuvvKLPP/9cH374oS5cuKB77rlHw4cPlyTVrVtXR48e1ZIlS9SmTRsjowMAAC/k9Wd+wsLCtGjRIjVu3Lh4zGKxyGKxKDs7W6mpqSVKTkJCgg4ePCiXy1XRcQEAgJfz+jM/drtd999/v9vYtm3bdPr0ab300kt6//33FR0d7TYfGRmp3NxcZWZmKjw83KP9+vh43gttNq/vlDCAkccFxySuh2MS3qgijg2vLz8/9eWXX+rFF19Up06dlJiYqLy8PPn5+bltc+3rgoICj/ZhtVoUFhZ8y1mB/2W3BxodAXDDMQlvVBHHZaUqPzt37tSoUaN07733KiUlRZLk7+9fouRc+zow0LMH0Ol0KTv7isc5bTYrf1RQQnZ2rhwOpyH75pjE9XBMwht5elza7YGlPmtUacrPO++8o1dffVWdO3fWn//85+KzOzExMcrIyHDbNiMjQ0FBQapSpYrH+ysqMuYPAm5fDoeT4wpehWMS3qgijstKcdF19erVmjRpkrp166YZM2a4XeZq0aKF9u/f77b93r17de+998pqrRTfHgAAqEBef+bn5MmTmjJlijp27Kj+/fvr/PnzxXMBAQHq0aOHkpOTlZKSouTkZP3tb3/TX//6Vy1ZssTA1AAAwFt5ffnZtm2bCgsLtWPHDu3YscNtLjk5WVOnTtWbb76padOm6a233tKdd96padOm8R4/AADgury+/AwYMEADBgz42W06dOigDh06VFAiAABQmbEoBgAAmArlBwAAmArlBwAAmArlBwAAmArlBwAAmArlBwAAmArlBwAAmArlBwAAmArlBwAAmArlBwAAmArlBwAA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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='Pclass',hue='Survived',data=data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "cace1115",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Sex\n",
       "male      266\n",
       "female    152\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Sex'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "36cd331b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Embarked\n",
       "S    270\n",
       "C    102\n",
       "Q     46\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Embarked'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "81d5a811",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "13d2f1de",
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Kelly, Mr. James</td>\n",
       "      <td>0</td>\n",
       "      <td>34.50000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330911</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
       "      <td>1</td>\n",
       "      <td>47.00000</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>363272</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>Myles, Mr. Thomas Francis</td>\n",
       "      <td>0</td>\n",
       "      <td>62.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>240276</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Wirz, Mr. Albert</td>\n",
       "      <td>0</td>\n",
       "      <td>27.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>315154</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
       "      <td>1</td>\n",
       "      <td>22.00000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3101298</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>1305</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Spector, Mr. Woolf</td>\n",
       "      <td>0</td>\n",
       "      <td>30.27259</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>A.5. 3236</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>414</th>\n",
       "      <td>1306</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Oliva y Ocana, Dona. Fermina</td>\n",
       "      <td>1</td>\n",
       "      <td>39.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17758</td>\n",
       "      <td>108.9000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>415</th>\n",
       "      <td>1307</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Saether, Mr. Simon Sivertsen</td>\n",
       "      <td>0</td>\n",
       "      <td>38.50000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>SOTON/O.Q. 3101262</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>416</th>\n",
       "      <td>1308</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Ware, Mr. Frederick</td>\n",
       "      <td>0</td>\n",
       "      <td>30.27259</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>359309</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>417</th>\n",
       "      <td>1309</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Peter, Master. Michael J</td>\n",
       "      <td>0</td>\n",
       "      <td>30.27259</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2668</td>\n",
       "      <td>22.3583</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>418 rows × 11 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass  \\\n",
       "0            892         0       3   \n",
       "1            893         1       3   \n",
       "2            894         0       2   \n",
       "3            895         0       3   \n",
       "4            896         1       3   \n",
       "..           ...       ...     ...   \n",
       "413         1305         0       3   \n",
       "414         1306         1       1   \n",
       "415         1307         0       3   \n",
       "416         1308         0       3   \n",
       "417         1309         0       3   \n",
       "\n",
       "                                             Name  Sex       Age  SibSp  \\\n",
       "0                                Kelly, Mr. James    0  34.50000      0   \n",
       "1                Wilkes, Mrs. James (Ellen Needs)    1  47.00000      1   \n",
       "2                       Myles, Mr. Thomas Francis    0  62.00000      0   \n",
       "3                                Wirz, Mr. Albert    0  27.00000      0   \n",
       "4    Hirvonen, Mrs. Alexander (Helga E Lindqvist)    1  22.00000      1   \n",
       "..                                            ...  ...       ...    ...   \n",
       "413                            Spector, Mr. Woolf    0  30.27259      0   \n",
       "414                  Oliva y Ocana, Dona. Fermina    1  39.00000      0   \n",
       "415                  Saether, Mr. Simon Sivertsen    0  38.50000      0   \n",
       "416                           Ware, Mr. Frederick    0  30.27259      0   \n",
       "417                      Peter, Master. Michael J    0  30.27259      1   \n",
       "\n",
       "     Parch              Ticket      Fare  Embarked  \n",
       "0        0              330911    7.8292         2  \n",
       "1        0              363272    7.0000         0  \n",
       "2        0              240276    9.6875         2  \n",
       "3        0              315154    8.6625         0  \n",
       "4        1             3101298   12.2875         0  \n",
       "..     ...                 ...       ...       ...  \n",
       "413      0           A.5. 3236    8.0500         0  \n",
       "414      0            PC 17758  108.9000         1  \n",
       "415      0  SOTON/O.Q. 3101262    7.2500         0  \n",
       "416      0              359309    8.0500         0  \n",
       "417      1                2668   22.3583         1  \n",
       "\n",
       "[418 rows x 11 columns]"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "ee4bf267",
   "metadata": {},
   "outputs": [],
   "source": [
    "X=data.drop(columns=['PassengerId','Name','Ticket'],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "af94ccaa",
   "metadata": {},
   "outputs": [],
   "source": [
    "Y=data['Survived']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "843d99d5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "     Survived  Pclass  Sex       Age  SibSp  Parch      Fare  Embarked\n",
      "0           0       3    0  34.50000      0      0    7.8292         2\n",
      "1           1       3    1  47.00000      1      0    7.0000         0\n",
      "2           0       2    0  62.00000      0      0    9.6875         2\n",
      "3           0       3    0  27.00000      0      0    8.6625         0\n",
      "4           1       3    1  22.00000      1      1   12.2875         0\n",
      "..        ...     ...  ...       ...    ...    ...       ...       ...\n",
      "413         0       3    0  30.27259      0      0    8.0500         0\n",
      "414         1       1    1  39.00000      0      0  108.9000         1\n",
      "415         0       3    0  38.50000      0      0    7.2500         0\n",
      "416         0       3    0  30.27259      0      0    8.0500         0\n",
      "417         0       3    0  30.27259      1      1   22.3583         1\n",
      "\n",
      "[418 rows x 8 columns]\n"
     ]
    }
   ],
   "source": [
    "print(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "51c5c3bd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0      0\n",
      "1      1\n",
      "2      0\n",
      "3      0\n",
      "4      1\n",
      "      ..\n",
      "413    0\n",
      "414    1\n",
      "415    0\n",
      "416    0\n",
      "417    0\n",
      "Name: Survived, Length: 418, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "d3852bef",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "e76bc76f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(418, 8) (334, 8) (84, 8)\n"
     ]
    }
   ],
   "source": [
    "print(X.shape,X_train.shape,X_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "d16662a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "model=LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "acfb9155",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Diwakar mishra\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\sklearn\\linear_model\\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "f1a94c25",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_prediction=model.predict(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "f67d2af6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 1 0 0 1 1 0 0 0 1 0 0 1 0 0 0 1 0 1 0 1 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0\n",
      " 1 1 1 0 0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1 0 1 0 1 1 1 0 1\n",
      " 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 1 0 1 0 0 0 0 0 0 0 1 0 1 1 1 0 1 0 1 0\n",
      " 1 1 0 0 0 0 1 1 0 1 0 0 1 1 0 1 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 1 0 1 1 0 0\n",
      " 0 0 1 1 1 0 0 1 1 0 1 1 0 0 0 0 0 0 0 1 1 0 0 1 1 1 1 0 1 0 0 0 0 1 0 1 1\n",
      " 1 0 1 0 0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 0 0 1 0 0 1 0 0\n",
      " 1 0 1 0 0 0 0 0 1 0 0 0 1 1 0 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1\n",
      " 0 1 1 1 1 0 0 0 1 1 0 0 1 0 1 1 0 0 0 0 1 0 0 0 0 0 1 0 0 1 1 0 1 1 0 0 0\n",
      " 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 1 1 0 0 0 1 1 1\n",
      " 1]\n"
     ]
    }
   ],
   "source": [
    "print(X_train_prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "c4025dd4",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data_accuracy=accuracy_score(Y_train,X_train_prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "44d5f934",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy Score of training data:  1.0\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy Score of training data: \",train_data_accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "d6fdfff6",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test_prediction=model.predict(X_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "8eced174",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 0 0 1 1 0 1 0 0 1 0 1 1 0 1 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 1 1 0 1 0 0 1\n",
      " 1 0 0 0 0 1 1 0 0 1 0 1 0 0 0 1 1 1 0 0 1 0 0 0 0 0 0 1 0 1 1 1 1 1 1 0 0\n",
      " 0 1 1 0 1 0 0 0 0 0]\n"
     ]
    }
   ],
   "source": [
    "print(X_test_prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "386e0ace",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_data_accuracy=accuracy_score(Y_test,X_test_prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "d14f0c7a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy score of testing data: 1.0\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy score of testing data:\",test_data_accuracy)"
   ]
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
   "version": "3.10.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
