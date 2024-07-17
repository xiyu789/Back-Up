import pandas as pd
from statsmodels.tsa.stattools import adfuller

#df = pd.read_excel("BoundTest_Data.xlsx")
df = pd.read_excel("DataBase.xlsx")
fed = df['Fed Rate']
deposit = df['Deposit Rate']
loan = df['Loan Rate']
RGDP = df['Real GDP'].dropna()
CPI = df['CPI'].dropna()
SPX = df['SPX'].dropna()

############# ADF ##############
def Teststationarity(df):
    dfinput = adfuller(df,regression='ct')
    dfoutput = pd.Series(dfinput[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    for key,value in dfinput[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

def perform_adf_test(series, name):
    print(f'{name} ADF Results')
    result = Teststationarity(series)
    print(result)
    print()

perform_adf_test(fed, 'Fed Rate')
perform_adf_test(deposit, 'Deposit Rate')
perform_adf_test(deposit.diff().dropna(), 'Diff Deposit Rate')
perform_adf_test(loan, 'Loan Rate')
perform_adf_test(loan.diff().dropna(), 'Diff Loan Rate')
perform_adf_test(RGDP, 'RGDP')


##### ARDL Bounds Test ##############
import os
os.environ['R_HOME'] = 'C:\\Program Files\\R\\R-4.3.2'
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

ro.r('Sys.setlocale("LC_ALL", "English_United States.UTF-8")')

pandas2ri.activate()
df.columns = [col.replace(' ', '_') for col in df.columns]
df['Deposit_Rate'] = df['Deposit_Rate'].diff().dropna()
df['Loan_Rate'] = df['Loan_Rate'].diff().dropna()


# R code
r_code = """
function(data_frame, dep_var, indep_vars) {
    library(dLagM)
    formula = as.formula(paste(dep_var, '~', indep_vars))
    result <- ardlBound(data = data_frame, formula = formula, case = 3, autoOrder = TRUE, ic = c('AIC', 'BIC'), max.p = 15, max.q = 15, ECM = TRUE, stability = TRUE)
    bounds <- result$bounds_test
    print(bounds)
}
"""


r_function = ro.r(r_code)
r_data = pandas2ri.py2rpy(df)
indep_var = ['Fed_Rate']

dep_var = ['Real_GDP']

print(f"ARDL Model for {dep_var} as dependent and {indep_var} as independent:")
results = r_function(r_data, dep_var, indep_var)
print("Model Summary:")
print(results[0])
print("Bounds Test Result:")
print(results[1])
print("\n")

