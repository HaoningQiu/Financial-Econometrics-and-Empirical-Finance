# -*- coding: utf-8 -*-
"""
@author: Edward


"""


"""
MGTF 404 HW5 Part One Suggested Solution, Junxiong Gao.Import packages

"""
#Part 1
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime as dt
import arch

"""
Read the REITS data until 2018, 2019 data is used later for Out of Sample

"""

data = pd.read_excel('reit_data_2019.xls',skiprows = 7)
data = data[2:-8]


data.index = data['Date']

del data['Date']
# define variables

Ret = data['Return']/100
delta_price = data['Return.1']/100
DY = data['Yield']/100
price = data['Index']
dividend = price*DY

# smooth

dividend_sm = dividend.rolling(12).mean()
DY_sm = dividend_sm/price 

# log

ret = np.log(Ret+1)
dy = np.log(DY)
dy_sm = np.log(DY_sm)

#%% Questioon 5

variable = np.array([Ret,ret,DY,DY_sm,dy,dy_sm]).transpose()
Variable = pd.DataFrame(variable,\
                        columns = ['Return','Log Return','Yield','Smoothed Yield'\
                                   ,'Log Yield','Log Smoothed Yield'])
Variable.index = data.index

mean = Variable.mean()*12

std = Variable.std()*np.sqrt(12)

skewness = Variable.skew()/(12**(1/2))

kurtosis = (Variable.kurt()+3)/12

acf = pd.DataFrame(columns = Variable.columns)

for x in acf.columns:
    acf.loc[:,x] = pd.Series(Variable[x].autocorr())

print('\n---------Question 5-----------')    
print('\nMean\n')
print(mean)
print('\nStandard deviation\n')
print(std)
print('\nAutocorrelation\n')
print(acf)
print('\nSkewness\n')
print(skewness)
print('\nKurtosis\n')
print(kurtosis) 

#%% Question 6
LagRet = pd.DataFrame(columns = np.arange(1,13))

for i in LagRet.columns:
    LagRet.iloc[:,i-1] = Ret.shift(i)

Lags = [1,2,12]
ARcoeff = pd.DataFrame()
ARpvalue = pd.DataFrame()
ARr2 = pd.DataFrame()

for i in Lags:
    result = sm.OLS(Ret[i:], sm.add_constant(LagRet.iloc[i:,:i])).fit()
    ARcoeff = pd.concat([ARcoeff, pd.DataFrame(result.params,columns=[i])],axis=1)
    ARpvalue = pd.concat([ARpvalue, pd.DataFrame(result.pvalues,columns=[i])],axis=1)
    ARr2 = pd.concat([ARr2, pd.DataFrame([result.rsquared],columns=[i])],axis=1)

print('\n---------Question 6-----------')
print('\ncoefficents:\n',ARcoeff)
print('\npvalues:\n',ARpvalue)
print('\nR2:\n',ARr2)

#%% Question 7
Lagret = pd.DataFrame(columns = np.arange(1,13))

for i in LagRet.columns:
    Lagret.iloc[:,i-1] = ret.shift(i)

Lags = [1,2,12]
arcoeff = pd.DataFrame()
arpvalue = pd.DataFrame()
arr2 = pd.DataFrame()

for i in Lags:
    result = sm.OLS(ret[i:], sm.add_constant(Lagret.iloc[i:,:i])).fit()
    arcoeff = pd.concat([arcoeff, pd.DataFrame(result.params,columns=[i])],axis=1)
    arpvalue = pd.concat([arpvalue, pd.DataFrame(result.pvalues,columns=[i])],axis=1)
    arr2 = pd.concat([arr2, pd.DataFrame([result.rsquared],columns=[i])],axis=1)
    
print('\n---------Question 7-----------')  
print('\ncoefficents:\n',arcoeff)
print('\npvalues:\n',arpvalue)
print('\nR2:\n',arr2)

#%% Question 9-12


"""
Predictive Regressions, simple return

"""
# AR for predictor

predictor1 = [DY,DY_sm]

T1 = np.size(DY,0)-1
phi1 = pd.DataFrame(columns = ['Yield','Smoothed Yield'])
u1 = pd.DataFrame()
for i in np.arange(len(predictor1)):
    x = predictor1[i]
    model = sm.OLS(x[1:],sm.add_constant(np.array(x[:-1])),missing = 'drop')
    res = model.fit()
    phi1.iloc[:,i] = pd.Series(res.params[1])
    u1= pd.concat([u1,res.resid],axis=1)
    
u1.columns = ['Yield','Smoothed Yield']    
Bias1 = -(1+3*phi1)/T1

phi_unbias1 = phi1-Bias1
    
# predictive reg
alpha1 = pd.DataFrame(columns = ['Yield','Smoothed Yield'])
beta1 = pd.DataFrame(columns = ['Yield','Smoothed Yield'])
e1 = pd.DataFrame()
pvalue1 = pd.DataFrame(columns = ['Yield','Smoothed Yield'])

for i in np.arange(len(predictor1)):
    x = predictor1[i]
    model = sm.OLS(Ret[1:],sm.add_constant(np.array(x[:-1])),missing = 'drop')
    res = model.fit()
    alpha1.iloc[:,i] = pd.Series(res.params[0])
    beta1.iloc[:,i] = pd.Series(res.params[1])
    pvalue1.iloc[:,i] = pd.Series(res.pvalues[1])
    e1= pd.concat([e1,res.resid],axis=1)
    
e1.columns = ['Yield','Smoothed Yield']

# bias
covar1 = []
for i in np.arange(len(predictor1)):
    S = np.ma.cov(u1.iloc[:,i].dropna(),e1.iloc[:,i].dropna())
    covar1 = np.hstack((covar1,S[0,1]))    
    
Bias_predict1 = Bias1*covar1/np.nanvar(u1,0)

beta_unbias1 = beta1-Bias_predict1


"""
Predictive Regressions, log return

"""
# AR for predictor

predictor = [dy,dy_sm]

T = np.size(dy,0)-1
phi = pd.DataFrame(columns = ['Yield','Smoothed Yield'])
u = pd.DataFrame()
for i in np.arange(len(predictor)):
    x = predictor[i]
    model = sm.OLS(x[1:],sm.add_constant(np.array(x[:-1])),missing = 'drop')
    res = model.fit()
    phi.iloc[:,i] = pd.Series(res.params[1])
    u= pd.concat([u,res.resid],axis=1)
    
u.columns = ['Yield','Smoothed Yield']    
Bias = -(1+3*phi)/T

phi_unbias = phi-Bias
    
# predictive reg
alpha = pd.DataFrame(columns = ['Yield','Smoothed Yield'])
beta = pd.DataFrame(columns = ['Yield','Smoothed Yield'])
e = pd.DataFrame()
pvalue = pd.DataFrame(columns = ['Yield','Smoothed Yield'])

for i in np.arange(len(predictor)):
    x = predictor[i]
    model = sm.OLS(ret[1:],sm.add_constant(np.array(x[:-1])),missing = 'drop')
    res = model.fit()
    alpha.iloc[:,i] = pd.Series(res.params[0])
    beta.iloc[:,i] = pd.Series(res.params[1])
    pvalue.iloc[:,i] = pd.Series(res.pvalues[1])
    e= pd.concat([e,res.resid],axis=1)
    
e.columns = ['Yield','Smoothed Yield']

# bias
covar = []
for i in np.arange(len(predictor)):
    S = np.ma.cov(u.iloc[:,i].dropna(),e.iloc[:,i].dropna())
    covar = np.hstack((covar,S[0,1]))    
    
Bias_predict = Bias*covar/np.nanvar(u,0)

beta_unbias = beta-Bias_predict

print('\n-----------Question 9-12------------')

print('\nalpha for a&b:\n',alpha1)
print('\nalpha for c&d:\n',alpha)
print('\nUnadjusted result (beta) for a&b:\n',beta1)
print('\nUnadjusted result (beta) for c&d:\n',beta)
print('\nP-values of beta for a&b:\n',pvalue1)
print('\nP-values of beta for c&d:\n',pvalue)
print('\nAdjusted result (beta) for a&b:\n',beta_unbias1)
print('\nAdjusted result (beta) for c&d:\n',beta_unbias)
print('\nBias in beta for a&b:\n',Bias_predict1)
print('\nBias in beta for c&d:\n',Bias_predict)

#%% Question 13
'''
# Out of Sample predict use unbiased beta
'''
x = pd.DataFrame(columns = ['Log Yield','Log Smoothed Yield'])
x['Log Yield'] = dy[(dy.index>='2017-10-31')&(dy.index<='2017-12-29')]
x['Log Smoothed Yield'] = dy_sm[(dy_sm.index>='2017-10-31')&(dy_sm.index<='2017-12-29')]
PredictVal = np.array(alpha)+np.array(beta_unbias)*x
PredictVal.index = ret[(ret.index>='2017-11-30')&(ret.index<='2018-01-31')].index

print('\nOut of Sample forecast using unbiased beta:\n',PredictVal)
# plot prediction with real value

fig = plt.figure()
plt.plot(PredictVal,'o-')
plt.plot(ret[(ret.index>='2017-11-30')&(ret.index<='2018-01-31')],'o-')
plt.legend(['Yield Predict','Log Smoothed Yield Predict','Log Return'])

# %% 1972-1993 subsample
"""
Redo Everything for two subsample, to save memory, did not create separate
result variables, run code by block please.

"""

data1 = pd.read_excel('reit_data_2019.xls',skiprows = 7)
data1.index = data1['Date']
subsample = [(data1.index.year>=1972)&(data1.index.year<=1993),(data1.index.year>=1994)&(data1.index.year<=2014)]
subsample_name = ['1972-1993 subsample','1994-2014 subsample']

del data1['Date']
for sub in [0,1]:
    subsamplei = subsample[sub]
    datai = data1[subsamplei]
    # define variables
    
    Ret1 = datai['Return']/100
    DY1 = datai['Yield']/100
    price1 = datai['Index']
    dividend1 = price1*DY1
    
    # smooth
    
    dividend_sm1 = dividend1.rolling(12).mean()
    DY_sm1 = dividend_sm1/price1
    
    # log
    
    ret1 = np.log(Ret1+1)
    dy1 = np.log(DY1)
    dy_sm1 = np.log(DY_sm1)
    
    
    #Question 6
    LagRet1 = pd.DataFrame(columns = np.arange(1,13))
    
    for i in LagRet1.columns:
        LagRet1.iloc[:,i-1] = Ret1.shift(i)
    
    Lags1 = [1,2,12]
    ARcoeff1 = pd.DataFrame()
    ARpvalue1 = pd.DataFrame()
    ARr21 = pd.DataFrame()
    
    for i in Lags:
        result = sm.OLS(Ret1[i:], sm.add_constant(LagRet1.iloc[i:,:i])).fit()
        ARcoeff1 = pd.concat([ARcoeff1, pd.DataFrame(result.params,columns=[i])],axis=1)
        ARpvalue1 = pd.concat([ARpvalue1, pd.DataFrame(result.pvalues,columns=[i])],axis=1)
        ARr21 = pd.concat([ARr21, pd.DataFrame([result.rsquared],columns=[i])],axis=1)
    
    print(f'\n-----------{subsample_name[sub]}-Q6------------')
    print('\ncoefficents:\n',ARcoeff1)
    print('\npvalues:\n',ARpvalue1)
    print('\nR2:\n',ARr21)
    
    #Question 7
    Lagret1 = pd.DataFrame(columns = np.arange(1,13))
    
    for i in Lagret1.columns:
        Lagret1.iloc[:,i-1] = ret1.shift(i)
    
    Lags1 = [1,2,12]
    arcoeff1 = pd.DataFrame()
    arpvalue1 = pd.DataFrame()
    arr21 = pd.DataFrame()
    
    for i in Lags1:
        result = sm.OLS(ret1[i:], sm.add_constant(Lagret1.iloc[i:,:i])).fit()
        arcoeff1 = pd.concat([arcoeff1, pd.DataFrame(result.params,columns=[i])],axis=1)
        arpvalue1 = pd.concat([arpvalue1, pd.DataFrame(result.pvalues,columns=[i])],axis=1)
        arr21 = pd.concat([arr21, pd.DataFrame([result.rsquared],columns=[i])],axis=1)
        
    print(f'\n-----------{subsample_name[sub]}-Q7------------') 
    print('\ncoefficents:\n',arcoeff1)
    print('\npvalues:\n',arpvalue1)
    print('\nR2:\n',arr21)
    
    """
    Predictive Regressions, simple return
    
    """
    # AR for predictor
    
    predictor11 = [DY1,DY_sm1]
    
    T11 = np.size(DY,0)-1
    phi11 = pd.DataFrame(columns = ['Yield','Smoothed Yield'])
    u11 = pd.DataFrame()
    for i in np.arange(len(predictor11)):
        x = predictor11[i]
        model = sm.OLS(x[1:],sm.add_constant(np.array(x[:-1])),missing = 'drop')
        res = model.fit()
        phi11.iloc[:,i] = pd.Series(res.params[1])
        u11= pd.concat([u11,res.resid],axis=1)
        
    u11.columns = ['Yield','Smoothed Yield']    
    Bias11 = -(1+3*phi11)/T11
    
    phi_unbias11 = phi11-Bias11
        
    # predictive reg
    alpha11 = pd.DataFrame(columns = ['Yield','Smoothed Yield'])
    beta11 = pd.DataFrame(columns = ['Yield','Smoothed Yield'])
    e11 = pd.DataFrame()
    pvalue11 = pd.DataFrame(columns = ['Yield','Smoothed Yield'])
    for i in np.arange(len(predictor11)):
        x = predictor11[i]
        model = sm.OLS(Ret1[1:],sm.add_constant(np.array(x[:-1])),missing = 'drop')
        res = model.fit()
        alpha11.iloc[:,i] = pd.Series(res.params[0])
        beta11.iloc[:,i] = pd.Series(res.params[1])
        pvalue11.iloc[:,i] = pd.Series(res.pvalues[1])
        e11= pd.concat([e11,res.resid],axis=1)
        
    e11.columns = ['Yield','Smoothed Yield']
    
    # bias
    covar11 = []
    for i in np.arange(len(predictor11)):
        S = np.ma.cov(u11.iloc[:,i].dropna(),e11.iloc[:,i].dropna())
        covar11 = np.hstack((covar11,S[0,1]))    
        
    Bias_predict11 = Bias11*covar11/np.nanvar(u11,0)
    
    beta_unbias11 = beta11-Bias_predict11
    """
    Predictive Regressions, log return
    
    """
    # AR for predictor
    
    predictor2 = [dy1,dy_sm1]
    
    T2 = np.size(dy,0)-1
    phi2 = pd.DataFrame(columns = ['Yield','Smoothed Yield'])
    u2 = pd.DataFrame()
    for i in np.arange(len(predictor2)):
        x = predictor2[i]
        model = sm.OLS(x[1:],sm.add_constant(np.array(x[:-1])),missing = 'drop')
        res = model.fit()
        phi2.iloc[:,i] = pd.Series(res.params[1])
        u2= pd.concat([u2,res.resid],axis=1)
        
    u2.columns = ['Yield','Smoothed Yield']    
    Bias2 = -(1+3*phi2)/T2
    
    phi_unbias2 = phi2-Bias2
        
    # predictive reg
    alpha2 = pd.DataFrame(columns = ['Yield','Smoothed Yield'])
    beta2 = pd.DataFrame(columns = ['Yield','Smoothed Yield'])
    e2 = pd.DataFrame()
    pvalue2 = pd.DataFrame(columns = ['Yield','Smoothed Yield'])
    
    for i in np.arange(len(predictor2)):
        x = predictor2[i]
        model = sm.OLS(ret1[1:],sm.add_constant(np.array(x[:-1])),missing = 'drop')
        res = model.fit()
        alpha2.iloc[:,i] = pd.Series(res.params[0])
        beta2.iloc[:,i] = pd.Series(res.params[1])
        pvalue2.iloc[:,i] = pd.Series(res.pvalues[1])
        e2= pd.concat([e2,res.resid],axis=1)
        
    e2.columns = ['Yield','Smoothed Yield']
    
    # bias
    covar2 = []
    for i in np.arange(len(predictor2)):
        S = np.ma.cov(u2.iloc[:,i].dropna(),e2.iloc[:,i].dropna())
        covar2 = np.hstack((covar2,S[0,1]))    
        
    Bias_predict2 = Bias2*covar2/np.nanvar(u2,0)
    
    beta_unbias2 = beta2-Bias_predict2
    
    print(f'\n-----------{subsample_name[sub]}-Q9----------')
    print('\nalpha for a&b:\n',alpha11)
    print('\nalpha for c&d:\n',alpha2)
    print('\nUnadjusted result (beta) for a&b:\n',beta11)
    print('\nUnadjusted result (beta) for c&d:\n',beta2)
    print('\nP-values of beta for a&b:\n',pvalue11)
    print('\nP-values of beta for c&d:\n',pvalue2)
    print('\nAdjusted result (beta) for a&b:\n',beta_unbias11)
    print('\nAdjusted result (beta) for c&d:\n',beta_unbias2)
    print('\nBias in beta for a&b:\n',Bias_predict11)
    print('\nBias in beta for c&d:\n',Bias_predict2)
    
    
    
    
    
    
    
    
data = pd.read_excel('vol_data_homework.xlsx',skiprows = 1)

data[data=='ND'] = np.nan

data = data.iloc[:,[0,1,3,5]]

data.index = data['Code']

del data['Code']

data.columns = ['SP500','$/Euro','Oil']

#%%Question 2 
"""
Compute Return

"""

Ret = 100*pd.DataFrame(np.log(np.asarray(np.array(data.iloc[1:,])/\
                          np.array(data.iloc[:-1,]),dtype = float))\
                   ,index = data[1:].index,columns = data.columns)

mean = Ret.mean()*252
mean
std = Ret.std()*np.sqrt(252)
std
skewness = Ret.skew()/(252**(1/2))
skewness
kurtosis = (Ret.kurt()+3)/252
kurtosis

#%%Question 3
"""
Compound to monthly return

"""
std_month = (Ret.groupby([Ret.index.year,Ret.index.month])\
                 .agg(np.nanstd))*np.sqrt(252)
Switch = pd.Series(Ret.index).groupby([Ret.index.year,Ret.index.month]).agg('last')
MonthIndex = pd.to_datetime(Switch)
std_month.index = MonthIndex

"""
Filter the return
"""

Ret = Ret.fillna(0)

#%%Question 4
# arch 1

vol_arch1 = pd.DataFrame(columns = Ret.columns,index = Ret.index)
arch1test_pval = pd.DataFrame(columns = Ret.columns)
ll_arch1 = pd.DataFrame(columns = Ret.columns)
coeff_arch1 = pd.DataFrame(columns = Ret.columns)
for x in Ret.columns:
   r = Ret.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 1, q = 0)
   res = am.fit()
   coeff_arch1.loc[:,x] = pd.Series(res.params)
   vol_arch1.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(252)
   arch1test_pval.loc[:,x]  = pd.Series(res.arch_lm_test(1).pval)
   ll_arch1.loc[:,x] = pd.Series(res.loglikelihood)
   
# arch 3
vol_arch3 = pd.DataFrame(columns = Ret.columns,index = Ret.index)
arch3test_pval = pd.DataFrame(columns = Ret.columns)
ll_arch3 = pd.DataFrame(columns = Ret.columns)
coeff_arch3 = pd.DataFrame(columns = Ret.columns)
for x in Ret.columns:
   r = Ret.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 3, q = 0)
   res = am.fit()
   coeff_arch3.loc[:,x] = pd.Series(res.params)
   vol_arch3.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(252)
   arch3test_pval.loc[:,x]  = pd.Series(res.arch_lm_test(3).pval)
   ll_arch3.loc[:,x] = pd.Series(res.loglikelihood)
  
# arch 12
vol_arch12 = pd.DataFrame(columns = Ret.columns,index = Ret.index)
arch12test_pval = pd.DataFrame(columns = Ret.columns)
ll_arch12 = pd.DataFrame(columns = Ret.columns)
coeff_arch12 = pd.DataFrame(columns = Ret.columns)
for x in Ret.columns:
   r = Ret.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 12, q = 0)
   res = am.fit()
   coeff_arch12.loc[:,x] = pd.Series(res.params)
   vol_arch12.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(252)
   arch12test_pval.loc[:,x]  = pd.Series(res.arch_lm_test(12).pval)
   ll_arch12.loc[:,x] = pd.Series(res.loglikelihood)
  
# f LR test
LRtest = (ll_arch12-ll_arch3)*2
chisq9 = sp.stats.chi2.ppf(0.95,9)

#%% Question 5

# garch 1

vol_garch11 = pd.DataFrame(columns = Ret.columns,index = Ret.index)
ll_garch1 = pd.DataFrame(columns = Ret.columns)
coeff_garch1 = pd.DataFrame(columns = Ret.columns)
for x in Ret.columns:
   r = Ret.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 1, q = 1)
   res = am.fit()
   coeff_garch1.loc[:,x] = pd.Series(res.params)
   vol_garch11.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(252)
   ll_garch1.loc[:,x] = pd.Series(res.loglikelihood)
   
# garch 2
vol_garch22 = pd.DataFrame(columns = Ret.columns,index = Ret.index)
ll_garch2 = pd.DataFrame(columns = Ret.columns)
coeff_garch2 = pd.DataFrame(columns = Ret.columns)
for x in Ret.columns:
   r = Ret.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 2, q = 2)
   res = am.fit()
   coeff_garch2.loc[:,x] = pd.Series(res.params)
   vol_garch22.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(252)
   ll_garch2.loc[:,x] = pd.Series(res.loglikelihood)


# d plot,resample daily vol to plot with monthly realized
   
fig = plt.figure()
plt.plot(std_month['SP500'])
plt.plot(vol_garch11.loc[std_month.index,'SP500'])
plt.plot(vol_garch22.loc[std_month.index,'SP500'])
plt.legend(['Realized Vol','GARCH(1,1)','GARCH(2,2)'],frameon = False,loc = (0,0.6))
plt.title('SP500')

fig = plt.figure()
plt.plot(std_month['$/Euro'])
plt.plot(vol_garch11.loc[std_month.index,'$/Euro'])
plt.plot(vol_garch22.loc[std_month.index,'$/Euro'])
plt.legend(['Realized Vol','GARCH(1,1)','GARCH(2,2)'],frameon = False,loc = (0,0.6))
plt.title('$/Euro')


fig = plt.figure()
plt.plot(std_month['Oil'])
plt.plot(vol_garch11.loc[std_month.index,'Oil'])
plt.plot(vol_garch22.loc[std_month.index,'Oil'])
plt.legend(['Realized Vol','GARCH(1,1)','GARCH(2,2)'],frameon = False,loc = (0,0.6))
plt.title('Oil')

# e 
LRtest2 = (ll_garch1-ll_arch1)*2
chisq1 = sp.stats.chi2.ppf(0.95,1)
chisq1
# f
LRtest3 = (ll_garch2-ll_garch1)*2
chisq2 = sp.stats.chi2.ppf(0.95,2)
chisq2
#%% Question 7

Ret_month = Ret.groupby([Ret.index.year,Ret.index.month]).agg('sum')
Ret_month.index = MonthIndex

# arch 1

vol_arch1_month = pd.DataFrame(columns = Ret_month.columns,index = Ret_month.index)
arch1test_pval_month = pd.DataFrame(columns = Ret_month.columns)
ll_arch1_month = pd.DataFrame(columns = Ret_month.columns)
coeff_arch1_month = pd.DataFrame(columns = Ret_month.columns)
for x in Ret_month.columns:
   r = Ret_month.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 1, q = 0)
   res = am.fit()
   coeff_arch1_month.loc[:,x] = pd.Series(res.params)
   vol_arch1_month.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(12)
   arch1test_pval_month.loc[:,x]  = pd.Series(res.arch_lm_test(1).pval)
   ll_arch1_month.loc[:,x] = pd.Series(res.loglikelihood)
   
# arch 3
vol_arch3_month = pd.DataFrame(columns = Ret_month.columns,index = Ret_month.index)
arch3test_pval_month = pd.DataFrame(columns = Ret_month.columns)
ll_arch3_month = pd.DataFrame(columns = Ret_month.columns)
coeff_arch3_month = pd.DataFrame(columns = Ret_month.columns)
for x in Ret_month.columns:
   r = Ret_month.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 3, q = 0)
   res = am.fit()
   coeff_arch3_month.loc[:,x] = pd.Series(res.params)
   vol_arch3_month.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(12)
   arch3test_pval_month.loc[:,x]  = pd.Series(res.arch_lm_test(3).pval)
   ll_arch3_month.loc[:,x] = pd.Series(res.loglikelihood)
  
# arch 12
vol_arch12_month = pd.DataFrame(columns = Ret_month.columns,index = Ret_month.index)
arch12test_pval_month = pd.DataFrame(columns = Ret_month.columns)
ll_arch12_month = pd.DataFrame(columns = Ret_month.columns)
coeff_arch12_month = pd.DataFrame(columns = Ret_month.columns)
for x in Ret_month.columns:
   r = Ret_month.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 12, q = 0)
   res = am.fit()
   coeff_arch12_month.loc[:,x] = pd.Series(res.params)
   vol_arch12_month.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(12)
   arch12test_pval_month.loc[:,x]  = pd.Series(res.arch_lm_test(12).pval)
   ll_arch12_month.loc[:,x] = pd.Series(res.loglikelihood)
  
# f LR test
LRtest_month = (ll_arch12_month-ll_arch3_month)*2
chisq9_month = sp.stats.chi2.ppf(0.95,9)


# garch 1

vol_garch11_month = pd.DataFrame(columns = Ret_month.columns,index = Ret_month.index)
ll_garch1_month = pd.DataFrame(columns = Ret_month.columns)
coeff_garch1_month = pd.DataFrame(columns = Ret_month.columns)
for x in Ret_month.columns:
   r = Ret_month.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 1, q = 1)
   res = am.fit()
   coeff_garch1_month.loc[:,x] = pd.Series(res.params)
   vol_garch11_month.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(12)
   ll_garch1_month.loc[:,x] = pd.Series(res.loglikelihood)
   
# garch 2
vol_garch22_month = pd.DataFrame(columns = Ret_month.columns,index = Ret_month.index)
ll_garch2_month = pd.DataFrame(columns = Ret_month.columns)
coeff_garch2_month = pd.DataFrame(columns = Ret_month.columns)
for x in Ret_month.columns:
   r = Ret_month.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 2, q = 2)
   res = am.fit()
   coeff_garch2_month.loc[:,x] = pd.Series(res.params)
   vol_garch22_month.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(12)
   ll_garch2_month.loc[:,x] = pd.Series(res.loglikelihood)


# d plot,resample daily vol to plot with monthly realized
   
fig = plt.figure()
plt.plot(std_month['SP500'])
plt.plot(vol_garch11_month.loc[std_month.index,'SP500'])
plt.plot(vol_garch22_month.loc[std_month.index,'SP500'])
plt.legend(['Realized Vol','GARCH(1,1)','GARCH(2,2)'],frameon = False,loc = (0,0.6))
plt.title('SP500_month')

fig = plt.figure()
plt.plot(std_month['$/Euro'])
plt.plot(vol_garch11_month.loc[std_month.index,'$/Euro'])
plt.plot(vol_garch22_month.loc[std_month.index,'$/Euro'])
plt.legend(['Realized Vol','GARCH(1,1)','GARCH(2,2)'],frameon = False,loc = (0,0.6))
plt.title('$/Euro_month')


fig = plt.figure()
plt.plot(std_month['Oil'])
plt.plot(vol_garch11_month.loc[std_month.index,'Oil'])
plt.plot(vol_garch22_month.loc[std_month.index,'Oil'])
plt.legend(['Realized Vol','GARCH(1,1)','GARCH(2,2)'],frameon = False,loc = (0,0.6))
plt.title('Oil_month')

# e 
LRtest2_month = (ll_garch1_month-ll_arch1_month)*2
chisq1_month = sp.stats.chi2.ppf(0.95,1)

# f
LRtest3_month = (ll_garch2_month-ll_garch1_month)*2
chisq2_month = sp.stats.chi2.ppf(0.95,2)





#Part 2

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime as dt
import arch

"""
Read the data

"""

data = pd.read_excel('vol_data_homework.xlsx',skiprows = 1)

data[data=='ND'] = np.nan

data = data.iloc[:,[0,1,3,5]]

data.index = data['Code']

del data['Code']

data.columns = ['SP500','$/Euro','Oil']

#%%Question 2 
"""
Compute Return

"""

Ret = 100*pd.DataFrame(np.log(np.asarray(np.array(data.iloc[1:,])/\
                          np.array(data.iloc[:-1,]),dtype = float))\
                   ,index = data[1:].index,columns = data.columns)

mean = Ret.mean()*252
mean
std = Ret.std()*np.sqrt(252)
std
skewness = Ret.skew()/(252**(1/2))
skewness
kurtosis = (Ret.kurt()+3)/252
kurtosis

#%%Question 3
"""
Compound to monthly return

"""
std_month = (Ret.groupby([Ret.index.year,Ret.index.month])\
                 .agg(np.nanstd))*np.sqrt(252)
Switch = pd.Series(Ret.index).groupby([Ret.index.year,Ret.index.month]).agg('last')
MonthIndex = pd.to_datetime(Switch)
std_month.index = MonthIndex

"""
Filter the return
"""

Ret = Ret.fillna(0)

#%%Question 4
# arch 1

vol_arch1 = pd.DataFrame(columns = Ret.columns,index = Ret.index)
arch1test_pval = pd.DataFrame(columns = Ret.columns)
ll_arch1 = pd.DataFrame(columns = Ret.columns)
coeff_arch1 = pd.DataFrame(columns = Ret.columns)
for x in Ret.columns:
   r = Ret.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 1, q = 0)
   res = am.fit()
   coeff_arch1.loc[:,x] = pd.Series(res.params)
   vol_arch1.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(252)
   arch1test_pval.loc[:,x]  = pd.Series(res.arch_lm_test(1).pval)
   ll_arch1.loc[:,x] = pd.Series(res.loglikelihood)
   
# arch 3
vol_arch3 = pd.DataFrame(columns = Ret.columns,index = Ret.index)
arch3test_pval = pd.DataFrame(columns = Ret.columns)
ll_arch3 = pd.DataFrame(columns = Ret.columns)
coeff_arch3 = pd.DataFrame(columns = Ret.columns)
for x in Ret.columns:
   r = Ret.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 3, q = 0)
   res = am.fit()
   coeff_arch3.loc[:,x] = pd.Series(res.params)
   vol_arch3.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(252)
   arch3test_pval.loc[:,x]  = pd.Series(res.arch_lm_test(3).pval)
   ll_arch3.loc[:,x] = pd.Series(res.loglikelihood)
  
# arch 12
vol_arch12 = pd.DataFrame(columns = Ret.columns,index = Ret.index)
arch12test_pval = pd.DataFrame(columns = Ret.columns)
ll_arch12 = pd.DataFrame(columns = Ret.columns)
coeff_arch12 = pd.DataFrame(columns = Ret.columns)
for x in Ret.columns:
   r = Ret.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 12, q = 0)
   res = am.fit()
   coeff_arch12.loc[:,x] = pd.Series(res.params)
   vol_arch12.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(252)
   arch12test_pval.loc[:,x]  = pd.Series(res.arch_lm_test(12).pval)
   ll_arch12.loc[:,x] = pd.Series(res.loglikelihood)
  
# f LR test
LRtest = (ll_arch12-ll_arch3)*2
chisq9 = sp.stats.chi2.ppf(0.95,9)

#%% Question 5

# garch 1

vol_garch11 = pd.DataFrame(columns = Ret.columns,index = Ret.index)
ll_garch1 = pd.DataFrame(columns = Ret.columns)
coeff_garch1 = pd.DataFrame(columns = Ret.columns)
for x in Ret.columns:
   r = Ret.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 1, q = 1)
   res = am.fit()
   coeff_garch1.loc[:,x] = pd.Series(res.params)
   vol_garch11.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(252)
   ll_garch1.loc[:,x] = pd.Series(res.loglikelihood)
   
# garch 2
vol_garch22 = pd.DataFrame(columns = Ret.columns,index = Ret.index)
ll_garch2 = pd.DataFrame(columns = Ret.columns)
coeff_garch2 = pd.DataFrame(columns = Ret.columns)
for x in Ret.columns:
   r = Ret.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 2, q = 2)
   res = am.fit()
   coeff_garch2.loc[:,x] = pd.Series(res.params)
   vol_garch22.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(252)
   ll_garch2.loc[:,x] = pd.Series(res.loglikelihood)


# d plot,resample daily vol to plot with monthly realized
   
fig = plt.figure()
plt.plot(std_month['SP500'])
plt.plot(vol_garch11.loc[std_month.index,'SP500'])
plt.plot(vol_garch22.loc[std_month.index,'SP500'])
plt.legend(['Realized Vol','GARCH(1,1)','GARCH(2,2)'],frameon = False,loc = (0,0.6))
plt.title('SP500')

fig = plt.figure()
plt.plot(std_month['$/Euro'])
plt.plot(vol_garch11.loc[std_month.index,'$/Euro'])
plt.plot(vol_garch22.loc[std_month.index,'$/Euro'])
plt.legend(['Realized Vol','GARCH(1,1)','GARCH(2,2)'],frameon = False,loc = (0,0.6))
plt.title('$/Euro')


fig = plt.figure()
plt.plot(std_month['Oil'])
plt.plot(vol_garch11.loc[std_month.index,'Oil'])
plt.plot(vol_garch22.loc[std_month.index,'Oil'])
plt.legend(['Realized Vol','GARCH(1,1)','GARCH(2,2)'],frameon = False,loc = (0,0.6))
plt.title('Oil')

# e 
LRtest2 = (ll_garch1-ll_arch1)*2
chisq1 = sp.stats.chi2.ppf(0.95,1)
chisq1
# f
LRtest3 = (ll_garch2-ll_garch1)*2
chisq2 = sp.stats.chi2.ppf(0.95,2)
chisq2
#%% Question 7

Ret_month = Ret.groupby([Ret.index.year,Ret.index.month]).agg('sum')
Ret_month.index = MonthIndex

# arch 1

vol_arch1_month = pd.DataFrame(columns = Ret_month.columns,index = Ret_month.index)
arch1test_pval_month = pd.DataFrame(columns = Ret_month.columns)
ll_arch1_month = pd.DataFrame(columns = Ret_month.columns)
coeff_arch1_month = pd.DataFrame(columns = Ret_month.columns)
for x in Ret_month.columns:
   r = Ret_month.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 1, q = 0)
   res = am.fit()
   coeff_arch1_month.loc[:,x] = pd.Series(res.params)
   vol_arch1_month.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(12)
   arch1test_pval_month.loc[:,x]  = pd.Series(res.arch_lm_test(1).pval)
   ll_arch1_month.loc[:,x] = pd.Series(res.loglikelihood)
   
# arch 3
vol_arch3_month = pd.DataFrame(columns = Ret_month.columns,index = Ret_month.index)
arch3test_pval_month = pd.DataFrame(columns = Ret_month.columns)
ll_arch3_month = pd.DataFrame(columns = Ret_month.columns)
coeff_arch3_month = pd.DataFrame(columns = Ret_month.columns)
for x in Ret_month.columns:
   r = Ret_month.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 3, q = 0)
   res = am.fit()
   coeff_arch3_month.loc[:,x] = pd.Series(res.params)
   vol_arch3_month.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(12)
   arch3test_pval_month.loc[:,x]  = pd.Series(res.arch_lm_test(3).pval)
   ll_arch3_month.loc[:,x] = pd.Series(res.loglikelihood)
  
# arch 12
vol_arch12_month = pd.DataFrame(columns = Ret_month.columns,index = Ret_month.index)
arch12test_pval_month = pd.DataFrame(columns = Ret_month.columns)
ll_arch12_month = pd.DataFrame(columns = Ret_month.columns)
coeff_arch12_month = pd.DataFrame(columns = Ret_month.columns)
for x in Ret_month.columns:
   r = Ret_month.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 12, q = 0)
   res = am.fit()
   coeff_arch12_month.loc[:,x] = pd.Series(res.params)
   vol_arch12_month.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(12)
   arch12test_pval_month.loc[:,x]  = pd.Series(res.arch_lm_test(12).pval)
   ll_arch12_month.loc[:,x] = pd.Series(res.loglikelihood)
  
# f LR test
LRtest_month = (ll_arch12_month-ll_arch3_month)*2
chisq9_month = sp.stats.chi2.ppf(0.95,9)


# garch 1

vol_garch11_month = pd.DataFrame(columns = Ret_month.columns,index = Ret_month.index)
ll_garch1_month = pd.DataFrame(columns = Ret_month.columns)
coeff_garch1_month = pd.DataFrame(columns = Ret_month.columns)
for x in Ret_month.columns:
   r = Ret_month.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 1, q = 1)
   res = am.fit()
   coeff_garch1_month.loc[:,x] = pd.Series(res.params)
   vol_garch11_month.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(12)
   ll_garch1_month.loc[:,x] = pd.Series(res.loglikelihood)
   
# garch 2
vol_garch22_month = pd.DataFrame(columns = Ret_month.columns,index = Ret_month.index)
ll_garch2_month = pd.DataFrame(columns = Ret_month.columns)
coeff_garch2_month = pd.DataFrame(columns = Ret_month.columns)
for x in Ret_month.columns:
   r = Ret_month.loc[:,x]
   r = r[r!=0]
   am = arch.arch_model(r, p = 2, q = 2)
   res = am.fit()
   coeff_garch2_month.loc[:,x] = pd.Series(res.params)
   vol_garch22_month.loc[:,x] = pd.Series(res.conditional_volatility)\
        *np.sqrt(12)
   ll_garch2_month.loc[:,x] = pd.Series(res.loglikelihood)


# d plot,resample daily vol to plot with monthly realized
   
fig = plt.figure()
plt.plot(std_month['SP500'])
plt.plot(vol_garch11_month.loc[std_month.index,'SP500'])
plt.plot(vol_garch22_month.loc[std_month.index,'SP500'])
plt.legend(['Realized Vol','GARCH(1,1)','GARCH(2,2)'],frameon = False,loc = (0,0.6))
plt.title('SP500_month')

fig = plt.figure()
plt.plot(std_month['$/Euro'])
plt.plot(vol_garch11_month.loc[std_month.index,'$/Euro'])
plt.plot(vol_garch22_month.loc[std_month.index,'$/Euro'])
plt.legend(['Realized Vol','GARCH(1,1)','GARCH(2,2)'],frameon = False,loc = (0,0.6))
plt.title('$/Euro_month')


fig = plt.figure()
plt.plot(std_month['Oil'])
plt.plot(vol_garch11_month.loc[std_month.index,'Oil'])
plt.plot(vol_garch22_month.loc[std_month.index,'Oil'])
plt.legend(['Realized Vol','GARCH(1,1)','GARCH(2,2)'],frameon = False,loc = (0,0.6))
plt.title('Oil_month')

# e 
LRtest2_month = (ll_garch1_month-ll_arch1_month)*2
chisq1_month = sp.stats.chi2.ppf(0.95,1)

# f
LRtest3_month = (ll_garch2_month-ll_garch1_month)*2
chisq2_month = sp.stats.chi2.ppf(0.95,2)