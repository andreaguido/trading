import numpy as np

# Calculate returns
rets = np.log(df / df.shift(1))

def port_ret(weights):
    return np.sum(rets.mean() * weights) * 252

def port_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

pret=[]
pvols=[]

def run_optimization(weights, port_ret):
    for p in range (5000):
        weights = np.random.random(noa)
        weights /= np.sum(weights)
        pret.append(port_ret(weights))
        pvols.append(port_vol(weights))

pret = np.array(pret)
pvols = np.array(pvols)


plt.figure(figsize=(10, 6))
plt.scatter(pvols, pret, c=pret / pvols, marker='o', cmap='coolwarm')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

## Max return

def min_func_sharpe(weights):
    return -port_ret(weights) / port_vol(weights)

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))
eweights = np.array(noa * [1. / noa,])

min_func_sharpe(eweights)

opts = sco.minimize(min_func_sharpe, eweights, method='SLSQP', bounds=bnds, constraints=cons)

print(opts)



for i in range(noa):

    if i == 0:

        print("Maximum return portfolio : \n")



    print(assets[i], opts['x'][i].round(3))

pass



print(

    "\n Returns : ", port_ret(opts['x']).round(3),

    "\n Volatility : ", port_vol(opts['x']).round(3),

    "Sharpe ration : ", port_ret(opts['x']) / port_vol(opts['x'])

)



### Volatility minimization

optv = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)



for i in range(noa):

    if i == 0:

        print("Min variance portfolio : \n")



    print(assets[i], optv['x'][i].round(3))

pass



print(

    "\n Returns : ", port_ret(optv['x']).round(3),

    "\n Volatility : ", port_vol(optv['x']).round(3),

    "Sharpe ration : ", port_ret(optv['x']) / port_vol(optv['x'])

)



optv['x'].round(3)

port_vol(optv['x']).round(3)

port_ret(optv['x']).round(3)

port_ret(optv['x']) / port_vol(optv['x'])



## Efficient Frontier

cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - tret}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

bnds = tuple((0, 1) for x in eweights)

trets = np.linspace(min(pret), max(np.append(pret, port_ret(opts['x']))), 50)

tvols = []



for tret in trets:

    res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)

    tvols.append(res['fun'])

tvols = np.array(tvols)



plt.figure(figsize=(10, 6))

plt.scatter(pvols, pret, c=pret / pvols, marker='.', alpha=0.8, cmap='coolwarm')

plt.plot(tvols, trets, 'b', lw=4.0)

plt.plot(port_vol(opts['x']), port_ret(opts['x']), 'y*', markersize=15.0)

plt.plot(port_vol(optv['x']), port_ret(optv['x']), 'r*', markersize=15.0)

plt.xlabel('expected volatility')

plt.ylabel('expected return')

plt.colorbar(label='Sharpe ratio')



## CAPM



ind = np.argmin(tvols)

evols = tvols[ind:]

erets = trets[ind:]

tck = sci.splrep(evols, erets)



def f(x):

    ''' Efficient frontier function (splines approximation). '''

    return sci.splev(x, tck, der=0)



def df(x):

    ''' First derivative of efficient frontier function. '''

    return sci.splev(x, tck, der=1)



def equations(p, rf=0.01):

    eq1 = rf - p[0]

    eq2 = rf + p[1] * p[2] - f(p[2])

    eq3 = p[1] - df(p[2])

    return eq1, eq2, eq3



opt = sco.fsolve(equations, [0.01, 2, 0.2])



opt



np.round(equations(opt), 6)



plt.figure(figsize=(10, 6))

plt.scatter(pvols, pret, c=(pret - 0.01) / pvols, marker='.', cmap='coolwarm')

plt.plot(evols, erets, 'b', lw=4.0)

cx = np.linspace(0.0, 0.3)

plt.plot(cx, opt[0] + opt[1] * cx, 'r', lw=1.5)

plt.plot(opt[2], f(opt[2]), 'y*', markersize=15.0)

plt.grid(True)

plt.axhline(0, color='k', ls='--', lw=2.0)

plt.axvline(0, color='k', ls='--', lw=2.0)

plt.xlabel('expected volatility')

plt.ylabel('expected return')

plt.colorbar(label='Sharpe ratio')



cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - f(opt[2])},

{'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

res = sco.minimize(port_vol, eweights, method='SLSQP',

bounds=bnds, constraints=cons)



res['x'].round(3)

port_ret(res['x'])

port_vol(res['x'])

port_ret(res['x']) / port_vol(res['x'])



## Max sharpe portfolio

w = np.linspace(0,1, 10)

retcapm = []

volcapm = []



for i in w:

    retcapm.append(i*port_ret(opts['x']) + (1-i)*0.01)

    volcapm.append(i*port_vol(opts['x']) + (1-i)*0)



plt.figure(figsize=(10, 6))

plt.scatter(pvols, pret, c=(pret - 0.01) / pvols, marker='.', cmap='coolwarm')

plt.plot(evols, erets, 'b', lw=4.0)

cx = np.linspace(0.0, 0.3)

plt.plot(volcapm, retcapm)

plt.grid(True)

plt.axhline(0, color='k', ls='--', lw=2.0)

plt.axvline(0, color='k', ls='--', lw=2.0)

plt.xlabel('expected volatility')

plt.ylabel('expected return')

plt.colorbar(label='Sharpe ratio')