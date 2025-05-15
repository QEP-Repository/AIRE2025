from functions.helpers import *
from functions.HDB523_STD5_preprocessing import *
from functions.fitness import *
import statsmodels.api as sm

# CONFIG FILE
config = dict()
config['runcontrol'] = dict()
config['userdata'] = dict()

config['runcontrol']['modality'] = 'all_db'
config['userdata']['add_var_to_fit'] = 'DELTA'
config['userdata']['predictor_names'] =  ['BT', 'TEV', 'AMIN', 'NEL', 'RGEO', 'KAPPAA', 'IP','MEFF', 'epsilon', 
                  'mu0', 'mp', 'e']
config['userdata']['db_filename'] = '../dataset/HDB5.2.3_STD5.xlsx'
config['userdata']['db_subset'] = 'STD5ELMYHITERlike'

# LOAD AND PROCESS DB
db = pd.read_excel(config['userdata']['db_filename'], header=0)
db = database_processing(db,config['userdata']['db_subset'])
Xtest, ytest, weights_test, Xtrain, ytrain, weights_train, var2add_train, var2add_test = prepare_data_HDB23_STD5(db, config)

# FIT 
gene = np.load('../saved_runs/win_x_right_ITPA5.npy')

pi_train = np.exp(np.log(Xtrain).dot(gene))
pi_test = np.exp(np.log(Xtest).dot(gene))

if config['userdata']['add_var_to_fit'] != None:
    pi_train = np.hstack((pi_train, var2add_train.reshape(var2add_train.shape[0],1)))
    pi_test = np.hstack((pi_test, var2add_test.reshape(var2add_test.shape[0],1)))

# get complexity
complexity = get_complexity(gene)

# log 
ytrain_log = np.log(ytrain)
ytest_log = np.log(ytest)
pi_train_log = np.log(pi_train)
pi_test_log = np.log(pi_test)

# log fit dimless var to target
model = sm.WLS(ytrain_log, sm.add_constant(pi_train_log), weights=weights_train).fit()
model_pred = model.get_prediction(sm.add_constant(pi_test_log))

# EVALUATE PERFORMANCES
ytest_log_pred = model_pred.predicted_mean
mse_mean = np.mean((ytest_log_pred - ytest_log)**2)
Rq_mean = 1-mse_mean/np.var(ytest_log)

ytest_log_pred_upper = model_pred.conf_int()[:,0]
mse_upper = np.mean((ytest_log_pred_upper - ytest_log)**2)
Rq_upper =  1-mse_upper/np.var(ytest_log)

ytest_log_pred_lower = model_pred.conf_int()[:,1]
mse_lower = np.mean((ytest_log_pred_lower - ytest_log)**2)
Rq_lower =  1-mse_lower/np.var(ytest_log)

scatter(ytest_log_pred,ytest_log, Rq_mean, mse_mean)

# EXTRAPOLATION TO ITER 
XIter = np.array([5.3, 8.6*10**3, 2, 10.3*10**19, 6.2, 1.7, 15*10**6,2.5,
                  db['epsilon'].values[1], db['mu0'].values[1], db['mp'].values[1],
                  db['e'].values[1]]).reshape(-1,1).T

pi_Iter = np.exp(np.log(XIter).dot(gene))
pi_Iter = np.append(pi_Iter, 1.48).reshape(1, -1)

pi_Iter_log = np.log(pi_Iter)

model_Iter_pred = model.get_prediction(sm.add_constant(pi_Iter_log, has_constant='add'))
ioncycfreq_Iter = 5.3/2.5/(2*math.pi)*10
tau_Iter = np.exp(model_Iter_pred.predicted_mean)/ioncycfreq_Iter
tau_Iter_upper = np.exp(model_Iter_pred.conf_int()[:,0])/ioncycfreq_Iter
tau_iter_lower = np.exp(model_Iter_pred.conf_int()[:,1])/ioncycfreq_Iter

plt.figure
plt.title('Fit with traditional variables')
plt.hist(ytest_log-ytest_log_pred)
plt.xlabel('residuals')
plt.ylabel('counts')
