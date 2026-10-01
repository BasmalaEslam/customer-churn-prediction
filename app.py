"""Synthetic telecom churn lab. Real deployment requires representative labelled data."""
import argparse, json
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
from common import ROOT, dump, serve

NUM = ['tenure_months','monthly_fee','support_tickets','usage_hours']
CAT = ['contract','payment']
FEATURES = NUM + CAT

def generate():
    rng = np.random.default_rng(42)
    n = 2200
    frame = pd.DataFrame({'customer_id':[f'C{i:05}' for i in range(n)],
        'tenure_months':rng.integers(1,73,n), 'monthly_fee':rng.uniform(150,1200,n).round(2),
        'support_tickets':rng.poisson(1.8,n), 'usage_hours':rng.gamma(3,8,n).round(2),
        'contract':rng.choice(['monthly','annual'],n,p=[.7,.3]),
        'payment':rng.choice(['automatic','manual'],n)})
    logit = -1.2-.035*frame.tenure_months+.0012*frame.monthly_fee+.5*frame.support_tickets-.055*frame.usage_hours+1.0*(frame.contract=='monthly')+.35*(frame.payment=='manual')
    frame['churn'] = (rng.random(n) < 1/(1+np.exp(-logit))).astype(int)
    frame.loc[rng.choice(n,50,replace=False),'usage_hours']=np.nan
    (ROOT/'data').mkdir(exist_ok=True)
    frame.to_csv(ROOT/'data/customers.csv',index=False)

def metric(y, probability, threshold):
    pred = probability >= threshold
    return {'roc_auc':float(roc_auc_score(y,probability)), 'pr_auc_average_precision':float(average_precision_score(y,probability)),
        'precision':float(precision_score(y,pred,zero_division=0)), 'recall':float(recall_score(y,pred,zero_division=0)),
        'f1':float(f1_score(y,pred,zero_division=0)), 'confusion_matrix':confusion_matrix(y,pred).tolist()}

def train():
    if not (ROOT/'data/customers.csv').exists(): generate()
    frame = pd.read_csv(ROOT/'data/customers.csv')
    if not set(FEATURES+['customer_id','churn']) <= set(frame): raise ValueError('CSV schema mismatch.')
    if frame.customer_id.duplicated().any(): raise ValueError('One row per unique customer required.')
    if not set(frame.churn.unique()) == {0,1}: raise ValueError('churn must contain 0 and 1.')
    X, y = frame[FEATURES], frame.churn
    Xdev, Xtest, ydev, ytest = train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    Xfit, Xval, yfit, yval = train_test_split(Xdev,ydev,test_size=.25,stratify=ydev,random_state=43)
    prep = ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),
        ('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),CAT)])
    models = {'majority_baseline':DummyClassifier(strategy='prior'), 'logistic':LogisticRegression(max_iter=1500),
        'random_forest':RandomForestClassifier(n_estimators=120,max_depth=8,min_samples_leaf=8,random_state=42,n_jobs=1)}
    candidates, fitted = {}, {}
    for name, estimator in models.items():
        pipe = Pipeline([('prepare',prep),('model',estimator)])
        cv = cross_val_score(pipe,Xfit,yfit,cv=StratifiedKFold(4,shuffle=True,random_state=42),scoring='roc_auc')
        pipe.fit(Xfit,yfit)
        candidates[name]={'cv_roc_auc_mean':float(cv.mean()),'cv_roc_auc_std':float(cv.std())}
        fitted[name]=pipe
    winner=max(candidates,key=lambda k:candidates[k]['cv_roc_auc_mean'])
    model=fitted[winner]
    probabilities=model.predict_proba(Xval)[:,1]
    thresholds=np.arange(.1,.91,.02)
    threshold=float(max(thresholds,key=lambda t:f1_score(yval,probabilities>=t,zero_division=0)))
    # Keep the fitted model unchanged after threshold tuning; test remains untouched until here.
    test_p=model.predict_proba(Xtest)[:,1]
    importance=permutation_importance(model,Xtest,ytest,n_repeats=6,random_state=42,scoring='roc_auc')
    report={'dataset':'bundled synthetic demo unless you replaced data/customers.csv','rows':len(frame),
        'split':{'fit':len(Xfit),'validation':len(Xval),'test':len(Xtest)},'selected_model':winner,
        'threshold_selected_on_validation':threshold,'candidates':candidates,
        'test':metric(ytest,test_p,threshold),'baseline_test':metric(ytest,fitted['majority_baseline'].predict_proba(Xtest)[:,1],.5),
        'permutation_importance_roc_auc_drop':dict(zip(FEATURES,map(float,importance.importances_mean))),
        'limitations':['Synthetic labels encode an invented mechanism. These metrics do not establish real customer performance.',
            'No causal claim: risk ranking does not prove a discount prevents churn.','Random split assumes independent customers; use time-based validation for production.']}
    (ROOT/'artifacts').mkdir(exist_ok=True)
    joblib.dump({'model':model,'threshold':threshold},ROOT/'artifacts/model.joblib')
    dump(ROOT/'reports/metrics.json',report)
    return report

def validate(value):
    if not isinstance(value,dict): raise ValueError('customer must be an object')
    for key in FEATURES:
        if key not in value: raise ValueError('Missing '+key)
    for key in NUM:
        v=value[key]
        if v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0):
            raise ValueError(key+' must be a non-negative finite number or null')
    if value['contract'] not in ['monthly','annual'] or value['payment'] not in ['manual','automatic']:
        raise ValueError('Unsupported contract/payment category')
    return {k:value[k] for k in FEATURES}

def predict(value):
    row=validate(value)
    bundle=joblib.load(ROOT/'artifacts/model.joblib')
    p=float(bundle['model'].predict_proba(pd.DataFrame([row]))[0,1])
    return {'churn_probability':round(p,4),'needs_review':bool(p>=bundle['threshold']),
        'threshold':bundle['threshold'],'suggested_action':'Human retention review' if p>=bundle['threshold'] else 'Monitor engagement',
        'data_scope':'synthetic demo model; not calibrated on your business'}

def dispatch(path,payload):
    if path=='/report': return json.loads((ROOT/'reports/metrics.json').read_text())
    if path=='/predict': return predict(payload['customer'])
    if path=='/batch':
        rows=payload['customers']
        if not isinstance(rows,list) or not 1<=len(rows)<=100: raise ValueError('Provide 1–100 customers')
        output=[{'customer_id':str(row.get('customer_id',i)),**predict(row)} for i,row in enumerate(rows)]
        return {'review_queue':sorted(output,key=lambda x:x['churn_probability'],reverse=True)}
    raise ValueError('Unknown endpoint')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','demo','serve']);args=p.parse_args()
    if args.command=='train': print(json.dumps(train(),indent=2))
    elif args.command=='demo':
        result=dispatch('/predict',{'customer':{'tenure_months':4,'monthly_fee':650,'support_tickets':5,'usage_hours':8,'contract':'monthly','payment':'manual'}})
        dump(ROOT/'reports/demo.json',result);print(json.dumps(result,indent=2))
    else: serve(dispatch,8101)
