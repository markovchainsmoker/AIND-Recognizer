import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=self.verbose).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        def catch(func, handle=lambda e : e, *args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return handle(e)
        #set up all models, then run the BIC method onn them
        model={k:self.base_model(k) for k in range(self.min_n_components, self.max_n_components)}
        def BIC(model):
            
            #if not model.monitor_.converged:
            #    return float('inf')
#            logL=catch(model.score(self.X, self.lengths))
            try:
                logL = model.score(self.X, self.lengths)
                # number of features
                n = self.X.shape[1] 
                # number of states
                k = model.n_components
	        # total parameter count
                p = k * (k - 1) + 2 * n * k
                logN = np.log(self.X.shape[0]) 
                return -2 * logL + p * logN
            except Exception as e:
                return float('inf')
        
        bic={k:BIC(v) for k,v in model.items() if v != None}
        return model[min(bic, key=lambda key: bic[key])]

class SelectorDIC(ModelSelector):
    #initialise these as empty class attributes to SelectorDIC
    models=dict()
    scores=dict()
    failed=dict()

    #static method since it needs not refer neither class nor instance 
    @staticmethod
    def fit_model(n_states,Xlengths):
        try:
            return GaussianHMM(n_components=n_states,covariance_type='diag',n_iter=1000,random_state=14,verbose=False).fit(*Xlengths)
        except Exception as e:
            return None

    @staticmethod
    def get_logL(model):
        try:
            return model.score()
        except Exception as e:
            return None

    #class method, precalculates all models to reduce incremental select time to close to zero
    @classmethod
    def init_models(cls, inst): 

        for n_states in range(inst.min_n_components, inst.max_n_components+1): 
            SelectorDIC.models[n_states]={k:SelectorDIC.fit_model(n_states=n_states,Xlengths=v) for k,v in inst.hwords.items()}
            SelectorDIC.scores[n_states]=dict()
            n=0
            for k,v in SelectorDIC.models[n_states].items():
                try:
                    SelectorDIC.scores[n_states][k]=v.score(*inst.hwords[k]) 
                except Exception as e:
                    n+=1
                    continue
            SelectorDIC.failed[n_states]=n
            
    #this init is just to add our init_models
    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,n_constant=3,
            min_n_components=2, max_n_components=10,random_state=14, verbose=False):
        super().__init__(all_word_sequences=all_word_sequences, all_word_Xlengths=all_word_Xlengths, this_word=this_word,
                n_constant=n_constant,min_n_components=min_n_components, max_n_components=max_n_components,random_state=random_state, verbose=False)
        if len(SelectorDIC.models)==0:
            self.init_models(self)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score,best_model = float('-inf'),None
        for n_states in range(self.min_n_components,self.max_n_components+1):
            models,scores=SelectorDIC.models[n_states],SelectorDIC.scores[n_states]            
            
            if(self.this_word in list(scores.keys())):
            
                score=scores[self.this_word]
                dic=score-np.mean([v for k,v in scores.items() if k!=self.this_word])
                dic2=score-(sum(scores.values())-score)/(len(scores.values())-1)
                assert abs(dic/dic2-1)<1e-6 , '{} {} {}'.format(n_states,dic,dic2)
                #two equivalent ways to set up the calculation
                     
                if dic>best_score:
                    best_score,best_model=dic,models[self.this_word]
        return best_model
 
 
class SelectorCV(ModelSelector):

    @staticmethod
    def fit_model(n_states,Xlengths):
        #warnings.filterwarnings('ignore',category=DepreciationWarning)
        try:
            return GaussianHMM(n_components=n_states,covariance_type='diag',n_iter=1000,random_state=14,verbose=False).fit(*Xlengths)
        except Exception as e:
            return None
 
    def select(self): 
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        n_splits = 3 
 
        #basic logL maximation but for splits using KFold 
        best_score, best_model = float("-inf"), None 
          
        for n_states in range(self.min_n_components, self.max_n_components + 1): 
            scores, n_splits = [], n_splits 
            model, logL = None, None 
            #we need at least as many sequences as n_splits   
            if(len(self.sequences) < n_splits): 
                break 
              
            split_method = KFold(random_state=self.random_state, n_splits=n_splits) 
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences): 
                Xlengths_train = combine_sequences(cv_train_idx, self.sequences) 
                Xlengths_test  = combine_sequences(cv_test_idx, self.sequences) 
                
                model=SelectorCV.fit_model(n_states=n_states,Xlengths=Xlengths_train)
                if(model):
                    try:
                        logL=model.score(*Xlengths_test)
                        scores.append(logL)
                    except Exception as e:
                        pass
            avg=np.mean(scores) if len(scores)>0 else float('-inf')
            if avg>best_score:
                best_score,best_model=avg,model
        return best_model if best_model is not None else self.base_model(self.n_constant)
                
