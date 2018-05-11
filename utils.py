import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def make_scatter(df):
    fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(21,5))
    for ax in axes:
        ax.set_ylim([-2.0,2.0])
        ax.set_xlim([-2.0,2.0])
                                    
    pairs=[('norm-lx','norm-ly'),('norm-rx','norm-ry')]
    for i in (0,1):
        df.reset_index().plot(kind='scatter',x=pairs[i][0],y=pairs[i][1],c='index',ax=axes[i])
    return fig                                
                        

def features_list(text):
    return [text+'-'+side+dim for side in ('r','l') for dim in ('x','y')]

def get_word(training,features,key,word):
    t=training[key]
    f=features[key]
    data,xlens=t.get_word_Xlengths(word)
    xi=np.cumsum(np.array([0]+xlens))
    return [pd.DataFrame(data[xi[i]:xi[i+1]],columns=f) for i in range(0,len(xi)-1)]

#def get_training_samples(feature_name='custom',words=['CHOCOLATE','VEGETABLE','STUDENT']):
#    training_sample=dict()
#    for word in words:
#        training_sample[word]=get_sample(training[feature_name],word)
#    return training_sample

def make_kde(samples,figsize=(12,12)):
    fig,axes=plt.subplots(nrows=2,ncols=2,figsize=figsize)
    for df in samples:
        try:
            df.iloc[:,0].plot.kde(title=df.iloc[:,0].name,ax=axes[0][0])
        except: 
            continue

        try:
            df.iloc[:,1].plot.kde(title=df.iloc[:,1].name,ax=axes[1][0])
        except:
            continue
        
        try:
            df.iloc[:,2].plot.kde(title=df.iloc[:,2].name,ax=axes[0][1])
        except:
            continue
        
        try:    
            df.iloc[:,3].plot.kde(title=df.iloc[:,3].name,ax=axes[1][1])
        except:
            continue

    return fig

def make_histogram(samples,figsize=(12,12),normed=False):
    fig,axes=plt.subplots(nrows=2,ncols=2,figsize=figsize)
    if normed:
        for ax in axes:
            ax[0].set_xlim([-3.0,3.0])
            ax[1].set_xlim([-3.0,3.0])
            ax[0].set_ylim([0,20])
            ax[1].set_ylim([0,20])


    for df in samples:
        df.iloc[:,0].plot.hist(title=df.iloc[:,0].name,ax=axes[0][0],stacked=True)
        df.iloc[:,1].plot.hist(title=df.iloc[:,1].name,ax=axes[1][0],stacked=True)
        df.iloc[:,2].plot.hist(title=df.iloc[:,2].name,ax=axes[0][1],stacked=True)
        df.iloc[:,3].plot.hist(title=df.iloc[:,3].name,ax=axes[1][1],stacked=True)
    return fig
 
def make_lines(samples,figsize=(12,12)):
    fig,axes=plt.subplots(nrows=2,ncols=2,figsize=figsize)
   

    for df in samples:
        df.iloc[:,0].plot(title=df.iloc[:,0].name,ax=axes[0][0])
        df.iloc[:,1].plot(title=df.iloc[:,1].name,ax=axes[1][0])
        df.iloc[:,2].plot(title=df.iloc[:,2].name,ax=axes[0][1])
        df.iloc[:,3].plot(title=df.iloc[:,3].name,ax=axes[1][1])
    return fig


def likelihood_comparison(train_a_word,features,demoword = 'BOOK'):
    model=dict()
    from collections import OrderedDict
    logL=OrderedDict()
    features_names=['grnd','polar','delta','norm','custom']
    for k in features_names:
        model[k], logL[k] = train_a_word(demoword, 3, features[k])
        print("model {} on {}: {} states, logL {:+1.2}".format(k,demoword, model[k].n_components,logL[k]))
    return model,logL

def likelihood_barchart(logL,word):
    objects=tuple(logL.keys())
    y_pos=np.arange(len(objects))
    values=list(logL.values())

    plt.barh(y_pos,values,align='center',alpha=0.5)
    plt.yticks(y_pos,objects)
    plt.xlabel('logL')
    plt.title('log likelihood by features ({})'.format(word))
    plt.show()

def dot_hmm(m):    
    import pydot    
    g = pydot.Dot()
    g.rankdir='LR';

    g.set_type('digraph')    
    g.set_node_defaults(fontname = "helvetica",fontsize=10)    
    g.set_edge_defaults(fontname = "helvetica",fontsize=9)        
 
    for i in range(0,len(m)):
        g.add_node(pydot.Node(name='state_'+str(i),label='state_'+str(i),penwidth=2,shape='square'))
    
    for i in range(0,len(m)):
        for j in range(0,len(m)):
            print('{}->{}:{}'.format(i,j,m[i,j]))
            g.add_edge(pydot.Edge('state_'+str(i),'state_'+str(j),label='{:0.2f}'.format(m[i,j]),penwidth=1))
        
    return g.to_string()

def show_model_stats(word, model):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))    
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])    
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        print()