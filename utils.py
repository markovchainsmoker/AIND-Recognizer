import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from asl_data import AslDb



def init():

    asl = AslDb()
    #dimensions
    hand=['right','left']
    side=['r','l']
    cartesian = ['x','y']
    polar = ['r','theta']

    #rename the raw data for consistency
    raw_names={h+'-'+c:'raw-'+h[0]+c for h in hand for c in cartesian}
    asl.df=asl.df.rename(columns=raw_names)

    cartesian_features=['grnd','norm','delta']
    features={k:[k+'-'+h[0]+c for h in hand for c in cartesian] for k in cartesian_features}

    features['polar']=['polar'+'-'+s+c for s in side for c in polar]
    #derive the features    
    for f in features['grnd']:
        asl.df[f]=asl.df['raw'+f[-3:]] - asl.df['nose-'+f[-1:]]

    df_means = asl.df.groupby('speaker').mean()
    df_std = asl.df.groupby('speaker').std()

    for f in features['norm']:
        ref='raw'+f[-3:]
        asl.df[f]=(asl.df[ref]-asl.df['speaker'].map(df_means[ref]))/asl.df['speaker'].map(df_std[ref])
    
    for f in features['delta']:
        ref='grnd'+f[-3:]
        asl.df[f]=(asl.df[ref].diff()).fillna(0)
    
    ref='grnd'
    asl.df['polar-rtheta']=(np.arctan2(asl.df[ref+'-rx'],asl.df[ref+'-ry']))
    asl.df['polar-ltheta']=(np.arctan2(asl.df[ref+'-lx'],asl.df[ref+'-ly']))
    asl.df['polar-rr']=np.sqrt(asl.df[ref+'-rx']**2+asl.df[ref+'-ry']**2)
    asl.df['polar-lr']=np.sqrt(asl.df[ref+'-lx']**2+asl.df[ref+'-ly']**2)
    training={k:asl.build_training(v) for k,v in features.items()}
    
    xlens=training['grnd'].get_all_Xlengths()
    lens_stats=[(k,len(v[1]),min(v[1]),sum(v[1])/len(v[1]),max(v[1]),max(v[1])-min(v[1])) for k,v in xlens.items()]
    words_stats=pd.DataFrame.from_records(lens_stats,columns=['word','count','min','avg','max','range']).set_index('word')
    words_stats['spread']=words_stats['range']/words_stats['avg']
    
    #include all words 
    min_len=0    
    words=words_stats[words_stats['min']>min_len].sort_values(by='count',ascending=False).index.tolist()

    samples=dict()
    for f in features:
        samples[f]={k:get_word(training,features,f,k) for k in words}

    threshold=1e-9
    separated={k:([s for s in v if min(s.std())<threshold],[s for s in v if min(s.std())>threshold]) for k,v in samples['norm'].items()}
    separated_stats=pd.DataFrame.from_records({k:(len(v[0]),len(v[1])) for k,v in separated.items()}).T.rename(columns={0:'single',1:'double'})	
    return asl,features,training,samples,words_stats.join(separated_stats),separated

def make_scatter(df):
    fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(21,5))
    for ax in axes:
        ax.set_ylim([-3.0,3.0])
        ax.set_xlim([-3.0,3.0])
                                    
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
    for ax in axes:
        ax[0].set_ylim([-3.0,3.0])
        ax[1].set_ylim([-3.0,3.0])

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

def plot_states(m,df):
    from matplotlib import cm
    x=df.get_values()
    hidden_states = m.predict(x)    
    fig, axs = plt.subplots(m.n_components,2, sharex=True, sharey=True,figsize=(21,5))
    colours = cm.rainbow(np.linspace(0, 1, m.n_components))
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax[0].plot(df.iloc[:,0][mask], ".-", c='black')
        ax[0].plot(df.iloc[:,1][mask], ".-", c='red')
        ax[1].plot(df.iloc[:,2][mask], ".-", c='black')
        ax[1].plot(df.iloc[:,3][mask], ".-", c='red')
        ax[0].set_title("{0}th hidden state".format(i))
        ax[1].set_title("{0}th hidden state".format(i))


def show_model_stats(word, model):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))    
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])    
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        print()

def replicate_model():

    from hmmlearn import hmm
    np.random.seed(42)

    model = hmm.GaussianHMM(n_components=3, covariance_type="full")
    model.startprob_ = np.array([0.6, 0.3, 0.1])
    model.transmat_ = np.array([[0.7, 0.2, 0.1],
                                 [0.3, 0.5, 0.2],
                                 [0.3, 0.3, 0.4]])
    model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
    model.covars_ = np.tile(np.identity(2), (3, 1, 1))
    X, Y = model.sample(100)
    remodel = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    remodel.fit(X)
    Z=remodel.predict(X)
    res=pd.DataFrame({'original':Y, 'predicted':Z})
    res['correct']=res['original']==res['predicted']
    res.correct.sum()
    assert remodel.monitor_.converged
    np.set_printoptions(precision=3)

    for m in model,remodel:
        print(m.transmat_)

def plot_states(model,n_samples=100,fig_size=(12,6),lim=3):
    from matplotlib import patches 
    X,Z=model.sample(n_samples)
    L,R=pd.DataFrame({'x':X[:,0],'y':X[:,1]}),pd.DataFrame({'x':X[:,2],'y':X[:,3]})
    fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(12,6),sharey=True)
    axes[0].plot(L.x, L.y, ".-", color='grey', ms=6,mfc="black", alpha=0.7)
    axes[1].plot(R.x, R.y, ".-",color='grey', ms=6,mfc="black", alpha=0.7)

    for i, m in enumerate(model.means_):
        x,y=m[0],m[1]
        axes[0].text(m[0], m[1], 'S%i' % (i + 1),
                size=10, horizontalalignment='center',
                bbox=dict(alpha=.7, facecolor='w'))
        variance=np.array(np.diag(model.covars_[i]))
        ellipse=patches.Ellipse((m[0], m[1]), np.sqrt(variance[0]),np.sqrt(variance[1]), color='r')
        axes[0].add_artist(ellipse)

        axes[1].text(m[2], m[3], 'S%i' % (i + 1),
                size=10, horizontalalignment='center',
                bbox=dict(alpha=.7, facecolor='w'))
        ellipse=patches.Ellipse((m[2], m[3]), np.sqrt(variance[2]),np.sqrt(variance[3]), color='r')
        axes[1].add_artist(ellipse)
        
        axes[0].set_xlim(-lim,lim)
        axes[0].set_ylim(-lim,lim)

        axes[1].set_xlim(-lim,lim)
        axes[1].set_ylim(-lim,lim)
    return fig


