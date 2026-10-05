import statistics

from tqdm import tqdm
from ExperimentTSB import experiment




fstatic =open("./results/resultsKR.csv", "a+")
fdyn =open("./results/results.csv", "a+")


def rumsingleexpkr(parms):
    #(k,r,window,slide,slidingWindow,features,normalize,i,dataset)
    k = parms[0]
    r = parms[1]
    window = parms[2]
    slide=parms[3]
    slidingWindow=parms[4]
    features=parms[5]
    normalize=parms[6]
    i = parms[7]
    dataset = parms[8]

    try:
        datasetname = f'{dataset}{i}_data.out'
        exp = experiment(slide=slide, window=window, k=k, r=r, filepath=datasetname, slidingWindow=slidingWindow,
                         onlystatic=True, features=features, normalize=normalize)
        if LOGRESULTS:
            fstatic.write(exp.static_to_write)
        return exp.staticf1
    except Exception as e:
        #print(e)
        return None


# Run Distance based outlier detection for all data files in dataset
# using given parameters, for multiple k and R parameters
def check_many_kt(slide=100, window=200,slidingWindow=None,features=False,normalize=False,dataset="./data/YAHOO/Yahoo_A2synthetic_"):
    maxall=0
    keepmax=None
    krparams=[]
    for k in [5, 10, 15, 20,30,40]:
        for r in [0.01,0.1, 0.5, 1, 2,5,10]:
            krparams.append((k,r))

    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool(3)

    for qq in tqdm(range(len(krparams))):
        k=krparams[qq][0]
        r=krparams[qq][1]
        params=[]
        for i in range(100):
            params.append((k,r,window,slide,slidingWindow,features,normalize,i,dataset))
        results = pool.map(rumsingleexpkr, params)
        statallf1=[res for res in results if res is not None]
        if len(statallf1)==0:
            continue
        if maxall<sum(statallf1)/len(statallf1):
            maxall=sum(statallf1)/len(statallf1)
            keepmax=(k,r)
            print(maxall)
    print(maxall)
    print(keepmax)
    return keepmax

def onerun(slide=100, window=200,slidingWindow=None,keepmax=(10,0.5),debug=False,features=False,normalize=False,dataset="./data/YAHOO/Yahoo_A2synthetic_"):
    dynallf1 = []
    statallf1 = []
    counter = 0

    for i in tqdm(range(100)):
        try:

            datasetname = f'{dataset}{i}_data.out'
            exp = experiment(slide=slide, window=window, k=keepmax[0], r=keepmax[1], filepath=datasetname, slidingWindow=slidingWindow,debug=debug,features=features,normalize=normalize)

            if LOGRESULTS:
                fstatic.write(exp.static_to_write)
                fdyn.write(exp.dyn_to_write)

            dynallf1.append(exp.dynamicf1)
            statallf1.append(exp.staticf1)
            counter += 1
            # print(f"static: {exp.staticf1}")
            # print(f"dyn: {exp.dynamicf1}")
        except Exception as e:
            #print(str(e))
            continue

    print(f" dyn mean: {sum(dynallf1) / counter}")
    print(f" dyn median: {statistics.median(dynallf1)}")

    print(f" static mean: {sum(statallf1) / counter}")
    print(f" static median: {statistics.median(statallf1)}")

    print(dynallf1)
    print(statallf1)


# Run technique to all data under the folder of given dataset using given parameters
def testYahoo(slide=100, window=200,slidingWindow=None,features=False,normalize=False,dataset="./data/YAHOO/Yahoo_A1real_"):
    # datasetname = f'./data/YAHOO/Yahoo_A1real_{i}_data.out'
    # datasetname = f'./data/YAHOO/YahooA3Benchmark-TS{i}_data.out'
    # datasetname = f'./data/YAHOO/YahooA4Benchmark-TS{i}_data.out'
    #"./data/YAHOO/Yahoo_A1synthetic_"

    keepmax=check_many_kt(slide=slide, window=window,slidingWindow=slidingWindow,features=features,normalize=normalize,dataset=dataset)
    onerun(slide=slide, window=window, slidingWindow=slidingWindow, keepmax=keepmax,features=features,normalize=normalize,dataset=dataset)



def run_all_for_dataset(window,slide,dataset):

    # # no features no normalize
    # #testYahoo(slide=slide, window=window, slidingWindow=2, features=False, normalize=False, dataset=dataset)
    # #testYahoo(slide=slide, window=window, slidingWindow=None, features=False, normalize=False,dataset=dataset)
    # #testYahoo(slide=slide, window=window, slidingWindow=10, features=False, normalize=False,dataset=dataset)
    #
    # # no features  normalize
    testYahoo(slide=slide, window=window, slidingWindow=2, features=False, normalize=True, dataset=dataset)
    testYahoo(slide=slide, window=window, slidingWindow=None, features=False, normalize=True,dataset=dataset)
    testYahoo(slide=slide, window=window, slidingWindow=10, features=False, normalize=True,dataset=dataset)
    #
    # # features
    # #testYahoo(slide=slide, window=window, slidingWindow=None, features=True, normalize=True,dataset=dataset)
    # #testYahoo(slide=slide, window=window, slidingWindow=10, features=True, normalize=True,dataset=dataset)
    #
    # #feature no normalize
    #testYahoo(slide=slide, window=window, slidingWindow=None, features=True, normalize=False, dataset=dataset)
    #testYahoo(slide=slide, window=window, slidingWindow=10, features=True, normalize=False,dataset=dataset)


import time
def timecalculation(k=40, r=0.3, window=200, slide=100, slidingWindow=10, features=False, normalize=False):
    totalstatic=0
    totaldyn=0
    for dataset in ["./data/YAHOO/Yahoo_A1real_","./data/YAHOO/Yahoo_A2synthetic_","./data/YAHOO/YahooA3Benchmark-TS","./data/YAHOO/YahooA4Benchmark-TS"]:
        for i in tqdm(range(100)):
            try:

                datasetname = f'{dataset}{i}_data.out'
                exp = experiment(slide=slide, window=window, k=k, r=r, filepath=datasetname, slidingWindow=slidingWindow,
                         onlystatic=False, features=features, normalize=normalize)

                totalstatic+=exp.statictime
                totaldyn+=exp.dyntime
            except Exception as e:
                continue
    print(f"dyn: {totaldyn} s")
    print(f"static: {totalstatic} s")
    ftime=open("./results/times.txt", "a+")
    ftime.write(f"{window},{slide},{k},{r},{slidingWindow},{features},{normalize},{totalstatic},{totaldyn}\n")
LOGRESULTS=True

def rundynall():
    for dataset in ["./data/YAHOO/Yahoo_A1real_", "./data/YAHOO/Yahoo_A2synthetic_", "./data/YAHOO/YahooA4Benchmark-TS",
                    "./data/YAHOO/YahooA3Benchmark-TS"]:
        for window in [200, 400]:
            for slidingWindow in [2, 10, None]:
                for normalize in [False, True]:
                    onerun(slide=window // 2, window=window, slidingWindow=slidingWindow, keepmax=(40, 1), features=False,
                           normalize=normalize, dataset=dataset)
                    print("ok")
    for dataset in ["./data/YAHOO/Yahoo_A1real_", "./data/YAHOO/Yahoo_A2synthetic_", "./data/YAHOO/YahooA4Benchmark-TS",
                    "./data/YAHOO/YahooA3Benchmark-TS"]:
        for window in [200, 400]:
            for slidingWindow in [10, None]:
                for normalize in [False, True]:
                    onerun(slide=window // 2, window=window, slidingWindow=slidingWindow, keepmax=(40, 1), features=True,
                           normalize=normalize, dataset=dataset)
                    print("ok")

if __name__ == "__main__" :
    #rundynall()
    #### TEST time ####
    # timecalculation(k=40, r=0.3, window=200, slide=100, slidingWindow=2, features=False, normalize=False)
    # timecalculation(k=40, r=0.3, window=200, slide=100, slidingWindow=10, features=False, normalize=False)
    # timecalculation(k=40, r=0.3, window=200, slide=100, slidingWindow=50, features=False, normalize=False)
    # timecalculation(k=40, r=0.3, window=200, slide=100, slidingWindow=100, features=False, normalize=False)

    #timecalculation(k=40, r=0.3, window=100, slide=50, slidingWindow=50, features=False, normalize=False)
    # timecalculation(k=40, r=0.3, window=400, slide=200, slidingWindow=10, features=False, normalize=False)
    # timecalculation(k=40, r=0.3, window=400, slide=200, slidingWindow=50, features=False, normalize=False)
    # timecalculation(k=40, r=0.3, window=400, slide=200, slidingWindow=100, features=False, normalize=False)

    # timecalculation(k=40, r=0.3, window=200, slide=100, slidingWindow=10, features=False, normalize=False)
    # timecalculation(k=30, r=0.3, window=200, slide=100, slidingWindow=10, features=False, normalize=False)
    # timecalculation(k=20, r=0.3, window=200, slide=100, slidingWindow=10, features=False, normalize=False)
    # timecalculation(k=10, r=0.3, window=200, slide=100, slidingWindow=10, features=False, normalize=False)
    # timecalculation(k=5, r=0.3, window=200, slide=100, slidingWindow=10, features=False, normalize=False)

    ######## For multiple combinations ###################################3
    #run_all_for_dataset(200, 100, "./data/YAHOO/Yahoo_A1real_")
    #run_all_for_dataset(200, 100, "./data/YAHOO/Yahoo_A2synthetic_")
    #run_all_for_dataset(200, 100, "./data/YAHOO/YahooA3Benchmark-TS")
    #run_all_for_dataset(200, 100, "./data/YAHOO/YahooA4Benchmark-TS")


    #### FOR a single combination ####################################3
    # testYahoo(slide=200, window=400, slidingWindow=10,features=True,normalize=False,dataset="./data/YAHOO/Yahoo_A1synthetic_")


    ### FOR Multuple KR using Distance based on all data files in dataset:

    # check_many_kt(slide=200, window=400, slidingWindow=10, features=True, normalize=True,
    #               dataset="./data/YAHOO/Yahoo_A2synthetic_")
    #check_many_kt(slide=200, window=400, slidingWindow=10, features=True, normalize=True,dataset="./data/YAHOO/Yahoo_A2synthetic_")
    #check_many_kt(slide=200, window=400, slidingWindow=10, features=True, normalize=True,dataset="./data/YAHOO/YahooA3Benchmark-TS")
    #check_many_kt(slide=200, window=400, slidingWindow=10, features=True, normalize=True,dataset="./data/YAHOO/YahooA4Benchmark-TS")
    #check_many_kt(slide=200, window=400, slidingWindow=10, features=True, normalize=True,dataset="./data/YAHOO/Yahoo_A1real_")

    ### FOR a single parametrization of DynamicKR and Distance based K-R on all data files in dataset:
    #onerun(slide=100, window=200,slidingWindow=10,keepmax=(40,0.5),debug=False,features=False,normalize=False,dataset="./data/YAHOO/Yahoo_A2synthetic_")
    #onerun(slide=100, window=200,slidingWindow=10,keepmax=(40,1.0),debug=False,features=False,normalize=False,dataset="./data/YAHOO/Yahoo_A2synthetic_")
    #onerun(slide=100, window=200,slidingWindow=10,keepmax=(15,0.5),debug=False,features=False,normalize=False,dataset="./data/YAHOO/Yahoo_A2synthetic_")




    # A single Experiment for a single data file using Distance based and Dynamic technique:
    experiment(slide=100, window=200,slidingWindow=10 ,k=40, r=0.5, filepath="./data/YAHOO/Yahoo_A2synthetic_40_data.out",
               onlystatic=True, features=False, normalize=False)





