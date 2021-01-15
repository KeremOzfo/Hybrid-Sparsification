import numpy as np
import matplotlib.pyplot as plt
from os import *
import math
from itertools import cycle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import pandas as pd

def special_adress(loc):
    adress=[]
    labels = []
    adress_loss = []
    #labels = ['\u03B1=0.85', '\u03B1=0.90', '\u03B1=0.95','\u03B1=1']
    folder = 'Results/{}/'.format(loc)
    folder = 'Results'
    for dir in listdir(folder):
        adress.append('{}/{}/{}/'.format(folder,dir,'acc'))
        labels.append(dir)
    #labels = ['Phi power =2.5','Phi power = 3','Phi power =4']

    return adress,labels

def compile_results(adress,labels):
    results = None
    f_results = []
    total_files = len(listdir(adress))
    for i, dir in enumerate(listdir(adress)):
        if dir[0:3] != 'sim':
            vec = np.load(adress + '/'+dir)
            final_result = vec[len(vec)-1]
            f_results.append(final_result)
            if i==0:
                results = vec/total_files
            else:
               results += vec/total_files
    avg = np.average(f_results)
    st_dev = np.std(f_results)
    return results, labels,avg,st_dev

def cycle_graph_props(colors,markers,linestyles):
    randoms =[]
    randc = np.random.randint(0,len(colors))
    randm = np.random.randint(0,len(markers))
    randl = np.random.randint(0,len(linestyles))
    m = markers[randm]
    c = colors[randc]
    l = linestyles[randl]
    np.delete(colors,randc)
    np.delete(markers,randm)
    np.delete(linestyles,randl)
    print(colors,markers,linestyles)
    return c,m,l


def avgs(sets):
    avgs =[]
    for set in sets:
        avg = np.zeros_like(set[0])
        avgs.append(avg)
    return avgs

def graph(data, legends,interval):
    marker = ['s', 'v', '+', 'o', '*']
    linestyle =['-', '--', '-.', ':']
    linecycler = cycle(linestyle)
    markercycler = cycle(marker)
    final_vals = []
    final_comm = []
    for d,legend in zip(data,legends):
        x_axis = []
        l = next(linecycler)
        m = next(markercycler)
        for i in range(0,len(d)):
            x_axis.append(i*interval)
        final_vals.append(d[len(d)-1])
        final_comm.append(x_axis[len(x_axis)-1])
        plt.plot(x_axis,d, marker= m ,linestyle = l ,markersize=2, label=legend)
    #plt.axis([5, 45,70 ,90])
    #plt.axis([145,155,88,92])
    #plt.axis([290, 300, 87, 95])
    #plt.axis([50, 100, 87, 95])
    y_low = min(final_vals) - 6
    y_high = max(final_vals) + 1.5
    plt.axis([0, max(final_comm), y_low, y_high])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    #plt.title('Majority Voting')
    plt.legend()
    plt.grid(True)
    plt.show()


def concateresults(dirsets,make_table=True,excell=False):
    all_results =[]
    table_vals = {'sim':[],'avg_acc':[],'std_dev':[]}
    adresses, labels = dirsets
    for set, label in zip(adresses,labels):
        results, label, avg, st_dev = compile_results(set,label)
        all_results.append(results)
        table_vals['sim'].append(label)
        table_vals['avg_acc'].append(avg)
        table_vals['std_dev'].append(st_dev)
    if make_table:
        table = pd.DataFrame(data=table_vals)
        if excell:
            with pd.ExcelWriter('hybrid_results.xlsx',
                                mode='a') as writer:
                table.to_excel(writer, sheet_name='09')
        print(table)
    return all_results, labels



intervels = 1
adress = 'results'
results, labels = concateresults(special_adress(adress))
#results = concateresults(locations)
graph(results,labels,intervels)
#data,legends = compile_results(loc)
#graph(data,labels,intervels)